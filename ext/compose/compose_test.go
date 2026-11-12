package compose_test

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/agent/agenttest"
	"github.com/yoren041102/rain/ext/compose"
	"github.com/yoren041102/rain/ext/tool/fromfunc"
	"github.com/yoren041102/rain/runner"
	"github.com/yoren041102/rain/schema"
)

func ids() func() string {
	var n atomic.Int64
	return func() string { return fmt.Sprintf("id%d", n.Add(1)) }
}

func react(t *testing.T, name string, m schema.Model, tools ...schema.Tool) agent.Agent {
	t.Helper()
	a, err := agent.NewReAct(agent.ReActConfig{Name: name, Model: m, Tools: tools})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func texts(evs []*schema.Event) []string {
	var out []string
	for _, e := range evs {
		if !e.Partial && e.Message != nil && e.Author != schema.AuthorUser && e.Message.Role == schema.RoleAssistant {
			out = append(out, e.Author+":"+e.Message.Text())
		}
	}
	return out
}

func TestSequential(t *testing.T) {
	m2 := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("two")}}}
	seq := compose.Sequential("seq", "", react(t, "a", &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("one")}}}), react(t, "b", m2))
	r, _ := runner.New(runner.Config{Agent: seq, IDFunc: ids()})
	evs, err := schema.Collect(r.Run(context.Background(), "u", "s", schema.UserText("go")))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(texts(evs), ","); got != "a:one,b:two" {
		t.Fatalf("got %s", got)
	}
	// b sees a's output as context
	found := false
	for _, m := range m2.Requests[0].Messages {
		if strings.Contains(m.Text(), "[a] said: one") {
			found = true
		}
	}
	if !found {
		t.Fatal("b did not see a's output")
	}
}

func TestParallelBranches(t *testing.T) {
	ma := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("A")}}}
	mb := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("B")}}}
	par := compose.Parallel("par", "", react(t, "a", ma), react(t, "b", mb))
	r, _ := runner.New(runner.Config{Agent: par, IDFunc: ids()})
	evs, err := schema.Collect(r.Run(context.Background(), "u", "s", schema.UserText("go")))
	if err != nil {
		t.Fatal(err)
	}
	got := texts(evs)
	sort.Strings(got)
	if strings.Join(got, ",") != "a:A,b:B" {
		t.Fatalf("got %v", got)
	}
	for _, e := range evs {
		if e.Author == "a" && e.Branch != "par.a" {
			t.Fatalf("branch = %q", e.Branch)
		}
	}
}

func TestParallelEarlyBreak(t *testing.T) {
	block := make(chan struct{})
	slow := schema.ModelFunc{ModelName: "slow", Fn: func(ctx context.Context, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
		return func(yield func(*schema.ModelResponse, error) bool) {
			select {
			case <-block:
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return
			}
			yield(&schema.ModelResponse{Message: schema.AssistantText("late")}, nil)
		}
	}}
	fast := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("fast")}}}
	par := compose.Parallel("par", "", react(t, "slow", slow), react(t, "fast", fast))
	r, _ := runner.New(runner.Config{Agent: par, IDFunc: ids()})
	for ev, err := range r.Run(context.Background(), "u", "s", schema.UserText("go")) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Author == "fast" {
			break // must cancel and join the slow goroutine
		}
	}
	close(block)
}

func TestLoopEscalate(t *testing.T) {
	var n atomic.Int32
	done := fromfunc.New("done", "", func(ctx context.Context, _ struct{}) (string, error) {
		tc, _ := agent.ToolContextFrom(ctx)
		if n.Add(1) == 2 {
			tc.Actions.Escalate = true
		}
		return "checked", nil
	})
	m := &agenttest.FakeModel{Fn: func(req *schema.ModelRequest) (*schema.Message, error) {
		last := req.Messages[len(req.Messages)-1]
		if last.Role == schema.RoleTool {
			return schema.AssistantText("iter"), nil
		}
		return agenttest.Call(agenttest.TC(fmt.Sprint("c", len(req.Messages)), "done", `{}`)), nil
	}}
	loop := compose.Loop("loop", "", 10, react(t, "worker", m, done))
	r, _ := runner.New(runner.Config{Agent: loop, IDFunc: ids()})
	if _, err := schema.Collect(r.Run(context.Background(), "u", "s", schema.UserText("go"))); err != nil {
		t.Fatal(err)
	}
	if n.Load() != 2 {
		t.Fatalf("tool ran %d times", n.Load())
	}
}

func TestResumeThroughSequential(t *testing.T) {
	approve := fromfunc.New("approve", "", func(ctx context.Context, _ struct{}) (string, error) {
		tc, _ := agent.ToolContextFrom(ctx)
		if !tc.Resuming() {
			return "", agent.Interrupt(ctx, "need approval", nil, nil)
		}
		return "approved", nil
	})
	mFirst := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("first")}}}
	mB := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "approve", `{}`))},
		{Message: schema.AssistantText("b done")},
	}}
	mC := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("c done")}}}
	seq := compose.Sequential("seq", "", react(t, "a", mFirst), react(t, "b", mB, approve), react(t, "c", mC))
	r, _ := runner.New(runner.Config{Agent: seq, IDFunc: ids()})
	ctx := context.Background()
	evs, err := schema.Collect(r.Run(ctx, "u", "s", schema.UserText("go")))
	if err != nil {
		t.Fatal(err)
	}
	it := evs[len(evs)-1].Actions.Interrupt
	if it == nil || it.ID != "seq/b/approve:c1" {
		t.Fatalf("interrupt = %+v", it)
	}
	evs, err = schema.Collect(r.Resume(ctx, "u", "s", schema.ResumeData{InterruptID: it.ID, Input: json.RawMessage(`true`)}))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(texts(evs), ","); got != "b:b done,c:c done" {
		t.Fatalf("after resume: %s", got)
	}
}
