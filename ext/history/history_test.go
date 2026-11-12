package history_test

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/agent/agenttest"
	"github.com/yoren041102/rain/ext/history"
	"github.com/yoren041102/rain/runner"
	"github.com/yoren041102/rain/schema"
)

func pass(got **schema.ModelRequest) agent.ModelHandler {
	return func(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
		*got = req
		return schema.FromSlice(&schema.ModelResponse{Message: schema.AssistantText("ok")})
	}
}

func conv() []*schema.Message {
	return []*schema.Message{
		schema.SystemText("sys"),
		schema.UserText("u1"),
		agenttest.Call(agenttest.TC("c1", "t", `{}`)),
		schema.ToolResults(&schema.ToolResultBlock{CallID: "c1"}),
		schema.AssistantText("a1"),
		schema.UserText("u2"),
		schema.AssistantText("a2"),
		schema.UserText("u3"),
	}
}

func TestKeepLastCutsAtTurnBoundary(t *testing.T) {
	var got *schema.ModelRequest
	h := history.KeepLast(4)(pass(&got))
	schema.Collect(h(context.Background(), nil, &schema.ModelRequest{Messages: conv()}))
	var roles []string
	for _, m := range got.Messages {
		roles = append(roles, m.Text())
	}
	// the last 4 would start at a1; the cut moves forward to the u2 turn
	if strings.Join(roles, ",") != "sys,u2,a2,u3" {
		t.Fatalf("got %v", roles)
	}
	if got.Messages[0].Role != schema.RoleSystem {
		t.Fatal("system dropped")
	}
	for i, m := range got.Messages {
		if m.Role == schema.RoleTool && (i == 0 || got.Messages[i-1].Role != schema.RoleAssistant) {
			t.Fatal("orphan tool result")
		}
	}
}

func TestTokenBudget(t *testing.T) {
	var got *schema.ModelRequest
	count := func(m *schema.Message) int { return 10 }
	h := history.TokenBudget(45, count)(pass(&got))
	schema.Collect(h(context.Background(), nil, &schema.ModelRequest{Messages: conv()}))
	var texts []string
	for _, m := range got.Messages {
		texts = append(texts, m.Text())
	}
	if strings.Join(texts, ",") != "sys,u2,a2,u3" {
		t.Fatalf("got %v", texts)
	}
}

func TestSummarizePersistsCompaction(t *testing.T) {
	var n atomic.Int64
	idf := func() string { return fmt.Sprintf("id%d", n.Add(1)) }
	summ := &agenttest.FakeModel{Fn: func(req *schema.ModelRequest) (*schema.Message, error) {
		return schema.AssistantText("SUMMARY"), nil
	}}
	main := &agenttest.FakeModel{Fn: func(req *schema.ModelRequest) (*schema.Message, error) {
		return schema.AssistantText("reply " + req.Messages[len(req.Messages)-1].Text()), nil
	}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: main})
	count := func(m *schema.Message) int { return 10 }
	r, _ := runner.New(runner.Config{Agent: a, IDFunc: idf},
		runner.WithModelMiddleware(history.Summarize(summ, 45, history.WithCounter(count), history.WithKeepTurns(1))))
	ctx := context.Background()
	for i := 1; i <= 4; i++ {
		if _, err := schema.Collect(r.Run(ctx, "u", "s", schema.UserText(fmt.Sprint("q", i)))); err != nil {
			t.Fatal(err)
		}
	}
	if len(summ.Requests) == 0 {
		t.Fatal("summarizer never called")
	}
	s, _ := r.Sessions().Get(ctx, "", "u", "s")
	evs, _ := agent.AllEvents(ctx, r.Sessions(), s, "")
	var comp *schema.Event
	for _, e := range evs {
		if e.Author == schema.CompactionAuthor {
			comp = e
		}
	}
	if comp == nil || comp.Actions.Compaction.Summary != "SUMMARY" || comp.Actions.Compaction.FirstKeptSeq == 0 {
		t.Fatalf("compaction event = %+v", comp)
	}
	last := main.Requests[len(main.Requests)-1]
	if !strings.Contains(last.Messages[0].Text(), "SUMMARY") {
		t.Fatalf("last request does not start with summary: %v", last.Messages)
	}
	for _, m := range last.Messages {
		if m.Text() == "q1" {
			t.Fatal("q1 should have been compacted away")
		}
	}
}
