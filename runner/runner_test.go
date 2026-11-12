package runner_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/agent/agenttest"
	"github.com/yoren041102/rain/ext/tool/fromfunc"
	"github.com/yoren041102/rain/runner"
	"github.com/yoren041102/rain/schema"
)

func ids() func() string {
	var n atomic.Int64
	return func() string { return fmt.Sprintf("id%d", n.Add(1)) }
}

func collect(t *testing.T, s schema.Stream[*schema.Event]) []*schema.Event {
	t.Helper()
	evs, err := schema.Collect(s)
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}
	return evs
}

func final(evs []*schema.Event) *schema.Event {
	for i := len(evs) - 1; i >= 0; i-- {
		if evs[i].Final {
			return evs[i]
		}
	}
	return nil
}

type weatherIn struct {
	City string `json:"city"`
}

func newRunner(t *testing.T, a agent.Agent, opts ...runner.Option) *runner.Runner {
	t.Helper()
	r, err := runner.New(runner.Config{AppName: "test", Agent: a, IDFunc: ids()}, opts...)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestReActToolLoop(t *testing.T) {
	weather := fromfunc.New("get_weather", "weather", func(ctx context.Context, in weatherIn) (string, error) {
		return "sunny in " + in.City, nil
	})
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "get_weather", `{"city":"Beijing"}`))},
		{Message: schema.AssistantText("It is sunny.")},
	}}
	a, err := agent.NewReAct(agent.ReActConfig{Name: "assistant", Instruction: "be nice", Model: m, Tools: []schema.Tool{weather}})
	if err != nil {
		t.Fatal(err)
	}
	r := newRunner(t, a)
	evs := collect(t, r.Run(context.Background(), "u", "s", schema.UserText("weather?")))
	f := final(evs)
	if f == nil || f.Message.Text() != "It is sunny." {
		t.Fatalf("final = %+v", f)
	}
	// second request must contain system, user, assistant call, tool result
	req := m.Requests[1]
	if req.Messages[0].Role != schema.RoleSystem || req.Messages[0].Text() != "be nice" {
		t.Fatalf("system = %v", req.Messages[0])
	}
	last := req.Messages[len(req.Messages)-1]
	if last.Role != schema.RoleTool || !strings.Contains((&schema.Message{Content: last.ToolResultBlocks()[0].Content}).Text(), "Beijing") {
		t.Fatalf("tool result = %v", last)
	}
	// persisted
	sess, _ := r.Sessions().Get(context.Background(), "test", "u", "s")
	all, _ := agent.AllEvents(context.Background(), r.Sessions(), sess, "")
	if len(all) != 4 {
		t.Fatalf("persisted %d events", len(all))
	}
	for i, e := range all {
		if e.Seq != int64(i+1) {
			t.Fatalf("seq %d at %d", e.Seq, i)
		}
	}
}

func TestInvalidArgsReturnedToModel(t *testing.T) {
	weather := fromfunc.New("get_weather", "weather", func(ctx context.Context, in weatherIn) (string, error) {
		t.Fatal("tool must not run")
		return "", nil
	})
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "get_weather", `{"city":42}`))},
		{Message: schema.AssistantText("ok")},
	}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "a", Model: m, Tools: []schema.Tool{weather}})
	collect(t, newRunner(t, a).Run(context.Background(), "u", "s", schema.UserText("x")))
	last := m.Requests[1].Messages[len(m.Requests[1].Messages)-1]
	rb := last.ToolResultBlocks()[0]
	if !rb.IsError || !strings.Contains((&schema.Message{Content: rb.Content}).Text(), "expected string") {
		t.Fatalf("result = %+v", rb)
	}
}

func TestInterruptAndResume(t *testing.T) {
	var runs atomic.Int32
	confirm := fromfunc.New("delete", "delete a file", func(ctx context.Context, in struct {
		Path string `json:"path"`
	}) (string, error) {
		runs.Add(1)
		tc, _ := agent.ToolContextFrom(ctx)
		var ans struct{ OK bool }
		if resumed, err := tc.ResumeInput(&ans); err != nil {
			return "", err
		} else if !resumed {
			return "", agent.Interrupt(ctx, "confirm", map[string]string{"path": in.Path}, nil)
		}
		if !ans.OK {
			return "cancelled", nil
		}
		return "deleted " + in.Path, nil
	})
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "delete", `{"path":"/tmp/a"}`))},
		{Message: schema.AssistantText("done")},
	}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: m, Tools: []schema.Tool{confirm}})
	r := newRunner(t, a)
	ctx := context.Background()
	evs := collect(t, r.Run(ctx, "u", "s", schema.UserText("delete it")))
	last := evs[len(evs)-1]
	it := last.Actions.Interrupt
	if it == nil || it.ID != "root/delete:c1" || it.CallID != "c1" {
		t.Fatalf("interrupt = %+v", it)
	}
	if string(it.Payload) != `{"path":"/tmp/a"}` {
		t.Fatalf("payload = %s", it.Payload)
	}
	// unknown interrupt
	if _, err := schema.Collect(r.Resume(ctx, "u", "s", schema.ResumeData{InterruptID: "nope"})); !errors.Is(err, runner.ErrNoInterrupt) {
		t.Fatalf("unknown resume: %v", err)
	}
	evs = collect(t, r.Resume(ctx, "u", "s", schema.ResumeData{InterruptID: it.ID, Input: json.RawMessage(`{"OK":true}`)}))
	if f := final(evs); f == nil || f.Message.Text() != "done" {
		t.Fatalf("final after resume = %+v", evs)
	}
	if runs.Load() != 2 {
		t.Fatalf("tool ran %d times", runs.Load())
	}
	tr := m.Requests[1].Messages[len(m.Requests[1].Messages)-1].ToolResultBlocks()[0]
	if (&schema.Message{Content: tr.Content}).Text() != "deleted /tmp/a" {
		t.Fatalf("result = %+v", tr)
	}
	// resuming again fails: already handled
	if _, err := schema.Collect(r.Resume(ctx, "u", "s", schema.ResumeData{InterruptID: it.ID})); !errors.Is(err, runner.ErrNoInterrupt) {
		t.Fatalf("double resume: %v", err)
	}
}

func TestPartialResumeAndDirectResult(t *testing.T) {
	var runsA, runsB, runsOK atomic.Int32
	mk := func(name string, n *atomic.Int32) schema.Tool {
		return fromfunc.New(name, name, func(ctx context.Context, _ struct{}) (string, error) {
			n.Add(1)
			tc, _ := agent.ToolContextFrom(ctx)
			if !tc.Resuming() {
				return "", agent.Interrupt(ctx, "approve", nil, nil)
			}
			return name + " approved", nil
		})
	}
	okTool := fromfunc.New("ok", "ok", func(ctx context.Context, _ struct{}) (string, error) {
		runsOK.Add(1)
		return "fine", nil
	})
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "a", `{}`), agenttest.TC("c2", "b", `{}`), agenttest.TC("c3", "ok", `{}`))},
		{Message: schema.AssistantText("all done")},
	}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: m, Tools: []schema.Tool{mk("a", &runsA), mk("b", &runsB), okTool}})
	r := newRunner(t, a)
	ctx := context.Background()
	evs := collect(t, r.Run(ctx, "u", "s", schema.UserText("go")))
	last := evs[len(evs)-1]
	its := last.Actions.AllInterrupts()
	if len(its) != 2 {
		t.Fatalf("interrupts = %d", len(its))
	}
	if res := last.Message.ToolResultBlocks(); len(res) != 1 || res[0].CallID != "c3" {
		t.Fatalf("completed results = %+v", res)
	}
	// resume only a: b must re-trigger
	evs = collect(t, r.Resume(ctx, "u", "s", schema.ResumeData{InterruptID: "root/a:c1"}))
	last = evs[len(evs)-1]
	if its := last.Actions.AllInterrupts(); len(its) != 1 || its[0].ID != "root/b:c2" {
		t.Fatalf("re-triggered = %+v", its)
	}
	// resume b with a direct result: tool not re-run
	evs = collect(t, r.Resume(ctx, "u", "s", schema.ResumeData{InterruptID: "root/b:c2", ToolResult: schema.TextResult("external")}))
	if f := final(evs); f == nil || f.Message.Text() != "all done" {
		t.Fatalf("final = %+v", evs)
	}
	if runsA.Load() != 2 || runsB.Load() != 1 || runsOK.Load() != 1 {
		t.Fatalf("runs a=%d b=%d ok=%d", runsA.Load(), runsB.Load(), runsOK.Load())
	}
	// model saw all three results in call order
	msgs := m.Requests[1].Messages
	var got []string
	for _, mm := range msgs {
		for _, rb := range mm.ToolResultBlocks() {
			got = append(got, rb.CallID+"="+(&schema.Message{Content: rb.Content}).Text())
		}
	}
	if strings.Join(got, ",") != "c1=a approved,c2=external,c3=fine" {
		t.Fatalf("results = %v", got)
	}
}

func TestDanglingRepair(t *testing.T) {
	var replays atomic.Int32
	safe := fromfunc.New("safe", "", func(ctx context.Context, _ struct{}) (string, error) {
		replays.Add(1)
		return "replayed", nil
	}, fromfunc.ReplaySafe())
	unsafe := fromfunc.New("unsafe", "", func(ctx context.Context, _ struct{}) (string, error) {
		t.Fatal("unsafe tool must not be replayed")
		return "", nil
	})
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("ok")}}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: m, Tools: []schema.Tool{safe, unsafe}})
	store := agent.NewInMemorySessionStore()
	ctx := context.Background()
	s, _ := store.Create(ctx, "test", "u", "s", nil)
	// simulate a crash after the intent was persisted
	_ = store.AppendEvents(ctx, s, []*schema.Event{
		{ID: "e1", Author: "user", Message: schema.UserText("hi")},
		{ID: "e2", Author: "root", Message: agenttest.Call(agenttest.TC("c1", "safe", `{}`), agenttest.TC("c2", "unsafe", `{}`))},
	}, 0)
	r, _ := runner.New(runner.Config{AppName: "test", Agent: a, Sessions: store, IDFunc: ids()})
	collect(t, r.Run(ctx, "u", "s", schema.UserText("again")))
	if replays.Load() != 1 {
		t.Fatalf("replays = %d", replays.Load())
	}
	var res []string
	for _, mm := range m.Requests[0].Messages {
		for _, rb := range mm.ToolResultBlocks() {
			res = append(res, (&schema.Message{Content: rb.Content}).Text())
		}
	}
	if len(res) != 2 || res[0] != "replayed" || res[1] != runner.InterruptedText {
		t.Fatalf("repaired = %v", res)
	}
}

func TestTransferAndActiveAgent(t *testing.T) {
	childModel := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: schema.AssistantText("child answer 1")},
		{Message: schema.AssistantText("child answer 2")},
	}}
	child, _ := agent.NewReAct(agent.ReActConfig{Name: "child", Description: "helper", Model: childModel})
	rootModel := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("t1", agent.TransferToolName, `{"agent_name":"child"}`))},
	}}
	root, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: rootModel, SubAgents: []agent.Agent{child}})
	r := newRunner(t, root)
	ctx := context.Background()
	evs := collect(t, r.Run(ctx, "u", "s", schema.UserText("help")))
	f := final(evs)
	if f == nil || f.Author != "child" || f.Message.Text() != "child answer 1" {
		t.Fatalf("final = %+v", f)
	}
	// transfer tool offered with enum
	if len(rootModel.Requests[0].Tools) != 1 || !strings.Contains(string(rootModel.Requests[0].Tools[0].Parameters), `"child"`) {
		t.Fatalf("tools = %+v", rootModel.Requests[0].Tools)
	}
	// child sees root's call as a foreign user message, not an assistant one
	for _, mm := range childModel.Requests[0].Messages {
		if mm.Role == schema.RoleAssistant || mm.Role == schema.RoleTool {
			t.Fatalf("child saw foreign %s message", mm.Role)
		}
	}
	// next turn starts at child
	evs = collect(t, r.Run(ctx, "u", "s", schema.UserText("more")))
	if f := final(evs); f == nil || f.Author != "child" || f.Message.Text() != "child answer 2" {
		t.Fatalf("second final = %+v", f)
	}
}

func TestStopAndSteer(t *testing.T) {
	var r *runner.Runner
	step := 0
	tool := fromfunc.New("work", "", func(ctx context.Context, _ struct{}) (string, error) {
		step++
		if step == 1 {
			_ = r.Steer(ctx, "u", "s", schema.UserText("also do X"))
		}
		if step == 2 {
			_ = r.Stop(ctx, "u", "s")
		}
		return "worked", nil
	})
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "work", `{}`))},
		{Message: agenttest.Call(agenttest.TC("c2", "work", `{}`))},
		{Message: schema.AssistantText("finished")},
	}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: m, Tools: []schema.Tool{tool}})
	r = newRunner(t, a)
	ctx := context.Background()
	evs := collect(t, r.Run(ctx, "u", "s", schema.UserText("go")))
	last := evs[len(evs)-1]
	if last.Actions.Interrupt == nil || last.Actions.Interrupt.Reason != agent.ReasonStop {
		t.Fatalf("expected stop interrupt, got %+v", last)
	}
	// the steering message reached the second model call
	found := false
	for _, mm := range m.Requests[1].Messages {
		if mm.Role == schema.RoleUser && mm.Text() == "also do X" {
			found = true
		}
	}
	if !found {
		t.Fatal("steering message not in request")
	}
	if err := r.Steer(ctx, "u", "s", schema.UserText("x")); !errors.Is(err, runner.ErrNotRunning) {
		t.Fatalf("steer idle: %v", err)
	}
	evs = collect(t, r.Resume(ctx, "u", "s", schema.ResumeData{InterruptID: last.Actions.Interrupt.ID}))
	if f := final(evs); f == nil || f.Message.Text() != "finished" {
		t.Fatalf("after stop resume: %+v", evs)
	}
}

type recPlugin struct {
	name string
	log  *[]string
}

func (p recPlugin) Name() string                { return p.name }
func (p recPlugin) Init(context.Context) error  { *p.log = append(*p.log, "init "+p.name); return nil }
func (p recPlugin) Close(context.Context) error { *p.log = append(*p.log, "close "+p.name); return nil }
func (p recPlugin) BeforeRun(context.Context, *agent.Invocation) error {
	*p.log = append(*p.log, "before "+p.name)
	return nil
}
func (p recPlugin) AfterRun(context.Context, *agent.Invocation, error) {
	*p.log = append(*p.log, "after "+p.name)
}
func (p recPlugin) OnEvent(_ context.Context, _ *agent.Invocation, ev *schema.Event) (*schema.Event, error) {
	if p.name == "b" && ev.Message != nil && ev.Message.Text() == "drop me" {
		return nil, nil
	}
	return ev, nil
}

func TestPluginsAndEarlyBreak(t *testing.T) {
	var log []string
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("drop me")}}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: m})
	r := newRunner(t, a, runner.WithPlugin(recPlugin{"a", &log}, recPlugin{"b", &log}))
	ctx := context.Background()
	evs := collect(t, r.Run(ctx, "u", "s", schema.UserText("hi")))
	if len(evs) != 1 { // only the user event; the answer was dropped
		t.Fatalf("events = %d", len(evs))
	}
	// early break still runs AfterRun and releases the session
	m.Turns = []agenttest.Turn{{Message: schema.AssistantText("x")}, {Message: schema.AssistantText("y")}}
	for ev, err := range r.Run(ctx, "u", "s", schema.UserText("again")) {
		if err != nil {
			t.Fatal(err)
		}
		if ev.Author != schema.AuthorUser {
			break
		}
	}
	collect(t, r.Run(ctx, "u", "s", schema.UserText("third")))
	_ = r.Close(ctx)
	want := "init a,init b,before a,before b,after b,after a,before a,before b,after b,after a,before a,before b,after b,after a,close b,close a"
	if got := strings.Join(log, ","); got != want {
		t.Fatalf("log =\n%s\nwant\n%s", got, want)
	}
}

func TestMiddlewareOrderAndState(t *testing.T) {
	var order []string
	mw := func(name string) agent.ModelMiddleware {
		return func(next agent.ModelHandler) agent.ModelHandler {
			return func(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
				order = append(order, name)
				return next(ctx, inv, req)
			}
		}
	}
	setter := fromfunc.New("remember", "", func(ctx context.Context, in struct {
		V string `json:"v"`
	}) (string, error) {
		tc, _ := agent.ToolContextFrom(ctx)
		tc.SetState("fav", in.V)
		tc.SetState("temp:scratch", "x")
		return "ok", nil
	})
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "remember", `{"v":"blue"}`))},
		{Message: schema.AssistantText("noted")},
	}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: m, Tools: []schema.Tool{setter}, OutputKey: "answer",
		Middlewares: agent.Middlewares{Model: []agent.ModelMiddleware{mw("local")}}})
	r := newRunner(t, a, runner.WithModelMiddleware(mw("global1"), mw("global2")))
	ctx := context.Background()
	collect(t, r.Run(ctx, "u", "s", schema.UserText("my fav is blue")))
	if strings.Join(order[:3], ",") != "global1,global2,local" {
		t.Fatalf("order = %v", order)
	}
	s, _ := r.Sessions().Get(ctx, "test", "u", "s")
	if v, _ := s.State().Get("fav"); v != "blue" {
		t.Fatalf("fav = %v", v)
	}
	if v, _ := s.State().Get("answer"); v != "noted" {
		t.Fatalf("answer = %v", v)
	}
	if _, ok := s.State().Get("temp:scratch"); ok {
		t.Fatal("temp key persisted")
	}
}

func TestBusySession(t *testing.T) {
	block := make(chan struct{})
	started := make(chan struct{})
	tool := fromfunc.New("slow", "", func(ctx context.Context, _ struct{}) (string, error) {
		close(started)
		<-block
		return "ok", nil
	})
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "slow", `{}`))},
		{Message: schema.AssistantText("done")},
	}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: m, Tools: []schema.Tool{tool}})
	r := newRunner(t, a)
	ctx := context.Background()
	done := make(chan error)
	go func() {
		_, err := schema.Collect(r.Run(ctx, "u", "s", schema.UserText("go")))
		done <- err
	}()
	<-started
	if _, err := schema.Collect(r.Run(ctx, "u", "s", schema.UserText("again"))); !errors.Is(err, runner.ErrBusy) {
		t.Fatalf("concurrent run: %v", err)
	}
	close(block)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestDuplicateAgentNames(t *testing.T) {
	m := &agenttest.FakeModel{}
	c, _ := agent.NewReAct(agent.ReActConfig{Name: "x", Model: m})
	root, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: m, SubAgents: []agent.Agent{c, c}})
	if _, err := runner.New(runner.Config{Agent: root}); err == nil {
		t.Fatal("expected duplicate name error")
	}
}
