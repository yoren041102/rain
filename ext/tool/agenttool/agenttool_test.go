package agenttool_test

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/agent/agenttest"
	"github.com/yoren041102/rain/ext/tool/agenttool"
	"github.com/yoren041102/rain/ext/tool/fromfunc"
	"github.com/yoren041102/rain/runner"
	"github.com/yoren041102/rain/schema"
)

func ids() func() string {
	var n atomic.Int64
	return func() string { return fmt.Sprintf("id%d", n.Add(1)) }
}

func TestAgentAsTool(t *testing.T) {
	innerM := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("42")}}}
	inner, _ := agent.NewReAct(agent.ReActConfig{Name: "calc", Description: "calculator", Model: innerM})
	outerM := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "calc", `{"request":"6*7"}`))},
		{Message: schema.AssistantText("The answer is 42")},
	}}
	outer, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: outerM, Tools: []schema.Tool{agenttool.New(inner)}})
	r, _ := runner.New(runner.Config{Agent: outer, IDFunc: ids()})
	evs, err := schema.Collect(r.Run(context.Background(), "u", "s", schema.UserText("what is 6*7")))
	if err != nil {
		t.Fatal(err)
	}
	if evs[len(evs)-1].Message.Text() != "The answer is 42" {
		t.Fatalf("final = %v", evs[len(evs)-1].Message)
	}
	if innerM.Requests[0].Messages[len(innerM.Requests[0].Messages)-1].Text() != "6*7" {
		t.Fatalf("inner input = %v", innerM.Requests[0].Messages)
	}
	res := outerM.Requests[1].Messages[len(outerM.Requests[1].Messages)-1].ToolResultBlocks()[0]
	if (&schema.Message{Content: res.Content}).Text() != "42" {
		t.Fatalf("tool result = %+v", res)
	}
	// inner agent's events are not in the outer session log
	for _, e := range evs {
		if e.Author == "calc" {
			t.Fatal("inner events leaked")
		}
	}
}

func TestInnerTransferDoesNotAffectParent(t *testing.T) {
	leafM := &agenttest.FakeModel{Turns: []agenttest.Turn{{Message: schema.AssistantText("leaf says hi")}}}
	leaf, _ := agent.NewReAct(agent.ReActConfig{Name: "leaf", Model: leafM})
	innerM := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("t1", agent.TransferToolName, `{"agent_name":"leaf"}`))},
	}}
	inner, _ := agent.NewReAct(agent.ReActConfig{Name: "inner", Model: innerM, SubAgents: []agent.Agent{leaf}})
	outerM := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "inner", `{"request":"x"}`))},
		{Message: schema.AssistantText("outer done")},
	}}
	outer, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: outerM, Tools: []schema.Tool{agenttool.New(inner)}})
	r, _ := runner.New(runner.Config{Agent: outer, IDFunc: ids()})
	ctx := context.Background()
	evs, err := schema.Collect(r.Run(ctx, "u", "s", schema.UserText("go")))
	if err != nil {
		t.Fatal(err)
	}
	if f := evs[len(evs)-1]; f.Author != "root" || f.Message.Text() != "outer done" {
		t.Fatalf("final = %+v", f)
	}
	s, _ := r.Sessions().Get(ctx, "", "u", "s")
	if v, ok := s.State().Get(agent.KeyActiveAgent); ok {
		t.Fatalf("active agent changed to %v", v)
	}
}

func TestInterruptInsideAgentTool(t *testing.T) {
	confirm := fromfunc.New("confirm", "", func(ctx context.Context, _ struct{}) (string, error) {
		tc, _ := agent.ToolContextFrom(ctx)
		var ok bool
		if resumed, _ := tc.ResumeInput(&ok); !resumed {
			return "", agent.Interrupt(ctx, "confirm?", map[string]string{"q": "sure?"}, nil)
		}
		if ok {
			return "confirmed", nil
		}
		return "rejected", nil
	})
	innerM := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("i1", "confirm", `{}`))},
		{Message: schema.AssistantText("inner finished")},
	}}
	inner, _ := agent.NewReAct(agent.ReActConfig{Name: "inner", Model: innerM, Tools: []schema.Tool{confirm}})
	outerM := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "inner", `{"request":"do"}`))},
		{Message: schema.AssistantText("outer finished")},
	}}
	outer, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: outerM, Tools: []schema.Tool{agenttool.New(inner)}})
	r, _ := runner.New(runner.Config{Agent: outer, IDFunc: ids()})
	ctx := context.Background()
	evs, err := schema.Collect(r.Run(ctx, "u", "s", schema.UserText("go")))
	if err != nil {
		t.Fatal(err)
	}
	it := evs[len(evs)-1].Actions.Interrupt
	if it == nil || it.ID != "root/inner:c1" || !strings.Contains(string(it.Payload), "sure?") {
		t.Fatalf("interrupt = %+v", it)
	}
	evs, err = schema.Collect(r.Resume(ctx, "u", "s", schema.ResumeData{InterruptID: it.ID, Input: json.RawMessage(`true`)}))
	if err != nil {
		t.Fatal(err)
	}
	if f := evs[len(evs)-1]; f.Message.Text() != "outer finished" {
		t.Fatalf("final = %+v", f.Message)
	}
	inRes := innerM.Requests[1].Messages[len(innerM.Requests[1].Messages)-1].ToolResultBlocks()[0]
	if (&schema.Message{Content: inRes.Content}).Text() != "confirmed" {
		t.Fatalf("inner tool result = %+v", inRes)
	}
}
