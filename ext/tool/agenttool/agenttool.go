// Package agenttool wraps an agent.Agent as a schema.Tool. The wrapped agent
// runs under the caller's invocation on an isolated branch, and its final
// message becomes the tool result. Transfer and escalate actions of the
// inner agent never affect the caller's control flow.
package agenttool

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
)

// Option configures the tool.
type Option func(*Tool)

// WithSkipSummarization makes the inner agent's answer the caller's final
// output without another model call.
func WithSkipSummarization() Option { return func(t *Tool) { t.skipSummarization = true } }

// WithSchema sets the input schema; by default the tool takes {"request": string}.
func WithSchema(s json.RawMessage) Option { return func(t *Tool) { t.params = s } }

// WithEventSink receives every non-partial event the inner agent produces.
func WithEventSink(f func(*schema.Event)) Option { return func(t *Tool) { t.sink = f } }

// Tool is an agent exposed as a tool.
type Tool struct {
	a                 agent.Agent
	params            json.RawMessage
	skipSummarization bool
	sink              func(*schema.Event)
}

// New wraps a.
func New(a agent.Agent, opts ...Option) *Tool {
	t := &Tool{a: a}
	for _, o := range opts {
		o(t)
	}
	if t.params == nil {
		t.params = json.RawMessage(`{"type":"object","properties":{"request":{"type":"string","description":"The request for the agent."}},"required":["request"]}`)
	}
	return t
}

func (t *Tool) Spec() schema.ToolSpec {
	return schema.ToolSpec{Name: t.a.Name(), Description: t.a.Description(), Parameters: t.params}
}

// Call runs the agent. Interrupts raised inside propagate as an interrupt of
// this tool call; the inner interrupt is stored in the interrupt state and
// re-delivered on resume.
func (t *Tool) Call(ctx context.Context, args json.RawMessage) (*schema.ToolResult, error) {
	tc, ok := agent.ToolContextFrom(ctx)
	if !ok || tc.Invocation == nil {
		return nil, errors.New("agenttool: must be called by an agent (no ToolContext)")
	}
	parent := tc.Invocation
	input := string(args)
	var req struct {
		Request string `json:"request"`
	}
	if json.Unmarshal(args, &req) == nil && req.Request != "" {
		input = req.Request
	}
	userMsg := schema.UserText(input)

	type saved struct {
		Events     []*schema.Event     `json:"events"`
		Interrupts []*schema.Interrupt `json:"interrupts"`
	}
	hist := agent.NewBufferHistory()
	inv := parent.WithAgent(t.a).WithBranch(parent.ChildBranch(t.a.Name() + ":" + tc.CallID))
	inv.UserMessage = userMsg
	inv.History = hist
	inv.State = agent.NewDeltaState(parent.State)
	inv.Resume, inv.OpenInterrupts = nil, nil
	inv.Steering = nil

	var st saved
	if found, err := tc.InterruptState(&st); err != nil {
		return nil, err
	} else if found && tc.Resume != nil {
		hist.Append(st.Events...)
		inv.OpenInterrupts = st.Interrupts
		var inner []schema.ResumeData
		if len(tc.Resume.Input) > 0 && json.Unmarshal(tc.Resume.Input, &inner) == nil && len(inner) > 0 {
			inv.Resume = inner
		} else {
			// forward the same answer to every inner interrupt
			for _, it := range st.Interrupts {
				d := *tc.Resume
				d.InterruptID = it.ID
				inv.Resume = append(inv.Resume, d)
			}
		}
	} else {
		ev := inv.NewEvent(userMsg)
		ev.Author = schema.AuthorUser
		hist.Append(ev)
	}

	var last *schema.Message
	var all []*schema.Event
	evs, _ := hist.Events(ctx)
	all = append(all, evs...)
	for ev, err := range agent.RunAgent(ctx, inv, t.a) {
		if err != nil {
			return nil, err
		}
		if ev.Partial {
			continue
		}
		hist.Append(ev)
		all = append(all, ev)
		if t.sink != nil {
			t.sink(ev)
		}
		if len(ev.Actions.StateDelta) > 0 {
			for k, v := range ev.Actions.StateDelta {
				tc.SetState(k, v)
			}
		}
		if its := ev.Actions.AllInterrupts(); len(its) > 0 {
			payload, _ := json.Marshal(its)
			return nil, agent.Interrupt(ctx, "agent_tool", json.RawMessage(payload), saved{Events: all, Interrupts: its})
		}
		if ev.Message != nil && ev.Message.Role == schema.RoleAssistant {
			last = ev.Message
		}
		if ev.Final && ev.Message != nil && ev.Message.Role == schema.RoleTool {
			last = ev.Message
		}
	}
	if t.skipSummarization {
		tc.Actions.SkipSummarization = true
	}
	if last == nil {
		return schema.TextResult(""), nil
	}
	if last.Role == schema.RoleTool {
		var blocks []schema.ContentBlock
		for _, r := range last.ToolResultBlocks() {
			blocks = append(blocks, r.Content...)
		}
		return &schema.ToolResult{Content: blocks}, nil
	}
	var blocks []schema.ContentBlock
	for _, b := range last.Content {
		switch b.(type) {
		case *schema.TextBlock, *schema.ImageBlock, *schema.FileBlock:
			blocks = append(blocks, b)
		}
	}
	return &schema.ToolResult{Content: blocks}, nil
}
