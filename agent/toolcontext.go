package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/yoren041102/rain/schema"
)

// ToolContext is available to tools via ToolContextFrom.
type ToolContext struct {
	Invocation *Invocation
	CallID     string
	ToolName   string
	// Actions lets tools request StateDelta, TransferTo, Escalate and
	// SkipSummarization.
	Actions *schema.EventActions
	// Interrupt is the interrupt this call raised earlier, when resuming.
	Interrupt *schema.Interrupt
	// Resume is the caller's answer, when resuming.
	Resume *schema.ResumeData
}

type toolCtxKey struct{}

// WithToolContext attaches tc to ctx.
func WithToolContext(ctx context.Context, tc *ToolContext) context.Context {
	return context.WithValue(ctx, toolCtxKey{}, tc)
}

// ToolContextFrom returns the ToolContext of the current tool call.
func ToolContextFrom(ctx context.Context) (*ToolContext, bool) {
	tc, ok := ctx.Value(toolCtxKey{}).(*ToolContext)
	return tc, ok && tc != nil
}

// SetState writes a session state key; the write is reported on the tool
// result event.
func (tc *ToolContext) SetState(key string, v any) {
	if tc.Actions.StateDelta == nil {
		tc.Actions.StateDelta = map[string]any{}
	}
	tc.Actions.StateDelta[key] = v
}

// GetState reads a session state key, seeing this call's own writes.
func (tc *ToolContext) GetState(key string) (any, bool) {
	if v, ok := tc.Actions.StateDelta[key]; ok {
		return v, v != nil
	}
	if tc.Invocation != nil && tc.Invocation.State != nil {
		return tc.Invocation.State.Get(key)
	}
	return nil, false
}

// Resuming reports whether this call is a re-execution after an interrupt.
func (tc *ToolContext) Resuming() bool { return tc.Resume != nil }

// ResumeInput decodes the caller's resume input into v. It returns false if
// the call is not resuming.
func (tc *ToolContext) ResumeInput(v any) (bool, error) {
	if tc.Resume == nil {
		return false, nil
	}
	if len(tc.Resume.Input) == 0 {
		return true, nil
	}
	return true, json.Unmarshal(tc.Resume.Input, v)
}

// InterruptState decodes the private state saved by the earlier interrupt.
func (tc *ToolContext) InterruptState(v any) (bool, error) {
	if tc.Interrupt == nil || len(tc.Interrupt.State) == 0 {
		return false, nil
	}
	return true, json.Unmarshal(tc.Interrupt.State, v)
}

// InterruptError is returned by tools (and agents) to pause execution.
type InterruptError struct {
	Reason  string
	Payload json.RawMessage
	State   json.RawMessage
}

func (e *InterruptError) Error() string { return "rain: interrupt: " + e.Reason }

// Interrupt builds an interrupt error. Tools return it to pause execution;
// payload is shown to the caller and state is handed back on resume. Both
// are JSON-encoded (json.RawMessage is used as-is).
func Interrupt(ctx context.Context, reason string, payload, state any) error {
	p, err := toRaw(payload)
	if err != nil {
		return fmt.Errorf("rain: encode interrupt payload: %w", err)
	}
	s, err := toRaw(state)
	if err != nil {
		return fmt.Errorf("rain: encode interrupt state: %w", err)
	}
	return &InterruptError{Reason: reason, Payload: p, State: s}
}

func toRaw(v any) (json.RawMessage, error) {
	switch x := v.(type) {
	case nil:
		return nil, nil
	case json.RawMessage:
		return x, nil
	case []byte:
		return json.RawMessage(x), nil
	}
	return json.Marshal(v)
}

// AsInterrupt reports whether err is (or wraps) an InterruptError.
func AsInterrupt(err error) (*InterruptError, bool) {
	var ie *InterruptError
	if errors.As(err, &ie) {
		return ie, true
	}
	return nil, false
}

// InterruptID builds the hierarchical interrupt address
// "root/child/tool:callID" (or "root/child" + suffix for agent-level ones).
func InterruptID(path []string, suffix string) string {
	base := strings.Join(path, "/")
	if suffix == "" {
		return base
	}
	return base + "/" + suffix
}

// NewInterrupt converts an InterruptError raised at the current agent into a
// schema.Interrupt. toolName/callID may be empty for agent-level interrupts.
func (inv *Invocation) NewInterrupt(ie *InterruptError, toolName, callID string) *schema.Interrupt {
	path := inv.Path()
	suffix := ""
	if callID != "" {
		suffix = toolName + ":" + callID
	} else if ie.Reason != "" {
		suffix = "_" + ie.Reason
	}
	return &schema.Interrupt{
		ID:      InterruptID(path, suffix),
		Path:    path,
		CallID:  callID,
		Reason:  ie.Reason,
		Payload: ie.Payload,
		State:   ie.State,
	}
}

// InterruptEvent builds an event that pauses execution with its.
func (inv *Invocation) InterruptEvent(msg *schema.Message, its ...*schema.Interrupt) *schema.Event {
	ev := inv.NewEvent(msg)
	if len(its) > 0 {
		ev.Actions.Interrupt = its[0]
		if len(its) > 1 {
			ev.Actions.Interrupts = its
		}
	}
	return ev
}

// RouteResume forwards a resuming invocation toward the agent that raised
// the interrupts. It returns ok == false when no open interrupt lies strictly
// below the current agent, in which case the caller handles execution itself.
func RouteResume(ctx context.Context, inv *Invocation) (schema.Stream[*schema.Event], bool) {
	if !inv.Resuming() || inv.Agent == nil {
		return nil, false
	}
	self := inv.Agent.Name()
	if len(inv.InterruptsFor(self)) > 0 {
		return nil, false
	}
	for _, it := range inv.OpenInterrupts {
		for i, p := range it.Path {
			if p != self || i+1 >= len(it.Path) {
				continue
			}
			next := it.Path[i+1]
			for _, c := range inv.Agent.SubAgents() {
				if c.Name() == next {
					return RunAgent(ctx, inv, c), true
				}
			}
			if inv.Tree != nil {
				if c := inv.Tree.Find(next); c != nil {
					return RunAgent(ctx, inv, c), true
				}
			}
		}
	}
	return nil, false
}
