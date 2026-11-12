package schema

import (
	"encoding/json"
	"time"
)

// Event is the unit of output for agents and the unit of persistence for
// sessions.
type Event struct {
	Seq          int64             `json:"seq"`
	ID           string            `json:"id"`
	InvocationID string            `json:"invocation_id,omitempty"`
	Author       string            `json:"author"`
	Branch       string            `json:"branch,omitempty"`
	Message      *Message          `json:"message,omitempty"`
	Partial      bool              `json:"partial,omitempty"`
	Final        bool              `json:"final,omitempty"`
	Actions      EventActions      `json:"actions"`
	Usage        *Usage            `json:"usage,omitempty"`
	TraceCarrier map[string]string `json:"trace_carrier,omitempty"`
	CreatedAt    time.Time         `json:"created_at"`
}

// EventActions carries side effects requested by an event.
type EventActions struct {
	StateDelta        map[string]any `json:"state_delta,omitempty"`
	TransferTo        string         `json:"transfer_to,omitempty"`
	Escalate          bool           `json:"escalate,omitempty"`
	SkipSummarization bool           `json:"skip_summarization,omitempty"`
	Interrupt         *Interrupt     `json:"interrupt,omitempty"`
	// Interrupts holds additional interrupts raised in the same pause
	// (parallel tool calls). Interrupt is always Interrupts[0] when set.
	Interrupts []*Interrupt `json:"interrupts,omitempty"`
	// Resumed lists interrupt IDs handled by this event (written by runner).
	Resumed []ResumeData `json:"resumed,omitempty"`
	// Compaction is set on events authored by CompactionAuthor.
	Compaction *Compaction `json:"compaction,omitempty"`
	// DiscardPartial tells consumers to drop partial chunks received so far
	// for the current model call (e.g. a retry is about to start).
	DiscardPartial bool `json:"discard_partial,omitempty"`
}

// AllInterrupts returns every interrupt carried by the actions.
func (a *EventActions) AllInterrupts() []*Interrupt {
	if len(a.Interrupts) > 0 {
		return a.Interrupts
	}
	if a.Interrupt != nil {
		return []*Interrupt{a.Interrupt}
	}
	return nil
}

// Interrupt describes a paused execution point.
type Interrupt struct {
	ID      string          `json:"id"`
	Path    []string        `json:"path,omitempty"`
	CallID  string          `json:"call_id,omitempty"`
	Reason  string          `json:"reason,omitempty"`
	Payload json.RawMessage `json:"payload,omitempty"`
	State   json.RawMessage `json:"state,omitempty"`
}

// ResumeData is the caller's answer to an Interrupt.
type ResumeData struct {
	InterruptID string          `json:"interrupt_id"`
	Input       json.RawMessage `json:"input,omitempty"`
	ToolResult  *ToolResult     `json:"tool_result,omitempty"`
}

// Compaction records a persisted history summary.
type Compaction struct {
	Summary      string `json:"summary"`
	FirstKeptSeq int64  `json:"first_kept_seq"`
}

// Reserved authors.
const (
	AuthorUser       = "user"
	CompactionAuthor = "_rain.compaction"
	SystemAuthor     = "_rain"
)

// Usage reports token consumption.
type Usage struct {
	InputTokens      int `json:"input_tokens,omitempty"`
	OutputTokens     int `json:"output_tokens,omitempty"`
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	ReasoningTokens  int `json:"reasoning_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}

// Add accumulates o into u.
func (u *Usage) Add(o *Usage) {
	if o == nil {
		return
	}
	u.InputTokens += o.InputTokens
	u.OutputTokens += o.OutputTokens
	u.CacheReadTokens += o.CacheReadTokens
	u.CacheWriteTokens += o.CacheWriteTokens
	u.ReasoningTokens += o.ReasoningTokens
	u.TotalTokens += o.TotalTokens
}

// Clone returns a copy of the event; the message is cloned, state delta is
// shallow-copied.
func (e *Event) Clone() *Event {
	if e == nil {
		return nil
	}
	c := *e
	c.Message = e.Message.Clone()
	if e.Actions.StateDelta != nil {
		c.Actions.StateDelta = make(map[string]any, len(e.Actions.StateDelta))
		for k, v := range e.Actions.StateDelta {
			c.Actions.StateDelta[k] = v
		}
	}
	if e.Usage != nil {
		u := *e.Usage
		c.Usage = &u
	}
	return &c
}
