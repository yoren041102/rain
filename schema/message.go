package schema

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Role is the author role of a message.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// StopReason records why a model stopped producing a message.
type StopReason string

const (
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "tool_use"
	StopReasonError   StopReason = "error"
	StopReasonAborted StopReason = "aborted"
)

// Origin records which provider / API / model produced a message.
type Origin struct {
	Provider string `json:"provider,omitempty"`
	API      string `json:"api,omitempty"`
	Model    string `json:"model,omitempty"`
}

// Same reports whether o and other identify the same provider, API and model.
func (o *Origin) Same(other Origin) bool {
	return o != nil && o.Provider == other.Provider && o.API == other.API && o.Model == other.Model
}

// Message is a single conversation turn.
type Message struct {
	Role       Role
	Content    []ContentBlock
	Origin     *Origin    // set by model adapters
	StopReason StopReason // set by model adapters
}

type messageJSON struct {
	Role       Role              `json:"role"`
	Content    []json.RawMessage `json:"content"`
	Origin     *Origin           `json:"origin,omitempty"`
	StopReason StopReason        `json:"stop_reason,omitempty"`
}

// MarshalJSON implements json.Marshaler.
func (m Message) MarshalJSON() ([]byte, error) {
	content, err := marshalBlocks(m.Content)
	if err != nil {
		return nil, err
	}
	return json.Marshal(messageJSON{Role: m.Role, Content: content, Origin: m.Origin, StopReason: m.StopReason})
}

// UnmarshalJSON implements json.Unmarshaler.
func (m *Message) UnmarshalJSON(data []byte) error {
	var mj messageJSON
	if err := json.Unmarshal(data, &mj); err != nil {
		return err
	}
	blocks, err := unmarshalBlocks(mj.Content)
	if err != nil {
		return err
	}
	*m = Message{Role: mj.Role, Content: blocks, Origin: mj.Origin, StopReason: mj.StopReason}
	return nil
}

// Clone returns a deep-enough copy of m: the block slice is copied, blocks
// themselves are shallow-copied.
func (m *Message) Clone() *Message {
	if m == nil {
		return nil
	}
	c := *m
	c.Content = make([]ContentBlock, len(m.Content))
	for i, b := range m.Content {
		c.Content[i] = cloneBlock(b)
	}
	if m.Origin != nil {
		o := *m.Origin
		c.Origin = &o
	}
	return &c
}

// Text concatenates every TextBlock of the message.
func (m *Message) Text() string {
	if m == nil {
		return ""
	}
	var sb strings.Builder
	for _, b := range m.Content {
		if t, ok := b.(*TextBlock); ok {
			sb.WriteString(t.Text)
		}
	}
	return sb.String()
}

// ToolCalls returns every ToolCallBlock of the message.
func (m *Message) ToolCalls() []*ToolCallBlock {
	if m == nil {
		return nil
	}
	var out []*ToolCallBlock
	for _, b := range m.Content {
		if c, ok := b.(*ToolCallBlock); ok {
			out = append(out, c)
		}
	}
	return out
}

// ToolResults returns every ToolResultBlock of the message.
func (m *Message) ToolResultBlocks() []*ToolResultBlock {
	if m == nil {
		return nil
	}
	var out []*ToolResultBlock
	for _, b := range m.Content {
		if r, ok := b.(*ToolResultBlock); ok {
			out = append(out, r)
		}
	}
	return out
}

func (m *Message) String() string {
	if m == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%s: %s", m.Role, m.Text())
}

// UserText builds a user message containing a single text block.
func UserText(s string) *Message {
	return &Message{Role: RoleUser, Content: []ContentBlock{&TextBlock{Text: s}}}
}

// SystemText builds a system message containing a single text block.
func SystemText(s string) *Message {
	return &Message{Role: RoleSystem, Content: []ContentBlock{&TextBlock{Text: s}}}
}

// AssistantText builds an assistant message containing a single text block.
func AssistantText(s string) *Message {
	return &Message{Role: RoleAssistant, Content: []ContentBlock{&TextBlock{Text: s}}, StopReason: StopReasonStop}
}

// ToolResults builds a tool message from result blocks.
func ToolResults(results ...*ToolResultBlock) *Message {
	m := &Message{Role: RoleTool}
	for _, r := range results {
		m.Content = append(m.Content, r)
	}
	return m
}

// TextResult builds a ToolResult with a single text block.
func TextResult(s string) *ToolResult {
	return &ToolResult{Content: []ContentBlock{&TextBlock{Text: s}}}
}

// ErrorResult builds an error ToolResult with a single text block.
func ErrorResult(s string) *ToolResult {
	return &ToolResult{Content: []ContentBlock{&TextBlock{Text: s}}, IsError: true}
}
