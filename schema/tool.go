package schema

import (
	"context"
	"encoding/json"
	"fmt"
)

// ToolSpec describes a tool to the model.
type ToolSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"` // JSON Schema
}

// Tool is a callable tool.
type Tool interface {
	Spec() ToolSpec
	Call(ctx context.Context, args json.RawMessage) (*ToolResult, error)
}

// ToolResult is the output of a tool call.
type ToolResult struct {
	Content []ContentBlock `json:"-"`
	Details []ContentBlock `json:"-"` // UI only
	IsError bool           `json:"is_error,omitempty"`
}

type toolResultJSON struct {
	Content json.RawMessage `json:"content,omitempty"`
	Details json.RawMessage `json:"details,omitempty"`
	IsError bool            `json:"is_error,omitempty"`
}

// MarshalJSON implements json.Marshaler.
func (r ToolResult) MarshalJSON() ([]byte, error) {
	var j toolResultJSON
	var err error
	j.IsError = r.IsError
	if r.Content != nil {
		if j.Content, err = MarshalBlocks(r.Content); err != nil {
			return nil, err
		}
	}
	if r.Details != nil {
		if j.Details, err = MarshalBlocks(r.Details); err != nil {
			return nil, err
		}
	}
	return json.Marshal(j)
}

// UnmarshalJSON implements json.Unmarshaler.
func (r *ToolResult) UnmarshalJSON(data []byte) error {
	var j toolResultJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return err
	}
	*r = ToolResult{IsError: j.IsError}
	var err error
	if len(j.Content) > 0 {
		if r.Content, err = UnmarshalBlocks(j.Content); err != nil {
			return err
		}
	}
	if len(j.Details) > 0 {
		if r.Details, err = UnmarshalBlocks(j.Details); err != nil {
			return err
		}
	}
	return nil
}

// Block converts the result into a ToolResultBlock for callID.
func (r *ToolResult) Block(callID string) *ToolResultBlock {
	if r == nil {
		return &ToolResultBlock{CallID: callID, Content: []ContentBlock{&TextBlock{Text: "No result provided"}}, IsError: true}
	}
	return &ToolResultBlock{CallID: callID, Content: cloneBlocks(r.Content), Details: cloneBlocks(r.Details), IsError: r.IsError}
}

// Optional marker interfaces.
type (
	ReadOnly        interface{ IsReadOnly() bool }
	ConcurrencySafe interface{ IsConcurrencySafe() bool }
	ReplaySafe      interface{ IsReplaySafe() bool }
)

// IsReadOnly reports whether t declares itself read-only.
func IsReadOnly(t Tool) bool { v, ok := t.(ReadOnly); return ok && v.IsReadOnly() }

// IsConcurrencySafe reports whether t declares itself safe to run in parallel.
func IsConcurrencySafe(t Tool) bool {
	v, ok := t.(ConcurrencySafe)
	return ok && v.IsConcurrencySafe()
}

// IsReplaySafe reports whether t declares itself safe to re-run.
func IsReplaySafe(t Tool) bool { v, ok := t.(ReplaySafe); return ok && v.IsReplaySafe() }

// Toolset provides a dynamic set of tools per model call.
type Toolset interface {
	Tools(ctx context.Context) ([]ToolSpec, error)
	ProcessRequest(ctx context.Context, req *ModelRequest) error
	Call(ctx context.Context, name string, args json.RawMessage) (*ToolResult, error)
}

// ToolsetTool exposes one tool of a Toolset as a Tool.
type ToolsetTool struct {
	Set      Toolset
	ToolSpec ToolSpec
}

func (t *ToolsetTool) Spec() ToolSpec { return t.ToolSpec }
func (t *ToolsetTool) Call(ctx context.Context, args json.RawMessage) (*ToolResult, error) {
	return t.Set.Call(ctx, t.ToolSpec.Name, args)
}

// StaticToolset is a Toolset over a fixed list of tools.
type StaticToolset struct{ List []Tool }

func (s *StaticToolset) Tools(context.Context) ([]ToolSpec, error) {
	out := make([]ToolSpec, len(s.List))
	for i, t := range s.List {
		out[i] = t.Spec()
	}
	return out, nil
}
func (s *StaticToolset) ProcessRequest(context.Context, *ModelRequest) error { return nil }
func (s *StaticToolset) Call(ctx context.Context, name string, args json.RawMessage) (*ToolResult, error) {
	for _, t := range s.List {
		if t.Spec().Name == name {
			return t.Call(ctx, args)
		}
	}
	return nil, fmt.Errorf("schema: tool %q not found", name)
}
