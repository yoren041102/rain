package schema

import (
	"context"
	"encoding/json"
)

// FinishReason is the provider-reported end-of-stream reason, mapped to
// StopReason on the aggregated message.
type FinishReason = StopReason

// ToolChoiceMode controls whether the model must call tools.
type ToolChoiceMode string

const (
	ToolChoiceAuto     ToolChoiceMode = ""
	ToolChoiceNone     ToolChoiceMode = "none"
	ToolChoiceRequired ToolChoiceMode = "required"
	ToolChoiceTool     ToolChoiceMode = "tool"
)

// ToolChoice selects tool-calling behaviour.
type ToolChoice struct {
	Mode ToolChoiceMode
	Name string // used with ToolChoiceTool
}

// ModelOptions are generation parameters.
type ModelOptions struct {
	Temperature  *float64
	TopP         *float64
	MaxTokens    int
	Stop         []string
	ToolChoice   ToolChoice
	OutputSchema json.RawMessage // JSON Schema for structured output
	Extra        map[string]any  // adapter-specific passthrough
}

// ModelRequest is the input to Model.Generate.
type ModelRequest struct {
	Messages []*Message
	Tools    []ToolSpec
	Options  ModelOptions
}

// Clone returns a copy whose slices can be modified independently.
func (r *ModelRequest) Clone() *ModelRequest {
	c := *r
	c.Messages = append([]*Message(nil), r.Messages...)
	c.Tools = append([]ToolSpec(nil), r.Tools...)
	return &c
}

// ModelResponse is one item of a model stream. Partial chunks carry only the
// delta for BlockIndex; the final item (Partial == false) carries the
// complete aggregated message.
type ModelResponse struct {
	Message      *Message
	Partial      bool
	BlockIndex   int
	BlockType    BlockType
	FinishReason FinishReason
	Usage        *Usage
}

// Model is the single interface model adapters implement.
type Model interface {
	Name() string
	Generate(ctx context.Context, req *ModelRequest) Stream[*ModelResponse]
}

// ModelFunc adapts a function to Model.
type ModelFunc struct {
	ModelName string
	Fn        func(ctx context.Context, req *ModelRequest) Stream[*ModelResponse]
}

func (m ModelFunc) Name() string { return m.ModelName }
func (m ModelFunc) Generate(ctx context.Context, req *ModelRequest) Stream[*ModelResponse] {
	return m.Fn(ctx, req)
}

// FinalResponse drains a model stream and returns its final aggregated
// response. If the stream only contained partial chunks, they are
// accumulated.
func FinalResponse(s Stream[*ModelResponse]) (*ModelResponse, error) {
	acc := NewAccumulator()
	var final *ModelResponse
	for r, err := range s {
		if err != nil {
			return nil, err
		}
		if r == nil {
			continue
		}
		if !r.Partial {
			final = r
			continue
		}
		acc.Add(r)
	}
	if final != nil {
		return final, nil
	}
	return &ModelResponse{Message: acc.Message(), Usage: acc.Usage(), FinishReason: acc.Message().StopReason}, nil
}

// OutputToolName is the synthetic tool adapters may force the model to call
// when they lack native structured output. Agents treat a call to it as the
// final structured answer.
const OutputToolName = "_rain_output"

// OriginProvider is optionally implemented by models to report the Origin
// they stamp on messages, so history can be prepared for replay.
type OriginProvider interface {
	Origin() Origin
}

// ModelOrigin returns m's Origin, falling back to {Model: m.Name()}.
func ModelOrigin(m Model) Origin {
	if p, ok := m.(OriginProvider); ok {
		return p.Origin()
	}
	return Origin{Model: m.Name()}
}
