// Package anthropicmodel adapts the Anthropic Messages API
// (github.com/anthropics/anthropic-sdk-go) to schema.Model.
package anthropicmodel

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/yoren041102/rain/schema"
)

const (
	provider = "anthropic"
	api      = "messages"
	// DefaultMaxTokens is used when ModelOptions.MaxTokens is zero; the
	// Messages API requires max_tokens.
	DefaultMaxTokens = 8192
)

// Option configures a Model.
type Option func(*Model)

// WithRequestOptions passes SDK request options to every call.
func WithRequestOptions(opts ...option.RequestOption) Option {
	return func(m *Model) { m.reqOpts = append(m.reqOpts, opts...) }
}

// WithParams lets callers set any SDK parameter (thinking, metadata, ...)
// after rain's mapping.
func WithParams(f func(*anthropic.MessageNewParams)) Option {
	return func(m *Model) { m.paramHooks = append(m.paramHooks, f) }
}

// WithThinking enables extended thinking with the given token budget.
func WithThinking(budget int64) Option {
	return WithParams(func(p *anthropic.MessageNewParams) {
		p.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
	})
}

// WithNativeStructuredOutput uses output_config.format for OutputSchema
// instead of the forced synthetic tool (default false: the tool fallback
// works on every model).
func WithNativeStructuredOutput() Option { return func(m *Model) { m.native = true } }

// Model is a schema.Model backed by the Anthropic Messages API.
type Model struct {
	client     anthropic.Client
	model      string
	reqOpts    []option.RequestOption
	paramHooks []func(*anthropic.MessageNewParams)
	native     bool
}

// New returns a Model for the given model name.
func New(client anthropic.Client, model string, opts ...Option) *Model {
	m := &Model{client: client, model: model}
	for _, o := range opts {
		o(m)
	}
	return m
}

func (m *Model) Name() string { return m.model }

// Origin implements schema.OriginProvider.
func (m *Model) Origin() schema.Origin {
	return schema.Origin{Provider: provider, API: api, Model: m.model}
}

// Generate streams a response.
func (m *Model) Generate(ctx context.Context, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
	return func(yield func(*schema.ModelResponse, error) bool) {
		params, forcedOutput, err := m.BuildParams(req)
		if err != nil {
			yield(nil, err)
			return
		}
		stream := m.client.Messages.NewStreaming(ctx, params, m.reqOpts...)
		defer stream.Close()

		acc := anthropic.Message{}
		blockTypes := map[int64]schema.BlockType{}
		for stream.Next() {
			ev := stream.Current()
			if err := acc.Accumulate(ev); err != nil {
				yield(nil, fmt.Errorf("anthropicmodel: %w", err))
				return
			}
			chunk := deltaChunk(ev, blockTypes)
			if chunk == nil {
				continue
			}
			if !yield(chunk, nil) {
				return
			}
		}
		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("anthropicmodel: %w", err))
			return
		}
		msg, err := m.convertResponse(&acc, forcedOutput)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(&schema.ModelResponse{Message: msg, FinishReason: msg.StopReason, Usage: convertUsage(acc.Usage)}, nil)
	}
}

func deltaChunk(ev anthropic.MessageStreamEventUnion, types map[int64]schema.BlockType) *schema.ModelResponse {
	idx := ev.Index
	var b schema.ContentBlock
	switch ev.Type {
	case "content_block_start":
		cb := ev.ContentBlock
		switch cb.Type {
		case "text":
			types[idx] = schema.BlockText
			if cb.Text == "" {
				return nil
			}
			b = &schema.TextBlock{Text: cb.Text}
		case "thinking":
			types[idx] = schema.BlockReasoning
			b = &schema.ReasoningBlock{Text: cb.Thinking, Signature: cb.Signature}
		case "tool_use":
			types[idx] = schema.BlockToolCall
			b = &schema.ToolCallBlock{ID: cb.ID, Name: cb.Name}
		default:
			return nil
		}
	case "content_block_delta":
		d := ev.Delta
		switch d.Type {
		case "text_delta":
			b = &schema.TextBlock{Text: d.Text}
		case "thinking_delta":
			b = &schema.ReasoningBlock{Text: d.Thinking}
		case "signature_delta":
			b = &schema.ReasoningBlock{Signature: d.Signature}
		case "input_json_delta":
			if d.PartialJSON == "" {
				return nil
			}
			b = &schema.ToolCallBlock{Arguments: json.RawMessage(d.PartialJSON)}
		default:
			return nil
		}
	default:
		return nil
	}
	return &schema.ModelResponse{
		Partial:    true,
		BlockIndex: int(idx),
		BlockType:  b.BlockType(),
		Message:    &schema.Message{Role: schema.RoleAssistant, Content: []schema.ContentBlock{b}},
	}
}

// BuildParams maps a rain request to SDK params. It reports whether the
// synthetic output tool was forced for structured output.
func (m *Model) BuildParams(req *schema.ModelRequest) (anthropic.MessageNewParams, bool, error) {
	p := anthropic.MessageNewParams{Model: m.model, MaxTokens: DefaultMaxTokens}
	o := req.Options
	if o.MaxTokens > 0 {
		p.MaxTokens = int64(o.MaxTokens)
	}
	if o.Temperature != nil {
		p.Temperature = anthropic.Float(*o.Temperature)
	}
	if o.TopP != nil {
		p.TopP = anthropic.Float(*o.TopP)
	}
	if len(o.Stop) > 0 {
		p.StopSequences = o.Stop
	}

	sameOrigin := m.Origin()
	for _, msg := range req.Messages {
		if msg.Role == schema.RoleSystem {
			for _, b := range msg.Content {
				if t, ok := b.(*schema.TextBlock); ok {
					tb := anthropic.TextBlockParam{Text: t.Text}
					applyCache(&tb.CacheControl, t.Meta)
					p.System = append(p.System, tb)
				}
			}
			continue
		}
		mp, err := convertMessage(msg, msg.Origin.Same(sameOrigin))
		if err != nil {
			return p, false, err
		}
		if len(mp.Content) == 0 {
			continue
		}
		// merge consecutive same-role messages (the API requires alternation)
		if n := len(p.Messages); n > 0 && p.Messages[n-1].Role == mp.Role {
			p.Messages[n-1].Content = append(p.Messages[n-1].Content, mp.Content...)
			continue
		}
		p.Messages = append(p.Messages, mp)
	}

	for _, t := range req.Tools {
		tp, err := convertTool(t)
		if err != nil {
			return p, false, err
		}
		p.Tools = append(p.Tools, anthropic.ToolUnionParam{OfTool: tp})
	}

	forced := false
	if len(o.OutputSchema) > 0 {
		if m.native {
			var s map[string]any
			if err := json.Unmarshal(o.OutputSchema, &s); err != nil {
				return p, false, fmt.Errorf("anthropicmodel: invalid output schema: %w", err)
			}
			p.OutputConfig.Format = anthropic.JSONOutputFormatParam{Schema: s}
		} else {
			tp, err := convertTool(schema.ToolSpec{Name: schema.OutputToolName, Description: "Return the final answer in the required structure.", Parameters: o.OutputSchema})
			if err != nil {
				return p, false, err
			}
			p.Tools = append(p.Tools, anthropic.ToolUnionParam{OfTool: tp})
			p.ToolChoice = anthropic.ToolChoiceUnionParam{OfTool: &anthropic.ToolChoiceToolParam{Name: schema.OutputToolName}}
			forced = true
		}
	}
	if !forced {
		switch o.ToolChoice.Mode {
		case schema.ToolChoiceNone:
			n := anthropic.NewToolChoiceNoneParam()
			p.ToolChoice = anthropic.ToolChoiceUnionParam{OfNone: &n}
		case schema.ToolChoiceRequired:
			p.ToolChoice = anthropic.ToolChoiceUnionParam{OfAny: &anthropic.ToolChoiceAnyParam{}}
		case schema.ToolChoiceTool:
			p.ToolChoice = anthropic.ToolChoiceUnionParam{OfTool: &anthropic.ToolChoiceToolParam{Name: o.ToolChoice.Name}}
		}
	}
	for _, h := range m.paramHooks {
		h(&p)
	}
	return p, forced, nil
}

func convertTool(t schema.ToolSpec) (*anthropic.ToolParam, error) {
	var s map[string]any
	if len(t.Parameters) > 0 {
		if err := json.Unmarshal(t.Parameters, &s); err != nil {
			return nil, fmt.Errorf("anthropicmodel: tool %q: invalid schema: %w", t.Name, err)
		}
	}
	in := anthropic.ToolInputSchemaParam{Properties: s["properties"]}
	if in.Properties == nil {
		in.Properties = map[string]any{}
	}
	if req, ok := s["required"].([]any); ok {
		for _, r := range req {
			if str, ok := r.(string); ok {
				in.Required = append(in.Required, str)
			}
		}
	}
	extra := map[string]any{}
	for k, v := range s {
		if k != "properties" && k != "required" && k != "type" {
			extra[k] = v
		}
	}
	if len(extra) > 0 {
		in.ExtraFields = extra
	}
	tp := &anthropic.ToolParam{Name: t.Name, InputSchema: in}
	if t.Description != "" {
		tp.Description = anthropic.String(t.Description)
	}
	return tp, nil
}

func applyCache(cc *anthropic.CacheControlEphemeralParam, meta schema.BlockMeta) {
	if raw, ok := meta.Extra["cache_control"]; ok {
		*cc = anthropic.NewCacheControlEphemeralParam()
		_ = cc.UnmarshalJSON(raw)
	}
}

func convertMessage(msg *schema.Message, sameModel bool) (anthropic.MessageParam, error) {
	role := anthropic.MessageParamRoleUser
	if msg.Role == schema.RoleAssistant {
		role = anthropic.MessageParamRoleAssistant
	}
	out := anthropic.MessageParam{Role: role}
	for _, b := range msg.Content {
		cb, ok, err := convertBlock(b, msg.Role, sameModel)
		if err != nil {
			return out, err
		}
		if ok {
			out.Content = append(out.Content, cb)
		}
	}
	return out, nil
}

func convertBlock(b schema.ContentBlock, role schema.Role, sameModel bool) (anthropic.ContentBlockParamUnion, bool, error) {
	switch v := b.(type) {
	case *schema.TextBlock:
		if v.Text == "" {
			return anthropic.ContentBlockParamUnion{}, false, nil
		}
		tb := &anthropic.TextBlockParam{Text: v.Text}
		applyCache(&tb.CacheControl, v.Meta)
		return anthropic.ContentBlockParamUnion{OfText: tb}, true, nil
	case *schema.ImageBlock:
		return imageBlock(v), true, nil
	case *schema.FileBlock:
		return fileBlock(v), true, nil
	case *schema.ReasoningBlock:
		if role != schema.RoleAssistant {
			return anthropic.ContentBlockParamUnion{}, false, nil
		}
		if v.Signature == "" {
			// thinking from another model: replay as plain text is not
			// allowed in assistant turns with thinking enabled, so drop it.
			return anthropic.ContentBlockParamUnion{}, false, nil
		}
		return anthropic.NewThinkingBlock(v.Signature, v.Text), true, nil
	case *schema.ToolCallBlock:
		var input any = map[string]any{}
		if len(v.Arguments) > 0 && json.Valid(v.Arguments) {
			input = json.RawMessage(v.Arguments)
		}
		tu := &anthropic.ToolUseBlockParam{ID: v.ID, Name: v.Name, Input: input}
		applyCache(&tu.CacheControl, v.Meta)
		return anthropic.ContentBlockParamUnion{OfToolUse: tu}, true, nil
	case *schema.ToolResultBlock:
		tr := &anthropic.ToolResultBlockParam{ToolUseID: v.CallID}
		if v.IsError {
			tr.IsError = anthropic.Bool(true)
		}
		for _, c := range v.Content {
			switch cv := c.(type) {
			case *schema.TextBlock:
				tr.Content = append(tr.Content, anthropic.ToolResultBlockParamContentUnion{OfText: &anthropic.TextBlockParam{Text: cv.Text}})
			case *schema.ImageBlock:
				img := imageBlock(cv)
				tr.Content = append(tr.Content, anthropic.ToolResultBlockParamContentUnion{OfImage: img.OfImage})
			}
		}
		applyCache(&tr.CacheControl, v.Meta)
		return anthropic.ContentBlockParamUnion{OfToolResult: tr}, true, nil
	case *schema.UnknownBlock:
		if !sameModel && !strings.HasPrefix(v.Type, "redacted_thinking") {
			return anthropic.ContentBlockParamUnion{}, false, nil
		}
		var u anthropic.ContentBlockParamUnion
		if err := u.UnmarshalJSON(v.Raw); err != nil {
			return u, false, nil
		}
		return u, true, nil
	}
	return anthropic.ContentBlockParamUnion{}, false, nil
}

func imageBlock(v *schema.ImageBlock) anthropic.ContentBlockParamUnion {
	if v.URL != "" && len(v.Data) == 0 {
		return anthropic.NewImageBlock(anthropic.URLImageSourceParam{URL: v.URL})
	}
	return anthropic.NewImageBlockBase64(v.MIMEType, base64.StdEncoding.EncodeToString(v.Data))
}

func fileBlock(v *schema.FileBlock) anthropic.ContentBlockParamUnion {
	switch {
	case v.URL != "" && len(v.Data) == 0:
		return anthropic.NewDocumentBlock(anthropic.URLPDFSourceParam{URL: v.URL})
	case strings.HasPrefix(v.MIMEType, "text/"):
		return anthropic.NewDocumentBlock(anthropic.PlainTextSourceParam{Data: string(v.Data)})
	default:
		return anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{Data: base64.StdEncoding.EncodeToString(v.Data)})
	}
}

func (m *Model) convertResponse(acc *anthropic.Message, forcedOutput bool) (*schema.Message, error) {
	o := m.Origin()
	msg := &schema.Message{Role: schema.RoleAssistant, Origin: &o}
	if acc.Model != "" {
		msg.Origin.Model = acc.Model
	}
	for _, c := range acc.Content {
		switch c.Type {
		case "text":
			tb := &schema.TextBlock{Text: c.Text}
			if len(c.Citations) > 0 {
				raw, _ := json.Marshal(c.Citations)
				tb.Meta.Extra = map[string]json.RawMessage{"citations": raw}
			}
			msg.Content = append(msg.Content, tb)
		case "thinking":
			msg.Content = append(msg.Content, &schema.ReasoningBlock{Text: c.Thinking, Signature: c.Signature})
		case "tool_use":
			args := c.Input
			if len(args) == 0 || !json.Valid(args) {
				args = json.RawMessage("{}")
			}
			if forcedOutput && c.Name == schema.OutputToolName {
				msg.Content = append(msg.Content, &schema.TextBlock{Text: string(args)})
				continue
			}
			msg.Content = append(msg.Content, &schema.ToolCallBlock{ID: c.ID, Name: c.Name, Arguments: append(json.RawMessage(nil), args...)})
		default:
			raw := json.RawMessage(c.RawJSON())
			if len(raw) == 0 {
				raw, _ = json.Marshal(c)
			}
			msg.Content = append(msg.Content, &schema.UnknownBlock{Type: c.Type, Raw: raw})
		}
	}
	msg.StopReason = mapStop(acc.StopReason)
	if forcedOutput && msg.StopReason == schema.StopReasonToolUse && len(msg.ToolCalls()) == 0 {
		msg.StopReason = schema.StopReasonStop
	}
	return msg, nil
}

func mapStop(r anthropic.StopReason) schema.StopReason {
	switch r {
	case anthropic.StopReasonMaxTokens, anthropic.StopReasonModelContextWindowExceeded:
		return schema.StopReasonLength
	case anthropic.StopReasonToolUse:
		return schema.StopReasonToolUse
	case "":
		return schema.StopReasonStop
	default: // end_turn, stop_sequence, pause_turn, refusal
		return schema.StopReasonStop
	}
}

func convertUsage(u anthropic.Usage) *schema.Usage {
	return &schema.Usage{
		InputTokens:      int(u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens),
		OutputTokens:     int(u.OutputTokens),
		CacheReadTokens:  int(u.CacheReadInputTokens),
		CacheWriteTokens: int(u.CacheCreationInputTokens),
		TotalTokens:      int(u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens + u.OutputTokens),
	}
}
