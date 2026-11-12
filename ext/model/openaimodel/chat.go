package openaimodel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/yoren041102/rain/schema"
)

// ---- request mapping ----

func (m *Model) chatParams(req *schema.ModelRequest) (openai.ChatCompletionNewParams, error) {
	p := openai.ChatCompletionNewParams{Model: m.model}
	msgs, err := chatMessages(req.Messages)
	if err != nil {
		return p, err
	}
	p.Messages = msgs
	p.StreamOptions.IncludeUsage = openai.Bool(true)

	for _, t := range req.Tools {
		params, err := schemaMap(t.Parameters)
		if err != nil {
			return p, fmt.Errorf("tool %q: %w", t.Name, err)
		}
		fd := openai.FunctionDefinitionParam{Name: t.Name, Parameters: openai.FunctionParameters(params)}
		if t.Description != "" {
			fd.Description = openai.String(t.Description)
		}
		p.Tools = append(p.Tools, openai.ChatCompletionToolParam{Function: fd})
	}

	o := req.Options
	if len(p.Tools) > 0 {
		switch o.ToolChoice.Mode {
		case schema.ToolChoiceAuto:
		case schema.ToolChoiceNone:
			p.ToolChoice.OfAuto = openai.String(string(openai.ChatCompletionToolChoiceOptionAutoNone))
		case schema.ToolChoiceRequired:
			p.ToolChoice.OfAuto = openai.String(string(openai.ChatCompletionToolChoiceOptionAutoRequired))
		case schema.ToolChoiceTool:
			if o.ToolChoice.Name == "" {
				return p, errors.New("tool choice mode \"tool\" requires a tool name")
			}
			p.ToolChoice = openai.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
				openai.ChatCompletionNamedToolChoiceFunctionParam{Name: o.ToolChoice.Name})
		default:
			return p, fmt.Errorf("unknown tool choice mode %q", o.ToolChoice.Mode)
		}
	}
	if o.Temperature != nil {
		p.Temperature = openai.Float(*o.Temperature)
	}
	if o.TopP != nil {
		p.TopP = openai.Float(*o.TopP)
	}
	if o.MaxTokens > 0 {
		p.MaxCompletionTokens = openai.Int(int64(o.MaxTokens))
	}
	if len(o.Stop) > 0 {
		p.Stop.OfStringArray = append([]string(nil), o.Stop...)
	}
	if len(o.OutputSchema) > 0 {
		if !json.Valid(o.OutputSchema) {
			return p, errors.New("invalid output schema JSON")
		}
		p.ResponseFormat.OfJSONSchema = &openai.ResponseFormatJSONSchemaParam{
			JSONSchema: openai.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:   "output",
				Schema: json.RawMessage(o.OutputSchema),
			},
		}
	}
	if len(o.Extra) > 0 {
		p.SetExtraFields(o.Extra)
	}
	for _, h := range m.chatHooks {
		h(&p)
	}
	return p, nil
}

func chatMessages(msgs []*schema.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	var out []openai.ChatCompletionMessageParamUnion
	for _, msg := range msgs {
		if msg == nil {
			continue
		}
		unknownOK := passUnknown(msg, ChatCompletions)
		switch msg.Role {
		case schema.RoleSystem:
			if text := joinText(msg.Content); text != "" {
				out = append(out, openai.SystemMessage(text))
			}
		case schema.RoleAssistant:
			out = append(out, chatAssistant(msg, unknownOK)...)
		default: // user, tool, or anything else
			var parts []openai.ChatCompletionContentPartUnionParam
			for _, b := range msg.Content {
				if r, ok := b.(*schema.ToolResultBlock); ok {
					out = append(out, openai.ToolMessage(toolResultText(r), r.CallID))
					continue
				}
				if part, ok := chatUserPart(b, unknownOK); ok {
					parts = append(parts, part)
				}
			}
			if len(parts) > 0 {
				out = append(out, chatUserMessage(parts))
			}
		}
	}
	return out, nil
}

func chatAssistant(msg *schema.Message, unknownOK bool) []openai.ChatCompletionMessageParamUnion {
	var (
		text     string
		parts    []openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion
		useParts bool
		calls    []openai.ChatCompletionMessageToolCallParam
		results  []openai.ChatCompletionMessageParamUnion
	)
	for _, b := range msg.Content {
		switch b := b.(type) {
		case *schema.TextBlock:
			text += b.Text
			parts = append(parts, openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion{
				OfText: &openai.ChatCompletionContentPartTextParam{Text: b.Text},
			})
		case *schema.ToolCallBlock:
			calls = append(calls, openai.ChatCompletionMessageToolCallParam{
				ID: b.ID,
				Function: openai.ChatCompletionMessageToolCallFunctionParam{
					Name:      b.Name,
					Arguments: argsString(b.Arguments),
				},
			})
		case *schema.ToolResultBlock:
			results = append(results, openai.ToolMessage(toolResultText(b), b.CallID))
		case *schema.UnknownBlock:
			if unknownOK && json.Valid(b.Raw) {
				parts = append(parts, param.Override[openai.ChatCompletionAssistantMessageParamContentArrayOfContentPartUnion](json.RawMessage(b.Raw)))
				useParts = true
			}
		}
		// ReasoningBlock, images and files are not representable in an
		// assistant chat message and are dropped.
	}
	var out []openai.ChatCompletionMessageParamUnion
	a := openai.ChatCompletionAssistantMessageParam{ToolCalls: calls}
	switch {
	case useParts:
		a.Content.OfArrayOfContentParts = parts
	case text != "":
		a.Content.OfString = openai.String(text)
	}
	if useParts || text != "" || len(calls) > 0 {
		out = append(out, openai.ChatCompletionMessageParamUnion{OfAssistant: &a})
	}
	return append(out, results...)
}

func chatUserPart(b schema.ContentBlock, unknownOK bool) (openai.ChatCompletionContentPartUnionParam, bool) {
	switch b := b.(type) {
	case *schema.TextBlock:
		return openai.TextContentPart(b.Text), true
	case *schema.ImageBlock:
		u := imageURL(b)
		if u == "" {
			return openai.ChatCompletionContentPartUnionParam{}, false
		}
		return openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
			URL:    u,
			Detail: metaString(b.Meta, "detail"),
		}), true
	case *schema.FileBlock:
		switch {
		case len(b.Data) > 0:
			return openai.FileContentPart(openai.ChatCompletionContentPartFileFileParam{
				FileData: openai.String(dataURL(b.MIMEType, b.Data)),
				Filename: openai.String(firstNonEmpty(b.Name, "file")),
			}), true
		case metaString(b.Meta, "file_id") != "":
			return openai.FileContentPart(openai.ChatCompletionContentPartFileFileParam{
				FileID: openai.String(metaString(b.Meta, "file_id")),
			}), true
		case b.URL != "":
			// Chat Completions cannot fetch file URLs; reference it as text.
			return openai.TextContentPart("[file: " + firstNonEmpty(b.Name+" ", "") + b.URL + "]"), true
		}
	case *schema.UnknownBlock:
		if unknownOK && json.Valid(b.Raw) {
			return param.Override[openai.ChatCompletionContentPartUnionParam](json.RawMessage(b.Raw)), true
		}
	}
	return openai.ChatCompletionContentPartUnionParam{}, false
}

func chatUserMessage(parts []openai.ChatCompletionContentPartUnionParam) openai.ChatCompletionMessageParamUnion {
	if len(parts) == 1 && parts[0].OfText != nil {
		return openai.UserMessage(parts[0].OfText.Text)
	}
	return openai.UserMessage(parts)
}

// ---- streaming ----

func (m *Model) generateChat(ctx context.Context, req *schema.ModelRequest, yield func(*schema.ModelResponse, error) bool) {
	params, err := m.chatParams(req)
	if err != nil {
		yield(nil, wrapErr(err))
		return
	}
	stream := m.client.Chat.Completions.NewStreaming(ctx, params, m.reqOpts...)
	defer stream.Close()

	st := &chatState{textIdx: -1, refusalIdx: -1, toolIdx: map[int64]int{}}
	for stream.Next() {
		for _, r := range st.add(stream.Current()) {
			if !yield(r, nil) {
				return
			}
		}
	}
	if err := stream.Err(); err != nil {
		yield(nil, wrapErr(err))
		return
	}
	if err := ctx.Err(); err != nil {
		yield(nil, wrapErr(err))
		return
	}
	yield(st.final(m.origin()), nil)
}

// chatState aggregates a Chat Completions stream. Only choice 0 is used.
type chatState struct {
	blocks     []schema.ContentBlock
	textIdx    int
	refusalIdx int
	toolIdx    map[int64]int // tool call index -> block index
	finish     string
	usage      *schema.Usage
}

func (s *chatState) alloc(b schema.ContentBlock) int {
	s.blocks = append(s.blocks, b)
	return len(s.blocks) - 1
}

func (s *chatState) add(chunk openai.ChatCompletionChunk) []*schema.ModelResponse {
	if chunk.JSON.Usage.Valid() {
		s.usage = chatUsage(chunk.Usage)
	}
	var out []*schema.ModelResponse
	for _, c := range chunk.Choices {
		if c.Index != 0 {
			continue
		}
		if c.FinishReason != "" {
			s.finish = c.FinishReason
		}
		d := c.Delta
		if d.Content != "" {
			if s.textIdx < 0 {
				s.textIdx = s.alloc(&schema.TextBlock{})
			}
			s.blocks[s.textIdx].(*schema.TextBlock).Text += d.Content
			out = append(out, partial(s.textIdx, &schema.TextBlock{Text: d.Content}))
		}
		if d.Refusal != "" {
			if s.refusalIdx < 0 {
				s.refusalIdx = s.alloc(&schema.TextBlock{})
			}
			s.blocks[s.refusalIdx].(*schema.TextBlock).Text += d.Refusal
			out = append(out, partial(s.refusalIdx, &schema.TextBlock{Text: d.Refusal}))
		}
		for _, tc := range d.ToolCalls {
			idx, ok := s.toolIdx[tc.Index]
			if !ok {
				idx = s.alloc(&schema.ToolCallBlock{})
				s.toolIdx[tc.Index] = idx
			}
			cur := s.blocks[idx].(*schema.ToolCallBlock)
			delta := &schema.ToolCallBlock{}
			if tc.ID != "" && cur.ID == "" {
				cur.ID = tc.ID
				delta.ID = tc.ID
			}
			if tc.Function.Name != "" {
				cur.Name += tc.Function.Name
				delta.Name = cur.Name
			}
			if tc.Function.Arguments != "" {
				cur.Arguments = append(cur.Arguments, tc.Function.Arguments...)
				delta.Arguments = json.RawMessage(tc.Function.Arguments)
			}
			if delta.ID != "" || delta.Name != "" || len(delta.Arguments) > 0 {
				out = append(out, partial(idx, delta))
			}
		}
	}
	return out
}

func (s *chatState) final(origin *schema.Origin) *schema.ModelResponse {
	stop := chatStopReason(s.finish)
	if s.finish == "" {
		stop = fallbackStop(s.blocks)
	}
	return &schema.ModelResponse{
		Message:      finishMessage(s.blocks, origin, stop),
		FinishReason: stop,
		Usage:        s.usage,
	}
}

func chatStopReason(finish string) schema.StopReason {
	switch finish {
	case "length":
		return schema.StopReasonLength
	case "tool_calls", "function_call":
		return schema.StopReasonToolUse
	default: // "stop", "content_filter", unknown
		return schema.StopReasonStop
	}
}

func chatUsage(u openai.CompletionUsage) *schema.Usage {
	return &schema.Usage{
		InputTokens:     int(u.PromptTokens),
		OutputTokens:    int(u.CompletionTokens),
		TotalTokens:     int(u.TotalTokens),
		CacheReadTokens: int(u.PromptTokensDetails.CachedTokens),
		ReasoningTokens: int(u.CompletionTokensDetails.ReasoningTokens),
	}
}
