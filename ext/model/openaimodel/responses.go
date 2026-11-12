package openaimodel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/yoren041102/rain/schema"
)

// ---- request mapping ----

func (m *Model) responsesParams(req *schema.ModelRequest) (responses.ResponseNewParams, error) {
	p := responses.ResponseNewParams{Model: m.model}
	p.Input.OfInputItemList = responsesInput(req.Messages)

	for _, t := range req.Tools {
		params, err := schemaMap(t.Parameters)
		if err != nil {
			return p, fmt.Errorf("tool %q: %w", t.Name, err)
		}
		ft := &responses.FunctionToolParam{Name: t.Name, Parameters: params, Strict: openai.Bool(false)}
		if t.Description != "" {
			ft.Description = openai.String(t.Description)
		}
		p.Tools = append(p.Tools, responses.ToolUnionParam{OfFunction: ft})
	}

	o := req.Options
	if len(p.Tools) > 0 {
		switch o.ToolChoice.Mode {
		case schema.ToolChoiceAuto:
		case schema.ToolChoiceNone:
			p.ToolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsNone)
		case schema.ToolChoiceRequired:
			p.ToolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsRequired)
		case schema.ToolChoiceTool:
			if o.ToolChoice.Name == "" {
				return p, errors.New("tool choice mode \"tool\" requires a tool name")
			}
			p.ToolChoice.OfFunctionTool = &responses.ToolChoiceFunctionParam{Name: o.ToolChoice.Name}
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
		p.MaxOutputTokens = openai.Int(int64(o.MaxTokens))
	}
	// The Responses API has no stop-sequence parameter; Options.Stop is ignored.
	if len(o.OutputSchema) > 0 {
		s, err := schemaMap(o.OutputSchema)
		if err != nil {
			return p, fmt.Errorf("output schema: %w", err)
		}
		p.Text.Format.OfJSONSchema = &responses.ResponseFormatTextJSONSchemaConfigParam{
			Name:   "output",
			Schema: s,
			Strict: openai.Bool(false),
		}
	}
	if len(o.Extra) > 0 {
		p.SetExtraFields(o.Extra)
	}
	for _, h := range m.respHooks {
		h(&p)
	}
	return p, nil
}

func responsesInput(msgs []*schema.Message) responses.ResponseInputParam {
	var out responses.ResponseInputParam
	for _, msg := range msgs {
		if msg == nil {
			continue
		}
		unknownOK := passUnknown(msg, Responses)
		switch msg.Role {
		case schema.RoleSystem:
			if text := joinText(msg.Content); text != "" {
				out = append(out, responses.ResponseInputItemParamOfMessage(text, responses.EasyInputMessageRoleSystem))
			}
		case schema.RoleAssistant:
			var text strings.Builder
			flush := func() {
				if text.Len() > 0 {
					out = append(out, responses.ResponseInputItemParamOfMessage(text.String(), responses.EasyInputMessageRoleAssistant))
					text.Reset()
				}
			}
			for _, b := range msg.Content {
				switch b := b.(type) {
				case *schema.TextBlock:
					text.WriteString(b.Text)
				case *schema.ToolCallBlock:
					flush()
					out = append(out, responses.ResponseInputItemParamOfFunctionCall(argsString(b.Arguments), b.ID, b.Name))
				case *schema.ToolResultBlock:
					flush()
					out = append(out, responses.ResponseInputItemParamOfFunctionCallOutput(b.CallID, toolResultText(b)))
				case *schema.UnknownBlock:
					if unknownOK && json.Valid(b.Raw) {
						flush()
						out = append(out, param.Override[responses.ResponseInputItemUnionParam](json.RawMessage(b.Raw)))
					}
				}
				// ReasoningBlock is dropped: replaying reasoning items requires
				// the original item IDs of every following output item.
			}
			flush()
		default: // user, tool, or anything else
			var parts responses.ResponseInputMessageContentListParam
			flush := func() {
				if len(parts) > 0 {
					out = append(out, responses.ResponseInputItemParamOfMessage(parts, responses.EasyInputMessageRoleUser))
					parts = nil
				}
			}
			for _, b := range msg.Content {
				switch b := b.(type) {
				case *schema.ToolResultBlock:
					flush()
					out = append(out, responses.ResponseInputItemParamOfFunctionCallOutput(b.CallID, toolResultText(b)))
				case *schema.UnknownBlock:
					if unknownOK && json.Valid(b.Raw) {
						flush()
						out = append(out, param.Override[responses.ResponseInputItemUnionParam](json.RawMessage(b.Raw)))
					}
				default:
					if part, ok := responsesUserPart(b); ok {
						parts = append(parts, part)
					}
				}
			}
			flush()
		}
	}
	return out
}

func responsesUserPart(b schema.ContentBlock) (responses.ResponseInputContentUnionParam, bool) {
	switch b := b.(type) {
	case *schema.TextBlock:
		return responses.ResponseInputContentParamOfInputText(b.Text), true
	case *schema.ImageBlock:
		u := imageURL(b)
		if u == "" {
			return responses.ResponseInputContentUnionParam{}, false
		}
		detail := responses.ResponseInputImageDetail(firstNonEmpty(metaString(b.Meta, "detail"), string(responses.ResponseInputImageDetailAuto)))
		return responses.ResponseInputContentUnionParam{OfInputImage: &responses.ResponseInputImageParam{
			Detail:   detail,
			ImageURL: openai.String(u),
		}}, true
	case *schema.FileBlock:
		f := &responses.ResponseInputFileParam{}
		switch {
		case len(b.Data) > 0:
			f.FileData = openai.String(dataURL(b.MIMEType, b.Data))
			f.Filename = openai.String(firstNonEmpty(b.Name, "file"))
		case metaString(b.Meta, "file_id") != "":
			f.FileID = openai.String(metaString(b.Meta, "file_id"))
		case b.URL != "":
			f.FileURL = openai.String(b.URL)
			if b.Name != "" {
				f.Filename = openai.String(b.Name)
			}
		default:
			return responses.ResponseInputContentUnionParam{}, false
		}
		return responses.ResponseInputContentUnionParam{OfInputFile: f}, true
	}
	return responses.ResponseInputContentUnionParam{}, false
}

// ---- streaming ----

func (m *Model) generateResponses(ctx context.Context, req *schema.ModelRequest, yield func(*schema.ModelResponse, error) bool) {
	params, err := m.responsesParams(req)
	if err != nil {
		yield(nil, wrapErr(err))
		return
	}
	stream := m.client.Responses.NewStreaming(ctx, params, m.reqOpts...)
	defer stream.Close()

	st := &respState{keys: map[respKey]int{}, summaryIdx: map[int64]int64{}}
	for stream.Next() {
		out, err := st.add(stream.Current())
		for _, r := range out {
			if !yield(r, nil) {
				return
			}
		}
		if err != nil {
			yield(nil, wrapErr(err))
			return
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

type respKind int

const (
	kindText respKind = iota
	kindToolCall
	kindReasoning
	kindOther
)

// respKey identifies a block: an output item (by output index) and, for
// message items, the content part within it.
type respKey struct {
	output  int64
	content int64
	kind    respKind
}

// respState aggregates a Responses API stream.
type respState struct {
	blocks     []schema.ContentBlock
	keys       map[respKey]int
	summaryIdx map[int64]int64 // output index -> last reasoning summary index
	status     responses.ResponseStatus
	incomplete string
	usage      *schema.Usage
}

func (s *respState) block(k respKey, mk func() schema.ContentBlock) (int, bool) {
	if idx, ok := s.keys[k]; ok {
		return idx, false
	}
	s.blocks = append(s.blocks, mk())
	idx := len(s.blocks) - 1
	s.keys[k] = idx
	return idx, true
}

func (s *respState) add(ev responses.ResponseStreamEventUnion) ([]*schema.ModelResponse, error) {
	var out []*schema.ModelResponse
	switch ev.Type {
	case "response.output_item.added":
		item := ev.AsResponseOutputItemAdded().Item
		if item.Type == "function_call" {
			idx, _ := s.block(respKey{ev.OutputIndex, 0, kindToolCall}, newToolCall)
			cur := s.blocks[idx].(*schema.ToolCallBlock)
			cur.ID, cur.Name = item.CallID, item.Name
			cur.Arguments = append(json.RawMessage(nil), item.Arguments...)
			out = append(out, partial(idx, &schema.ToolCallBlock{ID: item.CallID, Name: item.Name, Arguments: json.RawMessage(item.Arguments)}))
		}

	case "response.output_text.delta", "response.refusal.delta":
		var delta string
		if ev.Type == "response.refusal.delta" {
			delta = ev.AsResponseRefusalDelta().Delta
		} else {
			delta = ev.AsResponseOutputTextDelta().Delta
		}
		if delta == "" {
			break
		}
		idx, _ := s.block(respKey{ev.OutputIndex, ev.ContentIndex, kindText}, newText)
		s.blocks[idx].(*schema.TextBlock).Text += delta
		out = append(out, partial(idx, &schema.TextBlock{Text: delta}))

	case "response.function_call_arguments.delta":
		delta := ev.AsResponseFunctionCallArgumentsDelta().Delta
		if delta == "" {
			break
		}
		idx, _ := s.block(respKey{ev.OutputIndex, 0, kindToolCall}, newToolCall)
		cur := s.blocks[idx].(*schema.ToolCallBlock)
		cur.Arguments = append(cur.Arguments, delta...)
		out = append(out, partial(idx, &schema.ToolCallBlock{Arguments: json.RawMessage(delta)}))

	case "response.reasoning_summary_text.delta":
		e := ev.AsResponseReasoningSummaryTextDelta()
		if e.Delta == "" {
			break
		}
		idx, _ := s.block(respKey{ev.OutputIndex, 0, kindReasoning}, newReasoning)
		cur := s.blocks[idx].(*schema.ReasoningBlock)
		delta := e.Delta
		if last, ok := s.summaryIdx[ev.OutputIndex]; ok && last != e.SummaryIndex && cur.Text != "" {
			delta = "\n\n" + delta
		}
		s.summaryIdx[ev.OutputIndex] = e.SummaryIndex
		cur.Text += delta
		out = append(out, partial(idx, &schema.ReasoningBlock{Text: delta}))

	case "response.output_item.done":
		out = s.itemDone(ev.OutputIndex, ev.AsResponseOutputItemDone().Item)

	case "response.completed", "response.incomplete":
		r := ev.Response
		s.status = r.Status
		s.incomplete = r.IncompleteDetails.Reason
		if r.JSON.Usage.Valid() {
			s.usage = responsesUsage(r.Usage)
		}

	case "response.failed":
		r := ev.Response
		msg := r.Error.Message
		if msg == "" {
			msg = "response failed"
		}
		if r.Error.Code != "" {
			return out, fmt.Errorf("responses stream: %s: %s", r.Error.Code, msg)
		}
		return out, fmt.Errorf("responses stream: %s", msg)

	case "error":
		e := ev.AsError()
		if e.Code != "" {
			return out, fmt.Errorf("responses stream error: %s: %s", e.Code, e.Message)
		}
		return out, fmt.Errorf("responses stream error: %s", e.Message)
	}
	return out, nil
}

// itemDone applies the authoritative content of a completed output item.
func (s *respState) itemDone(outIdx int64, item responses.ResponseOutputItemUnion) []*schema.ModelResponse {
	var out []*schema.ModelResponse
	switch item.Type {
	case "message":
		for i, c := range item.Content {
			text := c.Text
			if c.Type == "refusal" {
				text = c.Refusal
			}
			idx, created := s.block(respKey{outIdx, int64(i), kindText}, newText)
			cur := s.blocks[idx].(*schema.TextBlock)
			if created && text != "" {
				out = append(out, partial(idx, &schema.TextBlock{Text: text}))
			}
			cur.Text = text
		}
	case "function_call":
		idx, created := s.block(respKey{outIdx, 0, kindToolCall}, newToolCall)
		cur := s.blocks[idx].(*schema.ToolCallBlock)
		if created {
			out = append(out, partial(idx, &schema.ToolCallBlock{ID: item.CallID, Name: item.Name, Arguments: json.RawMessage(item.Arguments)}))
		}
		cur.ID = firstNonEmpty(item.CallID, cur.ID)
		cur.Name = firstNonEmpty(item.Name, cur.Name)
		if item.Arguments != "" {
			cur.Arguments = json.RawMessage(item.Arguments)
		}
	case "reasoning":
		var parts []string
		for _, sm := range item.Summary {
			parts = append(parts, sm.Text)
		}
		text := strings.Join(parts, "\n\n")
		k := respKey{outIdx, 0, kindReasoning}
		if _, ok := s.keys[k]; !ok && text == "" && item.EncryptedContent == "" {
			break // nothing worth keeping
		}
		idx, created := s.block(k, newReasoning)
		cur := s.blocks[idx].(*schema.ReasoningBlock)
		if created || item.EncryptedContent != "" {
			d := &schema.ReasoningBlock{Signature: item.EncryptedContent}
			if created {
				d.Text = text
			}
			out = append(out, partial(idx, d))
		}
		if text != "" {
			cur.Text = text
		}
		cur.Signature = item.EncryptedContent
	default:
		// Built-in tool calls and other item types are preserved verbatim so
		// they can be replayed to the Responses API.
		raw := item.RawJSON()
		if raw == "" {
			break
		}
		b := &schema.UnknownBlock{Type: item.Type, Raw: json.RawMessage(raw)}
		idx, _ := s.block(respKey{outIdx, 0, kindOther}, func() schema.ContentBlock { return b })
		s.blocks[idx] = b
		out = append(out, partial(idx, &schema.UnknownBlock{Type: b.Type, Raw: b.Raw}))
	}
	return out
}

func (s *respState) final(origin *schema.Origin) *schema.ModelResponse {
	var stop schema.StopReason
	switch {
	case s.status == responses.ResponseStatusIncomplete && s.incomplete == "max_output_tokens":
		stop = schema.StopReasonLength
	case s.status == responses.ResponseStatusIncomplete:
		stop = schema.StopReasonStop
	default:
		stop = fallbackStop(s.blocks)
	}
	return &schema.ModelResponse{
		Message:      finishMessage(s.blocks, origin, stop),
		FinishReason: stop,
		Usage:        s.usage,
	}
}

func newText() schema.ContentBlock      { return &schema.TextBlock{} }
func newToolCall() schema.ContentBlock  { return &schema.ToolCallBlock{} }
func newReasoning() schema.ContentBlock { return &schema.ReasoningBlock{} }

func responsesUsage(u responses.ResponseUsage) *schema.Usage {
	return &schema.Usage{
		InputTokens:     int(u.InputTokens),
		OutputTokens:    int(u.OutputTokens),
		TotalTokens:     int(u.TotalTokens),
		CacheReadTokens: int(u.InputTokensDetails.CachedTokens),
		ReasoningTokens: int(u.OutputTokensDetails.ReasoningTokens),
	}
}
