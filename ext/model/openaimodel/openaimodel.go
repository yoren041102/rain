// Package openaimodel adapts the official OpenAI Go SDK (openai-go) to
// rain's schema.Model interface.
//
// Both the Chat Completions API (default) and the Responses API are
// supported; select one with WithAPI. Every call streams: partial responses
// carry per-block deltas, and the stream always ends with exactly one
// non-partial response holding the complete aggregated message.
package openaimodel

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"github.com/yoren041102/rain/schema"
)

// Provider is the Origin.Provider value stamped on messages produced by this
// adapter.
const Provider = "openai"

// API selects which OpenAI endpoint the adapter calls. Its value is also used
// as Origin.API.
type API string

const (
	// ChatCompletions uses POST /chat/completions (the default).
	ChatCompletions API = "chat_completions"
	// Responses uses POST /responses.
	Responses API = "responses"
)

// Option configures a Model.
type Option func(*Model)

// WithAPI selects the OpenAI API to use. The default is ChatCompletions.
func WithAPI(api API) Option {
	return func(m *Model) { m.api = api }
}

// WithRequestOptions appends SDK request options passed to every call.
func WithRequestOptions(opts ...option.RequestOption) Option {
	return func(m *Model) { m.reqOpts = append(m.reqOpts, opts...) }
}

// WithParams registers a hook that can modify the Chat Completions request
// parameters after rain's request has been mapped onto them. Hooks run in
// registration order.
func WithParams(fn func(*openai.ChatCompletionNewParams)) Option {
	return func(m *Model) {
		if fn != nil {
			m.chatHooks = append(m.chatHooks, fn)
		}
	}
}

// WithResponsesParams registers a hook that can modify the Responses API
// request parameters after rain's request has been mapped onto them. Hooks
// run in registration order.
func WithResponsesParams(fn func(*responses.ResponseNewParams)) Option {
	return func(m *Model) {
		if fn != nil {
			m.respHooks = append(m.respHooks, fn)
		}
	}
}

// Model implements schema.Model and schema.OriginProvider on top of an
// openai.Client.
type Model struct {
	client    openai.Client
	model     string
	api       API
	reqOpts   []option.RequestOption
	chatHooks []func(*openai.ChatCompletionNewParams)
	respHooks []func(*responses.ResponseNewParams)
}

var (
	_ schema.Model          = (*Model)(nil)
	_ schema.OriginProvider = (*Model)(nil)
)

// New returns a Model calling model (e.g. "gpt-4o") through client.
func New(client openai.Client, model string, opts ...Option) *Model {
	m := &Model{client: client, model: model, api: ChatCompletions}
	for _, o := range opts {
		if o != nil {
			o(m)
		}
	}
	return m
}

// Name returns the model name.
func (m *Model) Name() string { return m.model }

// API returns the API the model calls.
func (m *Model) API() API { return m.api }

// Origin reports the Origin stamped on generated messages.
func (m *Model) Origin() schema.Origin {
	return schema.Origin{Provider: Provider, API: string(m.api), Model: m.model}
}

// Generate streams a model response. See the package documentation for the
// streaming protocol.
func (m *Model) Generate(ctx context.Context, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
	return func(yield func(*schema.ModelResponse, error) bool) {
		if req == nil {
			yield(nil, wrapErr(errors.New("nil request")))
			return
		}
		switch m.api {
		case ChatCompletions, "":
			m.generateChat(ctx, req, yield)
		case Responses:
			m.generateResponses(ctx, req, yield)
		default:
			yield(nil, wrapErr(fmt.Errorf("unknown API %q", m.api)))
		}
	}
}

func wrapErr(err error) error { return fmt.Errorf("openaimodel: %w", err) }

func (m *Model) origin() *schema.Origin {
	o := m.Origin()
	return &o
}

// passUnknown reports whether UnknownBlocks of msg may be sent back to api:
// only blocks this adapter produced for the same API are replayed.
func passUnknown(msg *schema.Message, api API) bool {
	return msg.Origin != nil && msg.Origin.Provider == Provider && msg.Origin.API == string(api)
}

// ---- shared helpers ----

// argsString converts tool call arguments to the string form OpenAI expects.
func argsString(raw json.RawMessage) string {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return "{}"
	}
	// Invalid arguments are persisted by schema as a JSON string; unwrap it.
	if raw[0] == '"' {
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return s
		}
	}
	return string(raw)
}

// toolResultText flattens the model-facing content of a tool result to text.
func toolResultText(r *schema.ToolResultBlock) string {
	var parts []string
	for _, b := range r.Content {
		switch b := b.(type) {
		case *schema.TextBlock:
			parts = append(parts, b.Text)
		case *schema.ImageBlock:
			if b.URL != "" {
				parts = append(parts, "[image: "+b.URL+"]")
			} else {
				parts = append(parts, "[image]")
			}
		case *schema.FileBlock:
			parts = append(parts, "[file: "+firstNonEmpty(b.Name, b.URL, "attachment")+"]")
		}
	}
	return strings.Join(parts, "\n")
}

// joinText concatenates the text blocks of content.
func joinText(content []schema.ContentBlock) string {
	var sb strings.Builder
	for _, b := range content {
		if t, ok := b.(*schema.TextBlock); ok {
			sb.WriteString(t.Text)
		}
	}
	return sb.String()
}

func dataURL(mime string, data []byte) string {
	if mime == "" {
		mime = http.DetectContentType(data)
	}
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func imageURL(b *schema.ImageBlock) string {
	if b.URL != "" {
		return b.URL
	}
	if len(b.Data) > 0 {
		return dataURL(b.MIMEType, b.Data)
	}
	return ""
}

// metaString returns a string-valued BlockMeta.Extra entry.
func metaString(m schema.BlockMeta, key string) string {
	raw, ok := m.Extra[key]
	if !ok {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return ""
	}
	return s
}

func firstNonEmpty(ss ...string) string {
	for _, s := range ss {
		if s != "" {
			return s
		}
	}
	return ""
}

// schemaMap decodes a JSON Schema into the map form the SDK expects. An empty
// schema becomes an empty object schema.
func schemaMap(raw json.RawMessage) (map[string]any, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return map[string]any{"type": "object", "properties": map[string]any{}}, nil
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, fmt.Errorf("invalid JSON schema: %w", err)
	}
	return m, nil
}

// fallbackStop picks a stop reason when the provider reported none.
func fallbackStop(blocks []schema.ContentBlock) schema.StopReason {
	for _, b := range blocks {
		if _, ok := b.(*schema.ToolCallBlock); ok {
			return schema.StopReasonToolUse
		}
	}
	return schema.StopReasonStop
}

// finishMessage builds the final aggregated assistant message from blocks.
func finishMessage(blocks []schema.ContentBlock, origin *schema.Origin, stop schema.StopReason) *schema.Message {
	msg := &schema.Message{Role: schema.RoleAssistant, Origin: origin, StopReason: stop}
	for _, b := range blocks {
		if b == nil {
			continue
		}
		if tc, ok := b.(*schema.ToolCallBlock); ok {
			c := *tc
			c.Arguments = append(json.RawMessage(nil), tc.Arguments...)
			if len(bytes.TrimSpace(c.Arguments)) == 0 {
				c.Arguments = json.RawMessage("{}")
			}
			b = &c
		}
		msg.Content = append(msg.Content, b)
	}
	return msg
}

func partial(idx int, b schema.ContentBlock) *schema.ModelResponse {
	return &schema.ModelResponse{
		Message:    &schema.Message{Role: schema.RoleAssistant, Content: []schema.ContentBlock{b}},
		Partial:    true,
		BlockIndex: idx,
		BlockType:  b.BlockType(),
	}
}
