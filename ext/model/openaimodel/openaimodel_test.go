package openaimodel

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/yoren041102/rain/schema"
)

// ---- helpers ----

type sseEvent struct {
	name string // SSE "event:" field; empty for chat completions
	data string
}

type recorder struct {
	mu   sync.Mutex
	path string
	body []byte
}

func (r *recorder) json(t *testing.T) map[string]any {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	var m map[string]any
	if err := json.Unmarshal(r.body, &m); err != nil {
		t.Fatalf("request body is not JSON: %v\n%s", err, r.body)
	}
	return m
}

func sseServer(t *testing.T, events []sseEvent, done bool) (*httptest.Server, *recorder) {
	t.Helper()
	rec := &recorder{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		rec.mu.Lock()
		rec.path, rec.body = r.URL.Path, body
		rec.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			if e.name != "" {
				fmt.Fprintf(w, "event: %s\n", e.name)
			}
			fmt.Fprintf(w, "data: %s\n\n", e.data)
		}
		if done {
			fmt.Fprint(w, "data: [DONE]\n\n")
		}
	}))
	t.Cleanup(srv.Close)
	return srv, rec
}

func testClient(url string) openai.Client {
	return openai.NewClient(option.WithBaseURL(url), option.WithAPIKey("test-key"), option.WithMaxRetries(0))
}

func chunk(body string) sseEvent {
	return sseEvent{data: `{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o",` + body + `}`}
}

func collect(t *testing.T, s schema.Stream[*schema.ModelResponse]) (partials []*schema.ModelResponse, final *schema.ModelResponse) {
	t.Helper()
	for r, err := range s {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		if r.Partial {
			if final != nil {
				t.Fatalf("partial after final response")
			}
			partials = append(partials, r)
			continue
		}
		if final != nil {
			t.Fatalf("more than one final response")
		}
		final = r
	}
	if final == nil {
		t.Fatalf("no final response")
	}
	return partials, final
}

func get(t *testing.T, v any, path ...any) any {
	t.Helper()
	cur := v
	for _, p := range path {
		switch k := p.(type) {
		case string:
			m, ok := cur.(map[string]any)
			if !ok {
				t.Fatalf("path %v: %T is not an object", path, cur)
			}
			cur = m[k]
		case int:
			a, ok := cur.([]any)
			if !ok || k >= len(a) {
				t.Fatalf("path %v: cannot index %T (len) with %d", path, cur, k)
			}
			cur = a[k]
		}
	}
	return cur
}

func jsonEqual(t *testing.T, got, want string) {
	t.Helper()
	var g, w any
	if err := json.Unmarshal([]byte(got), &g); err != nil {
		t.Fatalf("got invalid JSON %q: %v", got, err)
	}
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("want invalid JSON %q: %v", want, err)
	}
	if !reflect.DeepEqual(g, w) {
		t.Fatalf("JSON mismatch\n got: %s\nwant: %s", got, want)
	}
}

func asJSON(t *testing.T, v any) map[string]any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, b)
	}
	return m
}

// ---- chat completions ----

func TestChatTextStream(t *testing.T) {
	srv, rec := sseServer(t, []sseEvent{
		chunk(`"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]`),
		chunk(`"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":1}}`),
	}, true)

	m := New(testClient(srv.URL), "gpt-4o")
	temp := 0.5
	req := &schema.ModelRequest{
		Messages: []*schema.Message{schema.SystemText("be brief"), schema.UserText("hi")},
		Options:  schema.ModelOptions{Temperature: &temp, MaxTokens: 100, Stop: []string{"END"}},
	}
	partials, final := collect(t, m.Generate(context.Background(), req))

	if len(partials) != 2 {
		t.Fatalf("got %d partials, want 2", len(partials))
	}
	for i, want := range []string{"Hel", "lo"} {
		p := partials[i]
		if p.BlockIndex != 0 || p.BlockType != schema.BlockText {
			t.Errorf("partial %d: index %d type %s", i, p.BlockIndex, p.BlockType)
		}
		if got := p.Message.Text(); got != want {
			t.Errorf("partial %d text %q, want %q", i, got, want)
		}
	}

	msg := final.Message
	if msg.Role != schema.RoleAssistant || msg.Text() != "Hello" || len(msg.Content) != 1 {
		t.Fatalf("final message = %+v", msg)
	}
	wantOrigin := schema.Origin{Provider: "openai", API: "chat_completions", Model: "gpt-4o"}
	if msg.Origin == nil || *msg.Origin != wantOrigin {
		t.Errorf("origin = %+v", msg.Origin)
	}
	if msg.StopReason != schema.StopReasonStop || final.FinishReason != schema.StopReasonStop {
		t.Errorf("stop = %q finish = %q", msg.StopReason, final.FinishReason)
	}
	wantUsage := schema.Usage{InputTokens: 10, OutputTokens: 2, TotalTokens: 12, CacheReadTokens: 4, ReasoningTokens: 1}
	if final.Usage == nil || *final.Usage != wantUsage {
		t.Errorf("usage = %+v, want %+v", final.Usage, wantUsage)
	}

	// The partials alone accumulate to the same content.
	acc := schema.NewAccumulator()
	for _, p := range partials {
		acc.Add(p)
	}
	if acc.Message().Text() != "Hello" {
		t.Errorf("accumulated partials = %q", acc.Message().Text())
	}

	if rec.path != "/chat/completions" {
		t.Errorf("path = %q", rec.path)
	}
	body := rec.json(t)
	if body["model"] != "gpt-4o" || body["stream"] != true {
		t.Errorf("model/stream = %v %v", body["model"], body["stream"])
	}
	if get(t, body, "stream_options", "include_usage") != true {
		t.Errorf("stream_options = %v", body["stream_options"])
	}
	if body["temperature"] != 0.5 || body["max_completion_tokens"] != float64(100) {
		t.Errorf("temperature/max_completion_tokens = %v %v", body["temperature"], body["max_completion_tokens"])
	}
	if !reflect.DeepEqual(body["stop"], []any{"END"}) {
		t.Errorf("stop = %v", body["stop"])
	}
	wantMsgs := []any{
		map[string]any{"role": "system", "content": "be brief"},
		map[string]any{"role": "user", "content": "hi"},
	}
	if !reflect.DeepEqual(body["messages"], wantMsgs) {
		t.Errorf("messages = %v", body["messages"])
	}
	if _, ok := body["tools"]; ok {
		t.Errorf("unexpected tools")
	}
}

func TestChatToolCallStream(t *testing.T) {
	srv, rec := sseServer(t, []sseEvent{
		chunk(`"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_time","arguments":"{}"}}]},"finish_reason":null}]`),
		chunk(`"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]`),
		chunk(`"choices":[],"usage":{"prompt_tokens":30,"completion_tokens":15,"total_tokens":45}`),
	}, true)

	m := New(testClient(srv.URL), "gpt-4o")
	req := &schema.ModelRequest{
		Messages: []*schema.Message{schema.UserText("weather in Paris?")},
		Tools: []schema.ToolSpec{
			{Name: "get_weather", Description: "Get weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)},
			{Name: "get_time"},
		},
		Options: schema.ModelOptions{ToolChoice: schema.ToolChoice{Mode: schema.ToolChoiceRequired}},
	}
	partials, final := collect(t, m.Generate(context.Background(), req))

	type want struct {
		idx            int
		id, name, args string
	}
	wants := []want{
		{0, "call_1", "get_weather", ""},
		{0, "", "", `{"ci`},
		{0, "", "", `ty":"Paris"}`},
		{1, "call_2", "get_time", `{}`},
	}
	if len(partials) != len(wants) {
		t.Fatalf("got %d partials, want %d", len(partials), len(wants))
	}
	for i, w := range wants {
		p := partials[i]
		if p.BlockIndex != w.idx || p.BlockType != schema.BlockToolCall {
			t.Errorf("partial %d: index %d type %s", i, p.BlockIndex, p.BlockType)
		}
		tc := p.Message.Content[0].(*schema.ToolCallBlock)
		if tc.ID != w.id || tc.Name != w.name || string(tc.Arguments) != w.args {
			t.Errorf("partial %d = %+v, want %+v", i, tc, w)
		}
	}

	msg := final.Message
	calls := msg.ToolCalls()
	if len(calls) != 2 {
		t.Fatalf("tool calls = %d", len(calls))
	}
	if calls[0].ID != "call_1" || calls[0].Name != "get_weather" {
		t.Errorf("call 0 = %+v", calls[0])
	}
	jsonEqual(t, string(calls[0].Arguments), `{"city":"Paris"}`)
	if calls[1].ID != "call_2" || calls[1].Name != "get_time" || string(calls[1].Arguments) != "{}" {
		t.Errorf("call 1 = %+v", calls[1])
	}
	if msg.StopReason != schema.StopReasonToolUse || final.FinishReason != schema.StopReasonToolUse {
		t.Errorf("stop = %q", msg.StopReason)
	}
	if final.Usage == nil || final.Usage.InputTokens != 30 || final.Usage.OutputTokens != 15 || final.Usage.TotalTokens != 45 {
		t.Errorf("usage = %+v", final.Usage)
	}

	acc := schema.NewAccumulator()
	for _, p := range partials {
		acc.Add(p)
	}
	accCalls := acc.Message().ToolCalls()
	if len(accCalls) != 2 || string(accCalls[0].Arguments) != `{"city":"Paris"}` || accCalls[0].Name != "get_weather" {
		t.Errorf("accumulated partial calls = %+v", accCalls)
	}

	body := rec.json(t)
	if body["tool_choice"] != "required" {
		t.Errorf("tool_choice = %v", body["tool_choice"])
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools = %v", body["tools"])
	}
	if get(t, tools, 0, "type") != "function" || get(t, tools, 0, "function", "name") != "get_weather" ||
		get(t, tools, 0, "function", "description") != "Get weather" {
		t.Errorf("tool 0 = %v", tools[0])
	}
	params, _ := json.Marshal(get(t, tools, 0, "function", "parameters"))
	jsonEqual(t, string(params), `{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)
	params, _ = json.Marshal(get(t, tools, 1, "function", "parameters"))
	jsonEqual(t, string(params), `{"type":"object","properties":{}}`)
}

func TestChatRequestMapping(t *testing.T) {
	m := New(openai.NewClient(option.WithAPIKey("x")), "gpt-4o",
		WithParams(func(p *openai.ChatCompletionNewParams) { p.User = openai.String("u-1") }))
	openaiOrigin := &schema.Origin{Provider: "openai", API: "chat_completions", Model: "gpt-4o"}
	req := &schema.ModelRequest{
		Messages: []*schema.Message{
			schema.SystemText("sys"),
			{Role: schema.RoleUser, Content: []schema.ContentBlock{
				&schema.TextBlock{Text: "look"},
				&schema.ImageBlock{URL: "https://example.com/a.png"},
				&schema.ImageBlock{MIMEType: "image/png", Data: []byte{1, 2, 3}},
			}},
			{Role: schema.RoleAssistant, Origin: &schema.Origin{Provider: "anthropic", API: "messages", Model: "claude"}, Content: []schema.ContentBlock{
				&schema.ReasoningBlock{Text: "hmm", Signature: "sig"},
				&schema.TextBlock{Text: "Checking."},
				&schema.ToolCallBlock{ID: "c1", Name: "search", Arguments: json.RawMessage(`{"q":"go"}`)},
				&schema.ToolCallBlock{ID: "c2", Name: "noargs"},
				&schema.UnknownBlock{Type: "server_tool_use", Raw: json.RawMessage(`{"type":"server_tool_use"}`)},
			}},
			schema.ToolResults(
				&schema.ToolResultBlock{CallID: "c1", Content: []schema.ContentBlock{&schema.TextBlock{Text: "line1"}, &schema.TextBlock{Text: "line2"}}, Details: []schema.ContentBlock{&schema.TextBlock{Text: "ui only"}}},
				&schema.ToolResultBlock{CallID: "c2", Content: []schema.ContentBlock{&schema.TextBlock{Text: "boom"}}, IsError: true},
			),
			{Role: schema.RoleAssistant, Origin: openaiOrigin, StopReason: schema.StopReasonStop, Content: []schema.ContentBlock{
				&schema.TextBlock{Text: "Sorry: "},
				&schema.UnknownBlock{Type: "refusal", Raw: json.RawMessage(`{"type":"refusal","refusal":"no"}`)},
			}},
		},
		Tools: []schema.ToolSpec{{Name: "search", Parameters: json.RawMessage(`{"type":"object"}`)}},
		Options: schema.ModelOptions{
			ToolChoice:   schema.ToolChoice{Mode: schema.ToolChoiceTool, Name: "search"},
			OutputSchema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}}}`),
			Extra:        map[string]any{"seed": 7},
		},
	}
	p, err := m.chatParams(req)
	if err != nil {
		t.Fatal(err)
	}
	body := asJSON(t, p)

	msgs, _ := body["messages"].([]any)
	if len(msgs) != 6 {
		b, _ := json.MarshalIndent(msgs, "", "  ")
		t.Fatalf("got %d messages, want 6:\n%s", len(msgs), b)
	}
	if !reflect.DeepEqual(msgs[0], map[string]any{"role": "system", "content": "sys"}) {
		t.Errorf("system = %v", msgs[0])
	}

	user := msgs[1]
	if get(t, user, "role") != "user" || get(t, user, "content", 0, "type") != "text" || get(t, user, "content", 0, "text") != "look" {
		t.Errorf("user = %v", user)
	}
	if get(t, user, "content", 1, "image_url", "url") != "https://example.com/a.png" {
		t.Errorf("image url part = %v", get(t, user, "content", 1))
	}
	if get(t, user, "content", 2, "image_url", "url") != "data:image/png;base64,AQID" {
		t.Errorf("image data part = %v", get(t, user, "content", 2))
	}

	asst := msgs[2]
	if get(t, asst, "role") != "assistant" || get(t, asst, "content") != "Checking." {
		t.Errorf("assistant = %v (reasoning and foreign unknown blocks must be dropped)", asst)
	}
	if get(t, asst, "tool_calls", 0, "id") != "c1" || get(t, asst, "tool_calls", 0, "type") != "function" ||
		get(t, asst, "tool_calls", 0, "function", "name") != "search" || get(t, asst, "tool_calls", 0, "function", "arguments") != `{"q":"go"}` {
		t.Errorf("tool call 0 = %v", get(t, asst, "tool_calls", 0))
	}
	if get(t, asst, "tool_calls", 1, "function", "arguments") != "{}" {
		t.Errorf("tool call 1 = %v", get(t, asst, "tool_calls", 1))
	}

	if !reflect.DeepEqual(msgs[3], map[string]any{"role": "tool", "tool_call_id": "c1", "content": "line1\nline2"}) {
		t.Errorf("tool result 1 = %v", msgs[3])
	}
	if !reflect.DeepEqual(msgs[4], map[string]any{"role": "tool", "tool_call_id": "c2", "content": "boom"}) {
		t.Errorf("tool result 2 = %v", msgs[4])
	}

	// Unknown blocks from the same provider/API are replayed verbatim.
	wantRefusal := map[string]any{"role": "assistant", "content": []any{
		map[string]any{"type": "text", "text": "Sorry: "},
		map[string]any{"type": "refusal", "refusal": "no"},
	}}
	if !reflect.DeepEqual(msgs[5], wantRefusal) {
		t.Errorf("openai assistant = %v", msgs[5])
	}

	if !reflect.DeepEqual(body["tool_choice"], map[string]any{"type": "function", "function": map[string]any{"name": "search"}}) {
		t.Errorf("tool_choice = %v", body["tool_choice"])
	}
	rf := body["response_format"]
	if get(t, rf, "type") != "json_schema" || get(t, rf, "json_schema", "name") != "output" ||
		get(t, rf, "json_schema", "schema", "properties", "answer", "type") != "string" {
		t.Errorf("response_format = %v", rf)
	}
	if body["seed"] != float64(7) || body["user"] != "u-1" {
		t.Errorf("seed/user = %v %v", body["seed"], body["user"])
	}
}

func TestChatAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"bad model","type":"invalid_request_error","code":"model_not_found"}}`)
	}))
	defer srv.Close()

	m := New(testClient(srv.URL), "nope")
	_, err := schema.FinalResponse(m.Generate(context.Background(), &schema.ModelRequest{Messages: []*schema.Message{schema.UserText("hi")}}))
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.HasPrefix(err.Error(), "openaimodel: ") {
		t.Errorf("error not wrapped: %v", err)
	}
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		t.Fatalf("errors.As(*openai.Error) failed for %T: %v", err, err)
	}
	if apiErr.StatusCode != http.StatusBadRequest || apiErr.Code != "model_not_found" {
		t.Errorf("api error = %d %q", apiErr.StatusCode, apiErr.Code)
	}
}

func TestChatEarlyBreakClosesStream(t *testing.T) {
	closed := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\n", chunk(`"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]`).data)
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
			close(closed)
		case <-time.After(5 * time.Second):
		}
	}))
	defer srv.Close()

	m := New(testClient(srv.URL), "gpt-4o")
	n := 0
	for r, err := range m.Generate(context.Background(), &schema.ModelRequest{Messages: []*schema.Message{schema.UserText("hi")}}) {
		if err != nil {
			t.Fatal(err)
		}
		if r.Partial {
			n++
			break
		}
	}
	if n != 1 {
		t.Fatalf("got %d partials", n)
	}
	select {
	case <-closed:
	case <-time.After(3 * time.Second):
		t.Fatal("server connection not closed after consumer break")
	}
}

func TestChatContextCanceled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	srv, _ := sseServer(t, nil, true)
	m := New(testClient(srv.URL), "gpt-4o")
	_, err := schema.FinalResponse(m.Generate(ctx, &schema.ModelRequest{Messages: []*schema.Message{schema.UserText("hi")}}))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

// ---- responses API ----

func respEvent(typ, body string) sseEvent {
	return sseEvent{name: typ, data: `{"type":"` + typ + `",` + body + `}`}
}

func TestResponsesStream(t *testing.T) {
	srv, rec := sseServer(t, []sseEvent{
		respEvent("response.created", `"sequence_number":0,"response":{"id":"resp_1","object":"response","status":"in_progress","output":[]}`),
		respEvent("response.output_item.added", `"sequence_number":1,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[]}`),
		respEvent("response.reasoning_summary_text.delta", `"sequence_number":2,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"Think"`),
		respEvent("response.reasoning_summary_text.delta", `"sequence_number":3,"item_id":"rs_1","output_index":0,"summary_index":0,"delta":"ing"`),
		respEvent("response.output_item.done", `"sequence_number":4,"output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"Thinking"}]}`),
		respEvent("response.output_item.added", `"sequence_number":5,"output_index":1,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}`),
		respEvent("response.output_text.delta", `"sequence_number":6,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"Hi ","logprobs":[]`),
		respEvent("response.output_text.delta", `"sequence_number":7,"item_id":"msg_1","output_index":1,"content_index":0,"delta":"there","logprobs":[]`),
		respEvent("response.output_item.done", `"sequence_number":8,"output_index":1,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hi there","annotations":[]}]}`),
		respEvent("response.output_item.added", `"sequence_number":9,"output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_9","name":"lookup","arguments":"","status":"in_progress"}`),
		respEvent("response.function_call_arguments.delta", `"sequence_number":10,"item_id":"fc_1","output_index":2,"delta":"{\"q\":"`),
		respEvent("response.function_call_arguments.delta", `"sequence_number":11,"item_id":"fc_1","output_index":2,"delta":"\"go\"}"`),
		respEvent("response.output_item.done", `"sequence_number":12,"output_index":2,"item":{"id":"fc_1","type":"function_call","call_id":"call_9","name":"lookup","arguments":"{\"q\":\"go\"}","status":"completed"}`),
		respEvent("response.output_item.done", `"sequence_number":13,"output_index":3,"item":{"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"go"}}`),
		respEvent("response.completed", `"sequence_number":14,"response":{"id":"resp_1","object":"response","status":"completed","output":[],"usage":{"input_tokens":20,"input_tokens_details":{"cached_tokens":5},"output_tokens":7,"output_tokens_details":{"reasoning_tokens":3},"total_tokens":27}}`),
	}, false)

	m := New(testClient(srv.URL), "o4-mini", WithAPI(Responses))
	respOrigin := &schema.Origin{Provider: "openai", API: "responses", Model: "o4-mini"}
	req := &schema.ModelRequest{
		Messages: []*schema.Message{
			schema.SystemText("sys"),
			schema.UserText("find go"),
			{Role: schema.RoleAssistant, Origin: respOrigin, Content: []schema.ContentBlock{
				&schema.ReasoningBlock{Text: "earlier", Signature: "enc"},
				&schema.TextBlock{Text: "Searching."},
				&schema.ToolCallBlock{ID: "call_1", Name: "lookup", Arguments: json.RawMessage(`{"q":"x"}`)},
				&schema.UnknownBlock{Type: "web_search_call", Raw: json.RawMessage(`{"id":"ws_0","type":"web_search_call","status":"completed"}`)},
			}},
			schema.ToolResults(&schema.ToolResultBlock{CallID: "call_1", Content: []schema.ContentBlock{&schema.TextBlock{Text: "result"}}}),
		},
		Tools: []schema.ToolSpec{{Name: "lookup", Description: "Look up", Parameters: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)}},
		Options: schema.ModelOptions{
			MaxTokens:    256,
			Stop:         []string{"ignored"},
			ToolChoice:   schema.ToolChoice{Mode: schema.ToolChoiceTool, Name: "lookup"},
			OutputSchema: json.RawMessage(`{"type":"object"}`),
		},
	}
	partials, final := collect(t, m.Generate(context.Background(), req))

	// Block layout: 0 reasoning, 1 text, 2 tool call, 3 unknown.
	type want struct {
		idx int
		typ schema.BlockType
	}
	wants := []want{
		{0, schema.BlockReasoning}, {0, schema.BlockReasoning},
		{1, schema.BlockText}, {1, schema.BlockText},
		{2, schema.BlockToolCall}, {2, schema.BlockToolCall}, {2, schema.BlockToolCall},
		{3, schema.BlockUnknown},
	}
	if len(partials) != len(wants) {
		for _, p := range partials {
			t.Logf("partial %d %s %#v", p.BlockIndex, p.BlockType, p.Message.Content[0])
		}
		t.Fatalf("got %d partials, want %d", len(partials), len(wants))
	}
	for i, w := range wants {
		if partials[i].BlockIndex != w.idx || partials[i].BlockType != w.typ {
			t.Errorf("partial %d: %d/%s, want %d/%s", i, partials[i].BlockIndex, partials[i].BlockType, w.idx, w.typ)
		}
	}
	if tc := partials[4].Message.Content[0].(*schema.ToolCallBlock); tc.ID != "call_9" || tc.Name != "lookup" {
		t.Errorf("first tool partial = %+v", tc)
	}

	msg := final.Message
	if len(msg.Content) != 4 {
		t.Fatalf("final content = %d blocks: %+v", len(msg.Content), msg.Content)
	}
	if r, ok := msg.Content[0].(*schema.ReasoningBlock); !ok || r.Text != "Thinking" {
		t.Errorf("block 0 = %#v", msg.Content[0])
	}
	if msg.Text() != "Hi there" {
		t.Errorf("text = %q", msg.Text())
	}
	tc, ok := msg.Content[2].(*schema.ToolCallBlock)
	if !ok || tc.ID != "call_9" || tc.Name != "lookup" {
		t.Fatalf("block 2 = %#v", msg.Content[2])
	}
	jsonEqual(t, string(tc.Arguments), `{"q":"go"}`)
	if u, ok := msg.Content[3].(*schema.UnknownBlock); !ok || u.Type != "web_search_call" || !strings.Contains(string(u.Raw), `"ws_1"`) {
		t.Errorf("block 3 = %#v", msg.Content[3])
	}
	if msg.Origin == nil || *msg.Origin != *respOrigin {
		t.Errorf("origin = %+v", msg.Origin)
	}
	if msg.StopReason != schema.StopReasonToolUse || final.FinishReason != schema.StopReasonToolUse {
		t.Errorf("stop = %q", msg.StopReason)
	}
	wantUsage := schema.Usage{InputTokens: 20, OutputTokens: 7, TotalTokens: 27, CacheReadTokens: 5, ReasoningTokens: 3}
	if final.Usage == nil || *final.Usage != wantUsage {
		t.Errorf("usage = %+v", final.Usage)
	}

	acc := schema.NewAccumulator()
	for _, p := range partials {
		acc.Add(p)
	}
	am := acc.Message()
	if am.Text() != "Hi there" || len(am.ToolCalls()) != 1 || string(am.ToolCalls()[0].Arguments) != `{"q":"go"}` {
		t.Errorf("accumulated partials = %+v", am.Content)
	}

	if rec.path != "/responses" {
		t.Errorf("path = %q", rec.path)
	}
	body := rec.json(t)
	if body["model"] != "o4-mini" || body["stream"] != true || body["max_output_tokens"] != float64(256) {
		t.Errorf("model/stream/max = %v %v %v", body["model"], body["stream"], body["max_output_tokens"])
	}
	if _, ok := body["stop"]; ok {
		t.Errorf("stop must not be sent to the Responses API")
	}
	input, _ := body["input"].([]any)
	wantInput := []any{
		map[string]any{"role": "system", "content": "sys"},
		map[string]any{"role": "user", "content": []any{map[string]any{"type": "input_text", "text": "find go"}}},
		map[string]any{"role": "assistant", "content": "Searching."},
		map[string]any{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": `{"q":"x"}`},
		map[string]any{"id": "ws_0", "type": "web_search_call", "status": "completed"},
		map[string]any{"type": "function_call_output", "call_id": "call_1", "output": "result"},
	}
	if !reflect.DeepEqual(input, wantInput) {
		b, _ := json.MarshalIndent(input, "", "  ")
		t.Errorf("input =\n%s", b)
	}
	tool := get(t, body, "tools", 0)
	if get(t, tool, "type") != "function" || get(t, tool, "name") != "lookup" || get(t, tool, "strict") != false ||
		get(t, tool, "parameters", "properties", "q", "type") != "string" {
		t.Errorf("tool = %v", tool)
	}
	if !reflect.DeepEqual(body["tool_choice"], map[string]any{"type": "function", "name": "lookup"}) {
		t.Errorf("tool_choice = %v", body["tool_choice"])
	}
	if get(t, body, "text", "format", "type") != "json_schema" || get(t, body, "text", "format", "name") != "output" {
		t.Errorf("text.format = %v", body["text"])
	}
}

func TestResponsesIncompleteAndFailed(t *testing.T) {
	srv, _ := sseServer(t, []sseEvent{
		respEvent("response.output_text.delta", `"sequence_number":1,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"partial","logprobs":[]`),
		respEvent("response.incomplete", `"sequence_number":2,"response":{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}`),
	}, false)
	m := New(testClient(srv.URL), "gpt-4.1", WithAPI(Responses))
	_, final := collect(t, m.Generate(context.Background(), &schema.ModelRequest{Messages: []*schema.Message{schema.UserText("hi")}}))
	if final.Message.StopReason != schema.StopReasonLength || final.Message.Text() != "partial" {
		t.Errorf("final = %+v", final.Message)
	}

	srv2, _ := sseServer(t, []sseEvent{
		respEvent("response.failed", `"sequence_number":1,"response":{"id":"r","status":"failed","error":{"code":"server_error","message":"kaput"},"output":[]}`),
	}, false)
	m2 := New(testClient(srv2.URL), "gpt-4.1", WithAPI(Responses))
	_, err := schema.FinalResponse(m2.Generate(context.Background(), &schema.ModelRequest{Messages: []*schema.Message{schema.UserText("hi")}}))
	if err == nil || !strings.Contains(err.Error(), "kaput") || !strings.HasPrefix(err.Error(), "openaimodel: ") {
		t.Errorf("err = %v", err)
	}
}

func TestOrigin(t *testing.T) {
	m := New(openai.NewClient(), "gpt-4o", WithAPI(Responses))
	if got := schema.ModelOrigin(m); got != (schema.Origin{Provider: "openai", API: "responses", Model: "gpt-4o"}) {
		t.Errorf("origin = %+v", got)
	}
	if m.Name() != "gpt-4o" {
		t.Errorf("name = %q", m.Name())
	}
}
