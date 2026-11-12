package anthropicmodel

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/yoren041102/rain/schema"
)

func sse(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		var m map[string]any
		_ = json.Unmarshal([]byte(e), &m)
		b.WriteString("event: " + m["type"].(string) + "\ndata: " + e + "\n\n")
	}
	return b.String()
}

func server(t *testing.T, body string, gotReq *map[string]any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if gotReq != nil {
			_ = json.Unmarshal(raw, gotReq)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func client(url string) anthropic.Client {
	return anthropic.NewClient(option.WithBaseURL(url), option.WithAPIKey("k"), option.WithMaxRetries(0))
}

const start = `{"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":10,"output_tokens":1}}}`

func TestStreamToolCall(t *testing.T) {
	body := sse(
		start,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Let me "}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"check."}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"tu1","name":"get_weather","input":{}}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":20}}`,
		`{"type":"message_stop"}`,
	)
	var req map[string]any
	srv := server(t, body, &req)
	m := New(client(srv.URL), "claude-test")
	temp := 0.5
	mreq := &schema.ModelRequest{
		Messages: []*schema.Message{
			schema.SystemText("sys"),
			schema.UserText("weather?"),
		},
		Tools:   []schema.ToolSpec{{Name: "get_weather", Description: "w", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)}},
		Options: schema.ModelOptions{Temperature: &temp, ToolChoice: schema.ToolChoice{Mode: schema.ToolChoiceRequired}},
	}
	var partials []*schema.ModelResponse
	var final *schema.ModelResponse
	for r, err := range m.Generate(context.Background(), mreq) {
		if err != nil {
			t.Fatal(err)
		}
		if r.Partial {
			partials = append(partials, r)
		} else {
			final = r
		}
	}
	if final == nil {
		t.Fatal("no final response")
	}
	msg := final.Message
	if msg.Text() != "Let me check." || msg.StopReason != schema.StopReasonToolUse {
		t.Fatalf("msg = %v stop=%s", msg, msg.StopReason)
	}
	calls := msg.ToolCalls()
	if len(calls) != 1 || calls[0].ID != "tu1" || string(calls[0].Arguments) != `{"city":"Paris"}` {
		t.Fatalf("calls = %+v", calls)
	}
	rb := msg.Content[0].(*schema.ReasoningBlock)
	if rb.Text != "hmm" || rb.Signature != "SIG" {
		t.Fatalf("reasoning = %+v", rb)
	}
	if msg.Origin == nil || msg.Origin.Provider != "anthropic" || msg.Origin.Model != "claude-test" {
		t.Fatalf("origin = %+v", msg.Origin)
	}
	if final.Usage.InputTokens != 10 || final.Usage.OutputTokens != 20 {
		t.Fatalf("usage = %+v", final.Usage)
	}
	// partials accumulate to the same message
	acc := schema.NewAccumulator()
	for _, p := range partials {
		acc.Add(p)
	}
	am := acc.Message()
	if am.Text() != "Let me check." || string(am.ToolCalls()[0].Arguments) != `{"city":"Paris"}` {
		t.Fatalf("accumulated = %+v", am)
	}
	// request mapping
	if req["system"].([]any)[0].(map[string]any)["text"] != "sys" {
		t.Fatalf("system = %v", req["system"])
	}
	if req["tool_choice"].(map[string]any)["type"] != "any" || req["temperature"] != 0.5 {
		t.Fatalf("req = %v", req)
	}
	tool := req["tools"].([]any)[0].(map[string]any)
	if tool["name"] != "get_weather" || tool["input_schema"].(map[string]any)["required"].([]any)[0] != "city" {
		t.Fatalf("tool = %v", tool)
	}
}

func TestBuildParamsHistory(t *testing.T) {
	m := New(anthropic.NewClient(option.WithAPIKey("k")), "claude-test")
	same := m.Origin()
	other := schema.Origin{Provider: "openai", Model: "gpt"}
	req := &schema.ModelRequest{Messages: []*schema.Message{
		schema.UserText("q"),
		{Role: schema.RoleAssistant, Origin: &same, Content: []schema.ContentBlock{
			&schema.ReasoningBlock{Text: "t", Signature: "S"},
			&schema.ToolCallBlock{ID: "a", Name: "f", Arguments: json.RawMessage(`{"x":1}`)},
		}},
		schema.ToolResults(&schema.ToolResultBlock{CallID: "a", IsError: true, Content: []schema.ContentBlock{&schema.TextBlock{Text: "bad"}}}),
		schema.UserText("more"),
		{Role: schema.RoleAssistant, Origin: &other, Content: []schema.ContentBlock{
			&schema.ReasoningBlock{Text: "foreign"}, &schema.TextBlock{Text: "ans"},
			&schema.UnknownBlock{Type: "x", Raw: json.RawMessage(`{"type":"x"}`)},
		}},
	}}
	p, _, err := m.BuildParams(req)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(p)
	var got struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	_ = json.Unmarshal(b, &got)
	// tool result + following user text merge into one user turn
	if len(got.Messages) != 4 {
		t.Fatalf("messages = %s", b)
	}
	if got.Messages[1].Content[0]["type"] != "thinking" || got.Messages[1].Content[1]["type"] != "tool_use" {
		t.Fatalf("assistant = %v", got.Messages[1].Content)
	}
	u := got.Messages[2]
	if u.Role != "user" || u.Content[0]["type"] != "tool_result" || u.Content[0]["is_error"] != true || u.Content[1]["text"] != "more" {
		t.Fatalf("user = %v", u)
	}
	if last := got.Messages[3].Content; len(last) != 1 || last[0]["text"] != "ans" {
		t.Fatalf("foreign assistant = %v", last)
	}
}

func TestStructuredOutputFallback(t *testing.T) {
	body := sse(
		start,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"o1","name":"_rain_output","input":{}}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"answer\":42}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}`,
		`{"type":"message_stop"}`,
	)
	var req map[string]any
	srv := server(t, body, &req)
	m := New(client(srv.URL), "claude-test")
	resp, err := schema.FinalResponse(m.Generate(context.Background(), &schema.ModelRequest{
		Messages: []*schema.Message{schema.UserText("q")},
		Options:  schema.ModelOptions{OutputSchema: json.RawMessage(`{"type":"object","properties":{"answer":{"type":"integer"}}}`)},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Text() != `{"answer":42}` || len(resp.Message.ToolCalls()) != 0 || resp.Message.StopReason != schema.StopReasonStop {
		t.Fatalf("msg = %+v", resp.Message)
	}
	if req["tool_choice"].(map[string]any)["name"] != "_rain_output" {
		t.Fatalf("tool_choice = %v", req["tool_choice"])
	}
}

func TestAPIErrorWrapped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"nope"}}`)
	}))
	defer srv.Close()
	m := New(client(srv.URL), "claude-test")
	_, err := schema.FinalResponse(m.Generate(context.Background(), &schema.ModelRequest{Messages: []*schema.Message{schema.UserText("q")}}))
	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 {
		t.Fatalf("err = %v", err)
	}
}
