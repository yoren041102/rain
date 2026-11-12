package schema

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestMessageJSONRoundTrip(t *testing.T) {
	m := &Message{
		Role: RoleAssistant,
		Content: []ContentBlock{
			&TextBlock{Text: "我查一下。", Meta: BlockMeta{Extra: map[string]json.RawMessage{"cache_control": json.RawMessage(`{"type":"ephemeral"}`)}}},
			&ReasoningBlock{Text: "think", Signature: "sig"},
			&ToolCallBlock{ID: "c1", Name: "search", Arguments: json.RawMessage(`{"q":"go iter"}`)},
			&ImageBlock{MIMEType: "image/png", Data: []byte{1, 2, 3}},
			&UnknownBlock{Type: "server_tool_use", Raw: json.RawMessage(`{"a":1}`)},
		},
		Origin:     &Origin{Provider: "anthropic", API: "messages", Model: "claude-sonnet-5-5"},
		StopReason: StopReasonToolUse,
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"type":"tool_call"`) || !strings.Contains(string(b), `"stop_reason":"tool_use"`) {
		t.Fatalf("json = %s", b)
	}
	var got Message
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	b2, _ := json.Marshal(&got)
	if string(b) != string(b2) {
		t.Fatalf("round trip mismatch:\n%s\n%s", b, b2)
	}
	if u, ok := got.Content[4].(*UnknownBlock); !ok || u.Type != "server_tool_use" || string(u.Raw) != `{"a":1}` {
		t.Fatalf("unknown = %#v", got.Content[4])
	}
	if string(got.Content[0].BlockMeta().Extra["cache_control"]) != `{"type":"ephemeral"}` {
		t.Fatal("meta lost")
	}
}

func TestUnknownTypeFromForeignJSON(t *testing.T) {
	var m Message
	if err := json.Unmarshal([]byte(`{"role":"assistant","content":[{"type":"citation","x":1}]}`), &m); err != nil {
		t.Fatal(err)
	}
	u := m.Content[0].(*UnknownBlock)
	if u.Type != "citation" || !strings.Contains(string(u.Raw), `"x":1`) {
		t.Fatalf("got %#v", u)
	}
}

func TestToolResultNested(t *testing.T) {
	m := ToolResults(&ToolResultBlock{CallID: "c1", Content: []ContentBlock{&TextBlock{Text: "ok"}}, Details: []ContentBlock{&TextBlock{Text: "ui"}}, IsError: true})
	b, _ := json.Marshal(m)
	var got Message
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	r := got.ToolResultBlocks()[0]
	if r.CallID != "c1" || !r.IsError || r.Content[0].(*TextBlock).Text != "ok" || r.Details[0].(*TextBlock).Text != "ui" {
		t.Fatalf("got %#v", r)
	}
}

func TestAccumulator(t *testing.T) {
	a := NewAccumulator()
	a.Add(&ModelResponse{Partial: true, BlockIndex: 0, Message: &Message{Content: []ContentBlock{&TextBlock{Text: "Hel"}}}})
	a.Add(&ModelResponse{Partial: true, BlockIndex: 0, Message: &Message{Content: []ContentBlock{&TextBlock{Text: "lo"}}}})
	a.Add(&ModelResponse{Partial: true, BlockIndex: 1, Message: &Message{Content: []ContentBlock{&ToolCallBlock{ID: "c1", Name: "f"}}}})
	a.Add(&ModelResponse{Partial: true, BlockIndex: 1, Message: &Message{Content: []ContentBlock{&ToolCallBlock{Arguments: json.RawMessage(`{"a":`)}}}})
	a.Add(&ModelResponse{Partial: true, BlockIndex: 1, Message: &Message{Content: []ContentBlock{&ToolCallBlock{Arguments: json.RawMessage(`1}`)}}}})
	a.Add(&ModelResponse{Partial: true, Usage: &Usage{InputTokens: 3, OutputTokens: 5}})
	m := a.Message()
	if m.Text() != "Hello" || len(m.ToolCalls()) != 1 || string(m.ToolCalls()[0].Arguments) != `{"a":1}` {
		t.Fatalf("got %+v", m)
	}
	if m.StopReason != StopReasonToolUse || a.Usage().OutputTokens != 5 {
		t.Fatalf("stop=%s usage=%+v", m.StopReason, a.Usage())
	}
}

func TestPrepareForModel(t *testing.T) {
	anth := Origin{Provider: "anthropic", API: "messages", Model: "m1"}
	other := Origin{Provider: "openai", API: "chat_completions", Model: "gpt"}
	msgs := []*Message{
		UserText("hi"),
		{Role: RoleAssistant, Origin: &anth, StopReason: StopReasonToolUse, Content: []ContentBlock{
			&ReasoningBlock{Text: "t", Signature: "sig"},
			&ToolCallBlock{ID: "c1", Name: "a"}, &ToolCallBlock{ID: "c2", Name: "b"},
		}},
		ToolResults(&ToolResultBlock{CallID: "c2", Content: []ContentBlock{&TextBlock{Text: "r2"}}}),
		{Role: RoleAssistant, StopReason: StopReasonError, Content: []ContentBlock{&TextBlock{Text: "broken"}}},
		UserText("again"),
	}
	out := PrepareForModel(msgs, other)
	if len(out) != 4 {
		t.Fatalf("len = %d", len(out))
	}
	if out[1].Content[0].(*ReasoningBlock).Signature != "" {
		t.Fatal("signature kept across models")
	}
	if msgs[1].Content[0].(*ReasoningBlock).Signature != "sig" {
		t.Fatal("input mutated")
	}
	res := out[2].ToolResultBlocks()
	if len(res) != 2 || res[0].CallID != "c1" || !res[0].IsError || res[1].CallID != "c2" {
		t.Fatalf("results = %+v", res)
	}
	same := PrepareForModel(msgs, anth)
	if same[1].Content[0].(*ReasoningBlock).Signature != "sig" {
		t.Fatal("signature dropped for same model")
	}
}

func TestValidateJSON(t *testing.T) {
	s := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string","minLength":1},"n":{"type":"integer","minimum":0},"mode":{"enum":["a","b"]},"tags":{"type":"array","items":{"type":"string"}}},"required":["city"],"additionalProperties":false}`)
	cases := []struct {
		in string
		ok bool
	}{
		{`{"city":"x"}`, true},
		{`{"city":"x","n":2,"mode":"a","tags":["t"]}`, true},
		{`{}`, false},
		{`{"city":1}`, false},
		{`{"city":"x","n":1.5}`, false},
		{`{"city":"x","n":-1}`, false},
		{`{"city":"x","mode":"c"}`, false},
		{`{"city":"x","tags":[1]}`, false},
		{`{"city":"x","extra":1}`, false},
		{`{"city":""}`, false},
		{`not json`, false},
	}
	for _, c := range cases {
		err := ValidateJSON(s, json.RawMessage(c.in))
		if (err == nil) != c.ok {
			t.Errorf("%s: err=%v", c.in, err)
		}
	}
	if ValidateJSON(nil, json.RawMessage(`{"x":1}`)) != nil {
		t.Error("empty schema should accept")
	}
}

func TestStreamHelpers(t *testing.T) {
	s := Map(Filter(FromSlice(1, 2, 3, 4), func(i int) bool { return i%2 == 0 }), func(i int) int { return i * 10 })
	got, err := Collect(s)
	if err != nil || len(got) != 2 || got[0] != 20 || got[1] != 40 {
		t.Fatalf("got %v %v", got, err)
	}
	boom := errors.New("boom")
	if _, err := Collect(Map(Error[int](boom), func(i int) int { return i })); !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	n := 0
	for range FromSlice(1, 2, 3) {
		n++
		break
	}
	if n != 1 {
		t.Fatal("break not honoured")
	}
}

func TestEventJSON(t *testing.T) {
	ev := &Event{Seq: 3, ID: "e", Author: "a", Message: UserText("x"), Actions: EventActions{
		StateDelta: map[string]any{"k": "v"},
		Interrupt:  &Interrupt{ID: "root/t:c1", Path: []string{"root"}, CallID: "c1", Payload: json.RawMessage(`{"p":1}`)},
	}}
	b, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	var got Event
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if got.Message.Text() != "x" || got.Actions.Interrupt.ID != "root/t:c1" || got.Actions.StateDelta["k"] != "v" {
		t.Fatalf("got %+v", got)
	}
	rd := ResumeData{InterruptID: "i", ToolResult: TextResult("done")}
	b, _ = json.Marshal(rd)
	var rd2 ResumeData
	if err := json.Unmarshal(b, &rd2); err != nil || rd2.ToolResult.Content[0].(*TextBlock).Text != "done" {
		t.Fatalf("resume data %s %v", b, err)
	}
}
