package otel

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

type echoTool struct {
	interrupt bool
	fail      bool
}

func (t *echoTool) Spec() schema.ToolSpec {
	return schema.ToolSpec{Name: "echo", Description: "echo", Parameters: json.RawMessage(`{"type":"object"}`)}
}

func (t *echoTool) Call(ctx context.Context, args json.RawMessage) (*schema.ToolResult, error) {
	if t.interrupt {
		return nil, agent.Interrupt(ctx, "confirm", map[string]string{"q": "SECRET-PAYLOAD"}, nil)
	}
	if t.fail {
		return nil, errors.New("tool exploded")
	}
	return schema.TextResult("echoed SECRET-RESULT"), nil
}

// scripted model: first call requests the echo tool, next call answers.
func newModel() schema.Model {
	return schema.ModelFunc{ModelName: "test-model", Fn: func(ctx context.Context, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
		return func(yield func(*schema.ModelResponse, error) bool) {
			last := req.Messages[len(req.Messages)-1]
			var msg *schema.Message
			if len(last.ToolResultBlocks()) > 0 {
				msg = &schema.Message{Role: schema.RoleAssistant, Content: []schema.ContentBlock{&schema.TextBlock{Text: "done SECRET-ANSWER"}}, StopReason: schema.StopReasonStop}
			} else {
				msg = &schema.Message{Role: schema.RoleAssistant, StopReason: schema.StopReasonToolUse,
					Content: []schema.ContentBlock{&schema.ToolCallBlock{ID: "call-1", Name: "echo", Arguments: json.RawMessage(`{"x":"SECRET-ARG"}`)}}}
			}
			if !yield(&schema.ModelResponse{Partial: true, Message: &schema.Message{Role: schema.RoleAssistant}}, nil) {
				return
			}
			yield(&schema.ModelResponse{Message: msg, FinishReason: msg.StopReason, Usage: &schema.Usage{InputTokens: 10, OutputTokens: 5}}, nil)
		}
	}}
}

type fakeSession struct{}

func (fakeSession) ID() string         { return "sess-1" }
func (fakeSession) AppName() string    { return "app" }
func (fakeSession) UserID() string     { return "u" }
func (fakeSession) State() agent.State { return nil }

type harness struct {
	rec    *tracetest.SpanRecorder
	tp     *sdktrace.TracerProvider
	reader *sdkmetric.ManualReader
	mp     *sdkmetric.MeterProvider
}

func newHarness() *harness {
	h := &harness{rec: tracetest.NewSpanRecorder(), reader: sdkmetric.NewManualReader()}
	h.tp = sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(h.rec))
	h.mp = sdkmetric.NewMeterProvider(sdkmetric.WithReader(h.reader))
	return h
}

func (h *harness) opts(extra ...Option) []Option {
	return append([]Option{WithTracerProvider(h.tp), WithMeterProvider(h.mp)}, extra...)
}

func run(t *testing.T, ctx context.Context, mws agent.Middlewares, tool schema.Tool) []*schema.Event {
	t.Helper()
	a, err := agent.NewReAct(agent.ReActConfig{Name: "assistant", Model: newModel(), Tools: []schema.Tool{tool}})
	if err != nil {
		t.Fatal(err)
	}
	tree, err := agent.NewTree(a)
	if err != nil {
		t.Fatal(err)
	}
	inv := &agent.Invocation{ID: "inv-1", Tree: tree, Session: fakeSession{}, UserMessage: schema.UserText("hello SECRET-INPUT"), Middlewares: mws}
	evs, err := schema.Collect(agent.RunAgent(ctx, inv, a))
	if err != nil {
		t.Fatal(err)
	}
	return evs
}

func spansByName(spans []sdktrace.ReadOnlySpan) map[string][]sdktrace.ReadOnlySpan {
	m := map[string][]sdktrace.ReadOnlySpan{}
	for _, s := range spans {
		m[s.Name()] = append(m[s.Name()], s)
	}
	return m
}

func attrMap(s sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	m := map[attribute.Key]attribute.Value{}
	for _, kv := range s.Attributes() {
		m[kv.Key] = kv.Value
	}
	return m
}

func allText(s sdktrace.ReadOnlySpan) string {
	var b strings.Builder
	for _, kv := range s.Attributes() {
		b.WriteString(kv.Value.Emit())
	}
	for _, e := range s.Events() {
		for _, kv := range e.Attributes {
			b.WriteString(kv.Value.Emit())
		}
	}
	return b.String()
}

func TestNestingAttributesNoContent(t *testing.T) {
	h := newHarness()
	run(t, context.Background(), Middlewares(h.opts()...), &echoTool{})

	spans := h.rec.Ended()
	by := spansByName(spans)
	ag, models, tools := by["rain.agent assistant"], by["rain.model test-model"], by["rain.tool echo"]
	if len(ag) != 1 || len(models) != 2 || len(tools) != 1 {
		t.Fatalf("spans = %v", by)
	}
	root := ag[0]
	for _, s := range append(models, tools...) {
		if s.Parent().SpanID() != root.SpanContext().SpanID() || s.SpanContext().TraceID() != root.SpanContext().TraceID() {
			t.Errorf("%s not a child of agent span", s.Name())
		}
	}

	ra := attrMap(root)
	if ra[AttrInvocationID].AsString() != "inv-1" || ra[AttrSessionID].AsString() != "sess-1" || ra[AttrAgent].AsString() != "assistant" {
		t.Errorf("agent attrs = %v", ra)
	}
	ma := attrMap(models[0])
	if ma[AttrRequestModel].AsString() != "test-model" || ma[AttrInputTokens].AsInt64() != 10 || ma[AttrOutputTokens].AsInt64() != 5 {
		t.Errorf("model attrs = %v", ma)
	}
	if fr := ma[AttrFinishReasons].AsStringSlice(); len(fr) != 1 || fr[0] != "tool_use" {
		t.Errorf("finish reasons = %v", fr)
	}
	ta := attrMap(tools[0])
	if ta[AttrToolCallID].AsString() != "call-1" || ta[AttrToolName].AsString() != "echo" || ta[AttrToolOutcome].AsString() != "ok" {
		t.Errorf("tool attrs = %v", ta)
	}
	for _, s := range spans {
		if strings.Contains(allText(s), "SECRET") {
			t.Errorf("span %s leaks content: %v", s.Name(), s.Attributes())
		}
		if s.Status().Code == codes.Error {
			t.Errorf("span %s has error status", s.Name())
		}
	}

	// metrics
	var rm metricdata.ResourceMetrics
	if err := h.reader.Collect(context.Background(), &rm); err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	var tokens int64
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			got[m.Name] = true
			if m.Name == MetricModelTokens {
				for _, dp := range m.Data.(metricdata.Sum[int64]).DataPoints {
					tokens += dp.Value
				}
			}
			if m.Name == MetricToolCalls {
				if dps := m.Data.(metricdata.Sum[int64]).DataPoints; len(dps) != 1 || dps[0].Value != 1 {
					t.Errorf("tool calls = %+v", dps)
				}
			}
		}
	}
	for _, n := range []string{MetricToolCalls, MetricToolDuration, MetricModelDuration, MetricModelTokens} {
		if !got[n] {
			t.Errorf("metric %s missing", n)
		}
	}
	if tokens != 30 {
		t.Errorf("tokens = %d, want 30", tokens)
	}
}

func TestWithContent(t *testing.T) {
	h := newHarness()
	run(t, context.Background(), Middlewares(h.opts(WithContent())...), &echoTool{})
	by := spansByName(h.rec.Ended())
	if txt := allText(by["rain.tool echo"][0]); !strings.Contains(txt, "SECRET-ARG") || !strings.Contains(txt, "SECRET-RESULT") {
		t.Errorf("tool content missing: %s", txt)
	}
	var sawIn, sawOut bool
	for _, s := range by["rain.model test-model"] {
		txt := allText(s)
		sawIn = sawIn || strings.Contains(txt, "SECRET-INPUT")
		sawOut = sawOut || strings.Contains(txt, "SECRET-ANSWER")
	}
	if !sawIn || !sawOut {
		t.Error("model content missing")
	}
}

func TestToolErrorRecorded(t *testing.T) {
	h := newHarness()
	mw := Tool(h.opts()...)
	h2 := mw(func(ctx context.Context, inv *agent.Invocation, call *schema.ToolCallBlock) (*schema.ToolResult, error) {
		return nil, errors.New("tool exploded")
	})
	if _, err := h2(context.Background(), &agent.Invocation{ID: "i"}, &schema.ToolCallBlock{ID: "c", Name: "echo"}); err == nil {
		t.Fatal("want error")
	}
	s := h.rec.Ended()[0]
	if s.Status().Code != codes.Error || len(s.Events()) == 0 || s.Events()[0].Name != "exception" {
		t.Errorf("status=%v events=%v", s.Status(), s.Events())
	}
}

func TestInterruptCarrierAndResume(t *testing.T) {
	h := newHarness()
	evs := run(t, context.Background(), Middlewares(h.opts()...), &echoTool{interrupt: true})
	var iev *schema.Event
	for _, ev := range evs {
		if len(ev.Actions.AllInterrupts()) > 0 {
			iev = ev
		}
	}
	if iev == nil || iev.TraceCarrier["traceparent"] == "" {
		t.Fatalf("interrupt event carrier missing: %+v", iev)
	}
	by := spansByName(h.rec.Ended())
	root := by["rain.agent assistant"][0]
	tool := by["rain.tool echo"][0]
	if tool.Status().Code == codes.Error || attrMap(tool)[AttrToolOutcome].AsString() != "interrupt" {
		t.Errorf("interrupt treated as error: %v %v", tool.Status(), attrMap(tool))
	}
	if !hasEvent(root, EventInterrupt) || root.Status().Code == codes.Error {
		t.Errorf("agent span: status %v events %v", root.Status(), root.Events())
	}
	// The carrier points at the agent span.
	sc := trace.SpanContextFromContext(propagation.TraceContext{}.Extract(context.Background(), propagation.MapCarrier(iev.TraceCarrier)))
	if sc.SpanID() != root.SpanContext().SpanID() {
		t.Errorf("carrier span %s != agent span %s", sc.SpanID(), root.SpanContext().SpanID())
	}

	// Resume: the runner puts the carrier into ctx; the new agent span is a
	// child of the original one and shares its trace.
	h2 := newHarness()
	ctx := agent.WithTraceCarrier(context.Background(), iev.TraceCarrier)
	run(t, ctx, Middlewares(h2.opts()...), &echoTool{})
	resumed := spansByName(h2.rec.Ended())["rain.agent assistant"][0]
	if resumed.SpanContext().TraceID() != root.SpanContext().TraceID() {
		t.Error("resumed run is on a different trace")
	}
	if resumed.Parent().SpanID() != root.SpanContext().SpanID() || !resumed.Parent().IsRemote() {
		t.Errorf("resumed parent = %v", resumed.Parent())
	}
}

func hasEvent(s sdktrace.ReadOnlySpan, name string) bool {
	for _, e := range s.Events() {
		if e.Name == name {
			return true
		}
	}
	return false
}

func TestEarlyBreakEndsSpans(t *testing.T) {
	h := newHarness()
	mws := Middlewares(h.opts()...)
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "assistant", Model: newModel(), Tools: []schema.Tool{&echoTool{}}})
	inv := &agent.Invocation{ID: "inv-1", UserMessage: schema.UserText("hi"), Middlewares: mws}
	for range agent.RunAgent(context.Background(), inv, a) {
		break
	}
	by := spansByName(h.rec.Ended())
	if len(by["rain.agent assistant"]) != 1 {
		t.Errorf("agent span not ended: %v", by)
	}
	if len(h.rec.Started()) != len(h.rec.Ended()) {
		t.Errorf("started %d, ended %d", len(h.rec.Started()), len(h.rec.Ended()))
	}

	// Model middleware alone: break after the first (partial) chunk.
	h2 := newHarness()
	mh := Model(h2.opts()...)(func(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
		return newModel().Generate(ctx, req)
	})
	for range mh(context.Background(), &agent.Invocation{ID: "i"}, &schema.ModelRequest{Messages: []*schema.Message{schema.UserText("x")}}) {
		break
	}
	if n := len(h2.rec.Ended()); n != 1 {
		t.Errorf("model span not ended after break: %d", n)
	}
}

func TestPlugin(t *testing.T) {
	h := newHarness()
	p := NewPlugin()
	if p.Name() != "otel" {
		t.Error(p.Name())
	}
	ctx, span := h.tp.Tracer("t").Start(context.Background(), "outer")
	defer span.End()

	plain := &schema.Event{}
	if ev, _ := p.OnEvent(ctx, nil, plain); ev.TraceCarrier != nil {
		t.Error("carrier set on non-interrupt event")
	}
	ev := &schema.Event{}
	ev.Actions.Interrupt = &schema.Interrupt{ID: "x"}
	ev, err := p.OnEvent(ctx, nil, ev)
	if err != nil || !strings.Contains(ev.TraceCarrier["traceparent"], span.SpanContext().SpanID().String()) {
		t.Errorf("carrier = %v err = %v", ev.TraceCarrier, err)
	}
	// Existing carriers are kept.
	ev2 := &schema.Event{TraceCarrier: map[string]string{"k": "v"}}
	ev2.Actions.Interrupt = &schema.Interrupt{ID: "y"}
	if ev2, _ = p.OnEvent(ctx, nil, ev2); ev2.TraceCarrier["k"] != "v" || len(ev2.TraceCarrier) != 1 {
		t.Errorf("carrier overwritten: %v", ev2.TraceCarrier)
	}
}
