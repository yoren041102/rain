// Package otel provides OpenTelemetry tracing and metrics for rain's agent,
// model and tool middleware layers.
//
// Spans:
//
//	rain.agent <agent name>   one agent run (including transfers below it)
//	rain.model <model name>   one model call
//	rain.tool <tool name>     one tool call
//
// By default span attributes carry only identifiers (invocation ID, session
// ID, agent name, branch, tool call ID, model name) and token counts. Message
// text, tool arguments and tool results are recorded only with WithContent.
//
// Interrupts are not errors: they are recorded as "rain.interrupt" span
// events, and the agent middleware writes the span context into
// Event.TraceCarrier of interrupt events so that a later Resume continues the
// same trace (the runner puts the persisted carrier back into the context via
// agent.WithTraceCarrier, which the agent middleware extracts).
//
// The package is named otel like the upstream module; import
// go.opentelemetry.io/otel under an alias when both are needed.
package otel

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
	gotel "go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	metricnoop "go.opentelemetry.io/otel/metric/noop"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
)

// ScopeName is the instrumentation scope used for the tracer and meter.
const ScopeName = "github.com/yoren041102/rain/ext/middleware/otel"

// Attribute keys.
const (
	AttrInvocationID = "rain.invocation_id"
	AttrSessionID    = "rain.session_id"
	AttrAgent        = "rain.agent"
	AttrBranch       = "rain.branch"

	AttrRequestModel  = "gen_ai.request.model"
	AttrResponseModel = "gen_ai.response.model"
	AttrProvider      = "gen_ai.provider.name"
	AttrOperation     = "gen_ai.operation.name"
	AttrInputTokens   = "gen_ai.usage.input_tokens"
	AttrOutputTokens  = "gen_ai.usage.output_tokens"
	AttrFinishReasons = "gen_ai.response.finish_reasons"
	AttrTokenType     = "gen_ai.token.type"
	AttrInputMessages = "gen_ai.input.messages"  // WithContent only
	AttrOutputMessage = "gen_ai.output.messages" // WithContent only

	AttrToolName      = "gen_ai.tool.name"
	AttrToolCallID    = "gen_ai.tool.call.id"
	AttrToolArguments = "gen_ai.tool.call.arguments" // WithContent only
	AttrToolResult    = "gen_ai.tool.call.result"    // WithContent only
	AttrToolIsError   = "rain.tool.is_error"
	AttrToolOutcome   = "rain.tool.outcome" // ok | error | interrupt

	AttrErrorType = "error.type"

	AttrInterruptID      = "rain.interrupt.id"
	AttrInterruptReason  = "rain.interrupt.reason"
	AttrInterruptCallID  = "rain.interrupt.call_id"
	AttrInterruptPayload = "rain.interrupt.payload" // WithContent only

	// EventInterrupt is the span event name recorded for interrupts.
	EventInterrupt = "rain.interrupt"
)

// Metric names.
const (
	MetricToolCalls     = "rain.tool.calls"
	MetricToolDuration  = "rain.tool.duration"
	MetricModelDuration = "rain.model.duration"
	MetricModelTokens   = "rain.model.tokens"
)

// Option configures the middlewares and the plugin.
type Option func(*config)

type config struct {
	tp      trace.TracerProvider
	mp      metric.MeterProvider
	prop    propagation.TextMapPropagator
	content bool
}

// WithTracerProvider sets the tracer provider (default: the global one).
func WithTracerProvider(tp trace.TracerProvider) Option { return func(c *config) { c.tp = tp } }

// WithMeterProvider sets the meter provider (default: the global one).
func WithMeterProvider(mp metric.MeterProvider) Option { return func(c *config) { c.mp = mp } }

// WithPropagator sets the propagator used for Event.TraceCarrier
// (default: propagation.TraceContext{}).
func WithPropagator(p propagation.TextMapPropagator) Option {
	return func(c *config) { c.prop = p }
}

// WithContent records message content, tool arguments, tool results and
// interrupt payloads as span attributes. Off by default.
func WithContent() Option { return func(c *config) { c.content = true } }

type instr struct {
	cfg    config
	tracer trace.Tracer

	toolCalls     metric.Int64Counter
	toolDuration  metric.Float64Histogram
	modelDuration metric.Float64Histogram
	modelTokens   metric.Int64Counter
}

func newInstr(opts []Option) *instr {
	var c config
	for _, o := range opts {
		o(&c)
	}
	if c.tp == nil {
		c.tp = gotel.GetTracerProvider()
	}
	if c.mp == nil {
		c.mp = gotel.GetMeterProvider()
	}
	if c.prop == nil {
		c.prop = propagation.TraceContext{}
	}
	in := &instr{cfg: c, tracer: c.tp.Tracer(ScopeName)}
	m := c.mp.Meter(ScopeName)
	noop := metricnoop.Meter{}
	var err error
	if in.toolCalls, err = m.Int64Counter(MetricToolCalls, metric.WithDescription("Number of tool calls"), metric.WithUnit("{call}")); err != nil {
		gotel.Handle(err)
		in.toolCalls, _ = noop.Int64Counter(MetricToolCalls)
	}
	if in.toolDuration, err = m.Float64Histogram(MetricToolDuration, metric.WithDescription("Tool call duration"), metric.WithUnit("s")); err != nil {
		gotel.Handle(err)
		in.toolDuration, _ = noop.Float64Histogram(MetricToolDuration)
	}
	if in.modelDuration, err = m.Float64Histogram(MetricModelDuration, metric.WithDescription("Model call duration"), metric.WithUnit("s")); err != nil {
		gotel.Handle(err)
		in.modelDuration, _ = noop.Float64Histogram(MetricModelDuration)
	}
	if in.modelTokens, err = m.Int64Counter(MetricModelTokens, metric.WithDescription("Tokens used by model calls"), metric.WithUnit("{token}")); err != nil {
		gotel.Handle(err)
		in.modelTokens, _ = noop.Int64Counter(MetricModelTokens)
	}
	return in
}

// Middlewares returns the agent, model and tool middlewares sharing one
// configuration.
func Middlewares(opts ...Option) agent.Middlewares {
	in := newInstr(opts)
	return agent.Middlewares{
		Agent: []agent.AgentMiddleware{in.agent()},
		Model: []agent.ModelMiddleware{in.model()},
		Tool:  []agent.ToolMiddleware{in.tool()},
	}
}

// Agent returns the agent-layer tracing middleware.
func Agent(opts ...Option) agent.AgentMiddleware { return newInstr(opts).agent() }

// Model returns the model-layer tracing and metrics middleware.
func Model(opts ...Option) agent.ModelMiddleware { return newInstr(opts).model() }

// Tool returns the tool-layer tracing and metrics middleware.
func Tool(opts ...Option) agent.ToolMiddleware { return newInstr(opts).tool() }

func baseAttrs(inv *agent.Invocation) []attribute.KeyValue {
	if inv == nil {
		return nil
	}
	attrs := []attribute.KeyValue{attribute.String(AttrInvocationID, inv.ID)}
	if inv.Session != nil {
		attrs = append(attrs, attribute.String(AttrSessionID, inv.Session.ID()))
	}
	if inv.Agent != nil {
		attrs = append(attrs, attribute.String(AttrAgent, inv.Agent.Name()))
	}
	if inv.Branch != "" {
		attrs = append(attrs, attribute.String(AttrBranch, inv.Branch))
	}
	return attrs
}

func agentName(inv *agent.Invocation) string {
	if inv != nil && inv.Agent != nil {
		return inv.Agent.Name()
	}
	return ""
}

func spanName(prefix, name string) string {
	if name == "" {
		return prefix
	}
	return prefix + " " + name
}

func errType(err error) string { return fmt.Sprintf("%T", err) }

func recordErr(span trace.Span, err error) {
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
	span.SetAttributes(attribute.String(AttrErrorType, errType(err)))
}

// rootKey marks contexts below an otel agent span, so nested agents (and
// sub-agents run through agent.RunAgent) neither re-extract the trace
// carrier nor overwrite it on interrupt events.
type rootKey struct{}

func (in *instr) agent() agent.AgentMiddleware {
	return func(next agent.AgentHandler) agent.AgentHandler {
		return func(ctx context.Context, inv *agent.Invocation) schema.Stream[*schema.Event] {
			return func(yield func(*schema.Event, error) bool) {
				root := ctx.Value(rootKey{}) == nil
				opts := []trace.SpanStartOption{trace.WithSpanKind(trace.SpanKindInternal), trace.WithAttributes(baseAttrs(inv)...)}
				if carrier := agent.TraceCarrierFrom(ctx); root && len(carrier) > 0 {
					prev := trace.SpanContextFromContext(ctx)
					ctx = in.cfg.prop.Extract(ctx, propagation.MapCarrier(carrier))
					if cur := trace.SpanContextFromContext(ctx); prev.IsValid() && !prev.Equal(cur) {
						opts = append(opts, trace.WithLinks(trace.Link{SpanContext: prev}))
					}
				}
				ctx, span := in.tracer.Start(ctx, spanName("rain.agent", agentName(inv)), opts...)
				defer span.End()
				if root {
					ctx = context.WithValue(ctx, rootKey{}, true)
				}
				var carrier map[string]string
				for ev, err := range next(ctx, inv) {
					if err != nil {
						if ie, ok := agent.AsInterrupt(err); ok {
							span.AddEvent(EventInterrupt, trace.WithAttributes(attribute.String(AttrInterruptReason, ie.Reason)))
						} else {
							recordErr(span, err)
						}
						yield(ev, err)
						return
					}
					if ev != nil {
						if its := ev.Actions.AllInterrupts(); len(its) > 0 {
							for _, it := range its {
								span.AddEvent(EventInterrupt, trace.WithAttributes(in.interruptAttrs(it)...))
							}
							if root && ev.TraceCarrier == nil {
								if carrier == nil {
									carrier = in.inject(ctx)
								}
								if len(carrier) > 0 {
									ev.TraceCarrier = cloneMap(carrier)
								}
							}
						}
					}
					if !yield(ev, nil) {
						return
					}
				}
			}
		}
	}
}

func (in *instr) interruptAttrs(it *schema.Interrupt) []attribute.KeyValue {
	attrs := []attribute.KeyValue{attribute.String(AttrInterruptID, it.ID)}
	if it.Reason != "" {
		attrs = append(attrs, attribute.String(AttrInterruptReason, it.Reason))
	}
	if it.CallID != "" {
		attrs = append(attrs, attribute.String(AttrInterruptCallID, it.CallID))
	}
	if in.cfg.content && len(it.Payload) > 0 {
		attrs = append(attrs, attribute.String(AttrInterruptPayload, string(it.Payload)))
	}
	return attrs
}

func (in *instr) inject(ctx context.Context) map[string]string {
	if !trace.SpanContextFromContext(ctx).IsValid() {
		return nil
	}
	c := propagation.MapCarrier{}
	in.cfg.prop.Inject(ctx, c)
	if len(c) == 0 {
		return nil
	}
	return map[string]string(c)
}

func cloneMap(m map[string]string) map[string]string {
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// modelName finds the model name for the invocation's agent: a ReActAgent's
// configured model, or any agent exposing Model() schema.Model.
func modelName(inv *agent.Invocation) string {
	if inv == nil || inv.Agent == nil {
		return ""
	}
	switch a := inv.Agent.(type) {
	case *agent.ReActAgent:
		if m := a.Config().Model; m != nil {
			return m.Name()
		}
	case interface{ Model() schema.Model }:
		if m := a.Model(); m != nil {
			return m.Name()
		}
	}
	return ""
}

func (in *instr) model() agent.ModelMiddleware {
	return func(next agent.ModelHandler) agent.ModelHandler {
		return func(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
			return func(yield func(*schema.ModelResponse, error) bool) {
				name := modelName(inv)
				attrs := append(baseAttrs(inv), attribute.String(AttrOperation, "chat"))
				if name != "" {
					attrs = append(attrs, attribute.String(AttrRequestModel, name))
				}
				if in.cfg.content && req != nil {
					if b, err := json.Marshal(req.Messages); err == nil {
						attrs = append(attrs, attribute.String(AttrInputMessages, string(b)))
					}
				}
				ctx, span := in.tracer.Start(ctx, spanName("rain.model", name), trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
				start := time.Now()
				var callErr error
				defer func() {
					mattrs := []attribute.KeyValue{attribute.String(AttrRequestModel, name), attribute.String(AttrAgent, agentName(inv))}
					if callErr != nil {
						mattrs = append(mattrs, attribute.String(AttrErrorType, errType(callErr)))
					}
					in.modelDuration.Record(ctx, time.Since(start).Seconds(), metric.WithAttributes(mattrs...))
					span.End()
				}()
				for r, err := range next(ctx, inv, req) {
					if err != nil {
						callErr = err
						recordErr(span, err)
						yield(r, err)
						return
					}
					if r != nil && !r.Partial {
						in.finishModel(ctx, span, inv, name, r)
					}
					if !yield(r, nil) {
						return
					}
				}
			}
		}
	}
}

func (in *instr) finishModel(ctx context.Context, span trace.Span, inv *agent.Invocation, name string, r *schema.ModelResponse) {
	reason := string(r.FinishReason)
	if r.Message != nil {
		if r.Message.StopReason != "" {
			reason = string(r.Message.StopReason)
		}
		if o := r.Message.Origin; o != nil {
			if o.Model != "" {
				span.SetAttributes(attribute.String(AttrResponseModel, o.Model))
				if name == "" {
					span.SetName(spanName("rain.model", o.Model))
					span.SetAttributes(attribute.String(AttrRequestModel, o.Model))
					name = o.Model
				}
			}
			if o.Provider != "" {
				span.SetAttributes(attribute.String(AttrProvider, o.Provider))
			}
		}
		if in.cfg.content {
			if b, err := json.Marshal([]*schema.Message{r.Message}); err == nil {
				span.SetAttributes(attribute.String(AttrOutputMessage, string(b)))
			}
		}
	}
	if reason != "" {
		span.SetAttributes(attribute.StringSlice(AttrFinishReasons, []string{reason}))
	}
	if u := r.Usage; u != nil {
		span.SetAttributes(attribute.Int(AttrInputTokens, u.InputTokens), attribute.Int(AttrOutputTokens, u.OutputTokens))
		base := []attribute.KeyValue{attribute.String(AttrRequestModel, name), attribute.String(AttrAgent, agentName(inv))}
		if u.InputTokens > 0 {
			in.modelTokens.Add(ctx, int64(u.InputTokens), metric.WithAttributes(append(base, attribute.String(AttrTokenType, "input"))...))
		}
		if u.OutputTokens > 0 {
			in.modelTokens.Add(ctx, int64(u.OutputTokens), metric.WithAttributes(append(base, attribute.String(AttrTokenType, "output"))...))
		}
	}
}

func (in *instr) tool() agent.ToolMiddleware {
	return func(next agent.ToolHandler) agent.ToolHandler {
		return func(ctx context.Context, inv *agent.Invocation, call *schema.ToolCallBlock) (res *schema.ToolResult, err error) {
			attrs := append(baseAttrs(inv), attribute.String(AttrOperation, "execute_tool"), attribute.String(AttrToolName, call.Name), attribute.String(AttrToolCallID, call.ID))
			if in.cfg.content {
				attrs = append(attrs, attribute.String(AttrToolArguments, string(call.Arguments)))
			}
			ctx, span := in.tracer.Start(ctx, spanName("rain.tool", call.Name), trace.WithSpanKind(trace.SpanKindInternal), trace.WithAttributes(attrs...))
			start := time.Now()
			outcome := "ok"
			defer func() {
				if p := recover(); p != nil {
					outcome = "error"
					span.SetStatus(codes.Error, fmt.Sprint(p))
					in.endTool(ctx, span, call, outcome, start)
					panic(p)
				}
				in.endTool(ctx, span, call, outcome, start)
			}()
			res, err = next(ctx, inv, call)
			switch {
			case err != nil:
				if ie, ok := agent.AsInterrupt(err); ok {
					outcome = "interrupt"
					span.AddEvent(EventInterrupt, trace.WithAttributes(attribute.String(AttrInterruptReason, ie.Reason), attribute.String(AttrInterruptCallID, call.ID)))
				} else {
					outcome = "error"
					recordErr(span, err)
				}
			case res != nil && res.IsError:
				outcome = "error"
				span.SetAttributes(attribute.Bool(AttrToolIsError, true))
				span.SetStatus(codes.Error, "tool returned an error result")
			}
			if in.cfg.content && res != nil {
				if b, jerr := json.Marshal(res); jerr == nil {
					span.SetAttributes(attribute.String(AttrToolResult, string(b)))
				}
			}
			return res, err
		}
	}
}

func (in *instr) endTool(ctx context.Context, span trace.Span, call *schema.ToolCallBlock, outcome string, start time.Time) {
	span.SetAttributes(attribute.String(AttrToolOutcome, outcome))
	mattrs := metric.WithAttributes(attribute.String(AttrToolName, call.Name), attribute.String(AttrToolOutcome, outcome))
	in.toolCalls.Add(ctx, 1, mattrs)
	in.toolDuration.Record(ctx, time.Since(start).Seconds(), mattrs)
	span.End()
}

// Plugin is a runner plugin (it matches runner.Plugin and runner.EventHook)
// that writes the span context found in the runner's context into
// Event.TraceCarrier of interrupt events that do not carry one yet.
//
// The agent middleware already stamps interrupt events with its own span,
// so the plugin matters mainly when the middlewares are not installed but
// the caller runs the runner under a span.
type Plugin struct{ in *instr }

// NewPlugin returns a Plugin. Only WithPropagator (and the providers, which
// it does not use) are relevant.
func NewPlugin(opts ...Option) *Plugin { return &Plugin{in: newInstr(opts)} }

// Name implements runner.Plugin.
func (p *Plugin) Name() string { return "otel" }

// OnEvent implements runner.EventHook.
func (p *Plugin) OnEvent(ctx context.Context, _ *agent.Invocation, ev *schema.Event) (*schema.Event, error) {
	if ev == nil || ev.TraceCarrier != nil || len(ev.Actions.AllInterrupts()) == 0 {
		return ev, nil
	}
	if c := p.in.inject(ctx); c != nil {
		ev.TraceCarrier = c
	}
	return ev, nil
}
