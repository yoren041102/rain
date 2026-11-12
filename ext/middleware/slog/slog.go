// Package slog provides agent, model and tool middlewares that log calls
// with log/slog. Message content is not logged unless WithContent is used.
package slog

import (
	"context"
	"log/slog"
	"time"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
)

// Option configures the logger middlewares.
type Option func(*cfg)

type cfg struct {
	content bool
	level   slog.Level
}

// WithContent logs message text and tool arguments.
func WithContent() Option { return func(c *cfg) { c.content = true } }

// WithLevel sets the level of success logs (errors use slog.LevelError).
func WithLevel(l slog.Level) Option { return func(c *cfg) { c.level = l } }

func build(opts []Option) cfg {
	c := cfg{level: slog.LevelInfo}
	for _, o := range opts {
		o(&c)
	}
	return c
}

func baseAttrs(inv *agent.Invocation) []any {
	attrs := []any{slog.String("invocation_id", inv.ID)}
	if inv.Agent != nil {
		attrs = append(attrs, slog.String("agent", inv.Agent.Name()))
	}
	if inv.Session != nil {
		attrs = append(attrs, slog.String("session_id", inv.Session.ID()))
	}
	if inv.Branch != "" {
		attrs = append(attrs, slog.String("branch", inv.Branch))
	}
	return attrs
}

// Middlewares returns all three middlewares logging to l.
func Middlewares(l *slog.Logger, opts ...Option) agent.Middlewares {
	return agent.Middlewares{
		Agent: []agent.AgentMiddleware{Agent(l, opts...)},
		Model: []agent.ModelMiddleware{Model(l, opts...)},
		Tool:  []agent.ToolMiddleware{Tool(l, opts...)},
	}
}

// Agent logs agent runs.
func Agent(l *slog.Logger, opts ...Option) agent.AgentMiddleware {
	c := build(opts)
	return func(next agent.AgentHandler) agent.AgentHandler {
		return func(ctx context.Context, inv *agent.Invocation) schema.Stream[*schema.Event] {
			return func(yield func(*schema.Event, error) bool) {
				start := time.Now()
				n := 0
				var runErr error
				defer func() {
					attrs := append(baseAttrs(inv), slog.Int("events", n), slog.Duration("duration", time.Since(start)))
					if runErr != nil {
						l.ErrorContext(ctx, "rain agent failed", append(attrs, slog.Any("error", runErr))...)
						return
					}
					l.Log(ctx, c.level, "rain agent finished", attrs...)
				}()
				for ev, err := range next(ctx, inv) {
					if err != nil {
						runErr = err
					} else if !ev.Partial {
						n++
					}
					if !yield(ev, err) || err != nil {
						return
					}
				}
			}
		}
	}
}

// Model logs model calls.
func Model(l *slog.Logger, opts ...Option) agent.ModelMiddleware {
	c := build(opts)
	return func(next agent.ModelHandler) agent.ModelHandler {
		return func(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
			return func(yield func(*schema.ModelResponse, error) bool) {
				start := time.Now()
				for r, err := range next(ctx, inv, req) {
					if err != nil {
						l.ErrorContext(ctx, "rain model failed", append(baseAttrs(inv), slog.Any("error", err), slog.Duration("duration", time.Since(start)))...)
						yield(nil, err)
						return
					}
					if !r.Partial {
						attrs := append(baseAttrs(inv), slog.Int("messages", len(req.Messages)), slog.Int("tools", len(req.Tools)), slog.Duration("duration", time.Since(start)))
						if r.Message != nil {
							attrs = append(attrs, slog.String("stop_reason", string(r.Message.StopReason)), slog.Int("tool_calls", len(r.Message.ToolCalls())))
							if c.content {
								attrs = append(attrs, slog.String("text", r.Message.Text()))
							}
						}
						if r.Usage != nil {
							attrs = append(attrs, slog.Int("input_tokens", r.Usage.InputTokens), slog.Int("output_tokens", r.Usage.OutputTokens))
						}
						l.Log(ctx, c.level, "rain model call", attrs...)
					}
					if !yield(r, nil) {
						return
					}
				}
			}
		}
	}
}

// Tool logs tool calls.
func Tool(l *slog.Logger, opts ...Option) agent.ToolMiddleware {
	c := build(opts)
	return func(next agent.ToolHandler) agent.ToolHandler {
		return func(ctx context.Context, inv *agent.Invocation, call *schema.ToolCallBlock) (*schema.ToolResult, error) {
			start := time.Now()
			res, err := next(ctx, inv, call)
			attrs := append(baseAttrs(inv), slog.String("tool", call.Name), slog.String("call_id", call.ID), slog.Duration("duration", time.Since(start)))
			if c.content {
				attrs = append(attrs, slog.String("arguments", string(call.Arguments)))
			}
			if err != nil {
				if _, ok := agent.AsInterrupt(err); ok {
					l.Log(ctx, c.level, "rain tool interrupted", attrs...)
				} else {
					l.ErrorContext(ctx, "rain tool failed", append(attrs, slog.Any("error", err))...)
				}
				return res, err
			}
			if res != nil {
				attrs = append(attrs, slog.Bool("is_error", res.IsError))
			}
			l.Log(ctx, c.level, "rain tool call", attrs...)
			return res, nil
		}
	}
}
