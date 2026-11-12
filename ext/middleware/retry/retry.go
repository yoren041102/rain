// Package retry provides a ModelMiddleware with exponential backoff.
package retry

import (
	"context"
	"errors"
	"math/rand/v2"
	"time"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
)

// Config configures retries.
type Config struct {
	MaxAttempts int           // total attempts including the first; default 3
	BaseDelay   time.Duration // default 500ms
	MaxDelay    time.Duration // default 30s
	// Retryable decides whether err is retried; default: everything except
	// context cancellation and interrupts.
	Retryable func(err error) bool
	// Rewrite may change the request before the next attempt (e.g. trim
	// history after a context-length error).
	Rewrite func(req *schema.ModelRequest, attempt int, err error) *schema.ModelRequest
	// Sleep is injectable for tests.
	Sleep func(ctx context.Context, d time.Duration) error
}

// New returns a retrying ModelMiddleware. When an attempt fails after
// partial chunks were yielded, a DiscardPartial event is emitted through
// inv.Emit so consumers can drop the chunks before the retry streams.
func New(cfg Config) agent.ModelMiddleware {
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = 3
	}
	if cfg.BaseDelay <= 0 {
		cfg.BaseDelay = 500 * time.Millisecond
	}
	if cfg.MaxDelay <= 0 {
		cfg.MaxDelay = 30 * time.Second
	}
	if cfg.Retryable == nil {
		cfg.Retryable = func(err error) bool {
			_, intr := agent.AsInterrupt(err)
			return !intr && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded)
		}
	}
	if cfg.Sleep == nil {
		cfg.Sleep = func(ctx context.Context, d time.Duration) error {
			t := time.NewTimer(d)
			defer t.Stop()
			select {
			case <-t.C:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
	return func(next agent.ModelHandler) agent.ModelHandler {
		return func(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
			return func(yield func(*schema.ModelResponse, error) bool) {
				cur := req
				for attempt := 1; ; attempt++ {
					yielded := false
					var failed error
					for r, err := range next(ctx, inv, cur) {
						if err != nil {
							failed = err
							break
						}
						yielded = true
						if !yield(r, nil) {
							return
						}
					}
					if failed == nil {
						return
					}
					if attempt >= cfg.MaxAttempts || !cfg.Retryable(failed) || ctx.Err() != nil {
						yield(nil, failed)
						return
					}
					if yielded && inv != nil && inv.Emit != nil {
						ev := inv.NewEvent(nil)
						ev.Partial = true
						ev.Actions.DiscardPartial = true
						inv.Emit(ev)
					}
					d := cfg.BaseDelay << (attempt - 1)
					if d > cfg.MaxDelay || d <= 0 {
						d = cfg.MaxDelay
					}
					d = d/2 + time.Duration(rand.Int64N(int64(d/2)+1))
					if err := cfg.Sleep(ctx, d); err != nil {
						yield(nil, err)
						return
					}
					if cfg.Rewrite != nil {
						if nr := cfg.Rewrite(cur, attempt, failed); nr != nil {
							cur = nr
						}
					}
				}
			}
		}
	}
}
