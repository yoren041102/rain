package runner

import (
	"context"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
)

// Plugin is a named extension; it implements any of the optional hook
// interfaces below.
type Plugin interface {
	Name() string
}

type (
	Initializer interface {
		Init(ctx context.Context) error
	}
	Closer interface {
		Close(ctx context.Context) error
	}
	RunHook interface {
		BeforeRun(ctx context.Context, inv *agent.Invocation) error
		AfterRun(ctx context.Context, inv *agent.Invocation, err error)
	}
	// EventHook may rewrite an event or drop it by returning nil.
	EventHook interface {
		OnEvent(ctx context.Context, inv *agent.Invocation, ev *schema.Event) (*schema.Event, error)
	}
	MiddlewareProvider interface{ Middlewares() agent.Middlewares }
)
