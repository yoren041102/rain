package agent

import (
	"context"
	"sync"

	"github.com/yoren041102/rain/schema"
)

// Handlers wrapped by middlewares.
type (
	AgentHandler func(ctx context.Context, inv *Invocation) schema.Stream[*schema.Event]
	ModelHandler func(ctx context.Context, inv *Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse]
	ToolHandler  func(ctx context.Context, inv *Invocation, call *schema.ToolCallBlock) (*schema.ToolResult, error)
)

// Middlewares wrap handlers; the first registered is the outermost.
type (
	AgentMiddleware func(next AgentHandler) AgentHandler
	ModelMiddleware func(next ModelHandler) ModelHandler
	ToolMiddleware  func(next ToolHandler) ToolHandler
)

// Middlewares groups the three middleware lists.
type Middlewares struct {
	Agent []AgentMiddleware
	Model []ModelMiddleware
	Tool  []ToolMiddleware
}

// Merge returns m followed by others (m stays outermost).
func (m Middlewares) Merge(others ...Middlewares) Middlewares {
	out := Middlewares{
		Agent: append([]AgentMiddleware(nil), m.Agent...),
		Model: append([]ModelMiddleware(nil), m.Model...),
		Tool:  append([]ToolMiddleware(nil), m.Tool...),
	}
	for _, o := range others {
		out.Agent = append(out.Agent, o.Agent...)
		out.Model = append(out.Model, o.Model...)
		out.Tool = append(out.Tool, o.Tool...)
	}
	return out
}

// RunAgent runs a with the invocation's agent middlewares (global outermost,
// then the agent's own), wires inv.Emit, and attaches pending state writes
// to outgoing events. Custom agents must use it to run sub-agents.
func RunAgent(ctx context.Context, inv *Invocation, a Agent) schema.Stream[*schema.Event] {
	return func(yield func(*schema.Event, error) bool) {
		child := inv.WithAgent(a)
		if child.State == nil {
			child.State = NewDeltaState(nil)
		}
		var (
			mu    sync.Mutex
			queue []*schema.Event
		)
		child.Emit = func(ev *schema.Event) {
			if ev == nil {
				return
			}
			child.fillEvent(ev)
			mu.Lock()
			queue = append(queue, ev)
			mu.Unlock()
		}
		attach := func(ev *schema.Event) {
			if d := child.State.TakeDelta(); len(d) > 0 {
				if ev.Actions.StateDelta == nil {
					ev.Actions.StateDelta = map[string]any{}
				}
				for k, v := range d {
					if _, exists := ev.Actions.StateDelta[k]; !exists {
						ev.Actions.StateDelta[k] = v
					}
				}
			}
		}
		flush := func() bool {
			mu.Lock()
			q := queue
			queue = nil
			mu.Unlock()
			for _, ev := range q {
				attach(ev)
				if !yield(ev, nil) {
					return false
				}
			}
			return true
		}

		var mws []AgentMiddleware
		mws = append(mws, inv.Middlewares.Agent...)
		if p, ok := a.(MiddlewareProvider); ok {
			mws = append(mws, p.AgentMiddlewares()...)
		}
		h := AgentHandler(func(ctx context.Context, inv *Invocation) schema.Stream[*schema.Event] {
			return inv.Agent.Run(ctx, inv)
		})
		for i := len(mws) - 1; i >= 0; i-- {
			h = mws[i](h)
		}
		for ev, err := range h(ctx, child) {
			if !flush() {
				return
			}
			if err != nil {
				yield(nil, err)
				return
			}
			if ev == nil {
				continue
			}
			child.fillEvent(ev)
			if !ev.Partial {
				attach(ev)
			}
			if !yield(ev, nil) {
				return
			}
		}
		if !flush() {
			return
		}
		// Writes made after the last event are delivered on a state-only event.
		if d := child.State.TakeDelta(); len(d) > 0 {
			ev := child.NewEvent(nil)
			ev.Actions.StateDelta = d
			yield(ev, nil)
		}
	}
}

// CallModel calls m through the invocation's model middlewares followed by
// local ones.
func CallModel(ctx context.Context, inv *Invocation, m schema.Model, req *schema.ModelRequest, local ...ModelMiddleware) schema.Stream[*schema.ModelResponse] {
	h := ModelHandler(func(ctx context.Context, _ *Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
		return m.Generate(ctx, req)
	})
	mws := append(append([]ModelMiddleware(nil), inv.Middlewares.Model...), local...)
	for i := len(mws) - 1; i >= 0; i-- {
		h = mws[i](h)
	}
	return h(ctx, inv, req)
}

// CallTool calls t through the invocation's tool middlewares followed by
// local ones. If ctx carries no ToolContext for call.ID, one is created.
func CallTool(ctx context.Context, inv *Invocation, t schema.Tool, call *schema.ToolCallBlock, local ...ToolMiddleware) (*schema.ToolResult, error) {
	if tc, ok := ToolContextFrom(ctx); !ok || tc.CallID != call.ID {
		ctx = WithToolContext(ctx, &ToolContext{Invocation: inv, CallID: call.ID, ToolName: call.Name, Actions: &schema.EventActions{}})
	}
	h := ToolHandler(func(ctx context.Context, _ *Invocation, call *schema.ToolCallBlock) (*schema.ToolResult, error) {
		return t.Call(ctx, call.Arguments)
	})
	mws := append(append([]ToolMiddleware(nil), inv.Middlewares.Tool...), local...)
	for i := len(mws) - 1; i >= 0; i-- {
		h = mws[i](h)
	}
	return h(ctx, inv, call)
}
