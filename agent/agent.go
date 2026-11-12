package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"strings"
	"sync"
	"time"

	"github.com/yoren041102/rain/schema"
)

// Agent is the minimal agent interface. Parent relationships are computed by
// Tree, not stored on agents.
type Agent interface {
	Name() string
	Description() string
	SubAgents() []Agent
	Run(ctx context.Context, inv *Invocation) schema.Stream[*schema.Event]
}

// MiddlewareProvider is optionally implemented by agents that carry their own
// agent-level middlewares; RunAgent applies their Agent middlewares inside the
// global ones.
type MiddlewareProvider interface {
	AgentMiddlewares() []AgentMiddleware
}

// ToolFinder is optionally implemented by agents so the runner can find a
// tool by name (used to re-run ReplaySafe tools during dangling repair).
type ToolFinder interface {
	FindTool(ctx context.Context, name string) schema.Tool
}

// History gives an agent access to the events visible to it.
type History interface {
	Events(ctx context.Context) ([]*schema.Event, error)
}

// HistoryFunc adapts a function to History.
type HistoryFunc func(ctx context.Context) ([]*schema.Event, error)

func (f HistoryFunc) Events(ctx context.Context) ([]*schema.Event, error) { return f(ctx) }

// BufferHistory is an in-memory History. Append is safe for concurrent use.
type BufferHistory struct {
	mu     sync.Mutex
	events []*schema.Event
}

// NewBufferHistory returns a buffer seeded with events.
func NewBufferHistory(events ...*schema.Event) *BufferHistory {
	return &BufferHistory{events: append([]*schema.Event(nil), events...)}
}

func (b *BufferHistory) Append(evs ...*schema.Event) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.events = append(b.events, evs...)
}

func (b *BufferHistory) Events(context.Context) ([]*schema.Event, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]*schema.Event(nil), b.events...), nil
}

// Invocation carries everything an agent needs for one Run or Resume. It is a
// concrete type passed alongside context.Context: adding fields does not
// break user mocks, and context wrappers do not lose it.
type Invocation struct {
	ID          string
	Agent       Agent
	Tree        *Tree
	Branch      string
	Session     Session
	State       *DeltaState // reads/writes session state; writes become StateDelta
	History     History
	Memory      Memory // optional
	UserMessage *schema.Message

	// Resume holds the caller's answers when this invocation resumes a
	// pause. OpenInterrupts lists every interrupt of the pause, including
	// those not selected for resumption (they must re-trigger).
	Resume         []schema.ResumeData
	OpenInterrupts []*schema.Interrupt

	Middlewares Middlewares

	// Emit lets middlewares and agents emit extra events into the current
	// agent's stream. Set by RunAgent; safe for concurrent use.
	Emit func(*schema.Event)

	// Steering drains messages injected by the user during execution.
	Steering func() []*schema.Message
	// StopRequested reports whether a safe-point stop was requested.
	StopRequested func() bool

	NewID func() string
	Now   func() time.Time
}

// WithAgent returns a shallow copy of inv targeting a.
func (inv *Invocation) WithAgent(a Agent) *Invocation {
	c := *inv
	c.Agent = a
	return &c
}

// WithBranch returns a shallow copy of inv on branch b.
func (inv *Invocation) WithBranch(b string) *Invocation {
	c := *inv
	c.Branch = b
	return &c
}

// WithoutResume returns a copy with resume information cleared.
func (inv *Invocation) WithoutResume() *Invocation {
	c := *inv
	c.Resume = nil
	c.OpenInterrupts = nil
	return &c
}

// ChildBranch returns the branch name for a child running in isolation.
func (inv *Invocation) ChildBranch(name string) string {
	if inv.Branch == "" {
		if inv.Agent != nil {
			return inv.Agent.Name() + "." + name
		}
		return name
	}
	return inv.Branch + "." + name
}

// GenID returns a new ID using inv.NewID or a random fallback.
func (inv *Invocation) GenID() string {
	if inv != nil && inv.NewID != nil {
		return inv.NewID()
	}
	return NewUUID()
}

// Time returns the current time using inv.Now or time.Now.
func (inv *Invocation) Time() time.Time {
	if inv != nil && inv.Now != nil {
		return inv.Now()
	}
	return time.Now()
}

// NewEvent creates an event authored by the current agent.
func (inv *Invocation) NewEvent(msg *schema.Message) *schema.Event {
	ev := &schema.Event{Message: msg}
	inv.fillEvent(ev)
	return ev
}

func (inv *Invocation) fillEvent(ev *schema.Event) {
	if ev.ID == "" {
		ev.ID = inv.GenID()
	}
	if ev.InvocationID == "" {
		ev.InvocationID = inv.ID
	}
	if ev.Author == "" && inv.Agent != nil {
		ev.Author = inv.Agent.Name()
	}
	if ev.Branch == "" {
		ev.Branch = inv.Branch
	}
	if ev.CreatedAt.IsZero() {
		ev.CreatedAt = inv.Time()
	}
}

// Path returns the agent path of the current agent.
func (inv *Invocation) Path() []string {
	if inv.Tree != nil && inv.Agent != nil {
		if p := inv.Tree.Path(inv.Agent); p != nil {
			return p
		}
	}
	if inv.Agent != nil {
		return []string{inv.Agent.Name()}
	}
	return nil
}

// ResumeFor returns the resume data for an interrupt ID.
func (inv *Invocation) ResumeFor(id string) (*schema.ResumeData, bool) {
	for i := range inv.Resume {
		if inv.Resume[i].InterruptID == id {
			return &inv.Resume[i], true
		}
	}
	return nil, false
}

// InterruptsFor returns the open interrupts raised by the agent named name.
func (inv *Invocation) InterruptsFor(name string) []*schema.Interrupt {
	var out []*schema.Interrupt
	for _, it := range inv.OpenInterrupts {
		if len(it.Path) > 0 && it.Path[len(it.Path)-1] == name {
			out = append(out, it)
		}
	}
	return out
}

// InterruptsUnder returns the open interrupts raised by name or any of its
// descendants.
func (inv *Invocation) InterruptsUnder(name string) []*schema.Interrupt {
	var out []*schema.Interrupt
	for _, it := range inv.OpenInterrupts {
		for _, p := range it.Path {
			if p == name {
				out = append(out, it)
				break
			}
		}
	}
	return out
}

// Resuming reports whether this invocation resumes a pause.
func (inv *Invocation) Resuming() bool { return len(inv.OpenInterrupts) > 0 }

// NewUUID returns a random RFC 4122 version 4 UUID.
func NewUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	h := hex.EncodeToString(b[:])
	return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:]
}

// BranchVisible reports whether an event on evBranch is visible to an
// invocation on invBranch: empty branches see / are seen by everything,
// otherwise the event branch must equal or be an ancestor of invBranch.
func BranchVisible(invBranch, evBranch string) bool {
	if invBranch == "" || evBranch == "" {
		return true
	}
	return invBranch == evBranch || strings.HasPrefix(invBranch, evBranch+".")
}

// ---- trace carrier ----

type traceKey struct{}

// WithTraceCarrier attaches a persisted trace carrier to ctx so tracing
// middlewares can continue a trace after Resume.
func WithTraceCarrier(ctx context.Context, carrier map[string]string) context.Context {
	return context.WithValue(ctx, traceKey{}, carrier)
}

// TraceCarrierFrom returns the trace carrier attached by WithTraceCarrier.
func TraceCarrierFrom(ctx context.Context) map[string]string {
	c, _ := ctx.Value(traceKey{}).(map[string]string)
	return c
}
