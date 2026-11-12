// Package runner is rain's execution layer: session orchestration, event
// persistence, dangling-call repair, interrupts and Resume, steering,
// safe-point stop and plugin lifecycle.
package runner

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
)

// InterruptedText is the synthesized result for dangling non-replayable calls.
const InterruptedText = "Execution was interrupted. The tool may have partially executed."

var (
	// ErrNoInterrupt is returned by Resume when an interrupt is unknown or
	// was superseded by a newer user message.
	ErrNoInterrupt = errors.New("rain: interrupt not found or no longer open")
	// ErrBusy is returned when a session already has a running invocation.
	ErrBusy = errors.New("rain: session is busy")
	// ErrNotRunning is returned by Steer/Stop when nothing is running.
	ErrNotRunning = errors.New("rain: session is not running")
	// ErrClosed is returned after Close.
	ErrClosed = errors.New("rain: runner closed")
)

// Config configures a Runner.
type Config struct {
	AppName  string
	Agent    agent.Agent        // root agent
	Sessions agent.SessionStore // default: in-memory
	Memory   agent.Memory       // optional
	IDFunc   func() string      // default: UUID v4
	Clock    func() time.Time   // default: time.Now
	// AutoCreateSession creates missing sessions on Run (default true when
	// Sessions is nil). Set DisableAutoCreate to turn it off.
	DisableAutoCreate bool
}

// Option configures optional runner behaviour.
type Option func(*Runner)

func WithAgentMiddleware(m ...agent.AgentMiddleware) Option {
	return func(r *Runner) { r.mws.Agent = append(r.mws.Agent, m...) }
}
func WithModelMiddleware(m ...agent.ModelMiddleware) Option {
	return func(r *Runner) { r.mws.Model = append(r.mws.Model, m...) }
}
func WithToolMiddleware(m ...agent.ToolMiddleware) Option {
	return func(r *Runner) { r.mws.Tool = append(r.mws.Tool, m...) }
}
func WithPlugin(p ...Plugin) Option {
	return func(r *Runner) { r.plugins = append(r.plugins, p...) }
}

// Runner orchestrates agents over sessions.
type Runner struct {
	cfg     Config
	tree    *agent.Tree
	mws     agent.Middlewares
	plugins []Plugin

	mu     sync.Mutex
	active map[string]*control
	closed bool
}

type control struct {
	mu       sync.Mutex
	steering []*schema.Message
	stop     bool
}

func (c *control) drain() []*schema.Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	m := c.steering
	c.steering = nil
	return m
}

func (c *control) stopRequested() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stop
}

// New builds the agent tree, initializes plugins and returns a Runner.
func New(cfg Config, opts ...Option) (*Runner, error) {
	if cfg.Agent == nil {
		return nil, errors.New("rain: runner needs a root agent")
	}
	tree, err := agent.NewTree(cfg.Agent)
	if err != nil {
		return nil, err
	}
	if cfg.Sessions == nil {
		cfg.Sessions = agent.NewInMemorySessionStore()
	}
	if cfg.IDFunc == nil {
		cfg.IDFunc = agent.NewUUID
	}
	if cfg.Clock == nil {
		cfg.Clock = time.Now
	}
	r := &Runner{cfg: cfg, tree: tree, active: map[string]*control{}}
	for _, o := range opts {
		o(r)
	}
	for i, p := range r.plugins {
		if in, ok := p.(Initializer); ok {
			if err := in.Init(context.Background()); err != nil {
				var errs []error
				errs = append(errs, fmt.Errorf("rain: plugin %q init: %w", p.Name(), err))
				for j := i - 1; j >= 0; j-- {
					if c, ok := r.plugins[j].(Closer); ok {
						if cerr := c.Close(context.Background()); cerr != nil {
							errs = append(errs, cerr)
						}
					}
				}
				return nil, errors.Join(errs...)
			}
		}
	}
	for _, p := range r.plugins {
		if mp, ok := p.(MiddlewareProvider); ok {
			r.mws = r.mws.Merge(mp.Middlewares())
		}
	}
	return r, nil
}

// Tree returns the agent tree.
func (r *Runner) Tree() *agent.Tree { return r.tree }

// Sessions returns the session store.
func (r *Runner) Sessions() agent.SessionStore { return r.cfg.Sessions }

// Close closes plugins in reverse order.
func (r *Runner) Close(ctx context.Context) error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return nil
	}
	r.closed = true
	r.mu.Unlock()
	var errs []error
	for i := len(r.plugins) - 1; i >= 0; i-- {
		if c, ok := r.plugins[i].(Closer); ok {
			if err := c.Close(ctx); err != nil {
				errs = append(errs, fmt.Errorf("rain: plugin %q close: %w", r.plugins[i].Name(), err))
			}
		}
	}
	return errors.Join(errs...)
}

func sessKey(user, sid string) string { return user + "\x00" + sid }

func (r *Runner) acquire(user, sid string) (*control, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, ErrClosed
	}
	k := sessKey(user, sid)
	if _, busy := r.active[k]; busy {
		return nil, ErrBusy
	}
	c := &control{}
	r.active[k] = c
	return c, nil
}

func (r *Runner) release(user, sid string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.active, sessKey(user, sid))
}

// Steer injects a message into a running invocation; the agent reads it
// before its next model call.
func (r *Runner) Steer(_ context.Context, userID, sessionID string, msg *schema.Message) error {
	r.mu.Lock()
	c, ok := r.active[sessKey(userID, sessionID)]
	r.mu.Unlock()
	if !ok {
		return ErrNotRunning
	}
	c.mu.Lock()
	c.steering = append(c.steering, msg)
	c.mu.Unlock()
	return nil
}

// Stop requests a stop at the next safe point (after a model call or after
// tool calls). The stopped execution can be continued with Resume.
func (r *Runner) Stop(_ context.Context, userID, sessionID string) error {
	r.mu.Lock()
	c, ok := r.active[sessKey(userID, sessionID)]
	r.mu.Unlock()
	if !ok {
		return ErrNotRunning
	}
	c.mu.Lock()
	c.stop = true
	c.mu.Unlock()
	return nil
}

// Run appends msg to the session and runs the starting agent.
func (r *Runner) Run(ctx context.Context, userID, sessionID string, msg *schema.Message) schema.Stream[*schema.Event] {
	return func(yield func(*schema.Event, error) bool) {
		ex, err := r.begin(ctx, userID, sessionID, true)
		if err != nil {
			yield(nil, err)
			return
		}
		defer ex.finish()
		if err := ex.repairDangling(ctx); err != nil {
			yield(nil, err)
			return
		}
		if msg != nil {
			ev := &schema.Event{ID: r.cfg.IDFunc(), InvocationID: ex.inv.ID, Author: schema.AuthorUser, Message: msg, CreatedAt: r.cfg.Clock()}
			if err := ex.persist(ctx, ev); err != nil {
				yield(nil, err)
				return
			}
			if !yield(ev, nil) {
				return
			}
		}
		ex.inv.UserMessage = msg
		ex.inv.Agent = r.startAgent(ex.sess)
		ex.execute(ctx, ex.inv.Agent, yield)
	}
}

// Resume answers open interrupts of the latest pause and continues
// execution. Interrupts not answered must re-trigger.
func (r *Runner) Resume(ctx context.Context, userID, sessionID string, data ...schema.ResumeData) schema.Stream[*schema.Event] {
	return func(yield func(*schema.Event, error) bool) {
		ex, err := r.begin(ctx, userID, sessionID, false)
		if err != nil {
			yield(nil, err)
			return
		}
		defer ex.finish()
		open, carrier, err := OpenInterrupts(ex.events)
		if err != nil {
			yield(nil, err)
			return
		}
		if len(open) == 0 {
			yield(nil, ErrNoInterrupt)
			return
		}
		known := map[string]*schema.Interrupt{}
		for _, it := range open {
			known[it.ID] = it
		}
		for _, d := range data {
			if known[d.InterruptID] == nil {
				yield(nil, fmt.Errorf("%w: %s", ErrNoInterrupt, d.InterruptID))
				return
			}
		}
		// record the resume intent first
		rec := &schema.Event{ID: r.cfg.IDFunc(), InvocationID: ex.inv.ID, Author: schema.SystemAuthor, CreatedAt: r.cfg.Clock(), TraceCarrier: carrier}
		rec.Actions.Resumed = data
		if rec.Actions.Resumed == nil {
			rec.Actions.Resumed = []schema.ResumeData{}
		}
		if err := ex.persist(ctx, rec); err != nil {
			yield(nil, err)
			return
		}
		ex.inv.Resume = data
		ex.inv.OpenInterrupts = open
		if carrier != nil {
			ctx = agent.WithTraceCarrier(ctx, carrier)
		}
		// Start from the agent common to all interrupt paths' roots: the
		// root of the tree; routing walks down Interrupt.Path.
		start := r.tree.Find(open[0].Path[0])
		if start == nil {
			start = r.tree.Root()
		}
		// If every interrupt shares a deeper common prefix, start there.
		start = r.commonAncestor(open, start)
		ex.inv.Agent = start
		ex.execute(ctx, start, yield)
	}
}

func (r *Runner) commonAncestor(open []*schema.Interrupt, fallback agent.Agent) agent.Agent {
	prefix := append([]string(nil), open[0].Path...)
	for _, it := range open[1:] {
		n := 0
		for n < len(prefix) && n < len(it.Path) && prefix[n] == it.Path[n] {
			n++
		}
		prefix = prefix[:n]
	}
	// Start at the shallowest agent that is an ancestor of all interrupts
	// and is itself in the tree. Starting at the first element preserves
	// agents that route resumes (compose agents) above the raiser.
	if len(prefix) > 0 {
		if a := r.tree.Find(prefix[0]); a != nil {
			return a
		}
	}
	return fallback
}

func (r *Runner) startAgent(s agent.Session) agent.Agent {
	if v, ok := s.State().Get(agent.KeyActiveAgent); ok {
		if name, ok := v.(string); ok {
			if a := r.tree.Find(name); a != nil {
				return a
			}
		}
	}
	return r.tree.Root()
}

// OpenInterrupts returns the interrupts of the latest pause that have not
// been resumed, provided no user message came after them.
func OpenInterrupts(events []*schema.Event) ([]*schema.Interrupt, map[string]string, error) {
	resumed := map[string]bool{}
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.Partial {
			continue
		}
		if ev.Author == schema.AuthorUser && ev.Message != nil {
			return nil, nil, nil
		}
		if ev.Actions.Resumed != nil {
			// A resume event closes the pause it answered, including
			// interrupts that were not selected (they re-trigger and are
			// raised again by newer events).
			return nil, nil, nil
		}
		its := ev.Actions.AllInterrupts()
		if len(its) == 0 {
			continue
		}
		// collect every interrupt of this pause (same invocation, contiguous)
		var out []*schema.Interrupt
		inv := ev.InvocationID
		carrier := ev.TraceCarrier
		for j := i; j >= 0; j-- {
			e := events[j]
			if e.InvocationID != inv || e.Actions.Resumed != nil || (e.Author == schema.AuthorUser && e.Message != nil) {
				break
			}
			for _, it := range e.Actions.AllInterrupts() {
				if !resumed[it.ID] {
					out = append(out, it)
					resumed[it.ID] = true
				}
			}
		}
		// keep the original order
		for a, b := 0, len(out)-1; a < b; a, b = a+1, b-1 {
			out[a], out[b] = out[b], out[a]
		}
		return out, carrier, nil
	}
	return nil, nil, nil
}

// ---- execution ----

type execution struct {
	r       *Runner
	userID  string
	sid     string
	sess    agent.Session
	events  []*schema.Event
	lastSeq int64
	inv     *agent.Invocation
	ctl     *control
	mu      sync.Mutex
	runErr  error
	ran     bool
}

func (r *Runner) begin(ctx context.Context, userID, sessionID string, create bool) (*execution, error) {
	ctl, err := r.acquire(userID, sessionID)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			r.release(userID, sessionID)
		}
	}()
	sess, err := r.cfg.Sessions.Get(ctx, r.cfg.AppName, userID, sessionID)
	if errors.Is(err, agent.ErrSessionNotFound) && create && !r.cfg.DisableAutoCreate {
		sess, err = r.cfg.Sessions.Create(ctx, r.cfg.AppName, userID, sessionID, nil)
	}
	if err != nil {
		return nil, err
	}
	events, err := agent.AllEvents(ctx, r.cfg.Sessions, sess, "")
	if err != nil {
		return nil, err
	}
	ex := &execution{r: r, userID: userID, sid: sessionID, sess: sess, events: events, ctl: ctl}
	if n := len(events); n > 0 {
		ex.lastSeq = events[n-1].Seq
	}
	ex.inv = &agent.Invocation{
		ID:            r.cfg.IDFunc(),
		Tree:          r.tree,
		Session:       sess,
		State:         agent.NewDeltaState(sess.State()),
		Memory:        r.cfg.Memory,
		Middlewares:   r.mws,
		Steering:      ctl.drain,
		StopRequested: ctl.stopRequested,
		NewID:         r.cfg.IDFunc,
		Now:           r.cfg.Clock,
	}
	ex.inv.History = agent.HistoryFunc(func(context.Context) ([]*schema.Event, error) {
		ex.mu.Lock()
		defer ex.mu.Unlock()
		return append([]*schema.Event(nil), ex.events...), nil
	})
	ok = true
	return ex, nil
}

func (ex *execution) finish() {
	r := ex.r
	if ex.ran {
		for i := len(r.plugins) - 1; i >= 0; i-- {
			if h, ok := r.plugins[i].(RunHook); ok {
				h.AfterRun(context.WithoutCancel(context.Background()), ex.inv, ex.runErr)
			}
		}
	}
	if r.cfg.Memory != nil {
		ex.saveMemory()
	}
	r.release(ex.userID, ex.sid)
}

func (ex *execution) saveMemory() {
	ctx := context.Background()
	var from int64
	if v, ok := ex.sess.State().Get(agent.KeyMemorySeq); ok {
		switch n := v.(type) {
		case int64:
			from = n
		case float64:
			from = int64(n)
		case int:
			from = int64(n)
		}
	}
	var batch []*schema.Event
	for _, ev := range ex.events {
		if ev.Seq > from && ev.Message != nil {
			batch = append(batch, ev)
		}
	}
	if len(batch) == 0 {
		return
	}
	if err := ex.r.cfg.Memory.Add(ctx, ex.r.cfg.AppName, ex.userID, ex.sid, batch); err != nil {
		return
	}
	mark := &schema.Event{ID: ex.r.cfg.IDFunc(), InvocationID: ex.inv.ID, Author: schema.SystemAuthor, CreatedAt: ex.r.cfg.Clock()}
	mark.Actions.StateDelta = map[string]any{agent.KeyMemorySeq: ex.lastSeq}
	_ = ex.persist(ctx, mark)
}

// persist strips temp: keys and appends ev to the session.
func (ex *execution) persist(ctx context.Context, ev *schema.Event) error {
	if d := ev.Actions.StateDelta; d != nil {
		for k := range d {
			if strings.HasPrefix(k, agent.TempPrefix) {
				delete(d, k)
			}
		}
		if len(d) == 0 {
			ev.Actions.StateDelta = nil
		}
	}
	ex.mu.Lock()
	defer ex.mu.Unlock()
	if err := ex.r.cfg.Sessions.AppendEvents(ctx, ex.sess, []*schema.Event{ev}, ex.lastSeq); err != nil {
		return err
	}
	ex.lastSeq = ev.Seq
	ex.events = append(ex.events, ev.Clone())
	return nil
}

func (ex *execution) execute(ctx context.Context, a agent.Agent, yield func(*schema.Event, error) bool) {
	r := ex.r
	inv := ex.inv
	for _, p := range r.plugins {
		if h, ok := p.(RunHook); ok {
			if err := h.BeforeRun(ctx, inv); err != nil {
				ex.ran = true
				ex.runErr = err
				yield(nil, err)
				return
			}
		}
	}
	ex.ran = true
	carrier := agent.TraceCarrierFrom(ctx)
	for ev, err := range agent.RunAgent(ctx, inv, a) {
		if err != nil {
			ex.runErr = err
			yield(nil, err)
			return
		}
		if ev == nil {
			continue
		}
		for _, p := range r.plugins {
			h, ok := p.(EventHook)
			if !ok {
				continue
			}
			nev, err := h.OnEvent(ctx, inv, ev)
			if err != nil {
				ex.runErr = err
				yield(nil, err)
				return
			}
			ev = nev
			if ev == nil {
				break
			}
		}
		if ev == nil {
			continue
		}
		if ev.Actions.TransferTo != "" && r.tree.Find(ev.Actions.TransferTo) != nil {
			if ev.Actions.StateDelta == nil {
				ev.Actions.StateDelta = map[string]any{}
			}
			ev.Actions.StateDelta[agent.KeyActiveAgent] = ev.Actions.TransferTo
		}
		if len(ev.Actions.AllInterrupts()) > 0 && ev.TraceCarrier == nil {
			ev.TraceCarrier = carrier
		}
		if !ev.Partial {
			if err := ex.persist(ctx, ev); err != nil {
				ex.runErr = err
				yield(nil, err)
				return
			}
		}
		if !yield(ev, nil) {
			return
		}
	}
}

// repairDangling closes tool calls at the end of the log that never got a
// result: ReplaySafe tools are re-run, others get an error result.
func (ex *execution) repairDangling(ctx context.Context) error {
	// Calls of a pending pause are closed too: Run is only reached with a
	// new user message (which supersedes the pause) or after Resume
	// has recorded its intent, so any call still unanswered here is dangling.
	answered := map[string]bool{}
	for _, ev := range ex.events {
		if ev.Message == nil {
			continue
		}
		for _, res := range ev.Message.ToolResultBlocks() {
			answered[res.CallID] = true
		}
	}
	for _, ev := range ex.events {
		if ev.Message == nil || ev.Message.Role != schema.RoleAssistant {
			continue
		}
		if ev.Message.StopReason == schema.StopReasonLength {
			continue
		}
		var dangling []*schema.ToolCallBlock
		for _, c := range ev.Message.ToolCalls() {
			if !answered[c.ID] && c.Name != schema.OutputToolName {
				dangling = append(dangling, c)
			}
		}
		if len(dangling) == 0 {
			continue
		}
		author := ev.Author
		a := ex.r.tree.Find(author)
		res := &schema.Message{Role: schema.RoleTool}
		for _, c := range dangling {
			var tool schema.Tool
			if f, ok := a.(agent.ToolFinder); ok && a != nil {
				tool = f.FindTool(ctx, c.Name)
			}
			if tool != nil && schema.IsReplaySafe(tool) {
				inv := ex.inv.WithAgent(a).WithBranch(ev.Branch)
				out, err := agent.CallTool(ctx, inv, tool, c)
				if err == nil {
					if out == nil {
						out = &schema.ToolResult{}
					}
					res.Content = append(res.Content, out.Block(c.ID))
					answered[c.ID] = true
					continue
				}
			}
			res.Content = append(res.Content, &schema.ToolResultBlock{CallID: c.ID, IsError: true,
				Content: []schema.ContentBlock{&schema.TextBlock{Text: InterruptedText}}})
			answered[c.ID] = true
		}
		fix := &schema.Event{ID: ex.r.cfg.IDFunc(), InvocationID: ex.inv.ID, Author: author, Branch: ev.Branch, Message: res, CreatedAt: ex.r.cfg.Clock()}
		if err := ex.persist(ctx, fix); err != nil {
			return err
		}
	}
	return nil
}
