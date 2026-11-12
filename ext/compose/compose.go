// Package compose provides orchestration agents: Sequential, Parallel and
// Loop. There is deliberately no graph engine.
package compose

import (
	"context"
	"errors"
	"sync"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
)

type base struct {
	name, desc string
	subs       []agent.Agent
}

func (b *base) Name() string             { return b.name }
func (b *base) Description() string      { return b.desc }
func (b *base) SubAgents() []agent.Agent { return b.subs }

// ---- Sequential ----

// SequentialAgent runs its sub-agents one after another on the same history.
type SequentialAgent struct{ base }

// Sequential returns an agent that runs subs in order. Execution stops early
// on an interrupt or an Escalate action.
func Sequential(name, description string, subs ...agent.Agent) *SequentialAgent {
	return &SequentialAgent{base{name, description, subs}}
}

func (s *SequentialAgent) Run(ctx context.Context, inv *agent.Invocation) schema.Stream[*schema.Event] {
	return func(yield func(*schema.Event, error) bool) {
		start := 0
		if inv.Resuming() {
			start = resumeIndex(inv, s.subs)
			if start < 0 {
				return
			}
		}
		for i := start; i < len(s.subs); i++ {
			sub := s.subs[i]
			cinv := inv
			if i != start {
				cinv = inv.WithoutResume()
			}
			stop := false
			for ev, err := range agent.RunAgent(ctx, cinv, sub) {
				if err != nil {
					yield(nil, err)
					return
				}
				if !yield(ev, nil) {
					return
				}
				if !ev.Partial && (len(ev.Actions.AllInterrupts()) > 0 || ev.Actions.Escalate || ev.Actions.TransferTo != "") {
					stop = true
				}
			}
			if stop {
				return
			}
		}
	}
}

// resumeIndex returns the index of the child holding the open interrupts,
// or -1 if none (nothing to resume).
func resumeIndex(inv *agent.Invocation, subs []agent.Agent) int {
	for i, s := range subs {
		if len(inv.InterruptsUnder(s.Name())) > 0 {
			return i
		}
	}
	return -1
}

// ---- Loop ----

// LoopAgent repeats its sub-agents until one escalates or MaxIterations.
type LoopAgent struct {
	base
	max int
}

// Loop returns an agent that runs subs in order repeatedly until an event
// carries Escalate, an interrupt occurs, or maxIterations rounds have run
// (0 means unlimited).
func Loop(name, description string, maxIterations int, subs ...agent.Agent) *LoopAgent {
	return &LoopAgent{base{name, description, subs}, maxIterations}
}

func (l *LoopAgent) Run(ctx context.Context, inv *agent.Invocation) schema.Stream[*schema.Event] {
	return func(yield func(*schema.Event, error) bool) {
		start := 0
		if inv.Resuming() {
			start = resumeIndex(inv, l.subs)
			if start < 0 {
				return
			}
		}
		cur := inv
		for round := 0; l.max == 0 || round < l.max; round++ {
			for i := start; i < len(l.subs); i++ {
				if err := ctx.Err(); err != nil {
					yield(nil, err)
					return
				}
				stop := false
				for ev, err := range agent.RunAgent(ctx, cur, l.subs[i]) {
					if err != nil {
						yield(nil, err)
						return
					}
					if !yield(ev, nil) {
						return
					}
					if !ev.Partial && (ev.Actions.Escalate || len(ev.Actions.AllInterrupts()) > 0) {
						stop = true
					}
				}
				cur = inv.WithoutResume()
				if stop {
					return
				}
			}
			start = 0
		}
	}
}

// ---- Parallel ----

// ParallelAgent runs its sub-agents concurrently, each on its own branch so
// they do not see each other's events.
type ParallelAgent struct{ base }

// Parallel returns an agent running subs concurrently. Events are yielded as
// they arrive. All goroutines are stopped and joined before Run returns,
// including when the consumer breaks.
func Parallel(name, description string, subs ...agent.Agent) *ParallelAgent {
	return &ParallelAgent{base{name, description, subs}}
}

type item struct {
	ev  *schema.Event
	err error
}

func (p *ParallelAgent) Run(ctx context.Context, inv *agent.Invocation) schema.Stream[*schema.Event] {
	return func(yield func(*schema.Event, error) bool) {
		ctx, cancel := context.WithCancel(ctx)
		var wg sync.WaitGroup
		defer func() {
			cancel()
			wg.Wait()
		}()
		ch := make(chan item)
		var errs []error
		var errMu sync.Mutex
		for _, sub := range p.subs {
			cinv := inv.WithBranch(inv.ChildBranch(sub.Name()))
			if inv.Resuming() {
				if len(inv.InterruptsUnder(sub.Name())) == 0 {
					continue // finished in the earlier run
				}
			}
			wg.Add(1)
			go func(sub agent.Agent, cinv *agent.Invocation) {
				defer wg.Done()
				for ev, err := range agent.RunAgent(ctx, cinv, sub) {
					if err != nil {
						if !errors.Is(err, context.Canceled) || ctx.Err() == nil {
							errMu.Lock()
							errs = append(errs, err)
							errMu.Unlock()
						}
						return
					}
					select {
					case ch <- item{ev: ev}:
					case <-ctx.Done():
						return
					}
				}
			}(sub, cinv)
		}
		go func() {
			wg.Wait()
			close(ch)
		}()
		for it := range ch {
			if !yield(it.ev, nil) {
				return
			}
		}
		if len(errs) > 0 {
			yield(nil, errors.Join(errs...))
		}
	}
}
