// Package history provides history-processing model middlewares. They
// rewrite ModelRequest.Messages only; session events are never modified,
// except that Summarize persists its summary as a compaction event.
package history

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
)

// split separates leading system messages from the rest.
func split(msgs []*schema.Message) (sys, rest []*schema.Message) {
	i := 0
	for i < len(msgs) && msgs[i].Role == schema.RoleSystem {
		i++
	}
	return msgs[:i], msgs[i:]
}

// turnStarts returns the indexes in rest where a user turn starts, i.e.
// safe cut points that never separate a tool call from its result.
func turnStarts(rest []*schema.Message) []int {
	var out []int
	for i, m := range rest {
		if m.Role == schema.RoleUser {
			out = append(out, i)
		}
	}
	return out
}

// cut keeps rest[from:], moving from forward to the next user-turn boundary
// so tool calls and results stay paired and at most len(rest)-from messages
// are kept. If no boundary follows, the latest turn is kept.
func cut(rest []*schema.Message, from int) []*schema.Message {
	if from <= 0 {
		return rest
	}
	starts := turnStarts(rest)
	for _, s := range starts {
		if s >= from {
			return rest[s:]
		}
	}
	if len(starts) > 0 {
		return rest[starts[len(starts)-1]:]
	}
	return rest
}

func rewrite(next agent.ModelHandler, f func(*schema.ModelRequest) []*schema.Message) agent.ModelHandler {
	return func(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
		r := req.Clone()
		r.Messages = f(req)
		return next(ctx, inv, r)
	}
}

// KeepLast keeps system messages plus at most the last n messages, cutting
// at a user-turn boundary so tool calls and results stay paired. The latest
// user turn is always kept, even if it is longer than n.
func KeepLast(n int) agent.ModelMiddleware {
	return func(next agent.ModelHandler) agent.ModelHandler {
		return rewrite(next, func(req *schema.ModelRequest) []*schema.Message {
			sys, rest := split(req.Messages)
			if len(rest) <= n {
				return req.Messages
			}
			return append(append([]*schema.Message(nil), sys...), cut(rest, len(rest)-n)...)
		})
	}
}

// Counter estimates the token count of a message.
type Counter func(*schema.Message) int

// ApproxCounter estimates tokens as characters/4 plus a small overhead.
func ApproxCounter(m *schema.Message) int {
	n := 4
	for _, b := range m.Content {
		switch v := b.(type) {
		case *schema.TextBlock:
			n += len(v.Text) / 4
		case *schema.ReasoningBlock:
			n += len(v.Text) / 4
		case *schema.ToolCallBlock:
			n += (len(v.Name) + len(v.Arguments)) / 4
		case *schema.ToolResultBlock:
			n += ApproxCounter(&schema.Message{Content: v.Content})
		case *schema.ImageBlock, *schema.FileBlock:
			n += 1000
		}
	}
	return n
}

// TokenBudget drops the oldest turns until the history fits within budget
// tokens (system messages are always kept). The latest turn is always kept.
func TokenBudget(budget int, counter Counter) agent.ModelMiddleware {
	if counter == nil {
		counter = ApproxCounter
	}
	return func(next agent.ModelHandler) agent.ModelHandler {
		return rewrite(next, func(req *schema.ModelRequest) []*schema.Message {
			sys, rest := split(req.Messages)
			total := 0
			for _, m := range req.Messages {
				total += counter(m)
			}
			if total <= budget {
				return req.Messages
			}
			starts := turnStarts(rest)
			for _, s := range starts {
				if s == 0 {
					continue
				}
				t := 0
				for _, m := range sys {
					t += counter(m)
				}
				for _, m := range rest[s:] {
					t += counter(m)
				}
				if t <= budget {
					return append(append([]*schema.Message(nil), sys...), rest[s:]...)
				}
			}
			if len(starts) > 0 {
				return append(append([]*schema.Message(nil), sys...), rest[starts[len(starts)-1]:]...)
			}
			return req.Messages
		})
	}
}

// SummarizeOption configures Summarize.
type SummarizeOption func(*summarizer)

// WithCounter sets the token counter (default ApproxCounter).
func WithCounter(c Counter) SummarizeOption { return func(s *summarizer) { s.counter = c } }

// WithKeepTurns sets how many recent user turns are kept verbatim (default 2).
func WithKeepTurns(n int) SummarizeOption { return func(s *summarizer) { s.keep = n } }

// WithPrompt overrides the summarization instruction.
func WithPrompt(p string) SummarizeOption { return func(s *summarizer) { s.prompt = p } }

type summarizer struct {
	model     schema.Model
	threshold int
	counter   Counter
	keep      int
	prompt    string
	mu        sync.Mutex
}

const defaultPrompt = "Summarize the conversation so far for an assistant that will continue it. Keep facts, decisions, open tasks, names, numbers and tool results that may matter later. Be concise."

// Summarize compacts the history with model once it exceeds threshold
// tokens. The summary is emitted as a compaction event (Author
// schema.CompactionAuthor) that records the first kept event Seq, so the
// runner persists it and later runs reuse it instead of recomputing. Cuts
// happen only at user-turn boundaries.
func Summarize(model schema.Model, threshold int, opts ...SummarizeOption) agent.ModelMiddleware {
	s := &summarizer{model: model, threshold: threshold, counter: ApproxCounter, keep: 2, prompt: defaultPrompt}
	for _, o := range opts {
		o(s)
	}
	return func(next agent.ModelHandler) agent.ModelHandler {
		return func(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
			r, err := s.compact(ctx, inv, req)
			if err != nil {
				return schema.Error[*schema.ModelResponse](err)
			}
			return next(ctx, inv, r)
		}
	}
}

func (s *summarizer) compact(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) (*schema.ModelRequest, error) {
	total := 0
	for _, m := range req.Messages {
		total += s.counter(m)
	}
	if total <= s.threshold || inv == nil || inv.History == nil {
		return req, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	events, err := inv.History.Events(ctx)
	if err != nil {
		return nil, err
	}
	// user-turn boundaries in the persisted log (visible to this branch)
	var bounds []int
	for i, ev := range events {
		if ev.Author == schema.AuthorUser && ev.Message != nil && ev.Seq > 0 && agent.BranchVisible(inv.Branch, ev.Branch) {
			bounds = append(bounds, i)
		}
	}
	if len(bounds) <= s.keep {
		return req, nil
	}
	cutIdx := bounds[len(bounds)-s.keep]
	firstKept := events[cutIdx].Seq
	// the previous compaction, if any, already covers older events
	var prevSummary string
	var prevFirst int64
	for i := len(events) - 1; i >= 0; i-- {
		if c := events[i].Actions.Compaction; c != nil && events[i].Author == schema.CompactionAuthor {
			prevSummary, prevFirst = c.Summary, c.FirstKeptSeq
			break
		}
	}
	if prevFirst >= firstKept {
		return req, nil
	}
	var older []*schema.Event
	for _, ev := range events[:cutIdx] {
		if ev.Seq >= prevFirst && ev.Author != schema.CompactionAuthor {
			older = append(older, ev)
		}
	}
	name := ""
	if inv.Agent != nil {
		name = inv.Agent.Name()
	}
	var transcript strings.Builder
	if prevSummary != "" {
		fmt.Fprintf(&transcript, "Earlier summary:\n%s\n\n", prevSummary)
	}
	for _, m := range agent.MessagesFor(older, name, inv.Branch) {
		fmt.Fprintf(&transcript, "%s: %s\n", m.Role, renderForSummary(m))
	}
	sreq := &schema.ModelRequest{Messages: []*schema.Message{
		schema.SystemText(s.prompt),
		schema.UserText(transcript.String()),
	}}
	resp, err := schema.FinalResponse(s.model.Generate(ctx, sreq))
	if err != nil {
		return nil, fmt.Errorf("history: summarize: %w", err)
	}
	summary := resp.Message.Text()
	if inv.Emit != nil {
		ev := inv.NewEvent(nil)
		ev.Author = schema.CompactionAuthor
		ev.Actions.Compaction = &schema.Compaction{Summary: summary, FirstKeptSeq: firstKept}
		inv.Emit(ev)
	}
	// rebuild the request from the kept events
	sys, _ := split(req.Messages)
	kept := agent.MessagesFor(events[cutIdx:], name, inv.Branch)
	// carry over messages from this run not yet in the log
	_, rest := split(req.Messages)
	logged := len(agent.MessagesFor(events, name, inv.Branch))
	if extra := len(rest) - logged; extra > 0 && extra <= len(rest) {
		kept = append(kept, rest[len(rest)-extra:]...)
	}
	r := req.Clone()
	r.Messages = append(append(append([]*schema.Message(nil), sys...), schema.UserText("Summary of the earlier conversation:\n"+summary)), schema.PrepareForModel(kept, schema.ModelOrigin(s.model))...)
	return r, nil
}

func renderForSummary(m *schema.Message) string {
	var parts []string
	for _, b := range m.Content {
		switch v := b.(type) {
		case *schema.TextBlock:
			parts = append(parts, v.Text)
		case *schema.ToolCallBlock:
			parts = append(parts, fmt.Sprintf("[call %s %s]", v.Name, v.Arguments))
		case *schema.ToolResultBlock:
			parts = append(parts, fmt.Sprintf("[result %s]", (&schema.Message{Content: v.Content}).Text()))
		}
	}
	return strings.Join(parts, " ")
}
