// Package agenttest holds conformance tests shared by every SessionStore
// backend, plus a scripted fake model for agent tests.
package agenttest

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/schema"
)

// TestSessionStore runs the SessionStore conformance suite.
func TestSessionStore(t *testing.T, factory func() agent.SessionStore) {
	ctx := context.Background()

	t.Run("CreateGetDelete", func(t *testing.T) {
		st := factory()
		s, err := st.Create(ctx, "app", "u", "s1", map[string]any{"k": "v", "temp:x": 1})
		if err != nil {
			t.Fatal(err)
		}
		if s.ID() != "s1" || s.AppName() != "app" || s.UserID() != "u" {
			t.Fatalf("bad identity: %s %s %s", s.ID(), s.AppName(), s.UserID())
		}
		if _, err := st.Create(ctx, "app", "u", "s1", nil); !errors.Is(err, agent.ErrSessionExists) {
			t.Fatalf("duplicate create: got %v", err)
		}
		g, err := st.Get(ctx, "app", "u", "s1")
		if err != nil {
			t.Fatal(err)
		}
		if v, _ := g.State().Get("k"); v != "v" {
			t.Fatalf("state k = %v", v)
		}
		if _, ok := g.State().Get("temp:x"); ok {
			t.Fatal("temp: key persisted")
		}
		if _, err := st.Get(ctx, "app", "other", "s1"); !errors.Is(err, agent.ErrSessionNotFound) {
			t.Fatalf("other user: got %v", err)
		}
		if err := st.Delete(ctx, "app", "u", "s1"); err != nil {
			t.Fatal(err)
		}
		if _, err := st.Get(ctx, "app", "u", "s1"); !errors.Is(err, agent.ErrSessionNotFound) {
			t.Fatalf("after delete: got %v", err)
		}
	})

	t.Run("GeneratedID", func(t *testing.T) {
		st := factory()
		s, err := st.Create(ctx, "app", "u", "", nil)
		if err != nil {
			t.Fatal(err)
		}
		if s.ID() == "" {
			t.Fatal("empty generated id")
		}
	})

	t.Run("AppendAndState", func(t *testing.T) {
		st := factory()
		s, _ := st.Create(ctx, "app", "u", "s", nil)
		e1 := &schema.Event{ID: "e1", Author: "user", Message: schema.UserText("hi")}
		e2 := &schema.Event{ID: "e2", Author: "a", Message: schema.AssistantText("yo"),
			Actions: schema.EventActions{StateDelta: map[string]any{"x": "1", "temp:t": "no"}}}
		if err := st.AppendEvents(ctx, s, []*schema.Event{e1, e2}, 0); err != nil {
			t.Fatal(err)
		}
		if e1.Seq != 1 || e2.Seq != 2 {
			t.Fatalf("seq = %d,%d", e1.Seq, e2.Seq)
		}
		if err := st.AppendEvents(ctx, s, []*schema.Event{{ID: "e3"}}, 1); !errors.Is(err, agent.ErrSeqConflict) {
			t.Fatalf("stale expectedSeq: got %v", err)
		}
		g, _ := st.Get(ctx, "app", "u", "s")
		if v, _ := g.State().Get("x"); v != "1" {
			t.Fatalf("x = %v", v)
		}
		if _, ok := g.State().Get("temp:t"); ok {
			t.Fatal("temp: key persisted")
		}
		// delete via nil
		if err := st.AppendEvents(ctx, s, []*schema.Event{{ID: "e3", Actions: schema.EventActions{StateDelta: map[string]any{"x": nil}}}}, 2); err != nil {
			t.Fatal(err)
		}
		g, _ = st.Get(ctx, "app", "u", "s")
		if _, ok := g.State().Get("x"); ok {
			t.Fatal("x not deleted")
		}
		page, err := st.Events(ctx, g, agent.EventsOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if len(page.Events) != 3 || page.Events[1].Message.Text() != "yo" {
			t.Fatalf("events = %v", page.Events)
		}
	})

	t.Run("PagingAndBranch", func(t *testing.T) {
		st := factory()
		s, _ := st.Create(ctx, "app", "u", "s", nil)
		var evs []*schema.Event
		for i := 0; i < 5; i++ {
			b := ""
			if i%2 == 1 {
				b = "root.x"
			}
			if i == 4 {
				b = "root.y"
			}
			evs = append(evs, &schema.Event{ID: fmt.Sprint(i), Branch: b})
		}
		if err := st.AppendEvents(ctx, s, evs, 0); err != nil {
			t.Fatal(err)
		}
		var got []string
		after := ""
		for {
			p, err := st.Events(ctx, s, agent.EventsOptions{After: after, Limit: 2})
			if err != nil {
				t.Fatal(err)
			}
			for _, e := range p.Events {
				got = append(got, e.ID)
			}
			if p.Cursor == "" {
				break
			}
			after = p.Cursor
		}
		if fmt.Sprint(got) != "[0 1 2 3 4]" {
			t.Fatalf("paged = %v", got)
		}
		p, _ := st.Events(ctx, s, agent.EventsOptions{Branch: "root.x"})
		var ids []string
		for _, e := range p.Events {
			ids = append(ids, e.ID)
		}
		if fmt.Sprint(ids) != "[0 1 2 3]" {
			t.Fatalf("branch filter = %v", ids)
		}
	})

	t.Run("ConcurrentAppend", func(t *testing.T) {
		st := factory()
		s, _ := st.Create(ctx, "app", "u", "s", nil)
		var wg sync.WaitGroup
		var mu sync.Mutex
		ok := 0
		for i := 0; i < 10; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				if st.AppendEvents(ctx, s, []*schema.Event{{ID: "x"}}, 0) == nil {
					mu.Lock()
					ok++
					mu.Unlock()
				}
			}()
		}
		wg.Wait()
		if ok != 1 {
			t.Fatalf("%d concurrent appends with the same expectedSeq succeeded", ok)
		}
	})
}

// Turn is one scripted model response.
type Turn struct {
	Message *schema.Message
	Err     error
	// Chunks, if set, are streamed as partial responses before Message.
	Chunks []*schema.ModelResponse
}

// FakeModel replays scripted turns and records requests.
type FakeModel struct {
	mu        sync.Mutex
	ModelName string
	Turns     []Turn
	Requests  []*schema.ModelRequest
	// Fn, if set, is used instead of Turns.
	Fn func(req *schema.ModelRequest) (*schema.Message, error)
}

func (m *FakeModel) Name() string {
	if m.ModelName == "" {
		return "fake"
	}
	return m.ModelName
}

func (m *FakeModel) Origin() schema.Origin {
	return schema.Origin{Provider: "fake", API: "fake", Model: m.Name()}
}

func (m *FakeModel) Generate(ctx context.Context, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
	return func(yield func(*schema.ModelResponse, error) bool) {
		m.mu.Lock()
		m.Requests = append(m.Requests, req)
		var turn Turn
		if m.Fn != nil {
			msg, err := m.Fn(req)
			turn = Turn{Message: msg, Err: err}
		} else if len(m.Turns) == 0 {
			m.mu.Unlock()
			yield(nil, errors.New("fake model: no more turns"))
			return
		} else {
			turn = m.Turns[0]
			m.Turns = m.Turns[1:]
		}
		m.mu.Unlock()
		for _, c := range turn.Chunks {
			if !yield(c, nil) {
				return
			}
		}
		if turn.Err != nil {
			yield(nil, turn.Err)
			return
		}
		msg := turn.Message.Clone()
		o := m.Origin()
		msg.Origin = &o
		if msg.Role == "" {
			msg.Role = schema.RoleAssistant
		}
		yield(&schema.ModelResponse{Message: msg, FinishReason: msg.StopReason}, nil)
	}
}

// Call builds an assistant message with tool calls.
func Call(calls ...*schema.ToolCallBlock) *schema.Message {
	m := &schema.Message{Role: schema.RoleAssistant, StopReason: schema.StopReasonToolUse}
	for _, c := range calls {
		m.Content = append(m.Content, c)
	}
	return m
}

// TC builds a ToolCallBlock.
func TC(id, name, args string) *schema.ToolCallBlock {
	return &schema.ToolCallBlock{ID: id, Name: name, Arguments: []byte(args)}
}
