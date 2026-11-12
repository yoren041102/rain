package agent

import (
	"context"
	"fmt"

	"strconv"
	"strings"
	"sync"

	"github.com/yoren041102/rain/schema"
)

type memSession struct {
	id, app, user string
	state         *MapState
}

func (s *memSession) ID() string      { return s.id }
func (s *memSession) AppName() string { return s.app }
func (s *memSession) UserID() string  { return s.user }
func (s *memSession) State() State    { return s.state }

type memRecord struct {
	state  map[string]any
	events []*schema.Event
}

type inMemoryStore struct {
	mu       sync.RWMutex
	sessions map[string]*memRecord
}

// NewInMemorySessionStore returns a SessionStore that keeps everything in
// memory. Sessions returned by Get hold a state snapshot.
func NewInMemorySessionStore() SessionStore {
	return &inMemoryStore{sessions: map[string]*memRecord{}}
}

func key(app, user, id string) string { return app + "\x00" + user + "\x00" + id }

func (st *inMemoryStore) Create(_ context.Context, app, user, id string, init map[string]any) (Session, error) {
	if id == "" {
		id = NewUUID()
	}
	st.mu.Lock()
	defer st.mu.Unlock()
	k := key(app, user, id)
	if _, ok := st.sessions[k]; ok {
		return nil, fmt.Errorf("%w: %s", ErrSessionExists, id)
	}
	state := map[string]any{}
	for k, v := range init {
		if !strings.HasPrefix(k, TempPrefix) {
			state[k] = v
		}
	}
	st.sessions[k] = &memRecord{state: state}
	return &memSession{id: id, app: app, user: user, state: NewMapState(state)}, nil
}

func (st *inMemoryStore) Get(_ context.Context, app, user, id string) (Session, error) {
	st.mu.RLock()
	defer st.mu.RUnlock()
	r, ok := st.sessions[key(app, user, id)]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrSessionNotFound, id)
	}
	return &memSession{id: id, app: app, user: user, state: NewMapState(r.state)}, nil
}

func (st *inMemoryStore) AppendEvents(_ context.Context, s Session, events []*schema.Event, expectedSeq int64) error {
	st.mu.Lock()
	defer st.mu.Unlock()
	r, ok := st.sessions[key(s.AppName(), s.UserID(), s.ID())]
	if !ok {
		return fmt.Errorf("%w: %s", ErrSessionNotFound, s.ID())
	}
	var last int64
	if n := len(r.events); n > 0 {
		last = r.events[n-1].Seq
	}
	if last != expectedSeq {
		return fmt.Errorf("%w: expected %d, have %d", ErrSeqConflict, expectedSeq, last)
	}
	for i, ev := range events {
		ev.Seq = expectedSeq + int64(i) + 1
		c := ev.Clone()
		c.Partial = false
		r.events = append(r.events, c)
		for k, v := range ev.Actions.StateDelta {
			if strings.HasPrefix(k, TempPrefix) {
				continue
			}
			if v == nil {
				delete(r.state, k)
			} else {
				r.state[k] = v
			}
			if ms, ok := s.(*memSession); ok {
				ms.state.Set(k, v)
			}
		}
	}
	return nil
}

func (st *inMemoryStore) Events(_ context.Context, s Session, opts EventsOptions) (*EventPage, error) {
	st.mu.RLock()
	defer st.mu.RUnlock()
	r, ok := st.sessions[key(s.AppName(), s.UserID(), s.ID())]
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrSessionNotFound, s.ID())
	}
	var after int64
	if opts.After != "" {
		v, err := strconv.ParseInt(opts.After, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("rain: invalid cursor %q", opts.After)
		}
		after = v
	}
	page := &EventPage{}
	for _, ev := range r.events {
		if ev.Seq <= after || !BranchVisible(opts.Branch, ev.Branch) {
			continue
		}
		if opts.Limit > 0 && len(page.Events) == opts.Limit {
			page.Cursor = strconv.FormatInt(page.Events[len(page.Events)-1].Seq, 10)
			return page, nil
		}
		page.Events = append(page.Events, ev.Clone())
	}
	return page, nil
}

func (st *inMemoryStore) Delete(_ context.Context, app, user, id string) error {
	st.mu.Lock()
	defer st.mu.Unlock()
	k := key(app, user, id)
	if _, ok := st.sessions[k]; !ok {
		return fmt.Errorf("%w: %s", ErrSessionNotFound, id)
	}
	delete(st.sessions, k)
	return nil
}

// ---- memory ----

type inMemoryMemory struct {
	mu      sync.RWMutex
	entries map[string][]MemoryEntry // app\x00user -> entries
	seen    map[string]bool          // event IDs
}

// NewInMemoryMemory returns a Memory with case-insensitive keyword search.
func NewInMemoryMemory() Memory {
	return &inMemoryMemory{entries: map[string][]MemoryEntry{}, seen: map[string]bool{}}
}

func (m *inMemoryMemory) Add(_ context.Context, app, user, sessionID string, events []*schema.Event) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	k := app + "\x00" + user
	for _, ev := range events {
		if ev.Message == nil || ev.Partial || ev.Message.Text() == "" {
			continue
		}
		if ev.ID != "" {
			if m.seen[ev.ID] {
				continue
			}
			m.seen[ev.ID] = true
		}
		m.entries[k] = append(m.entries[k], MemoryEntry{SessionID: sessionID, Author: ev.Author, Content: ev.Message.Clone(), CreatedAt: ev.CreatedAt})
	}
	return nil
}

func (m *inMemoryMemory) Search(_ context.Context, app, user, query string) ([]MemoryEntry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	words := strings.Fields(strings.ToLower(query))
	var out []MemoryEntry
	for _, e := range m.entries[app+"\x00"+user] {
		text := strings.ToLower(e.Content.Text())
		for _, w := range words {
			if strings.Contains(text, w) {
				out = append(out, e)
				break
			}
		}
	}
	return out, nil
}
