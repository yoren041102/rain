package agent

import (
	"iter"
	"maps"
	"sync"
)

// State is read/write session state.
type State interface {
	Get(key string) (any, bool)
	Set(key string, v any)
	All() iter.Seq2[string, any]
}

// Reserved state key prefixes and keys.
const (
	TempPrefix     = "temp:"
	InternalPrefix = "_rain."
	KeyActiveAgent = "_rain.active_agent"
	KeyMemorySeq   = "_rain.memory_seq"
)

// MapState is a simple concurrency-safe State.
type MapState struct {
	mu sync.RWMutex
	m  map[string]any
}

// NewMapState returns a MapState holding a copy of init.
func NewMapState(init map[string]any) *MapState {
	return &MapState{m: maps.Clone(init)}
}

func (s *MapState) Get(k string) (any, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.m[k]
	return v, ok
}

func (s *MapState) Set(k string, v any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.m == nil {
		s.m = map[string]any{}
	}
	if v == nil {
		delete(s.m, k)
		return
	}
	s.m[k] = v
}

func (s *MapState) All() iter.Seq2[string, any] {
	s.mu.RLock()
	snap := maps.Clone(s.m)
	s.mu.RUnlock()
	return func(yield func(string, any) bool) {
		for k, v := range snap {
			if !yield(k, v) {
				return
			}
		}
	}
}

// Map returns a copy of the state.
func (s *MapState) Map() map[string]any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return maps.Clone(s.m)
}

// DeltaState overlays local writes on a base State. Writes are recorded and
// handed out once by TakeDelta, which RunAgent attaches to the next event's
// StateDelta. Reads see local writes first. Setting nil records a deletion.
type DeltaState struct {
	mu      sync.Mutex
	base    State
	local   map[string]any
	deleted map[string]bool
	pending map[string]any
}

// NewDeltaState wraps base (which may be nil).
func NewDeltaState(base State) *DeltaState {
	return &DeltaState{base: base, local: map[string]any{}, deleted: map[string]bool{}, pending: map[string]any{}}
}

func (s *DeltaState) Get(k string) (any, bool) {
	s.mu.Lock()
	if v, ok := s.local[k]; ok {
		s.mu.Unlock()
		return v, true
	}
	if s.deleted[k] {
		s.mu.Unlock()
		return nil, false
	}
	s.mu.Unlock()
	if s.base == nil {
		return nil, false
	}
	return s.base.Get(k)
}

func (s *DeltaState) Set(k string, v any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if v == nil {
		delete(s.local, k)
		s.deleted[k] = true
	} else {
		s.local[k] = v
		delete(s.deleted, k)
	}
	s.pending[k] = v
}

// Apply records a delta coming from elsewhere (e.g. tool actions) as already
// reported: it updates reads without adding to the pending delta.
func (s *DeltaState) Apply(delta map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k, v := range delta {
		if v == nil {
			delete(s.local, k)
			s.deleted[k] = true
		} else {
			s.local[k] = v
			delete(s.deleted, k)
		}
	}
}

func (s *DeltaState) All() iter.Seq2[string, any] {
	merged := map[string]any{}
	if s.base != nil {
		for k, v := range s.base.All() {
			merged[k] = v
		}
	}
	s.mu.Lock()
	for k := range s.deleted {
		delete(merged, k)
	}
	for k, v := range s.local {
		merged[k] = v
	}
	s.mu.Unlock()
	return func(yield func(string, any) bool) {
		for k, v := range merged {
			if !yield(k, v) {
				return
			}
		}
	}
}

// TakeDelta returns and clears the writes not yet reported.
func (s *DeltaState) TakeDelta() map[string]any {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.pending) == 0 {
		return nil
	}
	d := s.pending
	s.pending = map[string]any{}
	return d
}
