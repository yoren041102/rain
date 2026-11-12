package agent

import (
	"context"
	"errors"
	"time"

	"github.com/yoren041102/rain/schema"
)

// Session identifies a conversation and exposes its state.
type Session interface {
	ID() string
	AppName() string
	UserID() string
	State() State
}

var (
	// ErrSessionNotFound is returned when a session does not exist.
	ErrSessionNotFound = errors.New("rain: session not found")
	// ErrSessionExists is returned when creating a session whose ID exists.
	ErrSessionExists = errors.New("rain: session already exists")
	// ErrSeqConflict is returned by AppendEvents when expectedSeq does not
	// match the session's last sequence number.
	ErrSeqConflict = errors.New("rain: event sequence conflict")
)

// SessionStore persists sessions and their event logs.
type SessionStore interface {
	// Create creates a session. An empty sessionID asks the store to
	// generate one.
	Create(ctx context.Context, appName, userID, sessionID string, init map[string]any) (Session, error)
	Get(ctx context.Context, appName, userID, sessionID string) (Session, error)
	// AppendEvents atomically appends events, assigning Seq values
	// expectedSeq+1, expectedSeq+2, ... and applying each StateDelta. It
	// returns ErrSeqConflict if the last stored Seq is not expectedSeq.
	AppendEvents(ctx context.Context, s Session, events []*schema.Event, expectedSeq int64) error
	Events(ctx context.Context, s Session, opts EventsOptions) (*EventPage, error)
	Delete(ctx context.Context, appName, userID, sessionID string) error
}

// EventsOptions filters and pages events.
type EventsOptions struct {
	Branch string // only events visible to this branch (see BranchVisible)
	After  string // cursor from a previous EventPage
	Limit  int    // 0 means no limit
}

// EventPage is one page of events.
type EventPage struct {
	Events []*schema.Event
	Cursor string // pass as After for the next page; empty when exhausted
}

// AllEvents reads every event of s, following cursors.
func AllEvents(ctx context.Context, store SessionStore, s Session, branch string) ([]*schema.Event, error) {
	var out []*schema.Event
	after := ""
	for {
		page, err := store.Events(ctx, s, EventsOptions{Branch: branch, After: after})
		if err != nil {
			return nil, err
		}
		out = append(out, page.Events...)
		if page.Cursor == "" || len(page.Events) == 0 {
			return out, nil
		}
		after = page.Cursor
	}
}

// MemoryEntry is one search hit.
type MemoryEntry struct {
	SessionID string
	Author    string
	Content   *schema.Message
	CreatedAt time.Time
}

// Memory is long-term, cross-session memory.
type Memory interface {
	Add(ctx context.Context, appName, userID, sessionID string, events []*schema.Event) error
	Search(ctx context.Context, appName, userID, query string) ([]MemoryEntry, error)
}
