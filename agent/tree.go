package agent

import (
	"errors"
	"fmt"
)

// Tree indexes an agent hierarchy.
type Tree struct {
	root    Agent
	byName  map[string]Agent
	parents map[string]Agent
}

// NewTree builds a tree rooted at root. It fails if two agents share a name
// (which includes the same instance appearing twice).
func NewTree(root Agent) (*Tree, error) {
	if root == nil {
		return nil, errors.New("rain: nil root agent")
	}
	t := &Tree{root: root, byName: map[string]Agent{}, parents: map[string]Agent{}}
	var walk func(a, parent Agent) error
	walk = func(a, parent Agent) error {
		if a == nil {
			return errors.New("rain: nil sub-agent")
		}
		name := a.Name()
		if name == "" {
			return errors.New("rain: agent with empty name")
		}
		if name == "user" || len(name) >= 6 && name[:6] == "_rain." {
			return fmt.Errorf("rain: reserved agent name %q", name)
		}
		if _, dup := t.byName[name]; dup {
			return fmt.Errorf("rain: duplicate agent name %q", name)
		}
		t.byName[name] = a
		if parent != nil {
			t.parents[name] = parent
		}
		for _, c := range a.SubAgents() {
			if err := walk(c, a); err != nil {
				return err
			}
		}
		return nil
	}
	if err := walk(root, nil); err != nil {
		return nil, err
	}
	return t, nil
}

func (t *Tree) Root() Agent { return t.root }

// Find returns the agent named name, or nil.
func (t *Tree) Find(name string) Agent { return t.byName[name] }

// Parent returns a's parent, or nil for the root or unknown agents.
func (t *Tree) Parent(a Agent) Agent {
	if a == nil {
		return nil
	}
	return t.parents[a.Name()]
}

// Path returns agent names from the root to a, or nil if a is not in t.
func (t *Tree) Path(a Agent) []string {
	if a == nil || t.byName[a.Name()] == nil {
		return nil
	}
	var rev []string
	for cur := a; cur != nil; cur = t.parents[cur.Name()] {
		rev = append(rev, cur.Name())
	}
	out := make([]string, len(rev))
	for i, n := range rev {
		out[len(rev)-1-i] = n
	}
	return out
}
