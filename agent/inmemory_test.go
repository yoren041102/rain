package agent_test

import (
	"testing"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/agent/agenttest"
)

func TestInMemorySessionStore(t *testing.T) {
	agenttest.TestSessionStore(t, agent.NewInMemorySessionStore)
}
