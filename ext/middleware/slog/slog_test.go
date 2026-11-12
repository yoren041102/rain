package slog_test

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/agent/agenttest"
	rainslog "github.com/yoren041102/rain/ext/middleware/slog"
	"github.com/yoren041102/rain/ext/tool/fromfunc"
	"github.com/yoren041102/rain/runner"
	"github.com/yoren041102/rain/schema"
)

func TestLogs(t *testing.T) {
	var buf bytes.Buffer
	l := slog.New(slog.NewTextHandler(&buf, nil))
	echo := fromfunc.New("echo", "", func(ctx context.Context, in struct {
		S string `json:"s"`
	}) (string, error) {
		return in.S, nil
	})
	m := &agenttest.FakeModel{Turns: []agenttest.Turn{
		{Message: agenttest.Call(agenttest.TC("c1", "echo", `{"s":"secret"}`))},
		{Message: schema.AssistantText("secret reply")},
	}}
	a, _ := agent.NewReAct(agent.ReActConfig{Name: "root", Model: m, Tools: []schema.Tool{echo}})
	mws := rainslog.Middlewares(l)
	r, _ := runner.New(runner.Config{Agent: a},
		runner.WithAgentMiddleware(mws.Agent...), runner.WithModelMiddleware(mws.Model...), runner.WithToolMiddleware(mws.Tool...))
	if _, err := schema.Collect(r.Run(context.Background(), "u", "s", schema.UserText("hi"))); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{"rain model call", "rain tool call", "tool=echo", "rain agent finished", "agent=root"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "secret") {
		t.Errorf("content logged without WithContent:\n%s", out)
	}
}
