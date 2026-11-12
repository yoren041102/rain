# rain

- Core module has **zero dependencies** (`go list -m all` prints one line).
- Everything is an event stream: `iter.Seq2[T, error]`, consumed with `for range`.
- Session state is derived from the event log; interrupts and crash recovery need no extra checkpoint store.

## Layout

| Package | Layer | Contents |
|---|---|---|
| `schema` | types | `Message`, sealed `ContentBlock`s with JSON codec, `Event`, `Interrupt`, `Stream`, `Accumulator`, `Model`, `Tool`, `Toolset`, `PrepareForModel`, `ValidateJSON` |
| `agent` | agent | `Agent`, `Invocation`, Agent/Model/Tool middlewares, `Tree` + transfer, `ReActAgent`, `Session`/`SessionStore`/`Memory` + in-memory impls |
| `agent/agenttest` | testing | `TestSessionStore` conformance suite, `FakeModel` |
| `runner` | execution | `Runner`: persistence, dangling-call repair, `Resume`, `Steer`, safe-point `Stop`, plugins |
| `ext/tool/fromfunc` | ext | typed function → tool, JSON Schema by reflection |
| `ext/tool/agenttool` | ext | agent → tool (isolated branch, interrupts propagate) |
| `ext/compose` | ext | `Sequential`, `Parallel`, `Loop` |
| `ext/history` | ext | `KeepLast`, `TokenBudget`, `Summarize` (persisted compaction) |
| `ext/middleware/retry`, `ext/middleware/slog` | ext | retry with backoff, slog logging |
| `ext/model/anthropicmodel` | separate module | Anthropic Messages API |
| `ext/model/openaimodel` | separate module | OpenAI Chat Completions / Responses |
| `ext/tool/mcp` (package `mcptool`) | separate module | MCP tools via mcp-go |
| `ext/middleware/otel` | separate module | OpenTelemetry traces + metrics |

## Quick start

```go
weather := fromfunc.New("get_weather", "查询城市天气",
    func(ctx context.Context, in struct{ City string `json:"city"` }) (string, error) {
        return "晴，22°C", nil
    })

a, _ := agent.NewReAct(agent.ReActConfig{
    Name: "assistant", Instruction: "你是一个助手。",
    Model: anthropicmodel.New(anthropic.NewClient(), "claude-sonnet-5-5"),
    Tools: []schema.Tool{weather},
})
r, _ := runner.New(runner.Config{AppName: "demo", Agent: a},
    runner.WithModelMiddleware(history.KeepLast(50)))

for ev, err := range r.Run(ctx, "u1", "s1", schema.UserText("北京天气怎么样？")) {
    if err != nil { return err }
    if ev.Actions.Interrupt != nil { /* show Payload, then r.Resume(...) */ }
    if !ev.Partial && ev.Message != nil { fmt.Println(ev.Author, ev.Message.Text()) }
}
```

Runnable without an API key: `go run ./examples/weather`.

### Human-in-the-loop

```go
func(ctx context.Context, in Args) (string, error) {
    tc, _ := agent.ToolContextFrom(ctx)
    var ok bool
    if resumed, _ := tc.ResumeInput(&ok); !resumed {
        return "", agent.Interrupt(ctx, "confirm", in, nil) // pauses the run
    }
    ...
}
// later:
r.Resume(ctx, user, sid, schema.ResumeData{InterruptID: it.ID, Input: json.RawMessage(`true`)})
```

Interrupt IDs are hierarchical (`root/sub/tool:call_id`); several can be resumed at once or one at a time,
and unanswered ones re-trigger. `ResumeData.ToolResult` supplies a result without re-running the tool.

## Development

The repo uses a `go.work` that includes every module (and maps the unpublished core version to `./`):

```bash
go test -race ./... ./ext/model/anthropicmodel/... ./ext/model/openaimodel/... ./ext/tool/mcp/... ./ext/middleware/otel/...
```

Extension modules declare the Go version their SDKs need (mcp-go: 1.25.5, otel: 1.26.0); the core module stays at 1.24.
