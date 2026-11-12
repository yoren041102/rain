package mcptool

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
	"github.com/yoren041102/rain/schema"
)

func newClient(t *testing.T) *client.Client {
	t.Helper()
	s := server.NewMCPServer("test-server", "1.0.0", server.WithToolCapabilities(true))
	s.AddTool(mcp.NewTool("echo",
		mcp.WithDescription("Echoes the input"),
		mcp.WithString("text", mcp.Required(), mcp.Description("text to echo")),
		mcp.WithReadOnlyHintAnnotation(true),
	), func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		text, err := req.RequireString("text")
		if err != nil {
			return mcp.NewToolResultError(err.Error()), nil
		}
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				mcp.NewTextContent("echo: " + text),
				mcp.NewImageContent(base64.StdEncoding.EncodeToString([]byte("PNG")), "image/png"),
				mcp.NewEmbeddedResource(mcp.TextResourceContents{URI: "mem://x", MIMEType: "text/plain", Text: "res"}),
				mcp.AudioContent{Type: "audio", Data: "AAAA", MIMEType: "audio/wav"},
			},
			StructuredContent: map[string]any{"echo": text},
		}, nil
	})
	s.AddTool(mcp.NewTool("fail", mcp.WithDescription("Always fails")),
		func(ctx context.Context, req mcp.CallToolRequest) (*mcp.CallToolResult, error) {
			return mcp.NewToolResultError("boom"), nil
		})

	c, err := client.NewInProcessClient(s)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	ctx := context.Background()
	if err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}
	init := mcp.InitializeRequest{}
	init.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	init.Params.ClientInfo = mcp.Implementation{Name: "rain-test", Version: "0"}
	if _, err := c.Initialize(ctx, init); err != nil {
		t.Fatal(err)
	}
	return c
}

func find(ts []schema.Tool, name string) schema.Tool {
	for _, t := range ts {
		if t.Spec().Name == name {
			return t
		}
	}
	return nil
}

func TestTools(t *testing.T) {
	c := newClient(t)
	ctx := context.Background()
	tools, err := Tools(ctx, c)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 2 {
		t.Fatalf("got %d tools", len(tools))
	}
	echo := find(tools, "echo")
	if echo == nil {
		t.Fatal("echo not found")
	}
	spec := echo.Spec()
	if spec.Description != "Echoes the input" {
		t.Errorf("description = %q", spec.Description)
	}
	var params struct {
		Type       string                     `json:"type"`
		Properties map[string]json.RawMessage `json:"properties"`
		Required   []string                   `json:"required"`
	}
	if err := json.Unmarshal(spec.Parameters, &params); err != nil {
		t.Fatalf("parameters %s: %v", spec.Parameters, err)
	}
	if params.Type != "object" || params.Properties["text"] == nil || len(params.Required) != 1 || params.Required[0] != "text" {
		t.Errorf("parameters = %s", spec.Parameters)
	}
	if !schema.IsReadOnly(echo) {
		t.Error("echo should be read-only")
	}
	if schema.IsReadOnly(find(tools, "fail")) {
		t.Error("fail should not be read-only")
	}

	res, err := echo.Call(ctx, json.RawMessage(`{"text":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.IsError {
		t.Fatal("unexpected IsError")
	}
	if len(res.Content) != 5 {
		t.Fatalf("got %d blocks: %#v", len(res.Content), res.Content)
	}
	if tb, ok := res.Content[0].(*schema.TextBlock); !ok || tb.Text != "echo: hi" {
		t.Errorf("block 0 = %#v", res.Content[0])
	}
	if ib, ok := res.Content[1].(*schema.ImageBlock); !ok || string(ib.Data) != "PNG" || ib.MIMEType != "image/png" {
		t.Errorf("block 1 = %#v", res.Content[1])
	}
	if tb, ok := res.Content[2].(*schema.TextBlock); !ok || tb.Text != "res" {
		t.Errorf("block 2 = %#v", res.Content[2])
	}
	if ub, ok := res.Content[3].(*schema.UnknownBlock); !ok || ub.Type != "mcp.audio" || !strings.Contains(string(ub.Raw), "AAAA") {
		t.Errorf("block 3 = %#v", res.Content[3])
	}
	if tb, ok := res.Content[4].(*schema.TextBlock); !ok || strings.ReplaceAll(tb.Text, " ", "") != `{"echo":"hi"}` {
		t.Errorf("block 4 = %#v", res.Content[4])
	}

	res, err = find(tools, "fail").Call(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !res.IsError {
		t.Error("expected IsError")
	}
	if tb, ok := res.Content[0].(*schema.TextBlock); !ok || tb.Text != "boom" {
		t.Errorf("error content = %#v", res.Content)
	}
}

func TestToolsetPrefixFilter(t *testing.T) {
	c := newClient(t)
	ctx := context.Background()
	ts := NewToolset(c, WithPrefix("srv_"), WithFilter(func(n string) bool { return n == "echo" }), WithCache(time.Minute))
	var _ schema.Toolset = ts

	specs, err := ts.Tools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 1 || specs[0].Name != "srv_echo" {
		t.Fatalf("specs = %#v", specs)
	}
	if err := ts.ProcessRequest(ctx, &schema.ModelRequest{}); err != nil {
		t.Fatal(err)
	}
	res, err := ts.Call(ctx, "srv_echo", json.RawMessage(`{"text":"x"}`))
	if err != nil {
		t.Fatal(err)
	}
	if res.Content[0].(*schema.TextBlock).Text != "echo: x" {
		t.Errorf("result = %#v", res.Content[0])
	}
	if _, err := ts.Call(ctx, "srv_fail", nil); err == nil {
		t.Error("filtered tool should not be callable")
	}
	if _, err := ts.Call(ctx, "echo", nil); err == nil {
		t.Error("unprefixed name should not be callable")
	}
	// Works as a schema.Tool via ToolsetTool, as ReActAgent does.
	tt := &schema.ToolsetTool{Set: ts, ToolSpec: specs[0]}
	if _, err := tt.Call(ctx, json.RawMessage(`{"text":"y"}`)); err != nil {
		t.Fatal(err)
	}
}

func TestConvertResultNil(t *testing.T) {
	if r := ConvertResult(nil); !r.IsError {
		t.Error("nil result should be an error")
	}
}
