// Package mcptool connects to an MCP server (via mark3labs/mcp-go) and
// exposes its tools as rain tools.
//
// The directory is named mcp, but the package is named mcptool so it can be
// imported next to github.com/mark3labs/mcp-go/mcp without an alias.
package mcptool

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	"github.com/yoren041102/rain/schema"
)

// Client is the subset of mcp-go's client.MCPClient used by this package.
// *client.Client and any client.MCPClient satisfy it. The client must
// already be started and initialized.
type Client interface {
	ListTools(ctx context.Context, request mcp.ListToolsRequest) (*mcp.ListToolsResult, error)
	CallTool(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error)
}

// Option configures a Toolset.
type Option func(*config)

type config struct {
	prefix   string
	filter   func(name string) bool
	cacheTTL time.Duration
	cache    bool
}

// WithPrefix prepends prefix to every tool name exposed to the model
// (e.g. "fs_" turns "read" into "fs_read"). Calls strip it again before
// reaching the server.
func WithPrefix(prefix string) Option { return func(c *config) { c.prefix = prefix } }

// WithFilter keeps only server tools for which keep returns true. keep
// receives the server-side name (without prefix).
func WithFilter(keep func(name string) bool) Option { return func(c *config) { c.filter = keep } }

// WithCache caches the tool list. ttl <= 0 caches forever (until
// Toolset.Refresh is called).
func WithCache(ttl time.Duration) Option {
	return func(c *config) { c.cache = true; c.cacheTTL = ttl }
}

func build(opts []Option) config {
	var c config
	for _, o := range opts {
		o(&c)
	}
	return c
}

// Toolset implements schema.Toolset over an MCP server.
type Toolset struct {
	c   Client
	cfg config

	mu       sync.Mutex
	cached   []*Tool
	cachedAt time.Time
}

var _ schema.Toolset = (*Toolset)(nil)

// NewToolset returns a Toolset backed by c. By default the server's tool
// list is fetched on every Tools call; use WithCache to avoid that.
func NewToolset(c Client, opts ...Option) *Toolset {
	return &Toolset{c: c, cfg: build(opts)}
}

// Refresh drops the cached tool list.
func (s *Toolset) Refresh() {
	s.mu.Lock()
	s.cached = nil
	s.mu.Unlock()
}

func (s *Toolset) list(ctx context.Context) ([]*Tool, error) {
	if s.cfg.cache {
		s.mu.Lock()
		if s.cached != nil && (s.cfg.cacheTTL <= 0 || time.Since(s.cachedAt) < s.cfg.cacheTTL) {
			out := s.cached
			s.mu.Unlock()
			return out, nil
		}
		s.mu.Unlock()
	}
	tools, err := listTools(ctx, s.c, s.cfg)
	if err != nil {
		return nil, err
	}
	if s.cfg.cache {
		s.mu.Lock()
		s.cached, s.cachedAt = tools, time.Now()
		s.mu.Unlock()
	}
	return tools, nil
}

// Tools implements schema.Toolset.
func (s *Toolset) Tools(ctx context.Context) ([]schema.ToolSpec, error) {
	tools, err := s.list(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]schema.ToolSpec, len(tools))
	for i, t := range tools {
		out[i] = t.spec
	}
	return out, nil
}

// ProcessRequest implements schema.Toolset; it does nothing.
func (s *Toolset) ProcessRequest(context.Context, *schema.ModelRequest) error { return nil }

// Call implements schema.Toolset. name is the rain-side (prefixed) name.
func (s *Toolset) Call(ctx context.Context, name string, args json.RawMessage) (*schema.ToolResult, error) {
	remote, ok := strings.CutPrefix(name, s.cfg.prefix)
	if !ok {
		return nil, fmt.Errorf("mcptool: tool %q not found", name)
	}
	if s.cfg.filter != nil && !s.cfg.filter(remote) {
		return nil, fmt.Errorf("mcptool: tool %q not found", name)
	}
	return callTool(ctx, s.c, remote, args)
}

// Tools lists the server's tools as schema.Tool values. Each returned tool
// is a *Tool and implements schema.ReadOnly / schema.ReplaySafe according
// to the server's readOnlyHint / idempotentHint annotations.
func Tools(ctx context.Context, c Client, opts ...Option) ([]schema.Tool, error) {
	tools, err := listTools(ctx, c, build(opts))
	if err != nil {
		return nil, err
	}
	out := make([]schema.Tool, len(tools))
	for i, t := range tools {
		out[i] = t
	}
	return out, nil
}

func listTools(ctx context.Context, c Client, cfg config) ([]*Tool, error) {
	res, err := c.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		return nil, fmt.Errorf("mcptool: list tools: %w", err)
	}
	if res == nil {
		return nil, nil
	}
	out := make([]*Tool, 0, len(res.Tools))
	for _, t := range res.Tools {
		if cfg.filter != nil && !cfg.filter(t.Name) {
			continue
		}
		nt, err := newTool(c, t, cfg.prefix)
		if err != nil {
			return nil, err
		}
		out = append(out, nt)
	}
	return out, nil
}

// Tool is one MCP server tool exposed as a schema.Tool.
type Tool struct {
	c          Client
	remoteName string
	spec       schema.ToolSpec
	readOnly   bool
	idempotent bool
	// MCP is the tool definition as reported by the server.
	MCP mcp.Tool
}

var (
	_ schema.Tool       = (*Tool)(nil)
	_ schema.ReadOnly   = (*Tool)(nil)
	_ schema.ReplaySafe = (*Tool)(nil)
)

func newTool(c Client, t mcp.Tool, prefix string) (*Tool, error) {
	params, err := inputSchema(t)
	if err != nil {
		return nil, fmt.Errorf("mcptool: tool %q: encode input schema: %w", t.Name, err)
	}
	return &Tool{
		c:          c,
		remoteName: t.Name,
		spec:       schema.ToolSpec{Name: prefix + t.Name, Description: t.Description, Parameters: params},
		readOnly:   t.Annotations.ReadOnlyHint != nil && *t.Annotations.ReadOnlyHint,
		idempotent: t.Annotations.IdempotentHint != nil && *t.Annotations.IdempotentHint,
		MCP:        t,
	}, nil
}

func inputSchema(t mcp.Tool) (json.RawMessage, error) {
	if len(t.RawInputSchema) > 0 {
		return append(json.RawMessage(nil), t.RawInputSchema...), nil
	}
	return json.Marshal(t.InputSchema)
}

// Spec implements schema.Tool.
func (t *Tool) Spec() schema.ToolSpec { return t.spec }

// IsReadOnly implements schema.ReadOnly (MCP readOnlyHint).
func (t *Tool) IsReadOnly() bool { return t.readOnly }

// IsReplaySafe implements schema.ReplaySafe (MCP idempotentHint).
func (t *Tool) IsReplaySafe() bool { return t.idempotent }

// Call implements schema.Tool.
func (t *Tool) Call(ctx context.Context, args json.RawMessage) (*schema.ToolResult, error) {
	return callTool(ctx, t.c, t.remoteName, args)
}

func callTool(ctx context.Context, c Client, name string, args json.RawMessage) (*schema.ToolResult, error) {
	req := mcp.CallToolRequest{}
	req.Params.Name = name
	if a := strings.TrimSpace(string(args)); a != "" && a != "null" {
		if !json.Valid(args) {
			return nil, fmt.Errorf("mcptool: tool %q: arguments are not valid JSON", name)
		}
		// RawArguments is re-emitted verbatim (no float64 precision loss);
		// Arguments is set for clients that inspect it.
		req.Params.RawArguments = append(json.RawMessage(nil), args...)
		var m map[string]any
		if json.Unmarshal(args, &m) == nil {
			req.Params.Arguments = m
		}
	}
	res, err := c.CallTool(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("mcptool: call %q: %w", name, err)
	}
	return ConvertResult(res), nil
}

// ConvertResult maps an MCP tool result to a rain ToolResult:
// text → TextBlock, image → ImageBlock, embedded text resource → TextBlock,
// anything else → UnknownBlock (raw MCP JSON). Structured content, if any, is
// appended as a JSON TextBlock. IsError is preserved.
func ConvertResult(res *mcp.CallToolResult) *schema.ToolResult {
	if res == nil {
		return &schema.ToolResult{Content: []schema.ContentBlock{&schema.TextBlock{Text: "No result provided"}}, IsError: true}
	}
	out := &schema.ToolResult{IsError: res.IsError}
	for _, c := range res.Content {
		out.Content = append(out.Content, convertContent(c))
	}
	if sc := structured(res); sc != nil {
		out.Content = append(out.Content, &schema.TextBlock{Text: string(sc)})
	}
	return out
}

func structured(res *mcp.CallToolResult) json.RawMessage {
	if len(res.RawStructuredContent) > 0 {
		return res.RawStructuredContent
	}
	if res.StructuredContent == nil {
		return nil
	}
	b, err := json.Marshal(res.StructuredContent)
	if err != nil {
		return nil
	}
	return b
}

func convertContent(c mcp.Content) schema.ContentBlock {
	switch v := c.(type) {
	case mcp.TextContent:
		return &schema.TextBlock{Text: v.Text}
	case *mcp.TextContent:
		return &schema.TextBlock{Text: v.Text}
	case mcp.ImageContent:
		if b := image(v); b != nil {
			return b
		}
	case *mcp.ImageContent:
		if b := image(*v); b != nil {
			return b
		}
	case mcp.EmbeddedResource:
		if b := resource(v); b != nil {
			return b
		}
	case *mcp.EmbeddedResource:
		if b := resource(*v); b != nil {
			return b
		}
	}
	return unknown(c)
}

func image(v mcp.ImageContent) schema.ContentBlock {
	data, err := base64.StdEncoding.DecodeString(v.Data)
	if err != nil {
		return nil
	}
	return &schema.ImageBlock{MIMEType: v.MIMEType, Data: data}
}

func resource(v mcp.EmbeddedResource) schema.ContentBlock {
	switch r := v.Resource.(type) {
	case mcp.TextResourceContents:
		return &schema.TextBlock{Text: r.Text}
	case *mcp.TextResourceContents:
		return &schema.TextBlock{Text: r.Text}
	}
	return nil
}

func unknown(c mcp.Content) schema.ContentBlock {
	raw, _ := json.Marshal(c)
	var head struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(raw, &head)
	typ := head.Type
	if typ == "" {
		typ = fmt.Sprintf("%T", c)
	}
	return &schema.UnknownBlock{Type: "mcp." + typ, Raw: raw}
}
