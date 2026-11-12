// Package fromfunc builds schema.Tool values from typed Go functions. The
// parameter JSON Schema is derived from the input type by reflection.
package fromfunc

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/yoren041102/rain/schema"
)

// Option configures a function tool.
type Option func(*options)

type options struct {
	readOnly, concurrencySafe, replaySafe bool
	params                                json.RawMessage
}

// ReadOnly marks the tool as not modifying external state.
func ReadOnly() Option { return func(o *options) { o.readOnly = true } }

// ConcurrencySafe marks the tool as safe to run in parallel.
func ConcurrencySafe() Option { return func(o *options) { o.concurrencySafe = true } }

// ReplaySafe marks the tool as safe to re-run after a crash.
func ReplaySafe() Option { return func(o *options) { o.replaySafe = true } }

// WithSchema overrides the derived parameter schema.
func WithSchema(s json.RawMessage) Option { return func(o *options) { o.params = s } }

// Tool is a function-backed tool.
type Tool[In, Out any] struct {
	spec schema.ToolSpec
	fn   func(context.Context, In) (Out, error)
	opts options
}

// New builds a tool from fn. The result is converted as follows:
//   - *schema.ToolResult is returned as-is;
//   - string becomes a text block;
//   - []schema.ContentBlock becomes the content;
//   - anything else is JSON-encoded into a text block.
func New[In, Out any](name, description string, fn func(ctx context.Context, in In) (Out, error), opts ...Option) *Tool[In, Out] {
	var o options
	for _, f := range opts {
		f(&o)
	}
	params := o.params
	if params == nil {
		s := SchemaFor(reflect.TypeFor[In]())
		if s["type"] != "object" {
			s = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		params, _ = json.Marshal(s)
	}
	return &Tool[In, Out]{spec: schema.ToolSpec{Name: name, Description: description, Parameters: params}, fn: fn, opts: o}
}

func (t *Tool[In, Out]) Spec() schema.ToolSpec   { return t.spec }
func (t *Tool[In, Out]) IsReadOnly() bool        { return t.opts.readOnly }
func (t *Tool[In, Out]) IsConcurrencySafe() bool { return t.opts.concurrencySafe }
func (t *Tool[In, Out]) IsReplaySafe() bool      { return t.opts.replaySafe }

func (t *Tool[In, Out]) Call(ctx context.Context, args json.RawMessage) (*schema.ToolResult, error) {
	var in In
	if len(args) > 0 && string(args) != "null" {
		if err := json.Unmarshal(args, &in); err != nil {
			return schema.ErrorResult(fmt.Sprintf("invalid arguments: %v", err)), nil
		}
	}
	out, err := t.fn(ctx, in)
	if err != nil {
		return nil, err
	}
	return toResult(out)
}

func toResult(v any) (*schema.ToolResult, error) {
	switch x := v.(type) {
	case *schema.ToolResult:
		return x, nil
	case schema.ToolResult:
		return &x, nil
	case string:
		return schema.TextResult(x), nil
	case []schema.ContentBlock:
		return &schema.ToolResult{Content: x}, nil
	case nil:
		return schema.TextResult(""), nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("fromfunc: encode result: %w", err)
	}
	return schema.TextResult(string(b)), nil
}

// SchemaFor derives a JSON Schema for t. Struct fields honour `json` tags
// (name, omitempty, "-"); fields without omitempty are required. The tags
// `description:"..."` / `jsonschema:"description=...,enum=a|b,required"`
// add descriptions and enums.
func SchemaFor(t reflect.Type) map[string]any {
	return schemaFor(t, map[reflect.Type]bool{})
}

var rawMessageType = reflect.TypeFor[json.RawMessage]()

func schemaFor(t reflect.Type, seen map[reflect.Type]bool) map[string]any {
	if t == nil {
		return map[string]any{}
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t == rawMessageType {
		return map[string]any{}
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}
	case reflect.Bool:
		return map[string]any{"type": "boolean"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}
	case reflect.Slice, reflect.Array:
		if t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string", "contentEncoding": "base64"}
		}
		return map[string]any{"type": "array", "items": schemaFor(t.Elem(), seen)}
	case reflect.Map:
		return map[string]any{"type": "object", "additionalProperties": schemaFor(t.Elem(), seen)}
	case reflect.Interface:
		return map[string]any{}
	case reflect.Struct:
		if seen[t] {
			return map[string]any{"type": "object"}
		}
		seen[t] = true
		defer delete(seen, t)
		props := map[string]any{}
		var required []string
		addFields(t, props, &required, seen)
		s := map[string]any{"type": "object", "properties": props}
		if len(required) > 0 {
			s["required"] = required
		}
		return s
	}
	return map[string]any{}
}

func addFields(t reflect.Type, props map[string]any, required *[]string, seen map[reflect.Type]bool) {
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, rest, _ := strings.Cut(tag, ",")
		if f.Anonymous && name == "" {
			ft := f.Type
			if ft.Kind() == reflect.Pointer {
				ft = ft.Elem()
			}
			if ft.Kind() == reflect.Struct {
				addFields(ft, props, required, seen)
				continue
			}
		}
		if !f.IsExported() {
			continue
		}
		if name == "" {
			name = f.Name
		}
		s := schemaFor(f.Type, seen)
		if d := f.Tag.Get("description"); d != "" {
			s["description"] = d
		}
		req := !strings.Contains(rest, "omitempty") && !strings.Contains(rest, "omitzero") && f.Type.Kind() != reflect.Pointer
		for _, part := range strings.Split(f.Tag.Get("jsonschema"), ",") {
			k, v, _ := strings.Cut(part, "=")
			switch strings.TrimSpace(k) {
			case "description":
				s["description"] = v
			case "enum":
				var enum []any
				for _, e := range strings.Split(v, "|") {
					enum = append(enum, e)
				}
				s["enum"] = enum
			case "required":
				req = true
			case "optional":
				req = false
			}
		}
		props[name] = s
		if req {
			*required = append(*required, name)
		}
	}
}
