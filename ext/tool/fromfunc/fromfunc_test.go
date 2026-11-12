package fromfunc

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/yoren041102/rain/schema"
)

type inner struct {
	X float64 `json:"x"`
}

type args struct {
	City   string            `json:"city" description:"city name"`
	Days   int               `json:"days,omitempty"`
	Unit   string            `json:"unit" jsonschema:"enum=c|f,optional"`
	Tags   []string          `json:"tags,omitempty"`
	Nested *inner            `json:"nested"`
	Meta   map[string]string `json:"meta,omitempty"`
	Skip   string            `json:"-"`
	hidden string
}

func TestSchema(t *testing.T) {
	s := SchemaFor(reflect.TypeFor[args]())
	b, _ := json.Marshal(s)
	want := `{"properties":{"city":{"description":"city name","type":"string"},"days":{"type":"integer"},"meta":{"additionalProperties":{"type":"string"},"type":"object"},"nested":{"properties":{"x":{"type":"number"}},"required":["x"],"type":"object"},"tags":{"items":{"type":"string"},"type":"array"},"unit":{"enum":["c","f"],"type":"string"}},"required":["city"],"type":"object"}`
	if string(b) != want {
		t.Fatalf("schema =\n%s\nwant\n%s", b, want)
	}
}

func TestCall(t *testing.T) {
	tool := New("t", "d", func(ctx context.Context, in args) (map[string]any, error) {
		return map[string]any{"city": in.City, "days": in.Days}, nil
	}, ReadOnly(), ConcurrencySafe())
	if !schema.IsReadOnly(tool) || !schema.IsConcurrencySafe(tool) || schema.IsReplaySafe(tool) {
		t.Fatal("markers")
	}
	res, err := tool.Call(context.Background(), json.RawMessage(`{"city":"x","days":3}`))
	if err != nil {
		t.Fatal(err)
	}
	if got := (&schema.Message{Content: res.Content}).Text(); got != `{"city":"x","days":3}` {
		t.Fatalf("got %s", got)
	}
	res, _ = tool.Call(context.Background(), json.RawMessage(`{"days":"bad"}`))
	if !res.IsError {
		t.Fatal("expected error result for bad args")
	}
	if err := schema.ValidateJSON(tool.Spec().Parameters, json.RawMessage(`{"city":"a","unit":"k"}`)); err == nil {
		t.Fatal("enum not enforced")
	}
}
