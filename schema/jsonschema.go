package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"unicode/utf8"
)

// ValidateJSON checks value against a JSON Schema. It supports the subset of
// JSON Schema that tool parameter schemas use in practice: type (single or
// list), properties, required, additionalProperties (bool or schema), items,
// enum, const, minimum/maximum, exclusiveMinimum/exclusiveMaximum,
// minLength/maxLength, minItems/maxItems, anyOf, oneOf, allOf. Unknown
// keywords are ignored. An empty schema accepts everything.
func ValidateJSON(schemaJSON, value json.RawMessage) error {
	if len(bytes.TrimSpace(schemaJSON)) == 0 {
		return nil
	}
	var s map[string]any
	if err := json.Unmarshal(schemaJSON, &s); err != nil {
		return fmt.Errorf("invalid schema: %w", err)
	}
	if len(bytes.TrimSpace(value)) == 0 {
		value = json.RawMessage("{}")
	}
	dec := json.NewDecoder(bytes.NewReader(value))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return validate(s, v, "$")
}

func validate(s map[string]any, v any, path string) error {
	if t, ok := s["type"]; ok {
		var types []string
		switch tt := t.(type) {
		case string:
			types = []string{tt}
		case []any:
			for _, x := range tt {
				if str, ok := x.(string); ok {
					types = append(types, str)
				}
			}
		}
		if len(types) > 0 {
			matched := false
			for _, typ := range types {
				if typeMatches(typ, v) {
					matched = true
					break
				}
			}
			if !matched {
				return fmt.Errorf("%s: expected %s, got %s", path, strings.Join(types, " or "), jsonTypeName(v))
			}
		}
	}
	if e, ok := s["enum"].([]any); ok {
		found := false
		for _, x := range e {
			if jsonEqual(x, v) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("%s: value not in enum %v", path, e)
		}
	}
	if c, ok := s["const"]; ok && !jsonEqual(c, v) {
		return fmt.Errorf("%s: value must equal %v", path, c)
	}
	for _, key := range []string{"allOf", "anyOf", "oneOf"} {
		subs, ok := s[key].([]any)
		if !ok {
			continue
		}
		n := 0
		var firstErr error
		for _, sub := range subs {
			sm, ok := sub.(map[string]any)
			if !ok {
				continue
			}
			if err := validate(sm, v, path); err != nil {
				if firstErr == nil {
					firstErr = err
				}
				if key == "allOf" {
					return err
				}
			} else {
				n++
			}
		}
		if key == "anyOf" && n == 0 && firstErr != nil {
			return fmt.Errorf("%s: matches none of anyOf: %w", path, firstErr)
		}
		if key == "oneOf" && n != 1 {
			return fmt.Errorf("%s: must match exactly one of oneOf, matched %d", path, n)
		}
	}
	switch val := v.(type) {
	case map[string]any:
		props, _ := s["properties"].(map[string]any)
		if req, ok := s["required"].([]any); ok {
			for _, r := range req {
				name, _ := r.(string)
				if _, ok := val[name]; !ok {
					return fmt.Errorf("%s: missing required property %q", path, name)
				}
			}
		}
		for k, pv := range val {
			if ps, ok := props[k].(map[string]any); ok {
				if err := validate(ps, pv, path+"."+k); err != nil {
					return err
				}
				continue
			}
			switch ap := s["additionalProperties"].(type) {
			case bool:
				if !ap {
					return fmt.Errorf("%s: unexpected property %q", path, k)
				}
			case map[string]any:
				if err := validate(ap, pv, path+"."+k); err != nil {
					return err
				}
			}
		}
	case []any:
		if n, ok := num(s["minItems"]); ok && float64(len(val)) < n {
			return fmt.Errorf("%s: expected at least %v items", path, n)
		}
		if n, ok := num(s["maxItems"]); ok && float64(len(val)) > n {
			return fmt.Errorf("%s: expected at most %v items", path, n)
		}
		if is, ok := s["items"].(map[string]any); ok {
			for i, item := range val {
				if err := validate(is, item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
					return err
				}
			}
		}
	case string:
		l := float64(utf8.RuneCountInString(val))
		if n, ok := num(s["minLength"]); ok && l < n {
			return fmt.Errorf("%s: shorter than %v", path, n)
		}
		if n, ok := num(s["maxLength"]); ok && l > n {
			return fmt.Errorf("%s: longer than %v", path, n)
		}
	case json.Number:
		f, _ := val.Float64()
		if n, ok := num(s["minimum"]); ok && f < n {
			return fmt.Errorf("%s: must be >= %v", path, n)
		}
		if n, ok := num(s["maximum"]); ok && f > n {
			return fmt.Errorf("%s: must be <= %v", path, n)
		}
		if n, ok := num(s["exclusiveMinimum"]); ok && f <= n {
			return fmt.Errorf("%s: must be > %v", path, n)
		}
		if n, ok := num(s["exclusiveMaximum"]); ok && f >= n {
			return fmt.Errorf("%s: must be < %v", path, n)
		}
	}
	return nil
}

func num(x any) (float64, bool) {
	switch n := x.(type) {
	case float64:
		return n, true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	}
	return 0, false
}

func typeMatches(typ string, v any) bool {
	switch typ {
	case "object":
		_, ok := v.(map[string]any)
		return ok
	case "array":
		_, ok := v.([]any)
		return ok
	case "string":
		_, ok := v.(string)
		return ok
	case "boolean":
		_, ok := v.(bool)
		return ok
	case "null":
		return v == nil
	case "number":
		_, ok := v.(json.Number)
		return ok
	case "integer":
		n, ok := v.(json.Number)
		if !ok {
			return false
		}
		f, err := n.Float64()
		return err == nil && f == math.Trunc(f)
	}
	return true
}

func jsonTypeName(v any) string {
	switch v.(type) {
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case bool:
		return "boolean"
	case nil:
		return "null"
	case json.Number:
		return "number"
	}
	return fmt.Sprintf("%T", v)
}

func jsonEqual(a, b any) bool {
	ab, _ := json.Marshal(normalizeNum(a))
	bb, _ := json.Marshal(normalizeNum(b))
	return bytes.Equal(ab, bb)
}

func normalizeNum(v any) any {
	switch n := v.(type) {
	case json.Number:
		f, _ := n.Float64()
		return f
	case map[string]any:
		m := make(map[string]any, len(n))
		for k, x := range n {
			m[k] = normalizeNum(x)
		}
		return m
	case []any:
		out := make([]any, len(n))
		for i, x := range n {
			out[i] = normalizeNum(x)
		}
		return out
	}
	return v
}
