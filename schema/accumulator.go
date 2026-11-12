package schema

import "encoding/json"

// Accumulator merges streaming deltas into a complete message.
//
// Delta semantics per block type, keyed by ModelResponse.BlockIndex:
//   - TextBlock / ReasoningBlock: Text is appended; a non-empty Signature
//     replaces the previous one.
//   - ToolCallBlock: ID and Name are set when non-empty; Arguments bytes are
//     appended.
//   - any other block: replaces the block at that index.
//
// Each partial ModelResponse is expected to carry the delta block(s) in
// Message.Content; the first block is applied at BlockIndex, subsequent ones
// at BlockIndex+1, ...
type Accumulator struct {
	msg    Message
	blocks map[int]ContentBlock
	order  []int
	usage  *Usage
}

// NewAccumulator returns an empty accumulator.
func NewAccumulator() *Accumulator {
	return &Accumulator{msg: Message{Role: RoleAssistant}, blocks: map[int]ContentBlock{}}
}

// Add applies one chunk. Non-partial responses replace the whole state.
func (a *Accumulator) Add(r *ModelResponse) {
	if r == nil {
		return
	}
	if r.Usage != nil {
		u := *r.Usage
		a.usage = &u // providers report cumulative usage
	}
	if r.FinishReason != "" {
		a.msg.StopReason = r.FinishReason
	}
	if r.Message == nil {
		return
	}
	if r.Message.Role != "" {
		a.msg.Role = r.Message.Role
	}
	if r.Message.Origin != nil {
		o := *r.Message.Origin
		a.msg.Origin = &o
	}
	if r.Message.StopReason != "" {
		a.msg.StopReason = r.Message.StopReason
	}
	if !r.Partial {
		a.blocks = map[int]ContentBlock{}
		a.order = nil
		for i, b := range r.Message.Content {
			a.set(i, cloneBlock(b))
		}
		return
	}
	for i, b := range r.Message.Content {
		a.apply(r.BlockIndex+i, b)
	}
}

func (a *Accumulator) set(i int, b ContentBlock) {
	if _, ok := a.blocks[i]; !ok {
		a.order = append(a.order, i)
	}
	a.blocks[i] = b
}

func (a *Accumulator) apply(i int, delta ContentBlock) {
	cur, ok := a.blocks[i]
	if !ok || cur.BlockType() != delta.BlockType() {
		a.set(i, cloneBlock(delta))
		return
	}
	switch d := delta.(type) {
	case *TextBlock:
		c := cur.(*TextBlock)
		c.Text += d.Text
		mergeMeta(&c.Meta, d.Meta)
	case *ReasoningBlock:
		c := cur.(*ReasoningBlock)
		c.Text += d.Text
		if d.Signature != "" {
			c.Signature = d.Signature
		}
		mergeMeta(&c.Meta, d.Meta)
	case *ToolCallBlock:
		c := cur.(*ToolCallBlock)
		if d.ID != "" {
			c.ID = d.ID
		}
		if d.Name != "" {
			c.Name = d.Name
		}
		c.Arguments = append(c.Arguments, d.Arguments...)
		mergeMeta(&c.Meta, d.Meta)
	default:
		a.set(i, cloneBlock(delta))
	}
}

func mergeMeta(dst *BlockMeta, src BlockMeta) {
	if len(src.Extra) == 0 {
		return
	}
	if dst.Extra == nil {
		dst.Extra = map[string]json.RawMessage{}
	}
	for k, v := range src.Extra {
		dst.Extra[k] = v
	}
}

// Message returns the accumulated message. Content is ordered by block index.
func (a *Accumulator) Message() *Message {
	m := a.msg
	idx := append([]int(nil), a.order...)
	// insertion sort; block counts are small
	for i := 1; i < len(idx); i++ {
		for j := i; j > 0 && idx[j] < idx[j-1]; j-- {
			idx[j], idx[j-1] = idx[j-1], idx[j]
		}
	}
	m.Content = make([]ContentBlock, 0, len(idx))
	for _, i := range idx {
		b := cloneBlock(a.blocks[i])
		if tc, ok := b.(*ToolCallBlock); ok && len(tc.Arguments) == 0 {
			tc.Arguments = json.RawMessage("{}")
		}
		m.Content = append(m.Content, b)
	}
	if m.StopReason == "" {
		if len(m.ToolCalls()) > 0 {
			m.StopReason = StopReasonToolUse
		} else {
			m.StopReason = StopReasonStop
		}
	}
	return &m
}

// Usage returns the latest usage reported.
func (a *Accumulator) Usage() *Usage {
	if a.usage == nil {
		return nil
	}
	u := *a.usage
	return &u
}
