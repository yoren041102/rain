package schema

// NoResultText is the content of synthesized results for tool calls that
// never received one.
const NoResultText = "No result provided"

// PrepareForModel adapts a history for replay against target:
//   - ReasoningBlock signatures are kept only when the message origin equals
//     target exactly.
//   - Assistant messages that ended with StopReasonError or StopReasonAborted
//     are dropped.
//   - Consecutive tool messages are merged into one.
//   - Tool calls without a result get an IsError result "No result provided".
//   - Tool results whose call is not in the history are dropped.
//
// The input slice and messages are not modified.
func PrepareForModel(msgs []*Message, target Origin) []*Message {
	out := make([]*Message, 0, len(msgs))
	var pending []*ToolCallBlock // calls of the latest assistant message
	var toolMsg *Message         // tool message being merged
	flush := func() {
		if len(pending) == 0 && toolMsg == nil {
			return
		}
		if toolMsg == nil {
			toolMsg = &Message{Role: RoleTool}
		}
		answered := map[string]bool{}
		for _, r := range toolMsg.ToolResultBlocks() {
			answered[r.CallID] = true
		}
		// order results by call order, synthesizing missing ones
		byID := map[string]*ToolResultBlock{}
		for _, r := range toolMsg.ToolResultBlocks() {
			byID[r.CallID] = r
		}
		ordered := &Message{Role: RoleTool}
		for _, c := range pending {
			if r, ok := byID[c.ID]; ok {
				ordered.Content = append(ordered.Content, r)
			} else {
				ordered.Content = append(ordered.Content, &ToolResultBlock{CallID: c.ID, Content: []ContentBlock{&TextBlock{Text: NoResultText}}, IsError: true})
			}
		}
		if len(ordered.Content) > 0 {
			out = append(out, ordered)
		}
		pending, toolMsg = nil, nil
	}
	for _, m := range msgs {
		if m == nil {
			continue
		}
		if m.Role == RoleAssistant && (m.StopReason == StopReasonError || m.StopReason == StopReasonAborted) {
			continue
		}
		if m.Role == RoleTool {
			if toolMsg == nil {
				toolMsg = &Message{Role: RoleTool}
			}
			for _, b := range m.Content {
				toolMsg.Content = append(toolMsg.Content, cloneBlock(b))
			}
			continue
		}
		flush()
		c := m.Clone()
		if m.Role == RoleAssistant {
			keepSig := m.Origin.Same(target)
			for _, b := range c.Content {
				if r, ok := b.(*ReasoningBlock); ok && !keepSig {
					r.Signature = ""
				}
			}
			pending = c.ToolCalls()
		}
		out = append(out, c)
	}
	flush()
	return out
}
