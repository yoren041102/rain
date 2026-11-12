package agent

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/yoren041102/rain/schema"
)

// MessagesFor converts events into the message history seen by the agent
// named self on branch:
//   - partial events, events without messages and events not visible on
//     branch are skipped;
//   - the latest compaction event replaces everything before its
//     FirstKeptSeq with a summary message;
//   - user and self messages are kept as-is;
//   - other agents' messages are rewritten as user messages prefixed with
//     "[name]", so providers never see consecutive assistant turns from
//     different authors.
func MessagesFor(events []*schema.Event, self, branch string) []*schema.Message {
	start := 0
	var summary string
	var firstKept int64
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.Author == schema.CompactionAuthor && ev.Actions.Compaction != nil && BranchVisible(branch, ev.Branch) {
			summary = ev.Actions.Compaction.Summary
			firstKept = ev.Actions.Compaction.FirstKeptSeq
			start = i + 1
			break
		}
	}
	var out []*schema.Message
	if summary != "" {
		out = append(out, schema.UserText("Summary of the earlier conversation:\n"+summary))
		// keep events from FirstKeptSeq even if they precede the compaction event
		var kept []*schema.Event
		for _, ev := range events[:start] {
			if ev.Seq >= firstKept && ev.Author != schema.CompactionAuthor {
				kept = append(kept, ev)
			}
		}
		events = append(kept, events[start:]...)
	}
	for _, ev := range events {
		if ev.Partial || ev.Message == nil || !BranchVisible(branch, ev.Branch) || ev.Author == schema.CompactionAuthor {
			continue
		}
		m := ev.Message
		switch {
		case ev.Author == schema.AuthorUser || ev.Author == self || m.Role == schema.RoleSystem:
			out = append(out, m)
		default:
			if r := foreignMessage(ev.Author, m); r != nil {
				out = append(out, r)
			}
		}
	}
	return out
}

func foreignMessage(author string, m *schema.Message) *schema.Message {
	var parts []string
	var media []schema.ContentBlock
	for _, b := range m.Content {
		switch v := b.(type) {
		case *schema.TextBlock:
			if strings.TrimSpace(v.Text) != "" {
				parts = append(parts, fmt.Sprintf("[%s] said: %s", author, v.Text))
			}
		case *schema.ToolCallBlock:
			parts = append(parts, fmt.Sprintf("[%s] called tool `%s` with arguments: %s", author, v.Name, compactJSON(v.Arguments)))
		case *schema.ToolResultBlock:
			parts = append(parts, fmt.Sprintf("[%s] tool call %s returned: %s", author, v.CallID, (&schema.Message{Content: v.Content}).Text()))
		case *schema.ImageBlock, *schema.FileBlock:
			media = append(media, b)
		}
	}
	if len(parts) == 0 && len(media) == 0 {
		return nil
	}
	msg := &schema.Message{Role: schema.RoleUser}
	msg.Content = append(msg.Content, &schema.TextBlock{Text: "For context:\n" + strings.Join(parts, "\n")})
	msg.Content = append(msg.Content, media...)
	return msg
}

func compactJSON(raw json.RawMessage) string {
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return string(raw)
	}
	b, _ := json.Marshal(v)
	return string(b)
}
