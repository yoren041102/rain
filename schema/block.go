package schema

import (
	"encoding/json"
	"fmt"
	"maps"
	"slices"
)

// BlockType identifies the kind of a ContentBlock.
type BlockType string

const (
	BlockText       BlockType = "text"
	BlockImage      BlockType = "image"
	BlockFile       BlockType = "file"
	BlockReasoning  BlockType = "reasoning"
	BlockToolCall   BlockType = "tool_call"
	BlockToolResult BlockType = "tool_result"
	BlockUnknown    BlockType = "unknown"
)

// ContentBlock is a sealed interface: only types in this package implement it,
// so a type switch over blocks can be exhaustive.
type ContentBlock interface {
	BlockType() BlockType
	BlockMeta() *BlockMeta
	sealed()
}

// BlockMeta stores vendor-specific extension fields (cache_control,
// thoughtSignature, ...). They round-trip through JSON unchanged.
type BlockMeta struct {
	Extra map[string]json.RawMessage
}

func (m BlockMeta) clone() BlockMeta {
	if m.Extra == nil {
		return m
	}
	return BlockMeta{Extra: maps.Clone(m.Extra)}
}

type TextBlock struct {
	Text string
	Meta BlockMeta
}

type ImageBlock struct {
	MIMEType string
	Data     []byte
	URL      string
	Meta     BlockMeta
}

type FileBlock struct {
	MIMEType string
	Name     string
	Data     []byte
	URL      string
	Meta     BlockMeta
}

type ReasoningBlock struct {
	Text      string
	Signature string
	Meta      BlockMeta
}

type ToolCallBlock struct {
	ID        string
	Name      string
	Arguments json.RawMessage
	Meta      BlockMeta
}

type ToolResultBlock struct {
	CallID  string
	Content []ContentBlock // sent to the model
	Details []ContentBlock // UI only, never sent to the model
	IsError bool
	Meta    BlockMeta
}

// UnknownBlock preserves content types rain does not understand.
type UnknownBlock struct {
	Type string
	Raw  json.RawMessage
	Meta BlockMeta
}

func (*TextBlock) BlockType() BlockType       { return BlockText }
func (*ImageBlock) BlockType() BlockType      { return BlockImage }
func (*FileBlock) BlockType() BlockType       { return BlockFile }
func (*ReasoningBlock) BlockType() BlockType  { return BlockReasoning }
func (*ToolCallBlock) BlockType() BlockType   { return BlockToolCall }
func (*ToolResultBlock) BlockType() BlockType { return BlockToolResult }
func (*UnknownBlock) BlockType() BlockType    { return BlockUnknown }

func (b *TextBlock) BlockMeta() *BlockMeta       { return &b.Meta }
func (b *ImageBlock) BlockMeta() *BlockMeta      { return &b.Meta }
func (b *FileBlock) BlockMeta() *BlockMeta       { return &b.Meta }
func (b *ReasoningBlock) BlockMeta() *BlockMeta  { return &b.Meta }
func (b *ToolCallBlock) BlockMeta() *BlockMeta   { return &b.Meta }
func (b *ToolResultBlock) BlockMeta() *BlockMeta { return &b.Meta }
func (b *UnknownBlock) BlockMeta() *BlockMeta    { return &b.Meta }

func (*TextBlock) sealed()       {}
func (*ImageBlock) sealed()      {}
func (*FileBlock) sealed()       {}
func (*ReasoningBlock) sealed()  {}
func (*ToolCallBlock) sealed()   {}
func (*ToolResultBlock) sealed() {}
func (*UnknownBlock) sealed()    {}

func cloneBlocks(bs []ContentBlock) []ContentBlock {
	if bs == nil {
		return nil
	}
	out := make([]ContentBlock, len(bs))
	for i, b := range bs {
		out[i] = cloneBlock(b)
	}
	return out
}

func cloneBlock(b ContentBlock) ContentBlock {
	switch v := b.(type) {
	case *TextBlock:
		c := *v
		c.Meta = v.Meta.clone()
		return &c
	case *ImageBlock:
		c := *v
		c.Data = slices.Clone(v.Data)
		c.Meta = v.Meta.clone()
		return &c
	case *FileBlock:
		c := *v
		c.Data = slices.Clone(v.Data)
		c.Meta = v.Meta.clone()
		return &c
	case *ReasoningBlock:
		c := *v
		c.Meta = v.Meta.clone()
		return &c
	case *ToolCallBlock:
		c := *v
		c.Arguments = slices.Clone(v.Arguments)
		c.Meta = v.Meta.clone()
		return &c
	case *ToolResultBlock:
		c := *v
		c.Content = cloneBlocks(v.Content)
		c.Details = cloneBlocks(v.Details)
		c.Meta = v.Meta.clone()
		return &c
	case *UnknownBlock:
		c := *v
		c.Raw = slices.Clone(v.Raw)
		c.Meta = v.Meta.clone()
		return &c
	}
	return b
}

// ---- JSON ----

type blockJSON struct {
	Type      string                     `json:"type"`
	Text      string                     `json:"text,omitempty"`
	MIMEType  string                     `json:"mime_type,omitempty"`
	Name      string                     `json:"name,omitempty"`
	Data      []byte                     `json:"data,omitempty"`
	URL       string                     `json:"url,omitempty"`
	Signature string                     `json:"signature,omitempty"`
	ID        string                     `json:"id,omitempty"`
	Arguments json.RawMessage            `json:"arguments,omitempty"`
	CallID    string                     `json:"call_id,omitempty"`
	Content   []json.RawMessage          `json:"content,omitempty"`
	Details   []json.RawMessage          `json:"details,omitempty"`
	IsError   bool                       `json:"is_error,omitempty"`
	Raw       json.RawMessage            `json:"raw,omitempty"`
	Extra     map[string]json.RawMessage `json:"extra,omitempty"`
}

// MarshalBlock encodes a single block with a "type" discriminator.
func MarshalBlock(b ContentBlock) ([]byte, error) {
	j := blockJSON{Type: string(b.BlockType()), Extra: b.BlockMeta().Extra}
	switch v := b.(type) {
	case *TextBlock:
		j.Text = v.Text
	case *ImageBlock:
		j.MIMEType, j.Data, j.URL = v.MIMEType, v.Data, v.URL
	case *FileBlock:
		j.MIMEType, j.Name, j.Data, j.URL = v.MIMEType, v.Name, v.Data, v.URL
	case *ReasoningBlock:
		j.Text, j.Signature = v.Text, v.Signature
	case *ToolCallBlock:
		j.ID, j.Name, j.Arguments = v.ID, v.Name, v.Arguments
		if len(j.Arguments) == 0 {
			j.Arguments = json.RawMessage("{}")
		} else if !json.Valid(j.Arguments) {
			s, _ := json.Marshal(string(v.Arguments))
			j.Arguments = s
		}
	case *ToolResultBlock:
		var err error
		j.CallID, j.IsError = v.CallID, v.IsError
		if j.Content, err = marshalBlocks(v.Content); err != nil {
			return nil, err
		}
		if j.Details, err = marshalBlocks(v.Details); err != nil {
			return nil, err
		}
	case *UnknownBlock:
		j.Type = BlockUnknownPrefix + v.Type
		j.Raw = v.Raw
	default:
		return nil, fmt.Errorf("schema: unsupported block %T", b)
	}
	return json.Marshal(j)
}

// BlockUnknownPrefix marks unknown vendor blocks in JSON so they can be
// restored exactly.
const BlockUnknownPrefix = "x-"

// UnmarshalBlock decodes a single block. Unrecognised types become
// UnknownBlock with the full raw payload preserved.
func UnmarshalBlock(data []byte) (ContentBlock, error) {
	var j blockJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return nil, err
	}
	meta := BlockMeta{Extra: j.Extra}
	switch BlockType(j.Type) {
	case BlockText:
		return &TextBlock{Text: j.Text, Meta: meta}, nil
	case BlockImage:
		return &ImageBlock{MIMEType: j.MIMEType, Data: j.Data, URL: j.URL, Meta: meta}, nil
	case BlockFile:
		return &FileBlock{MIMEType: j.MIMEType, Name: j.Name, Data: j.Data, URL: j.URL, Meta: meta}, nil
	case BlockReasoning:
		return &ReasoningBlock{Text: j.Text, Signature: j.Signature, Meta: meta}, nil
	case BlockToolCall:
		return &ToolCallBlock{ID: j.ID, Name: j.Name, Arguments: j.Arguments, Meta: meta}, nil
	case BlockToolResult:
		content, err := unmarshalBlocks(j.Content)
		if err != nil {
			return nil, err
		}
		details, err := unmarshalBlocks(j.Details)
		if err != nil {
			return nil, err
		}
		return &ToolResultBlock{CallID: j.CallID, Content: content, Details: details, IsError: j.IsError, Meta: meta}, nil
	}
	typ := j.Type
	raw := j.Raw
	if len(typ) > len(BlockUnknownPrefix) && typ[:len(BlockUnknownPrefix)] == BlockUnknownPrefix {
		typ = typ[len(BlockUnknownPrefix):]
	} else {
		raw = append(json.RawMessage(nil), data...)
		meta = BlockMeta{}
	}
	return &UnknownBlock{Type: typ, Raw: raw, Meta: meta}, nil
}

func marshalBlocks(bs []ContentBlock) ([]json.RawMessage, error) {
	if bs == nil {
		return nil, nil
	}
	out := make([]json.RawMessage, 0, len(bs))
	for _, b := range bs {
		raw, err := MarshalBlock(b)
		if err != nil {
			return nil, err
		}
		out = append(out, raw)
	}
	return out, nil
}

func unmarshalBlocks(raws []json.RawMessage) ([]ContentBlock, error) {
	if raws == nil {
		return nil, nil
	}
	out := make([]ContentBlock, 0, len(raws))
	for _, r := range raws {
		b, err := UnmarshalBlock(r)
		if err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, nil
}

// MarshalBlocks encodes a block slice as a JSON array.
func MarshalBlocks(bs []ContentBlock) ([]byte, error) {
	raws, err := marshalBlocks(bs)
	if err != nil {
		return nil, err
	}
	if raws == nil {
		raws = []json.RawMessage{}
	}
	return json.Marshal(raws)
}

// UnmarshalBlocks decodes a JSON array of blocks.
func UnmarshalBlocks(data []byte) ([]ContentBlock, error) {
	var raws []json.RawMessage
	if err := json.Unmarshal(data, &raws); err != nil {
		return nil, err
	}
	return unmarshalBlocks(raws)
}
