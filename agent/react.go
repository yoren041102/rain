package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/yoren041102/rain/schema"
)

// TransferToolName is the built-in transfer tool added to ReAct agents that
// have transfer targets.
const TransferToolName = "transfer_to_agent"

// Reasons used by agent-level interrupts.
const ReasonStop = "stop"

// ErrMaxIterations is returned when a ReAct agent exhausts MaxIterations or
// MaxModelCalls without producing a final answer.
var ErrMaxIterations = errors.New("rain: max iterations reached")

// ReActConfig configures NewReAct.
type ReActConfig struct {
	Name, Description string
	Instruction       string // system prompt
	// InstructionFunc, if set, computes the system prompt per model call and
	// takes precedence over Instruction.
	InstructionFunc                             func(ctx context.Context, inv *Invocation) (string, error)
	Model                                       schema.Model
	Tools                                       []schema.Tool
	Toolsets                                    []schema.Toolset
	SubAgents                                   []Agent
	MaxIterations                               int // default 20
	MaxModelCalls                               int // per Run; 0 means unlimited
	ParallelToolCalls                           bool
	AllowTransferToParent, AllowTransferToPeers bool
	OutputSchema                                json.RawMessage // structured output (JSON Schema)
	// OutputKey, if set, stores the final answer text in session state.
	OutputKey   string
	Options     schema.ModelOptions // base generation options
	Middlewares Middlewares         // agent-level middlewares
}

// ReActAgent is the built-in reason-and-act agent.
type ReActAgent struct {
	cfg ReActConfig
}

// NewReAct validates cfg and returns a ReAct agent.
func NewReAct(cfg ReActConfig) (*ReActAgent, error) {
	if cfg.Name == "" {
		return nil, errors.New("rain: ReAct agent needs a name")
	}
	if cfg.Model == nil {
		return nil, fmt.Errorf("rain: ReAct agent %q needs a model", cfg.Name)
	}
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 20
	}
	seen := map[string]bool{TransferToolName: true}
	for _, t := range cfg.Tools {
		n := t.Spec().Name
		if n == "" {
			return nil, fmt.Errorf("rain: agent %q has a tool with empty name", cfg.Name)
		}
		if seen[n] {
			return nil, fmt.Errorf("rain: agent %q has duplicate or reserved tool name %q", cfg.Name, n)
		}
		seen[n] = true
	}
	return &ReActAgent{cfg: cfg}, nil
}

func (a *ReActAgent) Name() string                        { return a.cfg.Name }
func (a *ReActAgent) Description() string                 { return a.cfg.Description }
func (a *ReActAgent) SubAgents() []Agent                  { return a.cfg.SubAgents }
func (a *ReActAgent) AgentMiddlewares() []AgentMiddleware { return a.cfg.Middlewares.Agent }
func (a *ReActAgent) Config() ReActConfig                 { return a.cfg }

// FindTool implements ToolFinder.
func (a *ReActAgent) FindTool(ctx context.Context, name string) schema.Tool {
	for _, t := range a.cfg.Tools {
		if t.Spec().Name == name {
			return t
		}
	}
	for _, ts := range a.cfg.Toolsets {
		specs, err := ts.Tools(ctx)
		if err != nil {
			continue
		}
		for _, s := range specs {
			if s.Name == name {
				return &schema.ToolsetTool{Set: ts, ToolSpec: s}
			}
		}
	}
	return nil
}

// Run implements Agent.
func (a *ReActAgent) Run(ctx context.Context, inv *Invocation) schema.Stream[*schema.Event] {
	return func(yield func(*schema.Event, error) bool) {
		if s, ok := RouteResume(ctx, inv); ok {
			for ev, err := range s {
				if !yield(ev, err) || err != nil {
					return
				}
			}
			return
		}
		r := &reactRun{a: a, inv: inv, yield: yield}
		r.run(ctx)
	}
}

type reactRun struct {
	a          *ReActAgent
	inv        *Invocation
	yield      func(*schema.Event, error) bool
	msgs       []*schema.Message
	modelCalls int
	stopped    bool // consumer broke out
}

func (r *reactRun) emit(ev *schema.Event) bool {
	if r.stopped {
		return false
	}
	if !r.yield(ev, nil) {
		r.stopped = true
		return false
	}
	if ev.Message != nil && !ev.Partial {
		r.msgs = append(r.msgs, ev.Message)
	}
	return true
}

func (r *reactRun) fail(err error) {
	if !r.stopped {
		r.yield(nil, err)
		r.stopped = true
	}
}

func (r *reactRun) run(ctx context.Context) {
	inv := r.inv
	if inv.History != nil {
		evs, err := inv.History.Events(ctx)
		if err != nil {
			r.fail(err)
			return
		}
		r.msgs = MessagesFor(evs, r.a.cfg.Name, inv.Branch)
	} else if inv.UserMessage != nil {
		r.msgs = []*schema.Message{inv.UserMessage}
	}

	if inv.Resuming() {
		done := r.resume(ctx)
		if done {
			return
		}
	}
	inv = inv.WithoutResume()
	r.inv = inv

	for iter := 0; iter < r.a.cfg.MaxIterations; iter++ {
		if err := ctx.Err(); err != nil {
			r.fail(err)
			return
		}
		if r.a.cfg.MaxModelCalls > 0 && r.modelCalls >= r.a.cfg.MaxModelCalls {
			break
		}
		if inv.Steering != nil {
			for _, m := range inv.Steering() {
				ev := inv.NewEvent(m)
				ev.Author = schema.AuthorUser
				if !r.emit(ev) {
					return
				}
			}
		}
		tools, err := r.collectTools(ctx)
		if err != nil {
			r.fail(err)
			return
		}
		req, err := r.buildRequest(ctx, tools)
		if err != nil {
			r.fail(err)
			return
		}
		r.modelCalls++
		resp, ok := r.callModel(ctx, req)
		if !ok {
			return
		}
		msg := resp.Message
		calls := msg.ToolCalls()

		// structured output via synthetic tool
		for _, c := range calls {
			if c.Name == schema.OutputToolName {
				msg = &schema.Message{Role: schema.RoleAssistant, Content: []schema.ContentBlock{&schema.TextBlock{Text: string(c.Arguments)}}, Origin: msg.Origin, StopReason: schema.StopReasonStop}
				calls = nil
				break
			}
		}

		ev := inv.NewEvent(msg)
		ev.Usage = resp.Usage
		final := len(calls) == 0 && msg.StopReason != schema.StopReasonLength
		if final {
			ev.Final = true
			if k := r.a.cfg.OutputKey; k != "" {
				inv.State.Set(k, msg.Text())
			}
		}
		if !r.emit(ev) {
			return
		}
		if final {
			return
		}
		if r.checkStop() {
			return
		}
		if msg.StopReason == schema.StopReasonLength {
			if len(calls) > 0 {
				res := &schema.Message{Role: schema.RoleTool}
				for _, c := range calls {
					res.Content = append(res.Content, &schema.ToolResultBlock{CallID: c.ID, IsError: true,
						Content: []schema.ContentBlock{&schema.TextBlock{Text: "Output was truncated (max tokens); this tool call was not executed."}}})
				}
				if !r.emit(inv.NewEvent(res)) {
					return
				}
			}
			continue
		}
		if r.executeAndContinue(ctx, calls, tools, nil) {
			return
		}
	}
	r.fail(fmt.Errorf("%w (agent %q)", ErrMaxIterations, r.a.cfg.Name))
}

// checkStop emits a stop interrupt at a safe point when one was requested.
func (r *reactRun) checkStop() bool {
	if r.inv.StopRequested == nil || !r.inv.StopRequested() {
		return false
	}
	it := r.inv.NewInterrupt(&InterruptError{Reason: ReasonStop}, "", "")
	r.emit(r.inv.InterruptEvent(nil, it))
	return true
}

// resumeInfo describes how one call is resumed.
type resumeInfo struct {
	interrupt *schema.Interrupt
	data      *schema.ResumeData // nil: not selected, must re-trigger
}

// resume handles interrupts raised by this agent. It returns true when the
// run is over (re-interrupted, transferred, finished or failed).
func (r *reactRun) resume(ctx context.Context) bool {
	inv := r.inv
	mine := inv.InterruptsFor(r.a.cfg.Name)
	byCall := map[string]resumeInfo{}
	var reraise []*schema.Interrupt
	for _, it := range mine {
		d, ok := inv.ResumeFor(it.ID)
		if it.CallID != "" {
			ri := resumeInfo{interrupt: it}
			if ok {
				ri.data = d
			}
			byCall[it.CallID] = ri
			continue
		}
		if !ok {
			reraise = append(reraise, it)
		}
	}
	if len(byCall) == 0 {
		if len(reraise) > 0 {
			r.emit(inv.InterruptEvent(nil, reraise...))
			return true
		}
		return false
	}
	// Find the assistant message holding the interrupted calls and the
	// calls that still lack results.
	answered := map[string]bool{}
	var pending []*schema.ToolCallBlock
	for i := len(r.msgs) - 1; i >= 0; i-- {
		m := r.msgs[i]
		if m.Role == schema.RoleTool {
			for _, res := range m.ToolResultBlocks() {
				answered[res.CallID] = true
			}
			continue
		}
		if m.Role != schema.RoleAssistant {
			continue
		}
		hit := false
		for _, c := range m.ToolCalls() {
			if _, ok := byCall[c.ID]; ok {
				hit = true
			}
		}
		if !hit {
			continue
		}
		for _, c := range m.ToolCalls() {
			if !answered[c.ID] {
				pending = append(pending, c)
			}
		}
		break
	}
	if len(pending) == 0 {
		if len(reraise) > 0 {
			r.emit(inv.InterruptEvent(nil, reraise...))
			return true
		}
		return false
	}
	tools, err := r.collectTools(ctx)
	if err != nil {
		r.fail(err)
		return true
	}
	return r.executeAndContinueExtra(ctx, pending, tools, byCall, reraise)
}

// toolEntry is a tool visible to the model in one iteration.
type toolEntry struct {
	spec schema.ToolSpec
	tool schema.Tool
}

func (r *reactRun) collectTools(ctx context.Context) ([]toolEntry, error) {
	var out []toolEntry
	seen := map[string]bool{}
	add := func(t schema.Tool, s schema.ToolSpec) {
		if seen[s.Name] {
			return
		}
		seen[s.Name] = true
		out = append(out, toolEntry{spec: s, tool: t})
	}
	for _, t := range r.a.cfg.Tools {
		add(t, t.Spec())
	}
	for _, ts := range r.a.cfg.Toolsets {
		specs, err := ts.Tools(ctx)
		if err != nil {
			return nil, fmt.Errorf("rain: toolset: %w", err)
		}
		for _, s := range specs {
			add(&schema.ToolsetTool{Set: ts, ToolSpec: s}, s)
		}
	}
	if targets := r.transferTargets(); len(targets) > 0 {
		t := &transferTool{targets: targets}
		add(t, t.Spec())
	}
	return out, nil
}

func (r *reactRun) transferTargets() []Agent {
	var out []Agent
	out = append(out, r.a.cfg.SubAgents...)
	tree := r.inv.Tree
	if tree == nil {
		return out
	}
	self := tree.Find(r.a.cfg.Name)
	if self == nil {
		self = r.a
	}
	parent := tree.Parent(self)
	if parent != nil && r.a.cfg.AllowTransferToParent {
		out = append(out, parent)
	}
	if parent != nil && r.a.cfg.AllowTransferToPeers {
		for _, p := range parent.SubAgents() {
			if p.Name() != r.a.cfg.Name {
				out = append(out, p)
			}
		}
	}
	return out
}

func (r *reactRun) buildRequest(ctx context.Context, tools []toolEntry) (*schema.ModelRequest, error) {
	instr := r.a.cfg.Instruction
	if r.a.cfg.InstructionFunc != nil {
		s, err := r.a.cfg.InstructionFunc(ctx, r.inv)
		if err != nil {
			return nil, err
		}
		instr = s
	}
	msgs := schema.PrepareForModel(r.msgs, schema.ModelOrigin(r.a.cfg.Model))
	req := &schema.ModelRequest{Options: r.a.cfg.Options}
	if instr != "" {
		req.Messages = append(req.Messages, schema.SystemText(instr))
	}
	req.Messages = append(req.Messages, msgs...)
	for _, t := range tools {
		req.Tools = append(req.Tools, t.spec)
	}
	if len(r.a.cfg.OutputSchema) > 0 {
		req.Options.OutputSchema = r.a.cfg.OutputSchema
	}
	for _, ts := range r.a.cfg.Toolsets {
		if err := ts.ProcessRequest(ctx, req); err != nil {
			return nil, fmt.Errorf("rain: toolset process request: %w", err)
		}
	}
	return req, nil
}

func (r *reactRun) callModel(ctx context.Context, req *schema.ModelRequest) (*schema.ModelResponse, bool) {
	acc := schema.NewAccumulator()
	var final *schema.ModelResponse
	for resp, err := range CallModel(ctx, r.inv, r.a.cfg.Model, req, r.a.cfg.Middlewares.Model...) {
		if err != nil {
			r.fail(err)
			return nil, false
		}
		if resp == nil {
			continue
		}
		if !resp.Partial {
			final = resp
			continue
		}
		acc.Add(resp)
		if resp.Message != nil {
			ev := r.inv.NewEvent(resp.Message)
			ev.Partial = true
			if !r.emit(ev) {
				return nil, false
			}
		}
	}
	if final == nil {
		m := acc.Message()
		if len(m.Content) == 0 {
			r.fail(fmt.Errorf("rain: model %q returned no response", r.a.cfg.Model.Name()))
			return nil, false
		}
		final = &schema.ModelResponse{Message: m, Usage: acc.Usage(), FinishReason: m.StopReason}
	}
	if final.Message == nil {
		r.fail(fmt.Errorf("rain: model %q returned a final response without message", r.a.cfg.Model.Name()))
		return nil, false
	}
	m := final.Message.Clone()
	if m.Role == "" {
		m.Role = schema.RoleAssistant
	}
	if m.StopReason == "" {
		m.StopReason = final.FinishReason
	}
	if m.StopReason == "" {
		if len(m.ToolCalls()) > 0 {
			m.StopReason = schema.StopReasonToolUse
		} else {
			m.StopReason = schema.StopReasonStop
		}
	}
	if m.Origin == nil {
		o := schema.ModelOrigin(r.a.cfg.Model)
		m.Origin = &o
	}
	out := *final
	out.Message = m
	return &out, true
}

func (r *reactRun) executeAndContinue(ctx context.Context, calls []*schema.ToolCallBlock, tools []toolEntry, resume map[string]resumeInfo) bool {
	return r.executeAndContinueExtra(ctx, calls, tools, resume, nil)
}

// executeAndContinueExtra runs calls, emits the tool result event and
// applies its actions. It returns true when the run is over.
func (r *reactRun) executeAndContinueExtra(ctx context.Context, calls []*schema.ToolCallBlock, tools []toolEntry, resume map[string]resumeInfo, extraInterrupts []*schema.Interrupt) bool {
	inv := r.inv
	results, actions, its := r.executeCalls(ctx, calls, tools, resume)
	its = append(its, extraInterrupts...)
	var msg *schema.Message
	if len(results) > 0 {
		msg = schema.ToolResults(results...)
	}
	ev := inv.InterruptEvent(msg, its...)
	ev.Actions.StateDelta = actions.StateDelta
	ev.Actions.TransferTo = actions.TransferTo
	ev.Actions.Escalate = actions.Escalate
	ev.Actions.SkipSummarization = actions.SkipSummarization
	if actions.SkipSummarization && len(its) == 0 {
		ev.Final = true
	}
	if msg != nil || len(its) > 0 || len(ev.Actions.StateDelta) > 0 || ev.Actions.TransferTo != "" || ev.Actions.Escalate {
		if !r.emit(ev) {
			return true
		}
	}
	if len(its) > 0 {
		return true
	}
	if actions.TransferTo != "" {
		target := r.findAgent(actions.TransferTo)
		if target == nil {
			r.fail(fmt.Errorf("rain: transfer target %q not found", actions.TransferTo))
			return true
		}
		for ev, err := range RunAgent(ctx, inv.WithoutResume(), target) {
			if err != nil {
				r.fail(err)
				return true
			}
			if !r.yield(ev, nil) {
				r.stopped = true
				return true
			}
		}
		return true
	}
	if actions.Escalate || actions.SkipSummarization {
		return true
	}
	return r.checkStop()
}

func (r *reactRun) findAgent(name string) Agent {
	if r.inv.Tree != nil {
		if a := r.inv.Tree.Find(name); a != nil {
			return a
		}
	}
	for _, a := range r.transferTargets() {
		if a.Name() == name {
			return a
		}
	}
	return nil
}

type callOutcome struct {
	result    *schema.ToolResultBlock
	actions   *schema.EventActions
	interrupt *schema.Interrupt
}

func (r *reactRun) executeCalls(ctx context.Context, calls []*schema.ToolCallBlock, tools []toolEntry, resume map[string]resumeInfo) ([]*schema.ToolResultBlock, schema.EventActions, []*schema.Interrupt) {
	byName := map[string]toolEntry{}
	for _, t := range tools {
		byName[t.spec.Name] = t
	}
	outcomes := make([]callOutcome, len(calls))
	run := func(i int) { outcomes[i] = r.executeCall(ctx, calls[i], byName, resume) }

	parallelOK := func(c *schema.ToolCallBlock) bool {
		if !r.a.cfg.ParallelToolCalls {
			return false
		}
		t, ok := byName[c.Name]
		return ok && schema.IsConcurrencySafe(t.tool)
	}
	for i := 0; i < len(calls); {
		if !parallelOK(calls[i]) {
			run(i)
			i++
			continue
		}
		j := i
		for j < len(calls) && parallelOK(calls[j]) {
			j++
		}
		var wg sync.WaitGroup
		for k := i; k < j; k++ {
			wg.Add(1)
			go func(k int) {
				defer wg.Done()
				run(k)
			}(k)
		}
		wg.Wait()
		i = j
	}

	var (
		results []*schema.ToolResultBlock
		actions schema.EventActions
		its     []*schema.Interrupt
	)
	for _, o := range outcomes {
		if o.interrupt != nil {
			its = append(its, o.interrupt)
			continue
		}
		if o.result != nil {
			results = append(results, o.result)
		}
		if a := o.actions; a != nil {
			if len(a.StateDelta) > 0 {
				if actions.StateDelta == nil {
					actions.StateDelta = map[string]any{}
				}
				for k, v := range a.StateDelta {
					actions.StateDelta[k] = v
				}
			}
			if a.TransferTo != "" {
				actions.TransferTo = a.TransferTo
			}
			actions.Escalate = actions.Escalate || a.Escalate
			actions.SkipSummarization = actions.SkipSummarization || a.SkipSummarization
		}
	}
	if len(actions.StateDelta) > 0 && r.inv.State != nil {
		r.inv.State.Apply(actions.StateDelta)
	}
	sort.SliceStable(its, func(a, b int) bool { return its[a].ID < its[b].ID })
	return results, actions, its
}

func (r *reactRun) executeCall(ctx context.Context, call *schema.ToolCallBlock, byName map[string]toolEntry, resume map[string]resumeInfo) callOutcome {
	inv := r.inv
	tc := &ToolContext{Invocation: inv, CallID: call.ID, ToolName: call.Name, Actions: &schema.EventActions{}}
	if ri, ok := resume[call.ID]; ok {
		if ri.data == nil {
			return callOutcome{interrupt: ri.interrupt}
		}
		if ri.data.ToolResult != nil {
			return callOutcome{result: ri.data.ToolResult.Block(call.ID)}
		}
		tc.Interrupt = ri.interrupt
		tc.Resume = ri.data
	}
	entry, ok := byName[call.Name]
	if !ok {
		return callOutcome{result: errBlock(call.ID, fmt.Sprintf("Tool %q not found. Available tools: %s", call.Name, toolNames(byName)))}
	}
	if err := schema.ValidateJSON(entry.spec.Parameters, call.Arguments); err != nil {
		return callOutcome{result: errBlock(call.ID, fmt.Sprintf("Invalid arguments for tool %q: %v\nArguments received: %s", call.Name, err, string(call.Arguments)))}
	}
	res, err := CallTool(WithToolContext(ctx, tc), inv, entry.tool, call, r.a.cfg.Middlewares.Tool...)
	if err != nil {
		if ie, ok := AsInterrupt(err); ok {
			return callOutcome{interrupt: inv.NewInterrupt(ie, call.Name, call.ID)}
		}
		return callOutcome{result: errBlock(call.ID, "Error: "+err.Error()), actions: tc.Actions}
	}
	if res == nil {
		res = &schema.ToolResult{}
	}
	return callOutcome{result: res.Block(call.ID), actions: tc.Actions}
}

func errBlock(callID, text string) *schema.ToolResultBlock {
	return &schema.ToolResultBlock{CallID: callID, IsError: true, Content: []schema.ContentBlock{&schema.TextBlock{Text: text}}}
}

func toolNames(m map[string]toolEntry) string {
	names := make([]string, 0, len(m))
	for n := range m {
		names = append(names, n)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// ---- transfer tool ----

type transferTool struct{ targets []Agent }

func (t *transferTool) Spec() schema.ToolSpec {
	var names []any
	var desc strings.Builder
	desc.WriteString("Transfer the conversation to another agent that is better suited to handle it. Available agents:\n")
	for _, a := range t.targets {
		names = append(names, a.Name())
		fmt.Fprintf(&desc, "- %s: %s\n", a.Name(), a.Description())
	}
	params, _ := json.Marshal(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"agent_name": map[string]any{"type": "string", "enum": names, "description": "Name of the agent to transfer to."},
		},
		"required": []string{"agent_name"},
	})
	return schema.ToolSpec{Name: TransferToolName, Description: desc.String(), Parameters: params}
}

func (t *transferTool) Call(ctx context.Context, args json.RawMessage) (*schema.ToolResult, error) {
	var in struct {
		AgentName string `json:"agent_name"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, err
	}
	for _, a := range t.targets {
		if a.Name() == in.AgentName {
			if tc, ok := ToolContextFrom(ctx); ok {
				tc.Actions.TransferTo = in.AgentName
			}
			return schema.TextResult("Transferred to " + in.AgentName), nil
		}
	}
	return schema.ErrorResult("Unknown agent " + in.AgentName), nil
}

func (t *transferTool) IsReadOnly() bool   { return true }
func (t *transferTool) IsReplaySafe() bool { return true }
