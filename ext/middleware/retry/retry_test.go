package retry_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/yoren041102/rain/agent"
	"github.com/yoren041102/rain/ext/middleware/retry"
	"github.com/yoren041102/rain/schema"
)

func TestRetry(t *testing.T) {
	attempts := 0
	next := func(ctx context.Context, inv *agent.Invocation, req *schema.ModelRequest) schema.Stream[*schema.ModelResponse] {
		attempts++
		return func(yield func(*schema.ModelResponse, error) bool) {
			if !yield(&schema.ModelResponse{Partial: true, Message: schema.AssistantText("par")}, nil) {
				return
			}
			if attempts < 3 {
				yield(nil, errors.New("503"))
				return
			}
			yield(&schema.ModelResponse{Message: schema.AssistantText("ok")}, nil)
		}
	}
	var emitted []*schema.Event
	inv := &agent.Invocation{ID: "i", Emit: func(e *schema.Event) { emitted = append(emitted, e) }}
	var slept []time.Duration
	mw := retry.New(retry.Config{MaxAttempts: 3, BaseDelay: time.Millisecond, Sleep: func(ctx context.Context, d time.Duration) error {
		slept = append(slept, d)
		return nil
	}})
	resp, err := schema.FinalResponse(mw(next)(context.Background(), inv, &schema.ModelRequest{}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Text() != "ok" || attempts != 3 || len(slept) != 2 {
		t.Fatalf("resp=%v attempts=%d slept=%v", resp.Message, attempts, slept)
	}
	if len(emitted) != 2 || !emitted[0].Actions.DiscardPartial {
		t.Fatalf("emitted = %+v", emitted)
	}
	// gives up
	attempts = -10
	_, err = schema.FinalResponse(retry.New(retry.Config{MaxAttempts: 2, Sleep: func(context.Context, time.Duration) error { return nil }})(next)(context.Background(), inv, &schema.ModelRequest{}))
	if err == nil || attempts != -8 {
		t.Fatalf("err=%v attempts=%d", err, attempts)
	}
}
