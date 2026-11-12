package schema

import "iter"

// Stream is the unified streaming type used throughout rain.
//
// Conventions:
//   - A producer yields a non-nil error at most once, and it is always the
//     last item of the stream.
//   - When yield returns false the consumer has stopped (break); the producer
//     must stop work and release resources.
//   - Cancellation is carried by context.Context, not by the stream.
type Stream[T any] = iter.Seq2[T, error]

// FromSlice returns a stream that yields every element of items.
func FromSlice[T any](items ...T) Stream[T] {
	return func(yield func(T, error) bool) {
		for _, it := range items {
			if !yield(it, nil) {
				return
			}
		}
	}
}

// Error returns a stream that yields only err.
func Error[T any](err error) Stream[T] {
	return func(yield func(T, error) bool) {
		var zero T
		yield(zero, err)
	}
}

// Collect drains s and returns all items. It stops at the first error and
// returns the items collected so far together with the error.
func Collect[T any](s Stream[T]) ([]T, error) {
	var out []T
	for v, err := range s {
		if err != nil {
			return out, err
		}
		out = append(out, v)
	}
	return out, nil
}

// Map transforms every item of s with f.
func Map[T, U any](s Stream[T], f func(T) U) Stream[U] {
	return func(yield func(U, error) bool) {
		for v, err := range s {
			if err != nil {
				var zero U
				yield(zero, err)
				return
			}
			if !yield(f(v), nil) {
				return
			}
		}
	}
}

// Filter keeps the items of s for which keep returns true. Errors always pass.
func Filter[T any](s Stream[T], keep func(T) bool) Stream[T] {
	return func(yield func(T, error) bool) {
		for v, err := range s {
			if err != nil {
				yield(v, err)
				return
			}
			if keep(v) && !yield(v, nil) {
				return
			}
		}
	}
}
