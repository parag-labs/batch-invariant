package batchinvariant

import "testing"

func TestFlipCaseShape(t *testing.T) {
	model, x, others := FlipCase()
	if model.Vocab() != 2 {
		t.Fatalf("vocab = %d, want 2", model.Vocab())
	}
	if len(x) != dIn {
		t.Fatalf("len(x) = %d, want %d", len(x), dIn)
	}
	if len(others) != 3 {
		t.Fatalf("len(others) = %d, want 3", len(others))
	}
	for i, v := range x {
		if v != 1.0 {
			t.Fatalf("x[%d] = %v, want 1.0", i, v)
		}
	}
}

func TestFlipCaseVariantFlipsToken(t *testing.T) {
	model, x, others := FlipCase()
	if got := Argmax(LogitsAlone(model, x, false)); got != 1 {
		t.Fatalf("variant alone token = %d, want 1", got)
	}
	if got := Argmax(LogitsInBatch(model, x, others, false)); got != 0 {
		t.Fatalf("variant batched token = %d, want 0", got)
	}
}

func TestFlipCaseInvariantKeepsToken(t *testing.T) {
	model, x, others := FlipCase()
	alone := Argmax(LogitsAlone(model, x, true))
	batched := Argmax(LogitsInBatch(model, x, others, true))
	if alone != 1 || batched != 1 {
		t.Fatalf("invariant tokens = (%d, %d), want (1, 1)", alone, batched)
	}
}
