package batchinvariant

import "testing"

func TestModelVocab(t *testing.T) {
	m := Model{W: Mat{{1, 2}, {3, 4}, {5, 6}}}
	if m.Vocab() != 3 {
		t.Fatalf("Vocab = %d, want 3", m.Vocab())
	}
}

func TestArgmaxFirstWinsOnTies(t *testing.T) {
	if got := Argmax(Vec{1.0, 3.0, 3.0, 2.0}); got != 1 {
		t.Fatalf("Argmax first-wins failed: got %d, want 1", got)
	}
}

func TestArgmaxSingleElement(t *testing.T) {
	if got := Argmax(Vec{5.0}); got != 0 {
		t.Fatalf("Argmax([5]) = %d, want 0", got)
	}
}

func TestArgmaxNegatives(t *testing.T) {
	if got := Argmax(Vec{-3.0, -1.0, -2.0}); got != 1 {
		t.Fatalf("Argmax negatives = %d, want 1", got)
	}
}

func TestLogitsAloneEqualsInvariantBatched(t *testing.T) {
	r := newLCG(7)
	model := Model{W: r.mixedRows(3, 16)}
	x := r.mixedRow(16)
	others := r.mixedRows(4, 16)
	if !vecEqual(LogitsAlone(model, x, true), LogitsInBatch(model, x, others, true)) {
		t.Fatalf("invariant alone vs batched differed")
	}
}

func TestTokenDependsOnBatchFalseForInvariantFlipCase(t *testing.T) {
	model, x, others := FlipCase()
	if TokenDependsOnBatch(model, x, others, true) {
		t.Fatalf("invariant kernel must not depend on batch")
	}
}

func TestTokenDependsOnBatchTrueForVariantFlipCase(t *testing.T) {
	model, x, others := FlipCase()
	if !TokenDependsOnBatch(model, x, others, false) {
		t.Fatalf("variant kernel must depend on batch for the flip case")
	}
}
