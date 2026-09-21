package batchinvariant

import "testing"

func TestSumFlatEmptyIsZero(t *testing.T) {
	if got := SumFlat(Vec{}); got != 0.0 {
		t.Fatalf("SumFlat(empty) = %v, want 0", got)
	}
}

func TestSumFlatSingleElement(t *testing.T) {
	if got := SumFlat(Vec{42.5}); got != 42.5 {
		t.Fatalf("SumFlat([42.5]) = %v, want 42.5", got)
	}
}

func TestSumFlatLeftToRight(t *testing.T) {
	if got := SumFlat(Vec{0.1, 0.2, 0.3, 0.4}); got != ((0.1+0.2)+0.3)+0.4 {
		t.Fatalf("SumFlat is not left-to-right: %v", got)
	}
}

func TestSplitEqualsFlatInExactMathButNotInFloat(t *testing.T) {
	vals := Vec{1e16, 1.0, -1e16, 1.0}
	if SumSplit(vals, 2) == SumFlat(vals) {
		t.Fatalf("expected different grouping to change the float sum")
	}
}

func TestSingleSplitIsFlat(t *testing.T) {
	vals := Vec{0.1, 0.2, 0.3, 0.4}
	if SumSplit(vals, 1) != SumFlat(vals) {
		t.Fatalf("SumSplit(_, 1) must equal SumFlat")
	}
}

func TestZeroOrNegativeSplitsIsFlat(t *testing.T) {
	vals := Vec{0.1, 0.2, 0.3}
	if SumSplit(vals, 0) != SumFlat(vals) {
		t.Fatalf("SumSplit(_, 0) must equal SumFlat")
	}
	if SumSplit(vals, -3) != SumFlat(vals) {
		t.Fatalf("SumSplit(_, negative) must equal SumFlat")
	}
}

func TestSumSplitEmptyIsZero(t *testing.T) {
	if got := SumSplit(Vec{}, 4); got != 0.0 {
		t.Fatalf("SumSplit(empty, 4) = %v, want 0", got)
	}
}

func TestSumSplitMoreSplitsThanElements(t *testing.T) {
	vals := Vec{1.0, 2.0, 3.0}
	// size = ceil(3/8) = 1, so three singleton partials plus empties skipped.
	if got := SumSplit(vals, 8); got != 6.0 {
		t.Fatalf("SumSplit([1,2,3], 8) = %v, want 6", got)
	}
}

func TestSplitsForBatch(t *testing.T) {
	cases := map[int]int{0: 8, 1: 8, 2: 4, 4: 2, 8: 1, 9: 1, 16: 1}
	for batch, want := range cases {
		if got := SplitsForBatch(batch); got != want {
			t.Fatalf("SplitsForBatch(%d) = %d, want %d", batch, got, want)
		}
	}
}

func TestSplitsForBatchVariesWithBatchSize(t *testing.T) {
	if SplitsForBatch(1) == SplitsForBatch(4) {
		t.Fatalf("split count must vary with batch size")
	}
}

func TestMatmulShapes(t *testing.T) {
	w := Mat{{1, 2}, {3, 4}, {5, 6}}
	out := MatmulInvariant(Mat{{1, 1}, {2, 2}}, w)
	if len(out) != 2 || len(out[0]) != 3 {
		t.Fatalf("unexpected shape %dx%d", len(out), len(out[0]))
	}
	// Row 0 = [1*1+1*2, 1*3+1*4, 1*5+1*6] = [3, 7, 11].
	if out[0][0] != 3 || out[0][1] != 7 || out[0][2] != 11 {
		t.Fatalf("wrong products: %v", out[0])
	}
}

func TestInvariantRowIdenticalAloneAndBatched(t *testing.T) {
	r := newLCG(1)
	model := Model{W: r.mixedRows(5, 16)}
	for trial := 0; trial < 200; trial++ {
		x := r.mixedRow(16)
		others := r.mixedRows(trial%7, 16)
		alone := LogitsAlone(model, x, true)
		batched := LogitsInBatch(model, x, others, true)
		if !vecEqual(alone, batched) {
			t.Fatalf("invariant kernel differed on trial %d", trial)
		}
	}
}

func TestInvariantPositionInBatchDoesNotMatter(t *testing.T) {
	r := newLCG(2)
	w := r.mixedRows(4, 16)
	x := r.mixedRow(16)
	neighbours := r.mixedRows(2, 16)
	first := MatmulInvariant(append(Mat{x}, neighbours...), w)[0]
	alone := MatmulInvariant(Mat{x}, w)[0]
	if !vecEqual(first, alone) {
		t.Fatalf("invariant row depended on batch position")
	}
}

func TestVariantRowChangesWithBatch(t *testing.T) {
	// The constructed flip case is guaranteed to differ under the variant kernel.
	model, x, others := FlipCase()
	if vecEqual(LogitsAlone(model, x, false), LogitsInBatch(model, x, others, false)) {
		t.Fatalf("variant kernel should be batch-dependent")
	}
}
