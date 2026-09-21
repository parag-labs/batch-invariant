// Package batchinvariant is a Go port of the batch-invariant reduction kernels.
//
// A matmul output is a dot product: a sum over the K dimension. In exact
// arithmetic the order of that sum does not matter. In floating point it does —
// (a + b) + c can differ from a + (b + c) in the last bit, and far more when
// large terms nearly cancel. Production kernels split the K reduction into
// chunks and combine partial sums, and how many chunks is often chosen from the
// batch shape. So a row's result can change depending on who else is in the
// batch — a real source of non-reproducible LLM inference.
//
// SumSplit reduces by grouping; MatmulVariant chooses the grouping from the
// batch size (so a row moves with its neighbours), while MatmulInvariant fixes
// the reduction order regardless of batch (so a row is always the same bits).
package batchinvariant

// Vec is a dense vector of float64 values.
type Vec = []float64

// Mat is a dense matrix stored as a slice of rows.
type Mat = [][]float64

// FixedSplits is a single, batch-independent reduction shape. The value does
// not matter for correctness — only that it never depends on the batch.
const FixedSplits = 4

// SumFlat returns the left-to-right sum of vals — one fixed order.
func SumFlat(vals Vec) float64 {
	acc := 0.0
	for _, v := range vals {
		acc += v
	}
	return acc
}

// SumSplit sums vals by partitioning them into nsplits contiguous groups,
// reducing each group left-to-right, then summing the partials. It is
// mathematically equal to SumFlat; in floating point, a different grouping is a
// different result. This models a split-K / tiled reduction.
func SumSplit(vals Vec, nsplits int) float64 {
	n := len(vals)
	if nsplits <= 1 || n == 0 {
		return SumFlat(vals)
	}
	size := (n + nsplits - 1) / nsplits // ceil(n / nsplits)
	partials := make(Vec, 0, nsplits)
	for g := 0; g < nsplits; g++ {
		start := g * size
		if start >= n {
			continue
		}
		end := start + size
		if end > n {
			end = n
		}
		partials = append(partials, SumFlat(vals[start:end]))
	}
	return SumFlat(partials)
}

// SplitsForBatch picks the K-split count from the batch size (fewer rows → split
// K further to keep every core busy). Because the result depends on this number,
// it makes the kernel batch-variant. This models shape-dependent tiling that
// real kernels do.
func SplitsForBatch(batchSize int) int {
	d := batchSize
	if d < 1 {
		d = 1
	}
	s := 8 / d
	if s < 1 {
		s = 1
	}
	return s
}

func matmul(xBatch Mat, w Mat, nsplits int) Mat {
	out := make(Mat, 0, len(xBatch))
	for _, x := range xBatch {
		row := make(Vec, len(w))
		for j := range w {
			products := make(Vec, len(x))
			for i := range x {
				products[i] = x[i] * w[j][i]
			}
			row[j] = SumSplit(products, nsplits)
		}
		out = append(out, row)
	}
	return out
}

// MatmulVariant computes logits = x @ wᵀ, reduced with a batch-size-dependent
// grouping. xBatch is [batch][d_in], w is [d_out][d_in]. The K reduction uses
// SplitsForBatch(len(xBatch)) groups — so a row's output depends on how many
// rows share its batch. This is the bug.
func MatmulVariant(xBatch Mat, w Mat) Mat {
	return matmul(xBatch, w, SplitsForBatch(len(xBatch)))
}

// MatmulInvariant computes logits = x @ wᵀ with a reduction order fixed
// independently of the batch. Every dot product is reduced with the same
// grouping whatever the batch size, so a row's output is identical whether it
// runs alone or alongside a thousand others. That is the guarantee.
func MatmulInvariant(xBatch Mat, w Mat) Mat {
	return matmul(xBatch, w, FixedSplits)
}
