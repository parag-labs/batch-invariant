package batchinvariant

// Model is the output projection: logits = x @ wᵀ. W is [vocab][d_in].
type Model struct {
	W Mat
}

// Vocab returns the number of output tokens (rows of W).
func (m Model) Vocab() int {
	return len(m.W)
}

func kernel(invariant bool) func(Mat, Mat) Mat {
	if invariant {
		return MatmulInvariant
	}
	return MatmulVariant
}

// LogitsAlone returns the logits for x decoded on its own (a batch of one).
func LogitsAlone(m Model, x Vec, invariant bool) Vec {
	return kernel(invariant)(Mat{x}, m.W)[0]
}

// LogitsInBatch returns the logits for x when it is decoded in a batch alongside
// others. x is placed first; the returned row is its logits. With the invariant
// kernel this equals LogitsAlone; with the variant kernel it may not.
func LogitsInBatch(m Model, x Vec, others Mat, invariant bool) Vec {
	batch := make(Mat, 0, len(others)+1)
	batch = append(batch, x)
	batch = append(batch, others...)
	return kernel(invariant)(batch, m.W)[0]
}

// Argmax returns the chosen next token: the index of the largest logit,
// first-wins on ties.
func Argmax(logits Vec) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

// TokenDependsOnBatch reports whether x's decoded token changes between running
// alone and running in a batch — the failure the invariant kernel makes
// impossible.
func TokenDependsOnBatch(m Model, x Vec, others Mat, invariant bool) bool {
	alone := Argmax(LogitsAlone(m, x, invariant))
	batched := Argmax(LogitsInBatch(m, x, others, invariant))
	return alone != batched
}
