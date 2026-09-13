# Design notes

## Problem and goals

Ask a model the same question twice and you expect the same answer. In a batched
inference server you don't always get it: a prompt's output can change depending
on **what else was in the batch** at that moment. Not because of temperature —
with greedy decoding, temperature 0 — but because the *arithmetic* changed. This
is a real, reported source of non-reproducible LLM inference, and it's subtle
enough that most people don't believe it until they see a token flip.

Goals:

- **Show the bug concretely** — not "logits differ by 1e-9" hand-waving, but a
  constructed case where the *decoded token* changes with the batch.
- **Fix it provably** — a kernel whose output for a row is bit-identical whatever
  the batch, asserted over thousands of batch compositions.
- **Explain the mechanism** honestly, in code small enough to read.

## Where the batch-dependence comes from

A matmul output is a dot product — a sum over the K dimension. In exact
arithmetic the order of that sum is irrelevant. In floating point it is not:
`(a + b) + c` can differ from `a + (b + c)` in the last bit, and with mixed
magnitudes (large positive and negative terms that nearly cancel) the difference
grows far past the last bit.

Production kernels don't reduce left-to-right. They **split the K reduction into
chunks** and combine partial sums — a split-K / tiled reduction — to spread work
across cores. Crucially, *how many chunks* is often chosen from the shape of the
whole batch: a smaller batch may split K further to keep every core busy. So the
grouping of the sum — and therefore its rounding — depends on the batch size.
Same row, different neighbours, different bits. Sometimes that difference lands
right on the boundary between two tokens' logits, and the argmax flips.

`kernels.py` models this exactly: `sum_split(vals, nsplits)` reduces by grouping,
and `splits_for_batch(batch_size)` picks the group count from the batch — a
plausible heuristic that is precisely the poison. `matmul_variant` uses it;
`matmul_invariant` ignores the batch and uses a **fixed** reduction shape.

## The fix

Batch-invariance is not exotic: **make the reduction order a property of the
problem, not of the batch.** `matmul_invariant` reduces every dot product with
the same fixed grouping regardless of how many rows share the batch, so a row's
output is a pure function of that row and the weights. The test suite asserts the
consequence directly — a row is bit-identical alone and in any batch — over
thousands of random compositions, and the constructed `flip_case` shows the token
that moved under the variant kernel staying put under the invariant one.

The real-world version of this fix is the same idea at kernel level: choose
reduction/tiling shapes that don't depend on batch dimensions (fixed split-K,
consistent accumulation order), accepting a little performance to buy
reproducibility.

## Trade-offs I made on purpose

- **Pure-Python float reductions, no numpy.** numpy's `sum` has its own pairwise
  reduction and SIMD, which would hide the very ordering I'm demonstrating.
  Explicit loops make the reduction order *exactly* what the code says, so the
  bug and its fix are legible and testable.
- **A constructed flip case, found by search.** Token flips are rare (they need
  two logits tied to within the reduction's rounding error), so `flip_case`
  reproduces one deterministically rather than pretending they're common. The
  point isn't frequency — it's that the probability is not zero, which for a
  reproducibility guarantee is already unacceptable.
- **The variant heuristic is realistic, not a strawman.** "Split K more when the
  batch is small" is a genuine load-balancing instinct; that's what makes it a
  good cautionary example.

## Non-goals

- **Not a GPU kernel.** This is a CPU model of the reduction-order effect, not a
  CUDA implementation. The mechanism and the fix are identical; the substrate is
  simplified so it's readable and runs in CI.
- **Not about nondeterministic hardware.** Atomics and run-to-run
  nondeterminism are a related but separate problem; here the variant kernel is
  perfectly deterministic *given the batch* — the batch itself is the hidden
  input.
- **Not a full attention stack.** The output projection matmul is enough to show
  a token flip; the same reasoning applies to every reduction in the model.

## Reference

The batch-invariance problem in LLM inference was crisply framed by Thinking
Machines' "Defeating Nondeterminism in LLM Inference" (2025); this repo is a
small, self-contained reconstruction of the core mechanism and fix.
