"""Reduction kernels — where batch-dependence sneaks in.

A single output of a matmul is a dot product: a sum over the K dimension. In
exact arithmetic the order of that sum doesn't matter. In floating point it does
— ``(a + b) + c`` can differ from ``a + (b + c)`` in the last bit. Production
inference kernels don't sum left-to-right; they split the K reduction into
chunks and combine the partial sums, and *how many chunks* is often chosen from
the shape of the whole batch (to balance work across cores). So the same row's
result can change depending on **who else is in the batch** — a real source of
non-reproducible LLM inference.

This module makes that concrete. ``sum_split`` reduces by grouping; a
batch-*variant* matmul chooses the grouping from the batch size, so a row's
result moves with its neighbours. A batch-*invariant* matmul fixes the reduction
order regardless of batch, so a row is always the same bits.
"""

from __future__ import annotations

from math import ceil

Vec = list[float]
Mat = list[list[float]]


def sum_flat(vals: Vec) -> float:
    """Left-to-right sum — one fixed order."""
    acc = 0.0
    for v in vals:
        acc += v
    return acc


def sum_split(vals: Vec, nsplits: int) -> float:
    """Sum by partitioning into ``nsplits`` contiguous groups, reducing each
    group left-to-right, then summing the partials. Mathematically equal to
    ``sum_flat``; in floating point, a different grouping is a different result.
    This models a split-K / tiled reduction."""
    n = len(vals)
    if nsplits <= 1 or n == 0:
        return sum_flat(vals)
    size = ceil(n / nsplits)
    partials: Vec = []
    for g in range(nsplits):
        chunk = vals[g * size : (g + 1) * size]
        if chunk:
            partials.append(sum_flat(chunk))
    return sum_flat(partials)


def splits_for_batch(batch_size: int) -> int:
    """A plausible-but-toxic heuristic: pick the K-split count from the batch
    size (fewer rows → split K further to keep every core busy). Because the
    result depends on this number, it makes the kernel batch-*variant*. This is
    the kind of shape-dependent tiling that real kernels do."""
    return max(1, 8 // max(1, batch_size))


def matmul_variant(x_batch: Mat, w: Mat) -> Mat:
    """logits = x @ wᵀ, reduced with a batch-size-dependent grouping.

    ``x_batch`` is ``[batch][d_in]``, ``w`` is ``[d_out][d_in]``. The K reduction
    uses ``splits_for_batch(len(x_batch))`` groups — so a row's output depends on
    how many rows share its batch. This is the bug.
    """
    nsplits = splits_for_batch(len(x_batch))
    out: Mat = []
    for x in x_batch:
        row = [sum_split([x[i] * w[j][i] for i in range(len(x))], nsplits) for j in range(len(w))]
        out.append(row)
    return out


def matmul_invariant(x_batch: Mat, w: Mat) -> Mat:
    """logits = x @ wᵀ with a reduction order fixed independently of the batch.

    Every dot product is reduced with the same grouping (a fixed split count)
    whatever the batch size, so a row's output is identical whether it runs alone
    or alongside a thousand others. That's the guarantee.
    """
    nsplits = FIXED_SPLITS
    out: Mat = []
    for x in x_batch:
        row = [sum_split([x[i] * w[j][i] for i in range(len(x))], nsplits) for j in range(len(w))]
        out.append(row)
    return out


# A single, batch-independent reduction shape. The value doesn't matter for
# correctness — only that it never depends on the batch.
FIXED_SPLITS = 4
