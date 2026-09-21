/**
 * Reduction kernels — where batch-dependence sneaks in.
 *
 * A matmul output is a dot product: a sum over the K dimension. In exact
 * arithmetic the order of that sum does not matter. In floating point it does —
 * `(a + b) + c` can differ from `a + (b + c)` in the last bit, and far more when
 * large terms nearly cancel. Production kernels split the K reduction into
 * chunks and combine partial sums, and how many chunks is often chosen from the
 * batch shape, so a row's result can change depending on who else is in the
 * batch.
 */

/** A dense vector of float64 values. */
export type Vec = number[];

/** A dense matrix stored as an array of rows. */
export type Mat = number[][];

/**
 * A single, batch-independent reduction shape. The value does not matter for
 * correctness — only that it never depends on the batch.
 */
export const FIXED_SPLITS = 4;

/** Left-to-right sum of `vals` — one fixed order. */
export function sumFlat(vals: Vec): number {
  let acc = 0.0;
  for (const v of vals) {
    acc += v;
  }
  return acc;
}

/**
 * Sum `vals` by partitioning them into `nsplits` contiguous groups, reducing
 * each group left-to-right, then summing the partials. Mathematically equal to
 * {@link sumFlat}; in floating point, a different grouping is a different
 * result. This models a split-K / tiled reduction.
 */
export function sumSplit(vals: Vec, nsplits: number): number {
  const n = vals.length;
  if (nsplits <= 1 || n === 0) {
    return sumFlat(vals);
  }
  const size = Math.ceil(n / nsplits);
  const partials: Vec = [];
  for (let g = 0; g < nsplits; g++) {
    const start = g * size;
    if (start >= n) {
      continue;
    }
    const end = Math.min(start + size, n);
    partials.push(sumFlat(vals.slice(start, end)));
  }
  return sumFlat(partials);
}

/**
 * Pick the K-split count from the batch size (fewer rows → split K further to
 * keep every core busy). Because the result depends on this number, it makes the
 * kernel batch-variant.
 */
export function splitsForBatch(batchSize: number): number {
  const d = Math.max(1, batchSize);
  return Math.max(1, Math.floor(8 / d));
}

function matmul(xBatch: Mat, w: Mat, nsplits: number): Mat {
  return xBatch.map((x) =>
    w.map((wj) => {
      const products = x.map((xi, i) => xi * wj[i]);
      return sumSplit(products, nsplits);
    }),
  );
}

/**
 * logits = x @ wᵀ, reduced with a batch-size-dependent grouping. `xBatch` is
 * [batch][d_in], `w` is [d_out][d_in]. The K reduction uses
 * `splitsForBatch(xBatch.length)` groups — so a row's output depends on how many
 * rows share its batch. This is the bug.
 */
export function matmulVariant(xBatch: Mat, w: Mat): Mat {
  return matmul(xBatch, w, splitsForBatch(xBatch.length));
}

/**
 * logits = x @ wᵀ with a reduction order fixed independently of the batch. Every
 * dot product is reduced with the same grouping whatever the batch size, so a
 * row's output is identical whether it runs alone or alongside a thousand
 * others. That is the guarantee.
 */
export function matmulInvariant(xBatch: Mat, w: Mat): Mat {
  return matmul(xBatch, w, FIXED_SPLITS);
}
