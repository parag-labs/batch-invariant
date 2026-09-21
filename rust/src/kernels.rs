//! Reduction kernels — where batch-dependence sneaks in.

/// A single, batch-independent reduction shape. The value does not matter for
/// correctness — only that it never depends on the batch.
pub const FIXED_SPLITS: usize = 4;

/// Left-to-right sum of `vals` — one fixed order.
pub fn sum_flat(vals: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in vals {
        acc += v;
    }
    acc
}

/// Sum `vals` by partitioning them into `nsplits` contiguous groups, reducing
/// each group left-to-right, then summing the partials. Mathematically equal to
/// [`sum_flat`]; in floating point, a different grouping is a different result.
/// This models a split-K / tiled reduction.
pub fn sum_split(vals: &[f64], nsplits: i64) -> f64 {
    let n = vals.len();
    if nsplits <= 1 || n == 0 {
        return sum_flat(vals);
    }
    let nsplits = nsplits as usize;
    let size = n.div_ceil(nsplits); // ceil(n / nsplits)
    let mut partials: Vec<f64> = Vec::with_capacity(nsplits);
    for g in 0..nsplits {
        let start = g * size;
        if start >= n {
            continue;
        }
        let end = usize::min(start + size, n);
        partials.push(sum_flat(&vals[start..end]));
    }
    sum_flat(&partials)
}

/// Pick the K-split count from the batch size (fewer rows → split K further to
/// keep every core busy). Because the result depends on this number, it makes
/// the kernel batch-variant. This models shape-dependent tiling that real
/// kernels do.
pub fn splits_for_batch(batch_size: usize) -> i64 {
    let d = batch_size.max(1) as i64;
    (8 / d).max(1)
}

fn matmul(x_batch: &[Vec<f64>], w: &[Vec<f64>], nsplits: i64) -> Vec<Vec<f64>> {
    x_batch
        .iter()
        .map(|x| {
            w.iter()
                .map(|wj| {
                    let products: Vec<f64> = x.iter().zip(wj.iter()).map(|(a, b)| a * b).collect();
                    sum_split(&products, nsplits)
                })
                .collect()
        })
        .collect()
}

/// `logits = x @ wᵀ`, reduced with a batch-size-dependent grouping. `x_batch` is
/// `[batch][d_in]`, `w` is `[d_out][d_in]`. The K reduction uses
/// `splits_for_batch(x_batch.len())` groups — so a row's output depends on how
/// many rows share its batch. This is the bug.
pub fn matmul_variant(x_batch: &[Vec<f64>], w: &[Vec<f64>]) -> Vec<Vec<f64>> {
    matmul(x_batch, w, splits_for_batch(x_batch.len()))
}

/// `logits = x @ wᵀ` with a reduction order fixed independently of the batch.
/// Every dot product is reduced with the same grouping whatever the batch size,
/// so a row's output is identical whether it runs alone or alongside a thousand
/// others. That is the guarantee.
pub fn matmul_invariant(x_batch: &[Vec<f64>], w: &[Vec<f64>]) -> Vec<Vec<f64>> {
    matmul(x_batch, w, FIXED_SPLITS as i64)
}
