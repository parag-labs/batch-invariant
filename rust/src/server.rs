//! A toy decode step, to show batch-dependence changing an actual token.

use crate::kernels::{matmul_invariant, matmul_variant};

/// The output projection: `logits = x @ wᵀ`. `w` is `[vocab][d_in]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub w: Vec<Vec<f64>>,
}

impl Model {
    /// Build a model from its output-projection weights.
    pub fn new(w: Vec<Vec<f64>>) -> Self {
        Model { w }
    }

    /// Number of output tokens (rows of `w`).
    pub fn vocab(&self) -> usize {
        self.w.len()
    }
}

fn run_kernel(invariant: bool, batch: &[Vec<f64>], w: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if invariant {
        matmul_invariant(batch, w)
    } else {
        matmul_variant(batch, w)
    }
}

/// Logits for `x` decoded on its own (a batch of one).
pub fn logits_alone(model: &Model, x: &[f64], invariant: bool) -> Vec<f64> {
    let batch = vec![x.to_vec()];
    run_kernel(invariant, &batch, &model.w).swap_remove(0)
}

/// Logits for `x` when it is decoded in a batch alongside `others`. `x` is placed
/// first; the returned row is its logits. With the invariant kernel this equals
/// [`logits_alone`]; with the variant kernel it may not.
pub fn logits_in_batch(model: &Model, x: &[f64], others: &[Vec<f64>], invariant: bool) -> Vec<f64> {
    let mut batch = Vec::with_capacity(others.len() + 1);
    batch.push(x.to_vec());
    batch.extend(others.iter().cloned());
    run_kernel(invariant, &batch, &model.w).swap_remove(0)
}

/// The chosen next token: index of the largest logit, first-wins on ties.
pub fn argmax(logits: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    best
}

/// Whether `x`'s decoded token changes between running alone and running in a
/// batch — the failure the invariant kernel makes impossible.
pub fn token_depends_on_batch(
    model: &Model,
    x: &[f64],
    others: &[Vec<f64>],
    invariant: bool,
) -> bool {
    let alone = argmax(&logits_alone(model, x, invariant));
    let batched = argmax(&logits_in_batch(model, x, others, invariant));
    alone != batched
}
