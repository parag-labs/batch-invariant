/**
 * A toy decode step, to show batch-dependence changing an actual token.
 */

import { type Mat, type Vec, matmulInvariant, matmulVariant } from "./kernels";

/** The output projection: logits = x @ wᵀ. `w` is [vocab][d_in]. */
export class Model {
  readonly w: Mat;

  constructor(w: Mat) {
    this.w = w;
  }

  /** The number of output tokens (rows of `w`). */
  get vocab(): number {
    return this.w.length;
  }
}

function runKernel(invariant: boolean, batch: Mat, w: Mat): Mat {
  return invariant ? matmulInvariant(batch, w) : matmulVariant(batch, w);
}

/** Logits for `x` decoded on its own (a batch of one). */
export function logitsAlone(model: Model, x: Vec, invariant: boolean): Vec {
  return runKernel(invariant, [x], model.w)[0];
}

/**
 * Logits for `x` when it is decoded in a batch alongside `others`. `x` is placed
 * first; the returned row is its logits. With the invariant kernel this equals
 * {@link logitsAlone}; with the variant kernel it may not.
 */
export function logitsInBatch(model: Model, x: Vec, others: Mat, invariant: boolean): Vec {
  return runKernel(invariant, [x, ...others], model.w)[0];
}

/** The chosen next token: index of the largest logit, first-wins on ties. */
export function argmax(logits: Vec): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) {
      best = i;
    }
  }
  return best;
}

/**
 * Whether `x`'s decoded token changes between running alone and running in a
 * batch — the failure the invariant kernel makes impossible.
 */
export function tokenDependsOnBatch(model: Model, x: Vec, others: Mat, invariant: boolean): boolean {
  const alone = argmax(logitsAlone(model, x, invariant));
  const batched = argmax(logitsInBatch(model, x, others, invariant));
  return alone !== batched;
}
