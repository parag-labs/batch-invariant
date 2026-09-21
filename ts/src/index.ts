/**
 * batch-invariant — inference whose output doesn't depend on the batch.
 *
 * A prompt's logits (and thus its decoded token) should never change because of
 * what else is being decoded alongside it. In real systems they can, because a
 * batched matmul's reduction order shifts with the batch shape. This package
 * shows the bug with a batch-variant kernel and fixes it with a batch-invariant
 * one, and proves the difference.
 */

export {
  type Mat,
  type Vec,
  FIXED_SPLITS,
  matmulInvariant,
  matmulVariant,
  splitsForBatch,
  sumFlat,
  sumSplit,
} from "./kernels";
export {
  Model,
  argmax,
  logitsAlone,
  logitsInBatch,
  tokenDependsOnBatch,
} from "./server";
export { type FlipCase, D_IN, flipCase } from "./flipCase";
