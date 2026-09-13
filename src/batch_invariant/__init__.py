"""batch-invariant — inference whose output doesn't depend on the batch.

A prompt's logits (and thus its decoded token) should never change because of
what else is being decoded alongside it. In real systems they can, because a
batched matmul's reduction order shifts with the batch shape. This package shows
the bug with a batch-*variant* kernel and fixes it with a batch-*invariant* one,
and proves the difference.

    from batch_invariant import matmul_invariant, matmul_variant, flip_case
"""

from .demo_data import flip_case
from .kernels import (
    FIXED_SPLITS,
    matmul_invariant,
    matmul_variant,
    splits_for_batch,
    sum_flat,
    sum_split,
)
from .server import (
    Model,
    argmax,
    logits_alone,
    logits_in_batch,
    token_depends_on_batch,
)

__all__ = [
    "FIXED_SPLITS",
    "Model",
    "argmax",
    "flip_case",
    "logits_alone",
    "logits_in_batch",
    "matmul_invariant",
    "matmul_variant",
    "splits_for_batch",
    "sum_flat",
    "sum_split",
    "token_depends_on_batch",
]
