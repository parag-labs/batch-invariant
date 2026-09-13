"""A concrete case where batching changes the decoded token.

Constructed deterministically (seed 514) in the cancellation regime where
reduction order matters: two vocabulary rows whose logits are tied to within the
reduction's rounding error. Run on its own, the *variant* kernel decodes one
token; run in a batch of four, its reduction is grouped differently and it
decodes a *different* token — the same prompt, a different answer, because of who
else was in the batch. The invariant kernel decodes the same token either way.

This is the demo the tests and CLI use to show the bug is real, then show it
gone.
"""

from __future__ import annotations

import random

from .kernels import Mat, Vec
from .server import Model

D_IN = 16


def flip_case() -> tuple[Model, Vec, list[Vec]]:
    """Return ``(model, x, others)`` where the variant kernel flips ``x``'s token
    between running alone and running batched with ``others``."""
    rng = random.Random(514)
    x: Vec = [1.0] * D_IN  # products equal the weights, keeping it transparent

    def big_or_small_row() -> Vec:
        # Even coords are large (±1e7), odd coords tiny (±1): summing them mixes
        # magnitudes so the reduction order genuinely moves the result.
        return [rng.uniform(-1e7, 1e7) if i % 2 == 0 else rng.uniform(-1.0, 1.0) for i in range(D_IN)]

    w0 = big_or_small_row()
    w1 = list(w0)
    j = rng.randrange(1, D_IN, 2)  # nudge one small coordinate to make a near-tie
    w1[j] += rng.uniform(-1e-6, 1e-6)
    w: Mat = [w0, w1]

    others: list[Vec] = [big_or_small_row() for _ in range(3)]
    return Model(w), x, others
