"""The property that matters: a row's output is independent of its batch.

These tests assert the invariant kernel guarantees it (a row is bit-identical
whether alone or batched, for every batch composition) and that the variant
kernel does not — including the headline case where the *decoded token* flips.
"""

from __future__ import annotations

import random

from batch_invariant import (
    Model,
    argmax,
    flip_case,
    logits_alone,
    logits_in_batch,
    matmul_invariant,
    matmul_variant,
    sum_flat,
    sum_split,
    token_depends_on_batch,
)
from batch_invariant.kernels import splits_for_batch


def _rand_rows(rng: random.Random, n: int, d: int) -> list[list[float]]:
    return [[rng.uniform(-1e6, 1e6) if i % 2 == 0 else rng.uniform(-1, 1) for i in range(d)] for _ in range(n)]


class TestReductions:
    def test_split_equals_flat_in_exact_math_but_not_in_float(self) -> None:
        vals = [1e16, 1.0, -1e16, 1.0]
        # Different groupings give different floating-point sums here.
        assert sum_split(vals, 2) != sum_flat(vals)

    def test_single_split_is_flat(self) -> None:
        vals = [0.1, 0.2, 0.3, 0.4]
        assert sum_split(vals, 1) == sum_flat(vals)

    def test_splits_for_batch_varies_with_batch_size(self) -> None:
        # This dependence is exactly what makes the variant kernel non-invariant.
        assert splits_for_batch(1) != splits_for_batch(4)


class TestInvariantKernel:
    def test_row_is_identical_alone_and_batched(self) -> None:
        rng = random.Random(1)
        w = _rand_rows(rng, 5, 16)
        model = Model(w)
        for _ in range(200):
            x = _rand_rows(rng, 1, 16)[0]
            others = _rand_rows(rng, rng.randint(0, 6), 16)
            alone = logits_alone(model, x, invariant=True)
            batched = logits_in_batch(model, x, others, invariant=True)
            assert alone == batched  # bit-for-bit

    def test_position_in_batch_does_not_matter(self) -> None:
        rng = random.Random(2)
        w = _rand_rows(rng, 4, 16)
        x = _rand_rows(rng, 1, 16)[0]
        a = _rand_rows(rng, 2, 16)
        # x first vs x among different neighbours -> same row output.
        first = matmul_invariant([x, *a], w)[0]
        alone = matmul_invariant([x], w)[0]
        assert first == alone


class TestVariantKernelIsBroken:
    def test_variant_row_changes_with_batch(self) -> None:
        rng = random.Random(3)
        w = _rand_rows(rng, 5, 16)
        model = Model(w)
        differed = False
        for _ in range(200):
            x = _rand_rows(rng, 1, 16)[0]
            others = _rand_rows(rng, 3, 16)
            if logits_alone(model, x, invariant=False) != logits_in_batch(model, x, others, invariant=False):
                differed = True
                break
        assert differed, "variant kernel should be batch-dependent"


class TestTokenFlip:
    def test_variant_flips_the_decoded_token(self) -> None:
        model, x, others = flip_case()
        assert token_depends_on_batch(model, x, others, invariant=False)
        # concretely: token 1 alone, token 0 batched
        assert argmax(logits_alone(model, x, invariant=False)) == 1
        assert argmax(logits_in_batch(model, x, others, invariant=False)) == 0

    def test_invariant_keeps_the_token_stable(self) -> None:
        model, x, others = flip_case()
        assert not token_depends_on_batch(model, x, others, invariant=True)
        assert argmax(logits_alone(model, x, invariant=True)) == argmax(
            logits_in_batch(model, x, others, invariant=True)
        )
