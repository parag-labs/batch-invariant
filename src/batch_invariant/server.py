"""A toy decode step, to show batch-dependence changing an actual token.

An LLM picks its next token by taking the argmax of the output logits. If the
logits for a prompt shift depending on what else is in the batch — because the
reduction order shifted — then the *same prompt* can decode to a *different
token* depending on its batch-mates. That's not a rounding curiosity; it's
non-reproducible generation. This module runs that experiment with the variant
and invariant kernels so the difference is visible as a flipped token.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kernels import Mat, Vec, matmul_invariant, matmul_variant


@dataclass(frozen=True)
class Model:
    """Just the output projection: logits = x @ wᵀ. ``w`` is ``[vocab][d_in]``."""

    w: Mat

    @property
    def vocab(self) -> int:
        return len(self.w)


def _kernel(invariant: bool):
    return matmul_invariant if invariant else matmul_variant


def logits_alone(model: Model, x: Vec, invariant: bool) -> Vec:
    """Logits for ``x`` decoded on its own (batch of one)."""
    return _kernel(invariant)([x], model.w)[0]


def logits_in_batch(model: Model, x: Vec, others: list[Vec], invariant: bool) -> Vec:
    """Logits for ``x`` when it's decoded in a batch alongside ``others``.

    ``x`` is placed first; the returned row is its logits. With the invariant
    kernel this equals :func:`logits_alone`; with the variant kernel it may not.
    """
    batch = [x, *others]
    return _kernel(invariant)(batch, model.w)[0]


def argmax(logits: Vec) -> int:
    """The chosen next token: index of the largest logit, first-wins on ties."""
    best = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[best]:
            best = i
    return best


def token_depends_on_batch(model: Model, x: Vec, others: list[Vec], invariant: bool) -> bool:
    """True if ``x``'s decoded token changes between running alone and running in
    a batch — the failure we want to prove impossible under the invariant kernel."""
    alone = argmax(logits_alone(model, x, invariant))
    batched = argmax(logits_in_batch(model, x, others, invariant))
    return alone != batched
