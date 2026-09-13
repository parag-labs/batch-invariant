"""Command-line demo for batch-invariant.

  batch-invariant demo      # show the token flip under the variant kernel, gone under the invariant one
  batch-invariant check     # stress: assert the invariant kernel is batch-independent
"""

from __future__ import annotations

import argparse
import random

from .demo_data import flip_case
from .kernels import splits_for_batch
from .server import Model, argmax, logits_alone, logits_in_batch


def _demo(_: argparse.Namespace) -> None:
    model, x, others = flip_case()
    print("A single prompt, decoded two ways.\n")
    for invariant in (False, True):
        name = "invariant" if invariant else "variant"
        alone = argmax(logits_alone(model, x, invariant=invariant))
        batched = argmax(logits_in_batch(model, x, others, invariant=invariant))
        flag = "  <-- same prompt, DIFFERENT token" if alone != batched else "  (stable)"
        print(f"  {name:<10} alone -> token {alone}   in a batch of {len(others) + 1} -> token {batched}{flag}")
    print(
        f"\nThe variant kernel splits its K-reduction into "
        f"{splits_for_batch(1)} groups alone but {splits_for_batch(4)} in the batch,"
        "\nso the rounding differs and the token flips. The invariant kernel fixes the"
        "\nreduction order regardless of batch, so the token can't move."
    )


def _check(_: argparse.Namespace) -> None:
    rng = random.Random(0)
    w = [[rng.uniform(-1e6, 1e6) if i % 2 == 0 else rng.uniform(-1, 1) for i in range(16)] for _ in range(6)]
    model = Model(w)
    trials = 5000
    for _ in range(trials):
        x = [rng.uniform(-1e6, 1e6) if i % 2 == 0 else rng.uniform(-1, 1) for i in range(16)]
        others = [[rng.uniform(-1, 1) for _ in range(16)] for _ in range(rng.randint(0, 8))]
        if logits_alone(model, x, invariant=True) != logits_in_batch(model, x, others, invariant=True):
            print("FAIL — invariant kernel was not batch-independent")
            raise SystemExit(1)
    print(f"PASS — invariant kernel is batch-independent across {trials} random batches.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="batch-invariant", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="show the token flip and its fix")
    sub.add_parser("check", help="stress the invariance guarantee")
    args = p.parse_args(argv)
    {"demo": _demo, "check": _check}[args.cmd](args)


if __name__ == "__main__":
    main()
