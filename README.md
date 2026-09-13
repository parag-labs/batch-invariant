# batch-invariant

**A prompt's answer must not depend on who else is in the batch — here's the bug
that breaks that, and a kernel that provably fixes it.**

Run a model twice on the same prompt with greedy decoding and you expect the same
token. In a batched inference server you don't always get it: the output can
change depending on **what else was being decoded alongside it** — not from
temperature, but because the *reduction order* inside a batched matmul shifted
with the batch shape. This repo shows that failure as a concrete **token flip**,
then fixes it and proves the fix over thousands of batches.

> It's a systems-correctness property most applied-AI work never checks: the same
> "the answer is a pure function of the input, not the schedule" discipline behind
> [deterministic-sim-testing](https://github.com/parag-labs/deterministic-sim-testing),
> applied to LLM inference.

```
$ batch-invariant demo
A single prompt, decoded two ways.

  variant    alone -> token 1   in a batch of 4 -> token 0  <-- same prompt, DIFFERENT token
  invariant  alone -> token 1   in a batch of 4 -> token 1  (stable)
```

---

## Why it happens

A matmul output is a sum over the K dimension, and in floating point the *order*
of that sum changes the last bits — more than the last bits when large terms
nearly cancel. Real kernels don't sum left-to-right; they **split the reduction
into chunks and combine partial sums**, and how many chunks is often chosen from
the batch shape (split K further when the batch is small, to keep every core
busy). So a row's rounding depends on its batch-mates. Occasionally that lands
exactly between two tokens' logits and the **argmax flips** — the same prompt
decodes to a different token.

```mermaid
flowchart TB
    P["prompt x"]:::blue --> V["variant kernel<br/>K-split = f(batch size)"]:::red
    P --> I["invariant kernel<br/>K-split fixed"]:::green
    V -->|"alone (8 splits)"| T1["token 1"]:::red
    V -->|"batched (2 splits)"| T0["token 0"]:::red
    I -->|"alone or batched"| S["token 1 — always"]:::green
    classDef blue fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a;
    classDef green fill:#dcfce7,stroke:#22c55e,color:#14532d;
    classDef red fill:#fee2e2,stroke:#ef4444,color:#7f1d1d;
```

## The fix

Make the reduction order a property of the *problem*, not of the *batch*.
`matmul_invariant` reduces every dot product with a **fixed** grouping regardless
of batch size, so a row's output is a pure function of that row and the weights —
bit-identical whether it runs alone or alongside a thousand others. The tests
assert exactly that, over thousands of random batch compositions, and the
constructed flip case shows the token that moved under the variant kernel staying
put under the invariant one.

## Use it

```bash
pip install -e ".[dev]"
pytest -q                 # 8 tests: invariance holds, variant breaks, token flips

batch-invariant demo      # show the flip under the variant kernel, gone under the invariant one
batch-invariant check     # stress: invariant kernel is batch-independent across 5000 batches
```

In code:

```python
from batch_invariant import matmul_invariant, flip_case, argmax, logits_alone, logits_in_batch

model, x, others = flip_case()
# variant: token depends on the batch; invariant: it doesn't
assert argmax(logits_alone(model, x, invariant=True)) == \
       argmax(logits_in_batch(model, x, others, invariant=True))
```

## Design decisions

- **Explicit float reductions, no numpy** — so the reduction order is exactly what
  the code says, making both the bug and the fix legible and testable.
- **A realistic variant heuristic** — "split K more when the batch is small" is a
  genuine load-balancing instinct, which is what makes it a fair cautionary tale.
- **A deterministic constructed flip** — token flips are rare (two logits tied
  within the reduction's rounding), so it's reproduced from a fixed seed rather
  than pretended common. Rare isn't good enough for a reproducibility guarantee.

Non-goals (not a GPU kernel, not hardware nondeterminism, not a full attention
stack) and the mechanism in full are in [DESIGN.md](DESIGN.md).

## Layout

```
batch-invariant/
├── src/batch_invariant/
│   ├── kernels.py     # sum_flat / sum_split, and the variant vs invariant matmul
│   ├── server.py      # a toy decode step (logits -> argmax token)
│   ├── demo_data.py   # the deterministic token-flip case
│   └── cli.py         # demo / check
└── tests/             # 8 tests: invariance, variant breakage, the token flip
```

## Reference

The batch-invariance problem was crisply framed by Thinking Machines' *Defeating
Nondeterminism in LLM Inference* (2025); this is a small, self-contained
reconstruction of the core mechanism and its fix.

## License

MIT — see [LICENSE](LICENSE).
