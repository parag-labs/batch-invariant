import { describe, expect, it } from "vitest";

import { Model, argmax, logitsAlone, logitsInBatch, tokenDependsOnBatch } from "./server";
import { flipCase } from "./flipCase";
import { Rng, vecEqual } from "./testUtil";

describe("Model", () => {
  it("counts rows as its vocab", () => {
    const m = new Model([
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ]);
    expect(m.vocab).toBe(3);
  });
});

describe("argmax", () => {
  it("is first-wins on ties", () => {
    expect(argmax([1.0, 3.0, 3.0, 2.0])).toBe(1);
  });

  it("handles a single element", () => {
    expect(argmax([5.0])).toBe(0);
  });

  it("handles all-negative logits", () => {
    expect(argmax([-3.0, -1.0, -2.0])).toBe(1);
  });
});

describe("logits", () => {
  it("are identical alone and batched under the invariant kernel", () => {
    const rng = new Rng(7);
    const model = new Model(rng.mixedRows(3, 16));
    const x = rng.mixedRow(16);
    const others = rng.mixedRows(4, 16);
    expect(vecEqual(logitsAlone(model, x, true), logitsInBatch(model, x, others, true))).toBe(true);
  });
});

describe("tokenDependsOnBatch", () => {
  it("is false for the invariant kernel on the flip case", () => {
    const { model, x, others } = flipCase();
    expect(tokenDependsOnBatch(model, x, others, true)).toBe(false);
  });

  it("is true for the variant kernel on the flip case", () => {
    const { model, x, others } = flipCase();
    expect(tokenDependsOnBatch(model, x, others, false)).toBe(true);
  });
});
