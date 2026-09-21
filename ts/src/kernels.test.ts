import { describe, expect, it } from "vitest";

import {
  FIXED_SPLITS,
  matmulInvariant,
  splitsForBatch,
  sumFlat,
  sumSplit,
} from "./kernels";
import { Model, logitsAlone, logitsInBatch } from "./server";
import { flipCase } from "./flipCase";
import { Rng, vecEqual } from "./testUtil";

describe("sumFlat", () => {
  it("is zero for an empty vector", () => {
    expect(sumFlat([])).toBe(0.0);
  });

  it("returns the single element", () => {
    expect(sumFlat([42.5])).toBe(42.5);
  });

  it("sums strictly left-to-right", () => {
    expect(sumFlat([0.1, 0.2, 0.3, 0.4])).toBe(0.1 + 0.2 + 0.3 + 0.4);
  });
});

describe("sumSplit", () => {
  it("differs from sumFlat in floating point when grouping changes", () => {
    const vals = [1e16, 1.0, -1e16, 1.0];
    expect(sumSplit(vals, 2)).not.toBe(sumFlat(vals));
  });

  it("equals sumFlat with a single split", () => {
    const vals = [0.1, 0.2, 0.3, 0.4];
    expect(sumSplit(vals, 1)).toBe(sumFlat(vals));
  });

  it("equals sumFlat for zero or negative splits", () => {
    const vals = [0.1, 0.2, 0.3];
    expect(sumSplit(vals, 0)).toBe(sumFlat(vals));
    expect(sumSplit(vals, -3)).toBe(sumFlat(vals));
  });

  it("is zero for an empty vector", () => {
    expect(sumSplit([], 4)).toBe(0.0);
  });

  it("handles more splits than elements", () => {
    expect(sumSplit([1.0, 2.0, 3.0], 8)).toBe(6.0);
  });
});

describe("splitsForBatch", () => {
  it.each([
    [0, 8],
    [1, 8],
    [2, 4],
    [4, 2],
    [8, 1],
    [9, 1],
    [16, 1],
  ])("maps batch %i to %i splits", (batch, expected) => {
    expect(splitsForBatch(batch)).toBe(expected);
  });

  it("varies with batch size", () => {
    expect(splitsForBatch(1)).not.toBe(splitsForBatch(4));
  });
});

describe("constants", () => {
  it("fixes the invariant split count at four", () => {
    expect(FIXED_SPLITS).toBe(4);
  });
});

describe("matmul", () => {
  it("computes the right products and shape", () => {
    const w = [
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0],
    ];
    const out = matmulInvariant(
      [
        [1.0, 1.0],
        [2.0, 2.0],
      ],
      w,
    );
    expect(out.length).toBe(2);
    expect(out[0]).toEqual([3.0, 7.0, 11.0]);
  });
});

describe("invariant kernel", () => {
  it("gives an identical row alone and batched over many batches", () => {
    const rng = new Rng(1);
    const model = new Model(rng.mixedRows(5, 16));
    for (let trial = 0; trial < 200; trial++) {
      const x = rng.mixedRow(16);
      const others = rng.mixedRows(trial % 7, 16);
      const alone = logitsAlone(model, x, true);
      const batched = logitsInBatch(model, x, others, true);
      expect(vecEqual(alone, batched)).toBe(true);
    }
  });
});

describe("variant kernel", () => {
  it("changes a row with the batch for the flip case", () => {
    const { model, x, others } = flipCase();
    expect(vecEqual(logitsAlone(model, x, false), logitsInBatch(model, x, others, false))).toBe(
      false,
    );
  });
});
