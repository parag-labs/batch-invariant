import { describe, expect, it } from "vitest";

import { argmax, logitsAlone, logitsInBatch } from "./server";
import { D_IN, flipCase } from "./flipCase";

describe("flipCase", () => {
  it("has the expected shape", () => {
    const { model, x, others } = flipCase();
    expect(model.vocab).toBe(2);
    expect(x.length).toBe(D_IN);
    expect(others.length).toBe(3);
    expect(x.every((v) => v === 1.0)).toBe(true);
  });

  it("flips the decoded token under the variant kernel", () => {
    const { model, x, others } = flipCase();
    expect(argmax(logitsAlone(model, x, false))).toBe(1);
    expect(argmax(logitsInBatch(model, x, others, false))).toBe(0);
  });

  it("keeps the token stable under the invariant kernel", () => {
    const { model, x, others } = flipCase();
    expect(argmax(logitsAlone(model, x, true))).toBe(1);
    expect(argmax(logitsInBatch(model, x, others, true))).toBe(1);
  });
});
