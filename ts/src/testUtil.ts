/**
 * A tiny deterministic PRNG (mulberry32) for reproducible test data. It does not
 * need to match CPython's RNG — the invariance property holds for any input, so
 * tests only need a repeatable stream.
 */

import { type Mat, type Vec } from "./kernels";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Rng {
  private readonly next: () => number;

  constructor(seed: number) {
    this.next = mulberry32(seed);
  }

  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /**
   * A row with large even coordinates (±1e6) and tiny odd coordinates (±1), so
   * summing them mixes magnitudes and reduction order matters.
   */
  mixedRow(d: number): Vec {
    const row: Vec = [];
    for (let i = 0; i < d; i++) {
      row.push(i % 2 === 0 ? this.uniform(-1e6, 1e6) : this.uniform(-1.0, 1.0));
    }
    return row;
  }

  mixedRows(n: number, d: number): Mat {
    const rows: Mat = [];
    for (let i = 0; i < n; i++) {
      rows.push(this.mixedRow(d));
    }
    return rows;
  }
}

export function vecEqual(a: Vec, b: Vec): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}
