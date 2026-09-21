package com.batchinvariant;

/**
 * A tiny deterministic PRNG for reproducible test data. It does not need to
 * match CPython's RNG — the invariance property holds for any input, so tests
 * only need a repeatable stream.
 */
final class Lcg {

    private long state;

    Lcg(long seed) {
        this.state = seed + 0x9e3779b97f4a7c15L;
    }

    private double nextDouble() {
        state = (state * 6364136223846793005L) + 1442695040888963407L;
        return (state >>> 11) / (double) (1L << 53);
    }

    double uniform(double lo, double hi) {
        return lo + ((hi - lo) * nextDouble());
    }

    double[] mixedRow(int d) {
        double[] row = new double[d];
        for (int i = 0; i < d; i++) {
            row[i] = i % 2 == 0 ? uniform(-1e6, 1e6) : uniform(-1.0, 1.0);
        }
        return row;
    }

    double[][] mixedRows(int n, int d) {
        double[][] rows = new double[n][];
        for (int i = 0; i < n; i++) {
            rows[i] = mixedRow(d);
        }
        return rows;
    }

    static boolean vecEqual(double[] a, double[] b) {
        if (a.length != b.length) {
            return false;
        }
        for (int i = 0; i < a.length; i++) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }
}
