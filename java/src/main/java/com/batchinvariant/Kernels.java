package com.batchinvariant;

/**
 * Reduction kernels — where batch-dependence sneaks in.
 *
 * <p>A matmul output is a dot product: a sum over the K dimension. In exact
 * arithmetic the order of that sum does not matter. In floating point it does —
 * {@code (a + b) + c} can differ from {@code a + (b + c)} in the last bit, and
 * far more when large terms nearly cancel. Production kernels split the K
 * reduction into chunks and combine partial sums, and how many chunks is often
 * chosen from the batch shape, so a row's result can change depending on who
 * else is in the batch.
 */
public final class Kernels {

    /**
     * A single, batch-independent reduction shape. The value does not matter for
     * correctness — only that it never depends on the batch.
     */
    public static final int FIXED_SPLITS = 4;

    private Kernels() {
    }

    /** Returns the left-to-right sum of {@code vals} — one fixed order. */
    public static double sumFlat(double[] vals) {
        double acc = 0.0;
        for (double v : vals) {
            acc += v;
        }
        return acc;
    }

    /**
     * Sums {@code vals} by partitioning them into {@code nsplits} contiguous
     * groups, reducing each group left-to-right, then summing the partials.
     * Mathematically equal to {@link #sumFlat}; in floating point, a different
     * grouping is a different result. This models a split-K / tiled reduction.
     */
    public static double sumSplit(double[] vals, int nsplits) {
        int n = vals.length;
        if (nsplits <= 1 || n == 0) {
            return sumFlat(vals);
        }
        int size = (n + nsplits - 1) / nsplits; // ceil(n / nsplits)
        double[] partials = new double[nsplits];
        int count = 0;
        for (int g = 0; g < nsplits; g++) {
            int start = g * size;
            if (start >= n) {
                continue;
            }
            int end = Math.min(start + size, n);
            double acc = 0.0;
            for (int i = start; i < end; i++) {
                acc += vals[i];
            }
            partials[count++] = acc;
        }
        double total = 0.0;
        for (int i = 0; i < count; i++) {
            total += partials[i];
        }
        return total;
    }

    /**
     * Picks the K-split count from the batch size (fewer rows → split K further
     * to keep every core busy). Because the result depends on this number, it
     * makes the kernel batch-variant.
     */
    public static int splitsForBatch(int batchSize) {
        int d = Math.max(1, batchSize);
        return Math.max(1, 8 / d);
    }

    private static double[][] matmul(double[][] xBatch, double[][] w, int nsplits) {
        double[][] out = new double[xBatch.length][];
        for (int r = 0; r < xBatch.length; r++) {
            double[] x = xBatch[r];
            double[] row = new double[w.length];
            for (int j = 0; j < w.length; j++) {
                double[] products = new double[x.length];
                for (int i = 0; i < x.length; i++) {
                    products[i] = x[i] * w[j][i];
                }
                row[j] = sumSplit(products, nsplits);
            }
            out[r] = row;
        }
        return out;
    }

    /**
     * logits = x @ wᵀ, reduced with a batch-size-dependent grouping.
     * {@code xBatch} is [batch][d_in], {@code w} is [d_out][d_in]. The K
     * reduction uses {@code splitsForBatch(xBatch.length)} groups — so a row's
     * output depends on how many rows share its batch. This is the bug.
     */
    public static double[][] matmulVariant(double[][] xBatch, double[][] w) {
        return matmul(xBatch, w, splitsForBatch(xBatch.length));
    }

    /**
     * logits = x @ wᵀ with a reduction order fixed independently of the batch.
     * Every dot product is reduced with the same grouping whatever the batch
     * size, so a row's output is identical whether it runs alone or alongside a
     * thousand others. That is the guarantee.
     */
    public static double[][] matmulInvariant(double[][] xBatch, double[][] w) {
        return matmul(xBatch, w, FIXED_SPLITS);
    }
}
