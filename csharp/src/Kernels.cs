using System;
using System.Collections.Generic;

namespace BatchInvariant;

/// <summary>
/// Reduction kernels — where batch-dependence sneaks in.
///
/// A matmul output is a dot product: a sum over the K dimension. In exact
/// arithmetic the order of that sum does not matter. In floating point it does —
/// (a + b) + c can differ from a + (b + c) in the last bit, and far more when
/// large terms nearly cancel. Production kernels split the K reduction into
/// chunks and combine partial sums, and how many chunks is often chosen from the
/// batch shape, so a row's result can change depending on who else is in the
/// batch.
/// </summary>
public static class Kernels
{
    /// <summary>
    /// A single, batch-independent reduction shape. The value does not matter
    /// for correctness — only that it never depends on the batch.
    /// </summary>
    public const int FixedSplits = 4;

    /// <summary>Left-to-right sum of <paramref name="vals"/> — one fixed order.</summary>
    public static double SumFlat(IReadOnlyList<double> vals)
    {
        double acc = 0.0;
        for (int i = 0; i < vals.Count; i++)
        {
            acc += vals[i];
        }
        return acc;
    }

    /// <summary>
    /// Sum <paramref name="vals"/> by partitioning them into
    /// <paramref name="nsplits"/> contiguous groups, reducing each group
    /// left-to-right, then summing the partials. Mathematically equal to
    /// <see cref="SumFlat"/>; in floating point, a different grouping is a
    /// different result. This models a split-K / tiled reduction.
    /// </summary>
    public static double SumSplit(IReadOnlyList<double> vals, int nsplits)
    {
        int n = vals.Count;
        if (nsplits <= 1 || n == 0)
        {
            return SumFlat(vals);
        }
        int size = (n + nsplits - 1) / nsplits; // ceil(n / nsplits)
        var partials = new List<double>(nsplits);
        for (int g = 0; g < nsplits; g++)
        {
            int start = g * size;
            if (start >= n)
            {
                continue;
            }
            int end = Math.Min(start + size, n);
            double acc = 0.0;
            for (int i = start; i < end; i++)
            {
                acc += vals[i];
            }
            partials.Add(acc);
        }
        return SumFlat(partials);
    }

    /// <summary>
    /// Pick the K-split count from the batch size (fewer rows → split K further
    /// to keep every core busy). Because the result depends on this number, it
    /// makes the kernel batch-variant.
    /// </summary>
    public static int SplitsForBatch(int batchSize)
    {
        int d = Math.Max(1, batchSize);
        return Math.Max(1, 8 / d);
    }

    private static double[][] Matmul(double[][] xBatch, double[][] w, int nsplits)
    {
        var output = new double[xBatch.Length][];
        for (int r = 0; r < xBatch.Length; r++)
        {
            double[] x = xBatch[r];
            var row = new double[w.Length];
            for (int j = 0; j < w.Length; j++)
            {
                var products = new double[x.Length];
                for (int i = 0; i < x.Length; i++)
                {
                    products[i] = x[i] * w[j][i];
                }
                row[j] = SumSplit(products, nsplits);
            }
            output[r] = row;
        }
        return output;
    }

    /// <summary>
    /// logits = x @ wᵀ, reduced with a batch-size-dependent grouping.
    /// <paramref name="xBatch"/> is [batch][d_in], <paramref name="w"/> is
    /// [d_out][d_in]. The K reduction uses <c>SplitsForBatch(xBatch.Length)</c>
    /// groups — so a row's output depends on how many rows share its batch. This
    /// is the bug.
    /// </summary>
    public static double[][] MatmulVariant(double[][] xBatch, double[][] w)
    {
        return Matmul(xBatch, w, SplitsForBatch(xBatch.Length));
    }

    /// <summary>
    /// logits = x @ wᵀ with a reduction order fixed independently of the batch.
    /// Every dot product is reduced with the same grouping whatever the batch
    /// size, so a row's output is identical whether it runs alone or alongside a
    /// thousand others. That is the guarantee.
    /// </summary>
    public static double[][] MatmulInvariant(double[][] xBatch, double[][] w)
    {
        return Matmul(xBatch, w, FixedSplits);
    }
}
