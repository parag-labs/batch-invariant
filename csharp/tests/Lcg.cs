using System.Collections.Generic;

namespace BatchInvariant.Tests;

/// <summary>
/// A tiny deterministic PRNG for reproducible test data. It does not need to
/// match CPython's RNG — the invariance property holds for any input, so tests
/// only need a repeatable stream.
/// </summary>
internal sealed class Lcg
{
    private ulong _state;

    public Lcg(ulong seed)
    {
        _state = seed + 0x9e3779b97f4a7c15UL;
    }

    private double NextDouble()
    {
        _state = (_state * 6364136223846793005UL) + 1442695040888963407UL;
        return (_state >> 11) / (double)(1UL << 53);
    }

    public double Uniform(double lo, double hi)
    {
        return lo + ((hi - lo) * NextDouble());
    }

    public double[] MixedRow(int d)
    {
        var row = new double[d];
        for (int i = 0; i < d; i++)
        {
            row[i] = i % 2 == 0 ? Uniform(-1e6, 1e6) : Uniform(-1.0, 1.0);
        }
        return row;
    }

    public double[][] MixedRows(int n, int d)
    {
        var rows = new double[n][];
        for (int i = 0; i < n; i++)
        {
            rows[i] = MixedRow(d);
        }
        return rows;
    }

    public static bool VecEqual(IReadOnlyList<double> a, IReadOnlyList<double> b)
    {
        if (a.Count != b.Count)
        {
            return false;
        }
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }
        return true;
    }
}
