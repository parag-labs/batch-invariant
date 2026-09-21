using System;
using System.Collections.Generic;

namespace BatchInvariant;

/// <summary>
/// The output projection: logits = x @ wᵀ. <c>W</c> is [vocab][d_in].
/// </summary>
public sealed class Model
{
    /// <summary>The output-projection weights, one row per vocabulary token.</summary>
    public double[][] W { get; }

    /// <summary>Create a model from its output-projection weights.</summary>
    public Model(double[][] w)
    {
        W = w;
    }

    /// <summary>The number of output tokens (rows of <see cref="W"/>).</summary>
    public int Vocab => W.Length;
}

/// <summary>
/// A toy decode step, to show batch-dependence changing an actual token.
/// </summary>
public static class Server
{
    private static double[][] RunKernel(bool invariant, double[][] batch, double[][] w)
    {
        return invariant ? Kernels.MatmulInvariant(batch, w) : Kernels.MatmulVariant(batch, w);
    }

    /// <summary>Logits for <paramref name="x"/> decoded on its own (a batch of one).</summary>
    public static double[] LogitsAlone(Model model, double[] x, bool invariant)
    {
        return RunKernel(invariant, new[] { x }, model.W)[0];
    }

    /// <summary>
    /// Logits for <paramref name="x"/> when it is decoded in a batch alongside
    /// <paramref name="others"/>. <paramref name="x"/> is placed first; the
    /// returned row is its logits. With the invariant kernel this equals
    /// <see cref="LogitsAlone"/>; with the variant kernel it may not.
    /// </summary>
    public static double[] LogitsInBatch(Model model, double[] x, double[][] others, bool invariant)
    {
        var batch = new double[others.Length + 1][];
        batch[0] = x;
        for (int i = 0; i < others.Length; i++)
        {
            batch[i + 1] = others[i];
        }
        return RunKernel(invariant, batch, model.W)[0];
    }

    /// <summary>
    /// The chosen next token: index of the largest logit, first-wins on ties.
    /// </summary>
    public static int Argmax(IReadOnlyList<double> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Count; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// Whether <paramref name="x"/>'s decoded token changes between running alone
    /// and running in a batch — the failure the invariant kernel makes impossible.
    /// </summary>
    public static bool TokenDependsOnBatch(Model model, double[] x, double[][] others, bool invariant)
    {
        int alone = Argmax(LogitsAlone(model, x, invariant));
        int batched = Argmax(LogitsInBatch(model, x, others, invariant));
        return alone != batched;
    }
}
