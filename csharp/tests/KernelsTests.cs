using Xunit;

namespace BatchInvariant.Tests;

public class KernelsTests
{
    [Fact]
    public void SumFlatEmptyIsZero()
    {
        Assert.Equal(0.0, Kernels.SumFlat(new double[] { }));
    }

    [Fact]
    public void SumFlatSingleElement()
    {
        Assert.Equal(42.5, Kernels.SumFlat(new[] { 42.5 }));
    }

    [Fact]
    public void SumFlatIsLeftToRight()
    {
        Assert.Equal(((0.1 + 0.2) + 0.3) + 0.4, Kernels.SumFlat(new[] { 0.1, 0.2, 0.3, 0.4 }));
    }

    [Fact]
    public void SplitDiffersFromFlatInFloat()
    {
        var vals = new[] { 1e16, 1.0, -1e16, 1.0 };
        Assert.NotEqual(Kernels.SumFlat(vals), Kernels.SumSplit(vals, 2));
    }

    [Fact]
    public void SingleSplitIsFlat()
    {
        var vals = new[] { 0.1, 0.2, 0.3, 0.4 };
        Assert.Equal(Kernels.SumFlat(vals), Kernels.SumSplit(vals, 1));
    }

    [Fact]
    public void ZeroOrNegativeSplitsIsFlat()
    {
        var vals = new[] { 0.1, 0.2, 0.3 };
        Assert.Equal(Kernels.SumFlat(vals), Kernels.SumSplit(vals, 0));
        Assert.Equal(Kernels.SumFlat(vals), Kernels.SumSplit(vals, -3));
    }

    [Fact]
    public void SumSplitEmptyIsZero()
    {
        Assert.Equal(0.0, Kernels.SumSplit(new double[] { }, 4));
    }

    [Fact]
    public void SumSplitMoreSplitsThanElements()
    {
        Assert.Equal(6.0, Kernels.SumSplit(new[] { 1.0, 2.0, 3.0 }, 8));
    }

    [Theory]
    [InlineData(0, 8)]
    [InlineData(1, 8)]
    [InlineData(2, 4)]
    [InlineData(4, 2)]
    [InlineData(8, 1)]
    [InlineData(9, 1)]
    [InlineData(16, 1)]
    public void SplitsForBatchKnownValues(int batch, int expected)
    {
        Assert.Equal(expected, Kernels.SplitsForBatch(batch));
    }

    [Fact]
    public void SplitsForBatchVariesWithBatchSize()
    {
        Assert.NotEqual(Kernels.SplitsForBatch(1), Kernels.SplitsForBatch(4));
    }

    [Fact]
    public void FixedSplitsIsFour()
    {
        Assert.Equal(4, Kernels.FixedSplits);
    }

    [Fact]
    public void MatmulProductsAndShape()
    {
        var w = new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 }, new[] { 5.0, 6.0 } };
        var outp = Kernels.MatmulInvariant(new[] { new[] { 1.0, 1.0 }, new[] { 2.0, 2.0 } }, w);
        Assert.Equal(2, outp.Length);
        Assert.Equal(new[] { 3.0, 7.0, 11.0 }, outp[0]);
    }

    [Fact]
    public void InvariantRowIdenticalAloneAndBatched()
    {
        var rng = new Lcg(1);
        var model = new Model(rng.MixedRows(5, 16));
        for (int trial = 0; trial < 200; trial++)
        {
            var x = rng.MixedRow(16);
            var others = rng.MixedRows(trial % 7, 16);
            var alone = Server.LogitsAlone(model, x, invariant: true);
            var batched = Server.LogitsInBatch(model, x, others, invariant: true);
            Assert.True(Lcg.VecEqual(alone, batched), $"invariant kernel differed on trial {trial}");
        }
    }

    [Fact]
    public void VariantRowChangesWithBatch()
    {
        var (model, x, others) = FlipCase.Build();
        Assert.False(Lcg.VecEqual(
            Server.LogitsAlone(model, x, invariant: false),
            Server.LogitsInBatch(model, x, others, invariant: false)));
    }
}
