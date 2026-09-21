using Xunit;

namespace BatchInvariant.Tests;

public class ServerTests
{
    [Fact]
    public void ModelVocabCountsRows()
    {
        var m = new Model(new[] { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 }, new[] { 5.0, 6.0 } });
        Assert.Equal(3, m.Vocab);
    }

    [Fact]
    public void ArgmaxFirstWinsOnTies()
    {
        Assert.Equal(1, Server.Argmax(new[] { 1.0, 3.0, 3.0, 2.0 }));
    }

    [Fact]
    public void ArgmaxSingleElement()
    {
        Assert.Equal(0, Server.Argmax(new[] { 5.0 }));
    }

    [Fact]
    public void ArgmaxAllNegative()
    {
        Assert.Equal(1, Server.Argmax(new[] { -3.0, -1.0, -2.0 }));
    }

    [Fact]
    public void LogitsAloneEqualsInvariantBatched()
    {
        var rng = new Lcg(7);
        var model = new Model(rng.MixedRows(3, 16));
        var x = rng.MixedRow(16);
        var others = rng.MixedRows(4, 16);
        Assert.True(Lcg.VecEqual(
            Server.LogitsAlone(model, x, invariant: true),
            Server.LogitsInBatch(model, x, others, invariant: true)));
    }

    [Fact]
    public void TokenStableUnderInvariantFlipCase()
    {
        var (model, x, others) = FlipCase.Build();
        Assert.False(Server.TokenDependsOnBatch(model, x, others, invariant: true));
    }

    [Fact]
    public void TokenMovesUnderVariantFlipCase()
    {
        var (model, x, others) = FlipCase.Build();
        Assert.True(Server.TokenDependsOnBatch(model, x, others, invariant: false));
    }
}
