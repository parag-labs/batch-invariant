using Xunit;

namespace BatchInvariant.Tests;

public class FlipCaseTests
{
    [Fact]
    public void FlipCaseShape()
    {
        var (model, x, others) = FlipCase.Build();
        Assert.Equal(2, model.Vocab);
        Assert.Equal(FlipCase.DIn, x.Length);
        Assert.Equal(3, others.Length);
        Assert.All(x, v => Assert.Equal(1.0, v));
    }

    [Fact]
    public void VariantFlipsTheDecodedToken()
    {
        var (model, x, others) = FlipCase.Build();
        Assert.Equal(1, Server.Argmax(Server.LogitsAlone(model, x, invariant: false)));
        Assert.Equal(0, Server.Argmax(Server.LogitsInBatch(model, x, others, invariant: false)));
    }

    [Fact]
    public void InvariantKeepsTheTokenStable()
    {
        var (model, x, others) = FlipCase.Build();
        Assert.Equal(1, Server.Argmax(Server.LogitsAlone(model, x, invariant: true)));
        Assert.Equal(1, Server.Argmax(Server.LogitsInBatch(model, x, others, invariant: true)));
    }
}
