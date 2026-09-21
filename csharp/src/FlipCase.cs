using System;

namespace BatchInvariant;

/// <summary>The deterministic constructed flip case.</summary>
public static class FlipCase
{
    /// <summary>Input dimension of the constructed flip case.</summary>
    public const int DIn = 16;

    /// <summary>
    /// Return the constructed flip case: a model, an input <c>x</c>, and the
    /// <c>others</c> it is batched with. Under the variant kernel <c>x</c>'s
    /// decoded token flips between running alone (token 1) and running batched
    /// (token 0); under the invariant kernel it stays at token 1.
    ///
    /// The Python reference builds this fixture with random.Random(514) in the
    /// cancellation regime where reduction order matters (even coordinates ~±1e7,
    /// odd coordinates ~±1, with one small coordinate of W[1] nudged to make a
    /// near-tie). Rather than re-derive CPython's Mersenne Twister, the exact
    /// round-trip constants of that construction are embedded here, so the token
    /// flip reproduces bit-for-bit.
    /// </summary>
    public static (Model Model, double[] X, double[][] Others) Build()
    {
        var x = new double[DIn];
        for (int i = 0; i < DIn; i++)
        {
            x[i] = 1.0;
        }

        double[] w0 =
        {
            -2837058.438388074, 0.26420929512244573, -5837462.8193018595, -0.17494963981494416,
            2938486.243101038, -0.13849493455405848, 1414600.9561483413, -0.3696167197232685,
            -3524830.7167600878, 0.7120987215333385, -9791492.533186175, 0.3472369745043211,
            -7726834.460507263, 0.6667423710526905, -2673068.1529440563, -0.24979974016852013,
        };
        double[] w1 =
        {
            -2837058.438388074, 0.26420929512244573, -5837462.8193018595, -0.17494963981494416,
            2938486.243101038, -0.13849493455405848, 1414600.9561483413, -0.36961671803294915,
            -3524830.7167600878, 0.7120987215333385, -9791492.533186175, 0.3472369745043211,
            -7726834.460507263, 0.6667423710526905, -2673068.1529440563, -0.24979974016852013,
        };
        double[][] others =
        {
            new[]
            {
                292122.4220953882, 0.08620621146572294, 1770221.2903503887, -0.7902030394118038,
                -7217155.468557179, 0.24946999371217848, 8873144.414676882, -0.589702840874115,
                -8196247.894585733, -0.30535118320943466, 9248112.91432828, 0.5347068628665839,
                9995925.557525683, -0.6974771862477742, 6713795.038392711, 0.7762102706535812,
            },
            new[]
            {
                -7465319.348107193, 0.32739161904649405, 8080276.9606556, 0.6096012994154612,
                5213409.496121431, -0.5823712477440066, -9280757.839520507, 0.543601618210793,
                6540111.4271609, 0.8110737822234138, -7363132.823341854, -0.4005024012375027,
                -7530586.256065372, 0.7644215568398569, -8262716.01622182, -0.044505499880406196,
            },
            new[]
            {
                5721791.491822636, 0.9724768265281529, 2881718.1018078756, -0.9159726869646703,
                9279586.975825846, -0.920573577312882, 5511774.20853916, -0.8060063100696118,
                -2650171.905299126, -0.030811189514146298, -4035974.461568672, -0.8411997933371969,
                -170731.31937074848, 0.24878003933057946, 7192418.797877323, -0.20054010428181912,
            },
        };

        return (new Model(new[] { w0, w1 }), x, others);
    }
}
