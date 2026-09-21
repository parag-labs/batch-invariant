package com.batchinvariant;

/** The output projection: logits = x @ wᵀ. {@code w} is [vocab][d_in]. */
public final class Model {

    private final double[][] w;

    /** Creates a model from its output-projection weights. */
    public Model(double[][] w) {
        this.w = w;
    }

    /** Returns the output-projection weights, one row per vocabulary token. */
    public double[][] w() {
        return w;
    }

    /** Returns the number of output tokens (rows of {@code w}). */
    public int vocab() {
        return w.length;
    }
}
