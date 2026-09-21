package com.batchinvariant;

/** A toy decode step, to show batch-dependence changing an actual token. */
public final class Server {

    private Server() {
    }

    private static double[][] runKernel(boolean invariant, double[][] batch, double[][] w) {
        return invariant ? Kernels.matmulInvariant(batch, w) : Kernels.matmulVariant(batch, w);
    }

    /** Returns the logits for {@code x} decoded on its own (a batch of one). */
    public static double[] logitsAlone(Model model, double[] x, boolean invariant) {
        return runKernel(invariant, new double[][] {x}, model.w())[0];
    }

    /**
     * Returns the logits for {@code x} when it is decoded in a batch alongside
     * {@code others}. {@code x} is placed first; the returned row is its logits.
     * With the invariant kernel this equals {@link #logitsAlone}; with the
     * variant kernel it may not.
     */
    public static double[] logitsInBatch(Model model, double[] x, double[][] others, boolean invariant) {
        double[][] batch = new double[others.length + 1][];
        batch[0] = x;
        System.arraycopy(others, 0, batch, 1, others.length);
        return runKernel(invariant, batch, model.w())[0];
    }

    /** Returns the chosen next token: index of the largest logit, first-wins on ties. */
    public static int argmax(double[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Reports whether {@code x}'s decoded token changes between running alone and
     * running in a batch — the failure the invariant kernel makes impossible.
     */
    public static boolean tokenDependsOnBatch(Model model, double[] x, double[][] others, boolean invariant) {
        int alone = argmax(logitsAlone(model, x, invariant));
        int batched = argmax(logitsInBatch(model, x, others, invariant));
        return alone != batched;
    }
}
