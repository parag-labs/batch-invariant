package com.batchinvariant;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class KernelsTest {

    @Test
    void sumFlatEmptyIsZero() {
        assertEquals(0.0, Kernels.sumFlat(new double[] {}));
    }

    @Test
    void sumFlatSingleElement() {
        assertEquals(42.5, Kernels.sumFlat(new double[] {42.5}));
    }

    @Test
    void sumFlatIsLeftToRight() {
        assertEquals(((0.1 + 0.2) + 0.3) + 0.4, Kernels.sumFlat(new double[] {0.1, 0.2, 0.3, 0.4}));
    }

    @Test
    void splitDiffersFromFlatInFloat() {
        double[] vals = {1e16, 1.0, -1e16, 1.0};
        assertNotEquals(Kernels.sumFlat(vals), Kernels.sumSplit(vals, 2));
    }

    @Test
    void singleSplitIsFlat() {
        double[] vals = {0.1, 0.2, 0.3, 0.4};
        assertEquals(Kernels.sumFlat(vals), Kernels.sumSplit(vals, 1));
    }

    @Test
    void zeroOrNegativeSplitsIsFlat() {
        double[] vals = {0.1, 0.2, 0.3};
        assertEquals(Kernels.sumFlat(vals), Kernels.sumSplit(vals, 0));
        assertEquals(Kernels.sumFlat(vals), Kernels.sumSplit(vals, -3));
    }

    @Test
    void sumSplitEmptyIsZero() {
        assertEquals(0.0, Kernels.sumSplit(new double[] {}, 4));
    }

    @Test
    void sumSplitMoreSplitsThanElements() {
        assertEquals(6.0, Kernels.sumSplit(new double[] {1.0, 2.0, 3.0}, 8));
    }

    @ParameterizedTest
    @CsvSource({"0,8", "1,8", "2,4", "4,2", "8,1", "9,1", "16,1"})
    void splitsForBatchKnownValues(int batch, int expected) {
        assertEquals(expected, Kernels.splitsForBatch(batch));
    }

    @Test
    void splitsForBatchVariesWithBatchSize() {
        assertNotEquals(Kernels.splitsForBatch(1), Kernels.splitsForBatch(4));
    }

    @Test
    void fixedSplitsIsFour() {
        assertEquals(4, Kernels.FIXED_SPLITS);
    }

    @Test
    void matmulProductsAndShape() {
        double[][] w = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        double[][] out = Kernels.matmulInvariant(new double[][] {{1.0, 1.0}, {2.0, 2.0}}, w);
        assertEquals(2, out.length);
        assertArrayEquals(new double[] {3.0, 7.0, 11.0}, out[0]);
    }

    @Test
    void invariantRowIdenticalAloneAndBatched() {
        Lcg rng = new Lcg(1);
        Model model = new Model(rng.mixedRows(5, 16));
        for (int trial = 0; trial < 200; trial++) {
            double[] x = rng.mixedRow(16);
            double[][] others = rng.mixedRows(trial % 7, 16);
            double[] alone = Server.logitsAlone(model, x, true);
            double[] batched = Server.logitsInBatch(model, x, others, true);
            assertTrue(Lcg.vecEqual(alone, batched), "invariant kernel differed on trial " + trial);
        }
    }

    @Test
    void variantRowChangesWithBatch() {
        FlipCase.Case c = FlipCase.build();
        assertFalse(Lcg.vecEqual(
                Server.logitsAlone(c.model(), c.x(), false),
                Server.logitsInBatch(c.model(), c.x(), c.others(), false)));
    }
}
