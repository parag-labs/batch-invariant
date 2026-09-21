package com.batchinvariant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ServerTest {

    @Test
    void modelVocabCountsRows() {
        Model m = new Model(new double[][] {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        assertEquals(3, m.vocab());
    }

    @Test
    void argmaxFirstWinsOnTies() {
        assertEquals(1, Server.argmax(new double[] {1.0, 3.0, 3.0, 2.0}));
    }

    @Test
    void argmaxSingleElement() {
        assertEquals(0, Server.argmax(new double[] {5.0}));
    }

    @Test
    void argmaxAllNegative() {
        assertEquals(1, Server.argmax(new double[] {-3.0, -1.0, -2.0}));
    }

    @Test
    void logitsAloneEqualsInvariantBatched() {
        Lcg rng = new Lcg(7);
        Model model = new Model(rng.mixedRows(3, 16));
        double[] x = rng.mixedRow(16);
        double[][] others = rng.mixedRows(4, 16);
        assertTrue(Lcg.vecEqual(
                Server.logitsAlone(model, x, true),
                Server.logitsInBatch(model, x, others, true)));
    }

    @Test
    void tokenStableUnderInvariantFlipCase() {
        FlipCase.Case c = FlipCase.build();
        assertFalse(Server.tokenDependsOnBatch(c.model(), c.x(), c.others(), true));
    }

    @Test
    void tokenMovesUnderVariantFlipCase() {
        FlipCase.Case c = FlipCase.build();
        assertTrue(Server.tokenDependsOnBatch(c.model(), c.x(), c.others(), false));
    }
}
