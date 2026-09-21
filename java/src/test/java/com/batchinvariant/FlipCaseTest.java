package com.batchinvariant;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class FlipCaseTest {

    @Test
    void flipCaseShape() {
        FlipCase.Case c = FlipCase.build();
        assertEquals(2, c.model().vocab());
        assertEquals(FlipCase.D_IN, c.x().length);
        assertEquals(3, c.others().length);
        for (double v : c.x()) {
            assertEquals(1.0, v);
        }
    }

    @Test
    void variantFlipsTheDecodedToken() {
        FlipCase.Case c = FlipCase.build();
        assertEquals(1, Server.argmax(Server.logitsAlone(c.model(), c.x(), false)));
        assertEquals(0, Server.argmax(Server.logitsInBatch(c.model(), c.x(), c.others(), false)));
    }

    @Test
    void invariantKeepsTheTokenStable() {
        FlipCase.Case c = FlipCase.build();
        assertEquals(1, Server.argmax(Server.logitsAlone(c.model(), c.x(), true)));
        assertEquals(1, Server.argmax(Server.logitsInBatch(c.model(), c.x(), c.others(), true)));
    }
}
