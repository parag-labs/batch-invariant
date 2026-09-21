//! Shared deterministic PRNG for reproducible test data. It does not need to
//! match CPython's RNG — the invariance property holds for any input, so tests
//! only need a repeatable stream.

pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg {
            state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.state >> 11) as f64 / ((1u64 << 53) as f64)
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// A row with large even coordinates (±1e6) and tiny odd coordinates (±1),
    /// so summing them mixes magnitudes and reduction order matters.
    pub fn mixed_row(&mut self, d: usize) -> Vec<f64> {
        (0..d)
            .map(|i| {
                if i % 2 == 0 {
                    self.uniform(-1e6, 1e6)
                } else {
                    self.uniform(-1.0, 1.0)
                }
            })
            .collect()
    }

    pub fn mixed_rows(&mut self, n: usize, d: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.mixed_row(d)).collect()
    }
}
