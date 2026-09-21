package batchinvariant

// lcg is a tiny deterministic PRNG for reproducible test data. It does not need
// to match CPython's RNG — the invariance property holds for any input, so tests
// only need a repeatable stream.
type lcg struct{ state uint64 }

func newLCG(seed uint64) *lcg { return &lcg{state: seed + 0x9e3779b97f4a7c15} }

func (r *lcg) next() float64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return float64(r.state>>11) / float64(uint64(1)<<53)
}

// uniform returns a value in [lo, hi).
func (r *lcg) uniform(lo, hi float64) float64 { return lo + (hi-lo)*r.next() }

// mixedRow builds a d-length row with large even coordinates (±1e6) and tiny odd
// coordinates (±1), so summing them mixes magnitudes and reduction order matters.
func (r *lcg) mixedRow(d int) Vec {
	row := make(Vec, d)
	for i := 0; i < d; i++ {
		if i%2 == 0 {
			row[i] = r.uniform(-1e6, 1e6)
		} else {
			row[i] = r.uniform(-1, 1)
		}
	}
	return row
}

func (r *lcg) mixedRows(n, d int) Mat {
	rows := make(Mat, n)
	for i := range rows {
		rows[i] = r.mixedRow(d)
	}
	return rows
}

func vecEqual(a, b Vec) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
