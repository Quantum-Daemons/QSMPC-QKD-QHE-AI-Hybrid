package agg

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

const parties, n, rounds = 4, 14, 15

// The BGV setting is exact: every round's aggregate is the integer sum, as the pairwise masks' is.
func TestBGVAggregateIsExact(t *testing.T) {
	s, err := Open("bgv-128", parties, []byte("crs"))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	for r := 0; r < rounds; r++ {
		ups := make([][]int64, parties)
		want := make([]int64, n)
		for k := range ups {
			ups[k] = make([]int64, n)
			for i := range ups[k] {
				ups[k][i] = rng.Int63n(1<<22) - 1<<21 // the masks' fixed point: |update * weight| * 2^20 well inside
				want[i] += ups[k][i]
			}
		}
		got, st, err := s.RoundInt(ups)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("round %d value %d: got %d, want %d", r, i, got[i], want[i])
			}
		}
		if st.CiphertextBytes == 0 || st.DecryptShareBytes == 0 {
			t.Fatalf("sizes not recorded: %+v", st)
		}
	}
}

// The CKKS settings are approximate; the bounds are a few times the errors measured on 2026-11-13 (4.1e-7 and 6.6e-6
// over 15 rounds of four 14-value updates), far below anything that moves a FedAvg model.
func TestCKKSAggregateErrors(t *testing.T) {
	for kind, bound := range map[string]float64{"ckks-demo": 5e-6, "ckks-l40": 1e-4} {
		s, err := Open(kind, parties, []byte("crs"))
		if err != nil {
			t.Fatal(kind, err)
		}
		rng := rand.New(rand.NewSource(2))
		worst := 0.0
		for r := 0; r < 3; r++ {
			ups := make([][]float64, parties)
			want := make([]float64, n)
			for k := range ups {
				ups[k] = make([]float64, n)
				for i := range ups[k] {
					ups[k][i] = (rng.Float64()*2 - 1) * 0.3
					want[i] += ups[k][i]
				}
			}
			got, _, err := s.RoundFloat(ups)
			if err != nil {
				t.Fatal(kind, err)
			}
			for i := range want {
				worst = math.Max(worst, math.Abs(got[i]-want[i]))
			}
		}
		if worst > bound {
			t.Fatalf("%s: largest error %.3g above %.3g", kind, worst, bound)
		}
	}
}

// The noise bound is the analytic worst case of four fresh encryptions summed, at N = 8192 and Lattigo's error bound
// 19.2: 4 (2 * 8192 * 4 * 19.2 + 19.2), about 2^22.26 - far above the 2^12.3 to 2^12.8 measured, as a worst case is.
func TestNoiseBoundIsTheWorstCase(t *testing.T) {
	s, err := Open("ckks-demo", parties, []byte("crs"))
	if err != nil {
		t.Fatal(err)
	}
	want := 4 * (2*8192*4*19.2 + 19.2)
	if got := NoiseBound(s.params, parties); math.Abs(got-want) > 1e-6 {
		t.Fatalf("NoiseBound %.1f, want %.1f", got, want)
	}
}

// Every setting's flooding meets its target over a whole share, N B/(2F+1) <= 2^-bits, with F = 2^f - 1, for the
// party counts the pages use (two to five), and is the least such f.
func TestFloodingIsSizedForItsTarget(t *testing.T) {
	for kind := range Settings() {
		for _, k := range []int{2, 4, 5} {
			s, err := Open(kind, k, []byte("crs"))
			if err != nil {
				t.Fatal(kind, k, err)
			}
			nb := float64(s.params.N()) * NoiseBound(s.params, k)
			f := s.Setting.FloodBits
			sd := math.Log2(nb) - math.Log2(math.Exp2(float64(f+1))-1)
			if sd > -s.Setting.StatSecBits {
				t.Fatalf("%s, %d parties: distance 2^%.2f above 2^-%.0f", kind, k, sd, s.Setting.StatSecBits)
			}
			if math.Log2(nb)-float64(f) <= -s.Setting.StatSecBits {
				t.Fatalf("%s, %d parties: f = %d is not the least that meets the target", kind, k, f)
			}
		}
	}
}

// The flooding is uniform on [-F, F]: every draw in range, half of them odd, half of them at least F/2 in size - the
// two properties the library's Gaussian samples did not have at these magnitudes (2026-11-13).
func TestFloodingIsUniform(t *testing.T) {
	for kind := range Settings() {
		s, err := Open(kind, parties, []byte("crs"))
		if err != nil {
			t.Fatal(kind, err)
		}
		rq := s.params.RingQ()
		p := rq.NewPoly()
		if err := s.addFlood(p, rq.Level(), false); err != nil {
			t.Fatal(kind, err)
		}
		coeffs := make([]*big.Int, rq.N())
		for i := range coeffs {
			coeffs[i] = new(big.Int)
		}
		rq.PolyToBigintCentered(p, 1, coeffs)
		odd, big2 := 0, 0
		half := new(big.Int).Rsh(s.flood, 1)
		for _, c := range coeffs {
			if new(big.Int).Abs(c).Cmp(s.flood) > 0 {
				t.Fatalf("%s: a draw %v outside [-F, F]", kind, c)
			}
			if c.Bit(0) == 1 {
				odd++
			}
			if new(big.Int).Abs(c).Cmp(half) >= 0 {
				big2++
			}
		}
		// 8,192 draws: a share of one half is within 0.5 +/- 0.03 with probability above 1 - 1e-7.
		for name, c := range map[string]int{"odd": odd, "at least F/2": big2} {
			if frac := float64(c) / float64(rq.N()); math.Abs(frac-0.5) > 0.03 {
				t.Fatalf("%s: %s %.3f, uniform gives 0.5", kind, name, frac)
			}
		}
	}
}
