package agg

import (
	"math"
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

// Every setting keeps the smudging bound below each prime of Q, or uses the sampler's big-integer branch: the
// condition under which Lattigo v6.2.0's Gaussian sampler is correct (tuneinsight/lattigo#596).
func TestSettingsAvoidTheSamplerDefect(t *testing.T) {
	for kind, st := range Settings() {
		if st.SigmaLog2 > 53 && 6*math.Exp2(st.SigmaLog2) > 18446744073709551615.0 {
			continue // the big.Int branch, which reduces each limb
		}
		for _, lq := range st.LogQ {
			if math.Log2(6)+st.SigmaLog2 >= float64(lq)-1 {
				t.Fatalf("%s: 6*sigma (2^%.1f) reaches a %d-bit prime", kind, math.Log2(6)+st.SigmaLog2, lq)
			}
		}
	}
}

// The statistical security of each setting's smudging is what the page prints: log2(sigma / noise bound).
func TestStatisticalSecurityIsStated(t *testing.T) {
	for kind, st := range Settings() {
		if got := st.SigmaLog2 - NoiseBoundLog2; math.Abs(got-st.StatSecBits) > 1e-9 {
			t.Fatalf("%s: states %.1f bits, sigma gives %.1f", kind, st.StatSecBits, got)
		}
	}
}
