package agg

import (
	"math"
	"math/big"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
)

const committeeWeights = 1 << 27

// The page's quantised position: two collateral balances and a scaled debt, at 2^20 units of each.
var committeeBalances = []int64{12 << 20, 3 << 19, 37 << 20}

func openTestCommittee(t *testing.T, n, thr int) (*Committee, *Recipient) {
	t.Helper()
	c, err := OpenCommittee(n, thr, []byte("committee crs"), committeeWeights)
	if err != nil {
		t.Fatal(err)
	}
	return c, c.NewRecipient()
}

func encryptAll(t *testing.T, c *Committee, xs []int64) []*rlwe.Ciphertext {
	t.Helper()
	cts := make([]*rlwe.Ciphertext, len(xs))
	for i, x := range xs {
		ct, err := c.Encrypt(x)
		if err != nil {
			t.Fatal(err)
		}
		cts[i] = ct
	}
	return cts
}

func margin(x, w, settled []int64) int64 {
	m := int64(0)
	for i := range x {
		m += w[i] * (x[i] - settled[i])
	}
	return m
}

// Carol's value and the pool's add up, modulo t, to the margin, for margins of either sign and zero, with a different
// active set each time; and only slot 0 carries anything.
func TestCommitteeGivesTheMarginToTheTwoShares(t *testing.T) {
	c, carol := openTestCommittee(t, 3, 2)
	cts := encryptAll(t, c, committeeBalances)
	tm := c.Setting.PlainModulus
	cases := []struct{ w, settled []int64 }{
		{[]int64{1305 << 9, 24536 << 9, -(1 << 19)}, []int64{0, 0, 0}},                // a healthy position: M > 0
		{[]int64{400 << 9, 9000 << 9, -(1 << 19) - 4000}, []int64{0, 0, 0}},           // after a crash: M < 0
		{[]int64{1305 << 9, 24536 << 9, -(1 << 19)}, []int64{2 << 20, 1 << 18, 5 << 20}}, // after a settlement
		{[]int64{37, 0, -12}, []int64{0, 0, 0}},                                        // M = 0 exactly
	}
	actives := [][]int{{0, 1}, {1, 2}, {0, 2}, {2, 0}}
	for k, cs := range cases {
		want := margin(committeeBalances, cs.w, cs.settled)
		ev, r, err := c.Evaluate(cts, cs.w, cs.settled)
		if err != nil {
			t.Fatal(k, err)
		}
		out, _, err := c.Reencrypt(ev, actives[k], carol.PK)
		if err != nil {
			t.Fatal(k, err)
		}
		held, rest, err := carol.Decrypt(out)
		if err != nil {
			t.Fatal(k, err)
		}
		if !rest {
			t.Fatalf("case %d: a slot other than 0 decrypted to something", k)
		}
		pool := (tm - r) % tm
		got := SignedMod((held+pool)%tm, tm)
		if got != want {
			t.Fatalf("case %d: the shares give %d, the margin is %d", k, got, want)
		}
		if k == 3 && want != 0 {
			t.Fatalf("case 3 should be a zero margin, is %d", want)
		}
	}
}

// The members share once for a given c1, whatever the active set; the pool's fresh encryption of zero makes two
// evaluations with the same weights two different c1, so each hour can still be checked.
func TestCommitteeRefusesASecondShareForTheSameCiphertext(t *testing.T) {
	c, carol := openTestCommittee(t, 3, 2)
	cts := encryptAll(t, c, committeeBalances)
	w, settled := []int64{1305 << 9, 24536 << 9, -(1 << 19)}, []int64{0, 0, 0}
	ev, _, err := c.Evaluate(cts, w, settled)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.Reencrypt(ev, []int{0, 1}, carol.PK); err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.Reencrypt(ev, []int{1, 2}, carol.PK); err == nil {
		t.Fatal("a second set of shares for the same ciphertext was not refused")
	}
	again, _, err := c.Evaluate(cts, w, settled)
	if err != nil {
		t.Fatal(err)
	}
	if again.Value[1].Equal(&ev.Value[1]) {
		t.Fatal("two evaluations with the same weights gave the same c1")
	}
	if _, _, err := c.Reencrypt(again, []int{1, 2}, carol.PK); err != nil {
		t.Fatal("the next hour's ciphertext was refused:", err)
	}
}

// collectiveSecret recombines the collective secret from an active set's additive shares: a test can, the page never does.
func collectiveSecret(t *testing.T, c *Committee, active []int) *rlwe.SecretKey {
	t.Helper()
	rp := c.params.Parameters
	pts := make([]multiparty.ShamirPublicPoint, len(active))
	for k, i := range active {
		pts[k] = c.points[i]
	}
	s := rlwe.NewSecretKey(rp)
	for _, i := range active {
		others := []multiparty.ShamirPublicPoint{}
		for _, p := range c.points {
			if p != c.points[i] {
				others = append(others, p)
			}
		}
		add := rlwe.NewSecretKey(rp)
		if err := multiparty.NewCombiner(rp, c.points[i], others, c.Setting.Threshold).GenAdditiveShare(pts, c.points[i], c.tsk[i], add); err != nil {
			t.Fatal(err)
		}
		rp.RingQ().Add(s.Value.Q, add.Value.Q, s.Value.Q)
	}
	return s
}

// noiseLog2 is log2 of the largest |coefficient| of what ct decrypts to under sk, less the encoding of slot 0 = v.
func noiseLog2(t *testing.T, c *Committee, sk *rlwe.SecretKey, ct *rlwe.Ciphertext, v uint64) float64 {
	t.Helper()
	rq := c.params.RingQ().AtLevel(ct.Level())
	raw := rlwe.NewDecryptor(c.params.Parameters, sk).DecryptNew(ct)
	want := bgv.NewPlaintext(c.params, ct.Level())
	*want.MetaData = *ct.MetaData
	if err := c.enc.Encode([]uint64{v}, want); err != nil {
		t.Fatal(err)
	}
	e := rq.NewPoly()
	rq.Sub(raw.Value, want.Value, e)
	if raw.IsNTT {
		rq.INTT(e, e)
	}
	coeffs := make([]*big.Int, rq.N())
	for i := range coeffs {
		coeffs[i] = new(big.Int)
	}
	rq.PolyToBigintCentered(e, 1, coeffs)
	max := new(big.Int)
	for _, x := range coeffs {
		if a := new(big.Int).Abs(x); a.Cmp(max) > 0 {
			max = a
		}
	}
	f, _ := new(big.Float).SetInt(max).Float64()
	return math.Log2(f)
}

// The evaluated ciphertext's noise is below the analytic bound for the weights used; after the switch, the noise
// under Carol's key is the floods' size, f + 1 bits for two members, and still decrypts.
func TestCommitteeNoiseIsWithinItsBounds(t *testing.T) {
	c, carol := openTestCommittee(t, 3, 2)
	cts := encryptAll(t, c, committeeBalances)
	w, settled := []int64{1305 << 9, 24536 << 9, -(1 << 19)}, []int64{0, 0, 0}
	ev, r, err := c.Evaluate(cts, w, settled)
	if err != nil {
		t.Fatal(err)
	}
	tm := c.Setting.PlainModulus
	m := margin(committeeBalances, w, settled)
	v := uint64((new(big.Int).Mod(new(big.Int).Add(big.NewInt(m), new(big.Int).SetUint64(r)), new(big.Int).SetUint64(tm))).Uint64())
	sumW := 0.0
	for _, x := range w {
		sumW += math.Abs(float64(x))
	}
	bound := math.Log2((sumW + 1) * NoiseBound(c.params.Parameters, 3) / 3)
	got := noiseLog2(t, c, collectiveSecret(t, c, []int{0, 2}), ev, v)
	t.Logf("evaluated noise 2^%.2f, bound 2^%.2f for these weights, 2^%.2f for the committee's", got, bound, c.Setting.NoiseBoundLog2)
	if got > bound {
		t.Fatalf("the evaluated noise 2^%.2f is above its bound 2^%.2f", got, bound)
	}
	out, _, err := c.Reencrypt(ev, []int{0, 2}, carol.PK)
	if err != nil {
		t.Fatal(err)
	}
	after := noiseLog2(t, c, carol.sk, out, v)
	limit := c.Setting.LogQ - 1 - math.Log2(float64(tm))
	t.Logf("noise under Carol's key 2^%.2f, f = %d, decryption limit 2^%.2f", after, c.Setting.FloodBits, limit)
	f := float64(c.Setting.FloodBits)
	if after < f || after >= f+2 || after >= limit {
		t.Fatalf("noise 2^%.2f after the switch: want the floods' f + 1 = %.0f bits, below 2^%.2f", after, f+1, limit)
	}
}

// The flooding meets 128 bits over a whole share for the committee's worst case and is the least f that does; Q fits
// the floods and stays within the HomomorphicEncryption.org standard's post-quantum bound at N = 16384 (411 bits);
// t is NTT-friendly for N and below 2^53.
func TestCommitteeParameters(t *testing.T) {
	for _, nt := range [][2]int{{3, 2}, {5, 3}, {7, 4}} {
		c, err := OpenCommittee(nt[0], nt[1], []byte("crs"), committeeWeights)
		if err != nil {
			t.Fatal(nt, err)
		}
		s := c.Setting
		nb := math.Log2(float64(c.params.N())) + s.NoiseBoundLog2 // log2 of N B
		sd := nb - math.Log2(math.Exp2(float64(s.FloodBits+1))-1)
		if sd > -CommitteeBits {
			t.Fatalf("%v: distance 2^%.2f above 2^-128", nt, sd)
		}
		if nb-float64(s.FloodBits) <= -CommitteeBits {
			t.Fatalf("%v: f = %d is not the least", nt, s.FloodBits)
		}
		if s.LogQ > 411 {
			t.Fatalf("%v: log Q = %.1f is past 411", nt, s.LogQ)
		}
		if s.PlainModulus >= 1<<53 || s.PlainModulus%(2<<CommitteeLogN) != 1 {
			t.Fatalf("%v: t = %d", nt, s.PlainModulus)
		}
		t.Logf("%d of %d: B 2^%.2f, f %d, log Q %.1f, t %d", nt[1], nt[0], s.NoiseBoundLog2, s.FloodBits, s.LogQ, s.PlainModulus)
	}
}

func TestCommitteeRefusesMalformedRequests(t *testing.T) {
	c, carol := openTestCommittee(t, 3, 2)
	cts := encryptAll(t, c, committeeBalances)
	if _, _, err := c.Evaluate(cts, []int64{committeeWeights, 1, 0}, []int64{0, 0, 0}); err == nil {
		t.Fatal("weights over the bound were accepted")
	}
	if _, _, err := c.Evaluate(cts, []int64{1, 2}, []int64{0, 0}); err == nil {
		t.Fatal("two weights for three ciphertexts were accepted")
	}
	for _, active := range [][]int{{0}, {0, 1, 2}, {1, 1}, {0, 3}, {-1, 0}} {
		ev, _, err := c.Evaluate(cts, []int64{1, 1, -1}, []int64{0, 0, 0})
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := c.Reencrypt(ev, active, carol.PK); err == nil {
			t.Fatalf("active set %v was accepted", active)
		}
	}
	for _, nt := range [][2]int{{1, 1}, {3, 0}, {3, 4}} {
		if _, err := OpenCommittee(nt[0], nt[1], []byte("crs"), committeeWeights); err == nil {
			t.Fatalf("a committee of %v was opened", nt)
		}
	}
	if _, err := OpenCommittee(3, 2, []byte("crs"), math.Exp2(41)); err == nil {
		t.Fatal("a weight bound past 2^40 was accepted")
	}
	if _, err := OpenCommittee(3, 2, make([]byte, 200), committeeWeights); err != nil {
		t.Fatal("a long seed for the common reference string was refused:", err)
	}
}

// Bob's ciphertexts and Carol's re-encrypted one survive the wire; a cut, an extra byte, the wrong count or a
// ciphertext of another shape is refused.
func TestCommitteeCiphertextsOnTheWire(t *testing.T) {
	c, carol := openTestCommittee(t, 3, 2)
	cts := encryptAll(t, c, committeeBalances)
	b, err := MarshalCiphertexts(cts)
	if err != nil {
		t.Fatal(err)
	}
	back, err := c.UnmarshalCiphertexts(b, len(cts))
	if err != nil {
		t.Fatal(err)
	}
	w, settled := []int64{1305 << 9, 24536 << 9, -(1 << 19)}, []int64{0, 0, 0}
	ev, r, err := c.Evaluate(back, w, settled)
	if err != nil {
		t.Fatal(err)
	}
	out, _, err := c.Reencrypt(ev, []int{0, 1}, carol.PK)
	if err != nil {
		t.Fatal(err)
	}
	ob, err := MarshalCiphertexts([]*rlwe.Ciphertext{out})
	if err != nil {
		t.Fatal(err)
	}
	got, err := c.UnmarshalCiphertexts(ob, 1)
	if err != nil {
		t.Fatal(err)
	}
	held, rest, err := carol.Decrypt(got[0])
	if err != nil || !rest {
		t.Fatal("the re-encrypted margin did not decrypt after the wire", err)
	}
	tm := c.Setting.PlainModulus
	if SignedMod((held+(tm-r)%tm)%tm, tm) != margin(committeeBalances, w, settled) {
		t.Fatal("the margin changed on the wire")
	}
	for name, bad := range map[string][]byte{
		"cut":        b[:len(b)-1],
		"extra":      append(append([]byte{}, b...), 0),
		"empty":      {},
		"count only": b[:4],
	} {
		if _, err := c.UnmarshalCiphertexts(bad, len(cts)); err == nil {
			t.Fatalf("a %s message was accepted", name)
		}
	}
	if _, err := c.UnmarshalCiphertexts(b, len(cts)-1); err == nil {
		t.Fatal("the wrong count was accepted")
	}
	lower := cts[0].CopyNew()
	lower.Resize(1, lower.Level()-1)
	lb, err := MarshalCiphertexts([]*rlwe.Ciphertext{lower})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.UnmarshalCiphertexts(lb, 1); err == nil {
		t.Fatal("a ciphertext at a lower level was accepted")
	}
}
