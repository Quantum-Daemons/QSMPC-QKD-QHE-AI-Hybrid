// Package agg aggregates four hospitals' FedAvg updates under Lattigo's N-out-of-N threshold encryption: a collective
// public key from each hospital's secret share, encryption under it, ciphertext addition by the aggregator, and one
// collective decryption per round by key-switching to the zero key, each hospital's share flooded with noise.
//
// Three settings, measured side by side on the medical page, each sized for a statistical security over the whole
// decryption share:
//
//	ckks-demo  CKKS, 16 bits (a demonstration), scale 2^80, Q {60, 60}
//	ckks-l40   CKKS, 40 bits, scale 2^100, Q {60, 60}
//	bgv-128    BGV, exact, 128 bits, Q {60, 60, 60, 20}, a 33-bit plaintext modulus; the hospitals quantise exactly as
//	           the pairwise masks do, so the aggregate is the masks' to the last bit
//
// THE FLOODING NOISE IS DRAWN HERE, uniform on [-F, F] for every coefficient of every share, from crypto/rand, and
// reduced into each limb exactly (Round 86, 2026-11-13). The smudging lemma is stated for this distribution (Asharov
// et al., EUROCRYPT 2012, Lemma 2.1), and its statistical distance to a shift by x has the closed form |x|/(2F+1), so F
// follows from the bound and the target: over N coefficients, N B/(2F+1) <= 2^-bits. B is the analytic worst case of
// the aggregate's noise (NoiseBound), not a measurement: worst-case estimation is what shared decryptions need (Guo,
// Nabokov, Suvanto and Johansson, USENIX Security 2024). Lattigo's own key-switching noise is left at its fresh level.
//
// Each aggregate is decrypted once, by construction: a round encrypts fresh ciphertexts, sums them inside
// aggregateAndDecrypt and returns only the plaintext, so no caller can ask a hospital for a second share of the same
// ciphertext. Lattigo's SECURITY.md says retrying a multiparty protocol must be considered insecure.
package agg

import (
	"crypto/rand"
	"errors"
	"math"
	"math/big"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// NoiseBound is the worst case, coefficient by coefficient, of the noise in the sum of `parties` fresh public-key
// encryptions under the collective key, before any smudging. One encryption decrypts with noise u e_pk + e0 + e1 s,
// where u is ternary and e0, e1 and each party's public-key error lie within the error bound b: the collective
// e_pk is a sum of `parties` errors and s a sum of `parties` ternary secrets, so a coefficient is at most
// N parties b (for u e_pk) + N b parties (for e1 s) + b, and `parties` encryptions sum to `parties` times that.
func NoiseBound(params rlwe.Parameters, parties int) float64 {
	b := params.Xe().(ring.DiscreteGaussian).Bound
	n, k := float64(params.N()), float64(parties)
	return k * (2*n*k*b + b)
}

// Setting describes one threshold configuration.
type Setting struct {
	Kind     string `json:"kind"`
	Scheme   string `json:"scheme"`
	LogN     int    `json:"logN"`
	LogQ     []int  `json:"logQ"`
	LogScale int    `json:"logScale,omitempty"`
	// StatSecBits is what the flooding is sized for: one decryption share is within statistical distance
	// 2^-StatSecBits of its simulation, over all of its N coefficients.
	StatSecBits float64 `json:"statSecBits"`
	// Set by Open, from the parameters and the number of parties: log2 of NoiseBound, and f with F = 2^f - 1.
	NoiseBoundLog2 float64 `json:"noiseBoundLog2,omitempty"`
	FloodBits      int     `json:"floodBits,omitempty"`
	PlainMod       uint64  `json:"plaintextModulus,omitempty"`
}

// Settings returns the three settings by kind.
func Settings() map[string]Setting {
	return map[string]Setting{
		"ckks-demo": {Kind: "ckks-demo", Scheme: "CKKS", LogN: 13, LogQ: []int{60, 60}, LogScale: 80, StatSecBits: 16},
		"ckks-l40": {Kind: "ckks-l40", Scheme: "CKKS", LogN: 13, LogQ: []int{60, 60}, LogScale: 100, StatSecBits: 40},
		// 200 bits of Q: inside the 202 the HomomorphicEncryption.org standard's post-quantum table allows at N = 8192
		// for 128-bit security (218 classically), with Q/(2t) about 2^166 above the four shares' flooding, at most
		// 4F = 2^165 because a uniform flood is bounded: the decryption cannot fail.
		"bgv-128": {Kind: "bgv-128", Scheme: "BGV", LogN: 13, LogQ: []int{60, 60, 60, 20}, StatSecBits: 128},
	}
}

// Session holds the hospitals' secret shares and the collective public key for one run.
type Session struct {
	Setting  Setting
	params   rlwe.Parameters
	ckksP    ckks.Parameters
	bgvP     bgv.Parameters
	sks      []*rlwe.SecretKey
	pk       *rlwe.PublicKey
	cks      multiparty.KeySwitchProtocol
	zero     *rlwe.SecretKey
	flood    *big.Int // F = 2^f - 1
	KeyShare int     // bytes of one hospital's public-key share
	KeyGenMs float64 // the whole collective key generation
}

// Open runs the collective key generation for `parties` hospitals. crsSeed is public: every party samples the same
// common reference polynomial from it.
func Open(kind string, parties int, crsSeed []byte) (*Session, error) {
	st, ok := Settings()[kind]
	if !ok {
		return nil, errors.New("unknown setting " + kind)
	}
	s := &Session{Setting: st}
	var err error
	switch st.Scheme {
	case "CKKS":
		s.ckksP, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{LogN: st.LogN, LogQ: st.LogQ, LogDefaultScale: st.LogScale})
		s.params = s.ckksP.Parameters
	case "BGV":
		g := ring.NewNTTFriendlyPrimesGenerator(33, uint64(2<<st.LogN))
		t, perr := g.NextUpstreamPrime()
		if perr != nil {
			return nil, perr
		}
		s.Setting.PlainMod = t
		s.bgvP, err = bgv.NewParametersFromLiteral(bgv.ParametersLiteral{LogN: st.LogN, LogQ: st.LogQ, PlaintextModulus: t})
		s.params = s.bgvP.Parameters
	}
	if err != nil {
		return nil, err
	}
	// F from the analytic bound and the target, over all N coefficients of a share: N B/(2F+1) <= 2^-bits, with
	// F = 2^f - 1 so that a draw is f+1 random bits rejected only when all are ones.
	B := NoiseBound(s.params, parties)
	nb := math.Log2(float64(s.params.N()) * B)
	f := int(math.Ceil(st.StatSecBits+nb)) - 1 // the least f with 2^(f+1) - 1 >= 2^bits N B
	if nb-math.Log2(math.Exp2(float64(f+1))-1) > -st.StatSecBits {
		f++
	}
	s.Setting.NoiseBoundLog2, s.Setting.FloodBits = math.Log2(B), f
	s.flood = new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), uint(f)), big.NewInt(1))
	// The aggregate must still decrypt with the K floods in it: below Q/(2t) for BGV, below Q/2 less the message
	// for CKKS. A setting that cannot is refused here rather than failing as a wrong number.
	logQ := 0.0
	for _, q := range s.params.Q() {
		logQ += math.Log2(float64(q))
	}
	// BGV: the K floods together below Q/(2t). CKKS: below a quarter of Q, leaving the rest to the scaled message.
	room := logQ - 2
	if st.Scheme == "BGV" {
		room = logQ - 1 - math.Log2(float64(s.Setting.PlainMod))
	}
	if float64(f)+math.Log2(float64(parties)) >= room {
		return nil, errors.New("the flooding for this statistical security does not fit in Q: add a prime or lower the target")
	}
	t0 := time.Now()
	kgen := rlwe.NewKeyGenerator(s.params)
	crs, err := sampling.NewKeyedPRNG(crsSeed)
	if err != nil {
		return nil, err
	}
	ckg := multiparty.NewPublicKeyGenProtocol(s.params)
	crp := ckg.SampleCRP(crs)
	var agg multiparty.PublicKeyGenShare
	for i := 0; i < parties; i++ {
		sk := kgen.GenSecretKeyNew()
		s.sks = append(s.sks, sk)
		sh := ckg.AllocateShare()
		ckg.GenShare(sk, crp, &sh)
		s.KeyShare = sh.BinarySize()
		if i == 0 {
			agg = sh
		} else {
			ckg.AggregateShares(agg, sh, &agg)
		}
	}
	s.pk = rlwe.NewPublicKey(s.params)
	ckg.GenPublicKey(agg, crp, s.pk)
	// Lattigo's own noise in a share stays at its fresh level (a flooding sigma of 0); the flooding is ours, in
	// aggregateAndDecrypt.
	s.cks, err = multiparty.NewKeySwitchProtocol(s.params, ring.DiscreteGaussian{Sigma: 0, Bound: 0})
	if err != nil {
		return nil, err
	}
	s.zero = rlwe.NewSecretKey(s.params)
	s.KeyGenMs = ms(t0)
	return s, nil
}

func ms(t time.Time) float64 { return float64(time.Since(t).Microseconds()) / 1000 }

// RoundStats are one round's costs.
type RoundStats struct {
	EncodeEncryptMs   float64 `json:"encodeEncryptMs"`   // per hospital
	AggregateMs       float64 `json:"aggregateMs"`       // the aggregator's additions
	DecryptShareMs    float64 `json:"decryptShareMs"`    // per hospital
	CombineDecodeMs   float64 `json:"combineDecodeMs"`   // key switch, decryption with the zero key, decoding
	CiphertextBytes   int     `json:"ciphertextBytes"`   // one hospital's encrypted update
	DecryptShareBytes int     `json:"decryptShareBytes"` // one hospital's decryption share
}

// RoundFloat aggregates float updates (CKKS settings): updates[k] is hospital k's weighted update.
func (s *Session) RoundFloat(updates [][]float64) ([]float64, RoundStats, error) {
	if s.Setting.Scheme != "CKKS" {
		return nil, RoundStats{}, errors.New("RoundFloat needs a CKKS setting")
	}
	var st RoundStats
	ecd := ckks.NewEncoder(s.ckksP)
	enc := rlwe.NewEncryptor(s.params, s.pk)
	cts := make([]*rlwe.Ciphertext, len(updates))
	t0 := time.Now()
	for k, u := range updates {
		pt := ckks.NewPlaintext(s.ckksP, s.ckksP.MaxLevel())
		if err := ecd.Encode(u, pt); err != nil {
			return nil, st, err
		}
		ct, err := enc.EncryptNew(pt)
		if err != nil {
			return nil, st, err
		}
		cts[k] = ct
	}
	st.EncodeEncryptMs = ms(t0) / float64(len(updates))
	st.CiphertextBytes = cts[0].BinarySize()
	ks, err := s.aggregateAndDecrypt(cts, &st)
	if err != nil {
		return nil, st, err
	}
	t0 = time.Now()
	pt := rlwe.NewDecryptor(s.params, s.zero).DecryptNew(ks)
	n := len(updates[0])
	out := make([]float64, n)
	if s.Setting.LogScale <= 53 {
		vals := make([]float64, s.ckksP.MaxSlots())
		if err := ecd.Decode(pt, vals); err != nil {
			return nil, st, err
		}
		copy(out, vals[:n])
	} else {
		vals := make([]*big.Float, s.ckksP.MaxSlots())
		for i := range vals {
			vals[i] = new(big.Float).SetPrec(uint(s.Setting.LogScale) + 64)
		}
		if err := ecd.Decode(pt, vals); err != nil {
			return nil, st, err
		}
		for i := 0; i < n; i++ {
			out[i], _ = vals[i].Float64()
		}
	}
	st.CombineDecodeMs += ms(t0)
	return out, st, nil
}

// RoundInt aggregates integer updates (the BGV setting), each the masks' fixed-point value; the sum is exact as long
// as it stays inside the plaintext modulus, which the caller's quantisation keeps far inside.
func (s *Session) RoundInt(updates [][]int64) ([]int64, RoundStats, error) {
	if s.Setting.Scheme != "BGV" {
		return nil, RoundStats{}, errors.New("RoundInt needs the BGV setting")
	}
	var st RoundStats
	ecd := bgv.NewEncoder(s.bgvP)
	enc := rlwe.NewEncryptor(s.params, s.pk)
	cts := make([]*rlwe.Ciphertext, len(updates))
	t0 := time.Now()
	for k, u := range updates {
		pt := bgv.NewPlaintext(s.bgvP, s.bgvP.MaxLevel())
		if err := ecd.Encode(u, pt); err != nil {
			return nil, st, err
		}
		ct, err := enc.EncryptNew(pt)
		if err != nil {
			return nil, st, err
		}
		cts[k] = ct
	}
	st.EncodeEncryptMs = ms(t0) / float64(len(updates))
	st.CiphertextBytes = cts[0].BinarySize()
	ks, err := s.aggregateAndDecrypt(cts, &st)
	if err != nil {
		return nil, st, err
	}
	t0 = time.Now()
	pt := rlwe.NewDecryptor(s.params, s.zero).DecryptNew(ks)
	vals := make([]int64, s.bgvP.MaxSlots())
	if err := ecd.Decode(pt, vals); err != nil {
		return nil, st, err
	}
	n := len(updates[0])
	out := make([]int64, n)
	t := int64(s.Setting.PlainMod)
	for i := 0; i < n; i++ {
		v := vals[i] % t
		if v > t/2 {
			v -= t
		} else if v < -t/2 {
			v += t
		}
		out[i] = v
	}
	st.CombineDecodeMs += ms(t0)
	return out, st, nil
}

func (s *Session) aggregateAndDecrypt(cts []*rlwe.Ciphertext, st *RoundStats) (*rlwe.Ciphertext, error) {
	t0 := time.Now()
	sum := cts[0].CopyNew()
	rq := s.params.RingQ().AtLevel(sum.Level())
	for _, ct := range cts[1:] {
		rq.Add(sum.Value[0], ct.Value[0], sum.Value[0])
		rq.Add(sum.Value[1], ct.Value[1], sum.Value[1])
	}
	st.AggregateMs = ms(t0)
	t0 = time.Now()
	var agg multiparty.KeySwitchShare
	for k, sk := range s.sks {
		sh := s.cks.AllocateShare(sum.Level())
		s.cks.GenShare(sk, s.zero, sum, &sh)
		if err := s.addFlood(sh.Value, sum.Level(), sum.IsNTT); err != nil {
			return nil, err
		}
		st.DecryptShareBytes = sh.BinarySize()
		if k == 0 {
			agg = sh
		} else if err := s.cks.AggregateShares(agg, sh, &agg); err != nil {
			return nil, err
		}
	}
	st.DecryptShareMs = ms(t0) / float64(len(s.sks))
	t0 = time.Now()
	ks := rlwe.NewCiphertext(s.params, 1, sum.Level())
	s.cks.KeySwitch(sum, agg, ks)
	st.CombineDecodeMs = ms(t0)
	return ks, nil
}

// addFlood adds to p, a share at levelQ, an integer uniform on [-F, F] in every coefficient, F = 2^f - 1: f+1 bits
// from crypto/rand, rejected when all are ones, so each of the 2F+1 values is equally likely, then reduced into every
// limb with Go's Euclidean modulus. The share is in the NTT domain when the ciphertext is, and the noise follows it.
func (s *Session) addFlood(p ring.Poly, levelQ int, isNTT bool) error {
	rq := s.params.RingQ().AtLevel(levelQ)
	f := s.Setting.FloodBits
	nb := (f + 1 + 7) / 8
	top := byte(0xff >> uint(8*nb-(f+1)))
	all := new(big.Int).Lsh(big.NewInt(1), uint(f+1))
	all.Sub(all, big.NewInt(1))
	moduli := rq.ModuliChain()[:levelQ+1]
	qs := make([]*big.Int, len(moduli))
	for j, q := range moduli {
		qs[j] = new(big.Int).SetUint64(q)
	}
	noise := rq.NewPoly()
	buf := make([]byte, nb)
	r, x, m := new(big.Int), new(big.Int), new(big.Int)
	for i := 0; i < rq.N(); i++ {
		for {
			if _, err := rand.Read(buf); err != nil {
				return err
			}
			buf[0] &= top
			r.SetBytes(buf)
			if r.Cmp(all) != 0 {
				break
			}
		}
		x.Sub(r, s.flood)
		for j := range moduli {
			noise.Coeffs[j][i] = m.Mod(x, qs[j]).Uint64()
		}
	}
	if isNTT {
		rq.NTT(noise, noise)
	}
	rq.Add(p, noise, p)
	return nil
}

