// Package agg aggregates four hospitals' FedAvg updates under Lattigo's N-out-of-N threshold encryption: a collective
// public key from each hospital's secret share, encryption under it, ciphertext addition by the aggregator, and one
// collective decryption per round by key-switching to the zero key, each hospital's share flooded with noise.
//
// Three settings, measured side by side on the medical page:
//
//	ckks-demo  CKKS, smudging sigma 2^30 (the value Lattigo's multiparty examples use), scale 2^60, Q {60, 60}
//	ckks-l40   CKKS, smudging for 40 bits of statistical security over the noise bound, scale 2^80, Q {60, 60}
//	bgv-128    BGV, exact, smudging for 128 bits, Q {60, 60, 60}, a 33-bit plaintext modulus; the hospitals quantise
//	           exactly as the pairwise masks do, so the aggregate is the masks' to the last bit
//
// Every prime of Q is kept above the smudging bound 6*sigma: below it, Lattigo v6.2.0's Gaussian sampler writes
// unreduced residues and the decryption fails (reproduced 2026-11-13, reported as tuneinsight/lattigo#596).
//
// Each aggregate is decrypted once, by construction: a round encrypts fresh ciphertexts, sums them inside
// aggregateAndDecrypt and returns only the plaintext, so no caller can ask a hospital for a second share of the same
// ciphertext. Lattigo's SECURITY.md says retrying a multiparty protocol must be considered insecure.
package agg

import (
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

// NoiseBoundLog2 bounds the aggregate ciphertext's noise before smudging, log2 of the largest coefficient: four
// fresh public-key encryptions summed measured 2^12.3 to 2^12.8 at N = 8192 (2026-11-13); 2^14 leaves a margin.
const NoiseBoundLog2 = 14

// Setting describes one threshold configuration.
type Setting struct {
	Kind        string  `json:"kind"`
	Scheme      string  `json:"scheme"`
	LogN        int     `json:"logN"`
	LogQ        []int   `json:"logQ"`
	LogScale    int     `json:"logScale,omitempty"`
	SigmaLog2   float64 `json:"sigmaLog2"`
	StatSecBits float64 `json:"statSecBits"`
	PlainMod    uint64  `json:"plaintextModulus,omitempty"`
}

// Settings returns the three settings by kind.
func Settings() map[string]Setting {
	return map[string]Setting{
		"ckks-demo": {Kind: "ckks-demo", Scheme: "CKKS", LogN: 13, LogQ: []int{60, 60}, LogScale: 60, SigmaLog2: 30,
			StatSecBits: 30 - NoiseBoundLog2},
		"ckks-l40": {Kind: "ckks-l40", Scheme: "CKKS", LogN: 13, LogQ: []int{60, 60}, LogScale: 80,
			SigmaLog2: 40 + NoiseBoundLog2, StatSecBits: 40},
		"bgv-128": {Kind: "bgv-128", Scheme: "BGV", LogN: 13, LogQ: []int{60, 60, 60}, SigmaLog2: 128 + NoiseBoundLog2,
			StatSecBits: 128},
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
	sigma := math.Exp2(st.SigmaLog2)
	minPrime := math.Inf(1)
	for _, q := range s.params.Q() {
		minPrime = math.Min(minPrime, float64(q))
	}
	// The sampler's big-integer branch, which reduces each limb, runs only when sigma > 2^53 and the bound exceeds
	// 2^64 - 1; in its integer branch a bound reaching a prime corrupts the shares (tuneinsight/lattigo#596).
	bigBranch := sigma > 0x20000000000000 && 6*sigma > 0xffffffffffffffff
	if !bigBranch && 6*sigma >= minPrime {
		return nil, errors.New("the smudging bound reaches a prime of Q; Lattigo v6.2.0's sampler would corrupt the shares")
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
	s.cks, err = multiparty.NewKeySwitchProtocol(s.params, ring.DiscreteGaussian{Sigma: sigma, Bound: 6 * sigma})
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
