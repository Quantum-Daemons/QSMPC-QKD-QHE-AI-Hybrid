package agg

// A t-of-n committee in place of the DeFi page's single key holder (Round 86, the owner's choice of 2026-11-13).
//
// The lending check's margin, M = sum_i W_i x_i, is linear in what the borrower holds, so it needs no ciphertext
// product and no rotation. The borrower encrypts each quantised balance x_i in slot 0 of its own BGV ciphertext under
// the committee's collective key, once. For each check the pool multiplies each by that hour's public integer weight
// W_i, adds them, subtracts what has been settled in the open, adds a mask r uniform modulo t and a fresh encryption
// of zero, all in slot 0 alone: added as a scalar, r would land in every slot, and the slots that decrypt to r alone
// would hand it to whoever decrypts. t of the n members then re-encrypt that one ciphertext to Carol, who runs the
// garbled circuit with the pool, by Lattigo's public-key switch, each member flooding its share as the aggregation
// does (uniform, from crypto/rand, sized from the worst case of the evaluated noise over a whole share). Carol decrypts
// M + r mod t, the pool holds t - r, and the circuit gives the sign of M; neither learns M.
//
// ONE SHARE PER CIPHERTEXT. A member's share depends only on the ciphertext's second component, c1. Lattigo's
// SECURITY.md (citing Okada et al., ePrint 2025/409, and Colin de Verdiere et al., ePrint 2026/031) says a member must
// never share twice for the same input, whatever the active set. Two hours with the same prices would give the same
// c1, so the pool adds a fresh encryption of zero to every evaluated ciphertext, and the members refuse any c1 they
// have shared for before, by its SHA-256.
//
// THE PARAMETERS. t is a 50-bit prime, so every value that reaches the page is below 2^53 and exact in a float64. With
// the weights' absolute sum up to 2^27 the worst case of the evaluated noise is about 2^48, and 128 bits over a share
// need f = 189; the floods' sum must stay below Q/(2t). At N = 8192 that would take a Q of more than 2^241, past the 202
// bits the HomomorphicEncryption.org standard's post-quantum table allows there. At N = 16384 it allows 411, and Q is
// 260 bits.
//
// WHAT IT DOES NOT DO. Lattigo's multiparty protocols are secure against passive members only, and are not robust: a
// member who withholds a share stops the check, and nothing proves a share was computed correctly. Every member runs
// in the browser tab, so the separation is notional.

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"math"
	"math/big"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// CommitteeLogN and CommitteePlainBits are the committee's ring degree and the size of its plaintext modulus.
const (
	CommitteeLogN      = 14
	CommitteePlainBits = 50
)

// CommitteeLogQ is Q for the committee, 260 bits at N = 16384.
var CommitteeLogQ = []int{60, 60, 60, 60, 20}

// CommitteeBits is the statistical security of each re-encryption share, over all of its coefficients.
const CommitteeBits = 128.0

// CommitteeSetting is what the page reports about one committee.
type CommitteeSetting struct {
	Members        int     `json:"members"`
	Threshold      int     `json:"threshold"`
	LogN           int     `json:"logN"`
	LogQ           float64 `json:"logQ"`
	PlainModulus   uint64  `json:"plainModulus"`
	MaxWeights     float64 `json:"maxWeights"`
	NoiseBoundLog2 float64 `json:"noiseBoundLog2"`
	FloodBits      int     `json:"floodBits"`
	StatSecBits    float64 `json:"statSecBits"`
}

// Committee holds the members' keys for one run, in this tab.
type Committee struct {
	Setting  CommitteeSetting
	params   bgv.Parameters
	pk       *rlwe.PublicKey
	points   []multiparty.ShamirPublicPoint
	tsk      []multiparty.ShamirSecretShare
	pcks     multiparty.PublicKeySwitchProtocol
	enc      *bgv.Encoder
	used     map[[32]byte]bool
	KeyGenMs float64
}

// OpenCommittee makes n members' keys, the collective public key and every member's Shamir share of the collective
// secret with threshold t, for evaluations whose weights sum to at most maxWeights in absolute value.
func OpenCommittee(n, t int, crsSeed []byte, maxWeights float64) (*Committee, error) {
	if n < 2 || t < 1 || t > n {
		return nil, errors.New("a committee has at least two members and a threshold between 1 and its size")
	}
	if !(maxWeights >= 1) || maxWeights > math.Exp2(40) {
		return nil, errors.New("the weights' bound must be between 1 and 2^40")
	}
	g := ring.NewNTTFriendlyPrimesGenerator(CommitteePlainBits, uint64(2<<CommitteeLogN))
	pt, err := g.NextUpstreamPrime()
	if err != nil {
		return nil, err
	}
	if pt >= 1<<53 {
		return nil, errors.New("the plaintext modulus must stay below 2^53")
	}
	params, err := bgv.NewParametersFromLiteral(bgv.ParametersLiteral{LogN: CommitteeLogN, LogQ: CommitteeLogQ, PlaintextModulus: pt})
	if err != nil {
		return nil, err
	}
	c := &Committee{params: params, enc: bgv.NewEncoder(params), used: map[[32]byte]bool{}}
	t0 := time.Now()
	rp := params.Parameters
	kgen := rlwe.NewKeyGenerator(rp)
	// The common reference string's seed, hashed to 32 bytes: the PRNG is keyed BLAKE2b, whose key is at most 64.
	seed := sha256.Sum256(crsSeed)
	crs, err := sampling.NewKeyedPRNG(seed[:])
	if err != nil {
		return nil, err
	}
	ckg := multiparty.NewPublicKeyGenProtocol(rp)
	crp := ckg.SampleCRP(crs)
	thr := multiparty.NewThresholdizer(rp)
	polys := make([]multiparty.ShamirPolynomial, n)
	c.points = make([]multiparty.ShamirPublicPoint, n)
	var agg multiparty.PublicKeyGenShare
	for i := range polys {
		c.points[i] = multiparty.ShamirPublicPoint(i + 1)
		sk := kgen.GenSecretKeyNew()
		sh := ckg.AllocateShare()
		ckg.GenShare(sk, crp, &sh)
		if i == 0 {
			agg = sh
		} else {
			ckg.AggregateShares(agg, sh, &agg)
		}
		if polys[i], err = thr.GenShamirPolynomial(t, sk); err != nil {
			return nil, err
		}
	}
	c.pk = rlwe.NewPublicKey(rp)
	ckg.GenPublicKey(agg, crp, c.pk)
	// Member j's share of the collective secret: the sum of every member's polynomial at j's point.
	c.tsk = make([]multiparty.ShamirSecretShare, n)
	for j := range c.tsk {
		c.tsk[j] = thr.AllocateThresholdSecretShare()
		for i := range polys {
			share := thr.AllocateThresholdSecretShare()
			thr.GenShamirSecretShare(c.points[j], polys[i], &share)
			if err := thr.AggregateShares(c.tsk[j], share, &c.tsk[j]); err != nil {
				return nil, err
			}
		}
	}
	// The worst case of the evaluated ciphertext's noise: each fresh encryption under the collective key of n members
	// decrypts with noise at most 2 N n b + b (NoiseBound over n ciphertexts, divided by n); the weights multiply the
	// balances' and the re-randomising encryption of zero adds one more.
	b := (maxWeights + 1) * NoiseBound(rp, n) / float64(n)
	f := leastFloodBits(CommitteeBits, rp.N(), b)
	logQ := 0.0
	for _, q := range rp.Q() {
		logQ += math.Log2(float64(q))
	}
	// Decryption needs the whole noise below Q/(2t): the t floods, the evaluated noise and the switch's own, with a
	// bit to spare.
	if float64(f)+math.Log2(float64(t))+2 >= logQ-1-math.Log2(float64(pt)) {
		return nil, errors.New("the committee's flooding does not fit in Q")
	}
	if c.pcks, err = multiparty.NewPublicKeySwitchProtocol(rp, ring.DiscreteGaussian{Sigma: rp.NoiseFreshSK(), Bound: 6 * rp.NoiseFreshSK()}); err != nil {
		return nil, err
	}
	c.Setting = CommitteeSetting{Members: n, Threshold: t, LogN: CommitteeLogN, LogQ: logQ, PlainModulus: pt,
		MaxWeights: maxWeights, NoiseBoundLog2: math.Log2(b), FloodBits: f, StatSecBits: CommitteeBits}
	c.KeyGenMs = ms(t0)
	return c, nil
}

// PublicKeyBytes is the size of the collective public key the borrower encrypts under.
func (c *Committee) PublicKeyBytes() int { return c.pk.BinarySize() }

// Encrypt is the borrower's: one value in slot 0 of its own ciphertext under the committee's key.
func (c *Committee) Encrypt(v int64) (*rlwe.Ciphertext, error) {
	pt := bgv.NewPlaintext(c.params, c.params.MaxLevel())
	if err := c.enc.Encode([]int64{v}, pt); err != nil {
		return nil, err
	}
	return rlwe.NewEncryptor(c.params, c.pk).EncryptNew(pt)
}

// Evaluate is the pool's: sum_i w_i (ct_i - settled_i), plus a mask r uniform modulo t and a fresh encryption of zero,
// all in slot 0. It returns the ciphertext and r; the pool's share of the margin is t - r.
func (c *Committee) Evaluate(cts []*rlwe.Ciphertext, w, settled []int64) (*rlwe.Ciphertext, uint64, error) {
	if len(cts) == 0 || len(cts) != len(w) || len(w) != len(settled) {
		return nil, 0, errors.New("one weight and one settled amount per ciphertext")
	}
	sum := 0.0
	for _, x := range w {
		sum += math.Abs(float64(x))
	}
	if sum > c.Setting.MaxWeights {
		return nil, 0, errors.New("the weights exceed the bound the flooding was sized for")
	}
	tBig := new(big.Int).SetUint64(c.params.PlaintextModulus())
	rBig, err := rand.Int(rand.Reader, tBig)
	if err != nil {
		return nil, 0, err
	}
	// The public constant: r less what has been settled, at this hour's weights, modulo t.
	k := new(big.Int).Set(rBig)
	for i := range w {
		k.Sub(k, new(big.Int).Mul(big.NewInt(w[i]), big.NewInt(settled[i])))
	}
	k.Mod(k, tBig)
	ev := bgv.NewEvaluator(c.params, nil)
	out, err := ev.MulNew(cts[0], w[0])
	if err != nil {
		return nil, 0, err
	}
	for i := 1; i < len(cts); i++ {
		if err := ev.MulThenAdd(cts[i], w[i], out); err != nil {
			return nil, 0, err
		}
	}
	pt := bgv.NewPlaintext(c.params, out.Level())
	if err := c.enc.Encode([]uint64{k.Uint64()}, pt); err != nil {
		return nil, 0, err
	}
	if err := ev.Add(out, pt, out); err != nil {
		return nil, 0, err
	}
	// A fresh encryption of zero, made as an encryption of the zero plaintext so that its metadata matches.
	zero, err := rlwe.NewEncryptor(c.params, c.pk).EncryptNew(bgv.NewPlaintext(c.params, out.Level()))
	if err != nil {
		return nil, 0, err
	}
	if err := ev.Add(out, zero, out); err != nil {
		return nil, 0, err
	}
	return out, rBig.Uint64(), nil
}

// ReencryptStats are one re-encryption's costs.
type ReencryptStats struct {
	ShareMs     float64 `json:"shareMs"`    // per member, the flooding included
	FloodMs     float64 `json:"floodMs"`    // per member
	ShareBytes  int     `json:"shareBytes"` // one member's share, to Carol
	C1Bytes     int     `json:"c1Bytes"`    // what each active member receives from the pool
	C0Bytes     int     `json:"c0Bytes"`    // what Carol receives from the pool
	CombineMs   float64 `json:"combineMs"`
	ActiveCount int     `json:"activeCount"`
}

// Reencrypt has the members at the given indices (exactly t of them) re-encrypt ct to pk by the public-key switch,
// each share flooded. The members share once for a given c1: a second request for it is refused whatever the
// active set.
func (c *Committee) Reencrypt(ct *rlwe.Ciphertext, active []int, pk *rlwe.PublicKey) (*rlwe.Ciphertext, ReencryptStats, error) {
	var st ReencryptStats
	if len(active) != c.Setting.Threshold {
		return nil, st, errors.New("exactly t members re-encrypt")
	}
	pts := make([]multiparty.ShamirPublicPoint, len(active))
	seen := map[int]bool{}
	for k, i := range active {
		if i < 0 || i >= len(c.points) || seen[i] {
			return nil, st, errors.New("the active members must be distinct members of the committee")
		}
		seen[i] = true
		pts[k] = c.points[i]
	}
	c1, err := ct.Value[1].MarshalBinary()
	if err != nil {
		return nil, st, err
	}
	digest := sha256.Sum256(c1)
	if c.used[digest] {
		return nil, st, errors.New("the members have already shared for this ciphertext; a second set of shares is refused")
	}
	c.used[digest] = true
	st.C1Bytes, st.C0Bytes, st.ActiveCount = ct.Value[1].BinarySize(), ct.Value[0].BinarySize(), len(active)
	rp := c.params.Parameters
	t0 := time.Now()
	var floodTotal float64
	var agg multiparty.PublicKeySwitchShare
	for k, i := range active {
		others := make([]multiparty.ShamirPublicPoint, 0, len(c.points)-1)
		for _, p := range c.points {
			if p != c.points[i] {
				others = append(others, p)
			}
		}
		cmb := multiparty.NewCombiner(rp, c.points[i], others, c.Setting.Threshold)
		add := rlwe.NewSecretKey(rp)
		if err := cmb.GenAdditiveShare(pts, c.points[i], c.tsk[i], add); err != nil {
			return nil, st, err
		}
		sh := c.pcks.AllocateShare(ct.Level())
		c.pcks.GenShare(add, pk, ct, &sh)
		tf := time.Now()
		if err := addUniformFlood(rp.RingQ().AtLevel(ct.Level()), sh.Value[0], c.Setting.FloodBits, ct.IsNTT); err != nil {
			return nil, st, err
		}
		floodTotal += ms(tf)
		st.ShareBytes = sh.BinarySize()
		if k == 0 {
			agg = sh
		} else if err := c.pcks.AggregateShares(agg, sh, &agg); err != nil {
			return nil, st, err
		}
	}
	st.ShareMs = ms(t0) / float64(len(active))
	st.FloodMs = floodTotal / float64(len(active))
	t0 = time.Now()
	out := rlwe.NewCiphertext(rp, 1, ct.Level())
	c.pcks.KeySwitch(ct, agg, out)
	st.CombineMs = ms(t0)
	return out, st, nil
}

// Recipient is Carol's own BGV key pair, to which the committee re-encrypts.
type Recipient struct {
	sk *rlwe.SecretKey
	PK *rlwe.PublicKey
	c  *Committee
}

// NewRecipient makes a key pair under the committee's parameters.
func (c *Committee) NewRecipient() *Recipient {
	sk, pk := rlwe.NewKeyGenerator(c.params.Parameters).GenKeyPairNew()
	return &Recipient{sk: sk, PK: pk, c: c}
}

// Decrypt returns slot 0 of ct, in [0, t), and whether every other slot is zero (it must be: only slot 0 was used).
func (r *Recipient) Decrypt(ct *rlwe.Ciphertext) (uint64, bool, error) {
	pt := rlwe.NewDecryptor(r.c.params.Parameters, r.sk).DecryptNew(ct)
	vals := make([]uint64, r.c.params.MaxSlots())
	if err := r.c.enc.Decode(pt, vals); err != nil {
		return 0, false, err
	}
	rest := true
	for _, v := range vals[1:] {
		if v != 0 {
			rest = false
			break
		}
	}
	return vals[0], rest, nil
}

// MaxCommitteeCiphertexts is the most ciphertexts one message may carry.
const MaxCommitteeCiphertexts = 16

// MarshalCiphertexts frames ciphertexts for the wire: a little-endian uint32 count, then each one's length and bytes.
func MarshalCiphertexts(cts []*rlwe.Ciphertext) ([]byte, error) {
	out := binary.LittleEndian.AppendUint32(nil, uint32(len(cts)))
	for _, ct := range cts {
		b, err := ct.MarshalBinary()
		if err != nil {
			return nil, err
		}
		out = binary.LittleEndian.AppendUint32(out, uint32(len(b)))
		out = append(out, b...)
	}
	return out, nil
}

// UnmarshalCiphertexts reads what MarshalCiphertexts wrote and checks each ciphertext's shape against the
// committee's parameters: degree 1, the top level, N coefficients, in the NTT domain and batched, as Encrypt and
// Reencrypt make them. Anything else, a short message or trailing bytes included, is refused.
func (c *Committee) UnmarshalCiphertexts(b []byte, want int) ([]*rlwe.Ciphertext, error) {
	if len(b) < 4 {
		return nil, errors.New("a ciphertext message too short for its count")
	}
	n := int(binary.LittleEndian.Uint32(b))
	if n != want || n < 1 || n > MaxCommitteeCiphertexts {
		return nil, errors.New("an unexpected number of ciphertexts")
	}
	b = b[4:]
	cts := make([]*rlwe.Ciphertext, n)
	for i := range cts {
		if len(b) < 4 {
			return nil, errors.New("a ciphertext message cut short")
		}
		l := int(binary.LittleEndian.Uint32(b))
		if l > len(b)-4 {
			return nil, errors.New("a ciphertext message cut short")
		}
		ct := new(rlwe.Ciphertext)
		if err := ct.UnmarshalBinary(b[4 : 4+l]); err != nil {
			return nil, err
		}
		if ct.Degree() != 1 || ct.Level() != c.params.MaxLevel() || ct.Value[0].N() != c.params.N() || !ct.IsNTT || !ct.IsBatched {
			return nil, errors.New("a ciphertext not of the committee's shape")
		}
		cts[i] = ct
		b = b[4+l:]
	}
	if len(b) != 0 {
		return nil, errors.New("trailing bytes after the ciphertexts")
	}
	return cts, nil
}

// SignedMod maps v in [0, t) to its representative in (-t/2, t/2].
func SignedMod(v, t uint64) int64 {
	if v > t/2 {
		return -int64(t - v)
	}
	return int64(v)
}
