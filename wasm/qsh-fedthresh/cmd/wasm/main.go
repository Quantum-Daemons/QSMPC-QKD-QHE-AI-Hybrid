//go:build js && wasm

// The browser's handle on agg: globalThis.qshFedThresh, with typed arrays passed as little-endian bytes.
package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"syscall/js"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"

	"qsh-fedthresh/agg"
)

var sessions = map[int]*agg.Session{}
var next = 1

// A committee run: the members' keys, Carol's key pair and the borrower's ciphertexts as the pool stores them.
type committeeRun struct {
	c     *agg.Committee
	carol *agg.Recipient
	cts   []*rlwe.Ciphertext
}

var committees = map[int]*committeeRun{}

// integers reads float64s that must hold integers below 2^53 in magnitude.
func integers(b []byte) ([]int64, error) {
	f := floats(b)
	out := make([]int64, len(f))
	for i, v := range f {
		if v != math.Trunc(v) || math.Abs(v) >= 1<<53 {
			return nil, errors.New("a value is not an integer below 2^53")
		}
		out[i] = int64(v)
	}
	return out, nil
}

func bytesOf(v js.Value) []byte {
	b := make([]byte, v.Get("byteLength").Int())
	js.CopyBytesToGo(b, v)
	return b
}

func floats(b []byte) []float64 {
	out := make([]float64, len(b)/8)
	for i := range out {
		out[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return out
}

func toJSBytes(f []float64) js.Value {
	b := make([]byte, 8*len(f))
	for i, v := range f {
		binary.LittleEndian.PutUint64(b[8*i:], math.Float64bits(v))
	}
	u := js.Global().Get("Uint8Array").New(len(b))
	js.CopyBytesToJS(u, b)
	return u
}

func errObj(err error) any { return map[string]any{"error": err.Error()} }

func msSince(t time.Time) float64 { return float64(time.Since(t).Microseconds()) / 1000 }

func toJSUint8(b []byte) js.Value {
	u := js.Global().Get("Uint8Array").New(len(b))
	js.CopyBytesToJS(u, b)
	return u
}

func statsJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func main() {
	api := map[string]any{
		"settings": js.FuncOf(func(this js.Value, args []js.Value) any { return statsJSON(agg.Settings()) }),
		// open(kind, parties, crsSeed Uint8Array) -> {id, setting, keyShareBytes, keyGenMs} | {error}
		"open": js.FuncOf(func(this js.Value, args []js.Value) any {
			s, err := agg.Open(args[0].String(), args[1].Int(), bytesOf(args[2]))
			if err != nil {
				return errObj(err)
			}
			id := next
			next++
			sessions[id] = s
			return map[string]any{"id": id, "setting": statsJSON(s.Setting), "keyShareBytes": s.KeyShare, "keyGenMs": s.KeyGenMs}
		}),
		// round(id, flat Uint8Array of float64, parties, n) -> {aggregate Uint8Array of float64, stats} | {error}
		// CKKS settings take the weighted updates as floats; the BGV setting takes them as integers held in float64.
		"round": js.FuncOf(func(this js.Value, args []js.Value) any {
			s, ok := sessions[args[0].Int()]
			if !ok {
				return map[string]any{"error": "no such session"}
			}
			flat, parties, n := floats(bytesOf(args[1])), args[2].Int(), args[3].Int()
			if len(flat) != parties*n {
				return map[string]any{"error": "the update array is not parties x n"}
			}
			if s.Setting.Scheme == "BGV" {
				ups := make([][]int64, parties)
				for k := range ups {
					ups[k] = make([]int64, n)
					for i := 0; i < n; i++ {
						ups[k][i] = int64(flat[k*n+i])
					}
				}
				out, st, err := s.RoundInt(ups)
				if err != nil {
					return errObj(err)
				}
				f := make([]float64, n)
				for i, v := range out {
					f[i] = float64(v)
				}
				return map[string]any{"aggregate": toJSBytes(f), "stats": statsJSON(st)}
			}
			ups := make([][]float64, parties)
			for k := range ups {
				ups[k] = flat[k*n : (k+1)*n]
			}
			out, st, err := s.RoundFloat(ups)
			if err != nil {
				return errObj(err)
			}
			return map[string]any{"aggregate": toJSBytes(out), "stats": statsJSON(st)}
		}),
		"close": js.FuncOf(func(this js.Value, args []js.Value) any { delete(sessions, args[0].Int()); return nil }),
		// committeeOpen(members, threshold, crsSeed Uint8Array, maxWeights) ->
		//   {id, setting, keyGenMs, publicKeyBytes, carolPublicKeyBytes, carolKeyGenMs} | {error}
		// The members' keys and Carol's own key pair, all in this tab.
		"committeeOpen": js.FuncOf(func(this js.Value, args []js.Value) any {
			c, err := agg.OpenCommittee(args[0].Int(), args[1].Int(), bytesOf(args[2]), args[3].Float())
			if err != nil {
				return errObj(err)
			}
			t0 := time.Now()
			carol := c.NewRecipient()
			carolMs := msSince(t0)
			id := next
			next++
			committees[id] = &committeeRun{c: c, carol: carol}
			return map[string]any{"id": id, "setting": statsJSON(c.Setting), "keyGenMs": c.KeyGenMs,
				"publicKeyBytes": c.PublicKeyBytes(), "carolPublicKeyBytes": carol.PK.BinarySize(), "carolKeyGenMs": carolMs}
		}),
		// committeeEncrypt(id, balances Uint8Array of float64 integers) -> {ciphertexts Uint8Array, encryptMs} | {error}
		// The borrower's side: each balance in slot 0 of its own ciphertext under the committee's key, framed for the wire.
		"committeeEncrypt": js.FuncOf(func(this js.Value, args []js.Value) any {
			run, ok := committees[args[0].Int()]
			if !ok {
				return map[string]any{"error": "no such committee"}
			}
			xs, err := integers(bytesOf(args[1]))
			if err != nil {
				return errObj(err)
			}
			t0 := time.Now()
			cts := make([]*rlwe.Ciphertext, len(xs))
			for i, x := range xs {
				if cts[i], err = run.c.Encrypt(x); err != nil {
					return errObj(err)
				}
			}
			b, err := agg.MarshalCiphertexts(cts)
			if err != nil {
				return errObj(err)
			}
			return map[string]any{"ciphertexts": toJSUint8(b), "encryptMs": msSince(t0)}
		}),
		// committeeStore(id, ciphertexts Uint8Array, count) -> {count} | {error}. The pool's side, once per run: what it
		// opened from the borrower's sealed submission, checked for shape.
		"committeeStore": js.FuncOf(func(this js.Value, args []js.Value) any {
			run, ok := committees[args[0].Int()]
			if !ok {
				return map[string]any{"error": "no such committee"}
			}
			if run.cts != nil {
				return map[string]any{"error": "the pool already holds the borrower's ciphertexts"}
			}
			cts, err := run.c.UnmarshalCiphertexts(bytesOf(args[1]), args[2].Int())
			if err != nil {
				return errObj(err)
			}
			run.cts = cts
			return map[string]any{"count": len(cts)}
		}),
		// committeeCheck(id, weights, settled (Uint8Array of float64 integers), active Int32Array) ->
		//   {switched Uint8Array, poolShare, evaluateMs, evaluatedBytes, reencrypt} | {error}
		// The pool evaluates and masks; the active members re-encrypt to Carol; the pool combines their shares into
		// the ciphertext Carol receives. poolShare is t - r: it and what Carol decrypts add to the margin modulo t.
		"committeeCheck": js.FuncOf(func(this js.Value, args []js.Value) any {
			run, ok := committees[args[0].Int()]
			if !ok || run.cts == nil {
				return map[string]any{"error": "no such committee, or no ciphertexts stored"}
			}
			w, err := integers(bytesOf(args[1]))
			if err != nil {
				return errObj(err)
			}
			settled, err := integers(bytesOf(args[2]))
			if err != nil {
				return errObj(err)
			}
			active := make([]int, args[3].Get("length").Int())
			for i := range active {
				active[i] = args[3].Index(i).Int()
			}
			t0 := time.Now()
			ev, r, err := run.c.Evaluate(run.cts, w, settled)
			if err != nil {
				return errObj(err)
			}
			evaluateMs := msSince(t0)
			out, st, err := run.c.Reencrypt(ev, active, run.carol.PK)
			if err != nil {
				return errObj(err)
			}
			b, err := agg.MarshalCiphertexts([]*rlwe.Ciphertext{out})
			if err != nil {
				return errObj(err)
			}
			tm := run.c.Setting.PlainModulus
			return map[string]any{"switched": toJSUint8(b), "poolShare": float64((tm - r) % tm), "evaluateMs": evaluateMs,
				"evaluatedBytes": ev.BinarySize(), "reencrypt": statsJSON(st)}
		}),
		// committeeDecrypt(id, switched Uint8Array) -> {held, restZero, decryptMs} | {error}. Carol's side: held is
		// M + r mod t, and restZero says every other slot decrypted to zero.
		"committeeDecrypt": js.FuncOf(func(this js.Value, args []js.Value) any {
			run, ok := committees[args[0].Int()]
			if !ok {
				return map[string]any{"error": "no such committee"}
			}
			t0 := time.Now()
			cts, err := run.c.UnmarshalCiphertexts(bytesOf(args[1]), 1)
			if err != nil {
				return errObj(err)
			}
			held, rest, err := run.carol.Decrypt(cts[0])
			if err != nil {
				return errObj(err)
			}
			return map[string]any{"held": float64(held), "restZero": rest, "decryptMs": msSince(t0)}
		}),
		"committeeClose": js.FuncOf(func(this js.Value, args []js.Value) any { delete(committees, args[0].Int()); return nil }),
	}
	js.Global().Set("qshFedThresh", js.ValueOf(api))
	select {}
}
