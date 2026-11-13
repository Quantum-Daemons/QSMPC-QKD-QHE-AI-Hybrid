//go:build js && wasm

// The browser's handle on agg: globalThis.qshFedThresh, with typed arrays passed as little-endian bytes.
package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"syscall/js"

	"qsh-fedthresh/agg"
)

var sessions = map[int]*agg.Session{}
var next = 1

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
	}
	js.Global().Set("qshFedThresh", js.ValueOf(api))
	select {}
}
