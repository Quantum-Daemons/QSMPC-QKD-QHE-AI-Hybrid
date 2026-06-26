// In-browser post-quantum crypto demo (pure JS, no server compute) using
// @noble/post-quantum: ML-KEM-768 key encapsulation + ML-DSA-65 signatures.
// This is what the public demo runs client-side on the tiny VPS.
import { ml_kem768 } from "@noble/post-quantum/ml-kem";
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa";

const hex = (u: Uint8Array, n = 16) =>
  Array.from(u.slice(0, n)).map((b) => b.toString(16).padStart(2, "0")).join("");

export interface KemDemo {
  algorithm: string;
  publicKeyBytes: number;
  cipherTextBytes: number;
  sharedSecretBytes: number;
  agreement: boolean;
  sharedSecretPrefix: string;
  elapsedMs: number;
}

export function mlkemDemo(): KemDemo {
  const t0 = performance.now();
  const bob = ml_kem768.keygen();
  const { cipherText, sharedSecret } = ml_kem768.encapsulate(bob.publicKey);
  const recovered = ml_kem768.decapsulate(cipherText, bob.secretKey);
  const agreement =
    sharedSecret.length === recovered.length &&
    sharedSecret.every((b, i) => b === recovered[i]);
  return {
    algorithm: "ML-KEM-768 (FIPS 203)",
    publicKeyBytes: bob.publicKey.length,
    cipherTextBytes: cipherText.length,
    sharedSecretBytes: sharedSecret.length,
    agreement,
    sharedSecretPrefix: hex(sharedSecret),
    elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
  };
}

export interface SigDemo {
  algorithm: string;
  publicKeyBytes: number;
  signatureBytes: number;
  validSignature: boolean;
  rejectsTampered: boolean;
  elapsedMs: number;
}

export function mldsaDemo(message = "QSH orchestration message"): SigDemo {
  const t0 = performance.now();
  const enc = new TextEncoder();
  const seed = crypto.getRandomValues(new Uint8Array(32));
  const { publicKey, secretKey } = ml_dsa65.keygen(seed);
  const msg = enc.encode(message);
  const sig = ml_dsa65.sign(secretKey, msg);
  const valid = ml_dsa65.verify(publicKey, msg, sig);
  const tampered = ml_dsa65.verify(publicKey, enc.encode(message + "!"), sig);
  return {
    algorithm: "ML-DSA-65 (FIPS 204)",
    publicKeyBytes: publicKey.length,
    signatureBytes: sig.length,
    validSignature: valid,
    rejectsTampered: !tampered,
    elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
  };
}
