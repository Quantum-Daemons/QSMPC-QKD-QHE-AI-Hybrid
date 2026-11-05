// A signed, time-stamped record that a downloaded text is AI-generated, and the check that reads it
// back. The Commission's Code of Practice on Transparency of AI-generated Content applies its
// Sub-measure 1.1.1 to content "exported in a data format that supports attaching metadata", and the
// panel's JSON download is one: the record is that metadata, and `checkProvenance` is its detection
// mechanism (Measure 2.1). The format is specified in docs/ai-act-art50-gap-analysis.md;
// `scripts/verify_ai_provenance.py` is a second implementation of the check, in Python.
//
// WHAT THE SIGNATURE SHOWS. The key is generated here, for one file, and thrown away, because a public
// page has nowhere safe to keep a private key (the Sub-measure's own exception for a "local deployment
// scenario"). So it shows that the record and the text have not changed since signing, and not who
// signed them: anyone can edit a copy and sign it again under a new key, or delete the record. The time
// is the visitor's clock. The file says all of this, in `ai_provenance_note`.
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";
import { stableStringify } from "./clientMcp";

export const PROVENANCE_SPEC = "qsh/ai-provenance/v1";
const ALG = "ML-DSA-65";

export interface SignedProvenance { body: string; sig: string; alg: string; public_key: string; }

export interface ProvenanceInput {
  text: string; engine: string; model: string;
  generatedAt: string;   // when the model finished, ISO 8601
  signedAt?: Date;       // now, unless a caller fixes it
}

const enc = new TextEncoder();
const toHex = (u: Uint8Array) => Array.from(u).map((b) => b.toString(16).padStart(2, "0")).join("");
const fromHex = (h: string): Uint8Array => {
  if (h.length % 2 || /[^0-9a-f]/i.test(h)) throw new Error("not hex");
  return new Uint8Array((h.match(/.{2}/g) ?? []).map((x) => parseInt(x, 16)));
};

async function sha256Hex(text: string): Promise<string> {
  return toHex(new Uint8Array(await crypto.subtle.digest("SHA-256", enc.encode(text))));
}

export async function signProvenance(input: ProvenanceInput): Promise<SignedProvenance> {
  const body = stableStringify({
    spec: PROVENANCE_SPEC,
    ai_generated: true,
    ai_system: "QSH Agentic AI panel",
    provider: "Quantum Daemons",
    source: "https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid",
    generator: { engine: input.engine, model: input.model, runtime: "in-browser" },
    generated_at: input.generatedAt,
    signed_at: (input.signedAt ?? new Date()).toISOString(),
    content_field: "run.text",
    content_sha256: await sha256Hex(input.text),
  });
  const { publicKey, secretKey } = ml_dsa65.keygen(crypto.getRandomValues(new Uint8Array(32)));
  const sig = ml_dsa65.sign(enc.encode(body), secretKey);
  secretKey.fill(0);
  return { body, sig: toHex(sig), alg: ALG, public_key: toHex(publicKey) };
}

export interface ProvenanceCheck {
  found: boolean;                     // the file has a record to check
  signature_valid: boolean;           // the signature verifies under the key the file carries
  text_matches: boolean;              // run.text hashes to the record's content_sha256
  copy_matches: boolean | null;       // the MCP envelope's copy of the answer equals run.text; null if none
  ai_generated: boolean | null;
  record: Record<string, any> | null; // the signed fields, once the signature has verified
  problem: string | null;
}

/** Read a saved file's record back. Takes the parsed JSON; every field of it is untrusted. */
export async function checkProvenance(doc: any): Promise<ProvenanceCheck> {
  const fail = (found: boolean, problem: string): ProvenanceCheck => ({
    found, signature_valid: false, text_matches: false, copy_matches: null, ai_generated: null,
    record: null, problem });
  const p = doc?.ai_provenance;
  if (!p || typeof p !== "object") return fail(false, "This file has no AI-generated record.");
  if (typeof p.body !== "string" || typeof p.sig !== "string" || typeof p.public_key !== "string") {
    return fail(true, "The record is incomplete.");
  }
  if (p.alg !== ALG) return fail(true, `The record names ${String(p.alg)}, not ${ALG}.`);
  let ok = false;
  try {
    ok = ml_dsa65.verify(fromHex(p.sig), enc.encode(p.body), fromHex(p.public_key));
  } catch {
    ok = false;
  }
  if (!ok) return fail(true, "The signature does not verify: the record was changed after signing.");
  let record: Record<string, any>;
  try {
    record = JSON.parse(p.body);
  } catch {
    return fail(true, "The signed body is not JSON.");
  }
  if (record?.spec !== PROVENANCE_SPEC) return fail(true, `The record is not ${PROVENANCE_SPEC}.`);
  const text = doc?.run?.text;
  const textMatches = typeof text === "string" && (await sha256Hex(text)) === record.content_sha256;
  const answer = doc?.mcp?.echoedPayload?.answer;
  return {
    found: true, signature_valid: true, text_matches: textMatches,
    copy_matches: typeof answer === "string" ? answer === text : null,
    ai_generated: record.ai_generated === true, record,
    problem: textMatches ? null : "The record verifies, but run.text is not the text it was signed over.",
  };
}
