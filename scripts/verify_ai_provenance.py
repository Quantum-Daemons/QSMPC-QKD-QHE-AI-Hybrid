#!/usr/bin/env python3
"""Check the AI-generated record in a file downloaded from the Agentic AI panel.

A second implementation of `checkProvenance` in `web/src/lib/aiProvenance.ts`, to the format in
`docs/ai-act-art50-gap-analysis.md`, so the check does not depend on the page that signed the file.

    python scripts/verify_ai_provenance.py qsh-agentic-finance_aml.json

Exit status 0 means the signature verifies and `run.text` is the text it was signed over. That shows the
text and the record are unchanged since signing, and not who signed them: the key travels in the file,
so anyone can edit a copy and sign it again. Nothing here looks at the text itself, so a file with no
record is "no record", not "not AI-generated".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPEC = "qsh/ai-provenance/v1"
ALG = "ML-DSA-65"


def check(doc: object) -> dict:
    """The same fields as the browser's `ProvenanceCheck`. Every field of ``doc`` is untrusted."""
    out = {"found": False, "signature_valid": False, "text_matches": False, "copy_matches": None,
           "ai_generated": None, "record": None, "problem": None}
    record = doc.get("ai_provenance") if isinstance(doc, dict) else None
    if not isinstance(record, dict):
        return {**out, "problem": "This file has no AI-generated record."}
    out["found"] = True
    body, sig, public_key = record.get("body"), record.get("sig"), record.get("public_key")
    if not all(isinstance(v, str) for v in (body, sig, public_key)):
        return {**out, "problem": "The record is incomplete."}
    if record.get("alg") != ALG:
        return {**out, "problem": f"The record names {record.get('alg')}, not {ALG}."}

    from qsh.pqc.signatures import verify

    try:
        valid = verify(body.encode(), bytes.fromhex(sig), bytes.fromhex(public_key), ALG)
    except ValueError:
        valid = False
    if not valid:
        return {**out, "problem": "The signature does not verify: the record was changed after signing."}
    try:
        signed = json.loads(body)
    except ValueError:
        return {**out, "problem": "The signed body is not JSON."}
    if not isinstance(signed, dict) or signed.get("spec") != SPEC:
        return {**out, "problem": f"The record is not {SPEC}."}

    run = doc.get("run")
    text = run.get("text") if isinstance(run, dict) else None
    matches = isinstance(text, str) and hashlib.sha256(text.encode()).hexdigest() == signed.get("content_sha256")
    mcp = doc.get("mcp")
    payload = mcp.get("echoedPayload") if isinstance(mcp, dict) else None
    answer = payload.get("answer") if isinstance(payload, dict) else None
    return {**out, "signature_valid": True, "text_matches": matches,
            "copy_matches": answer == text if isinstance(answer, str) else None,
            "ai_generated": signed.get("ai_generated") is True, "record": signed,
            "problem": None if matches else "The record verifies, but run.text is not the text it was signed over."}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        doc = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read {argv[1]} as JSON: {exc}")
        return 2
    result = check(doc)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["signature_valid"] and result["text_matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
