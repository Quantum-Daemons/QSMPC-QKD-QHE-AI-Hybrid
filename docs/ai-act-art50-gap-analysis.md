# EU AI Act Article 50 gap analysis

Non-signatories to the Commission's Code of Practice are "expected to demonstrate how they have
complied with their obligations under Article 50(2), (4) and (5) AI Act through other adequate means"
and "should carry out a gap analysis that compares the measures they have implemented with the
measures set out by a code of practice that is assessed as adequate" (Guidelines on Article 50,
C(2026) 5054 final, 20 July 2026, point 148). This is that document. The Article 50 section of
[`docs/pqc-roadmap.md`](pqc-roadmap.md) points here and carries no analysis of its own. Readings of the
Code below are this project's own; no authority has confirmed them.

## The instruments

| Instrument | Status | Date |
|---|---|---|
| Regulation (EU) 2024/1689 (AI Act), Article 50 | applicable | 2 August 2026 |
| Code of Practice on Transparency of AI-generated Content | final, assessed **adequate** 8-9 July 2026 | 10 June 2026 |
| Guidelines on Article 50, C(2026) 5054 final | adopted | 20 July 2026 |
| Regulation (EU) 2026/1744 (Digital Omnibus on AI), new Art. 111(4) | in force | 27 July 2026 |

## What this system does

One AI system is in scope: the Agentic AI panel, which runs a small language model **in the
visitor's browser** (Transformers.js or WebLLM), displays short free-form text, and offers that text
in a JSON download.

## Article 50(1) - interaction disclosure: MET

Implemented since Round 21, shown unconditionally rather than behind a disclosure toggle, because
the requirement is that the information reach the person, not that it be available to them.
Measured contrast ratio 12.42:1, above the 4.5:1 that WCAG 2.2 SC 1.4.3 (level AA) requires for normal text.

**No transitional relief applies to 50(1)** and none is claimed - the Guidelines (point 153) deny the
Art. 111(4) transition to the interactive half of a hybrid system.

## Article 50(2) - marking and detection

Two surfaces, with two answers. Until Round 86 this document answered for the first and said only a
future "download as PDF" button would change it, while the panel already offered a JSON download.

### The text on the page: no marking layer applies

The Code of Practice supplies the reasoning, and each step is quoted from it:

1. The output is **free-form text** - the glossary's own example is "the text that is shown on a
   website or within a chat".
2. Free-form text cannot carry metadata: "given that free-form text cannot transport metadata, a
   single-layer of marking as described in Sub-measure 1.1.2 is considered sufficient". So
   watermarking would be the only available layer.
3. Sub-measure 1.1.2 excepts short text: watermarking applies "with the exception of very short
   text", and the glossary defines very short text as "text shorter than 200 tokens".
4. **This system generates 96 tokens (Transformers.js) or 160 (WebLLM).** Both are under 200.

So no machine-readable marking is owed on the page. This is a Code provision rather than a statutory
one, and the underlying justification is sound independently: marking short text is provably limited -
Christ, Gunn and Zamir (COLT 2024) prove that "low empirical entropy outputs are not watermarked in any
undetectable watermarking scheme for any model", and Kirchenbauer et al.'s own Appendix G reports z of
0.25-0.40 for short greedy output against a detection threshold of 4.

### The JSON download: signed metadata, implemented

Sub-measure 1.1.1 applies "if content is generated, manipulated or exported in a data format that
supports attaching metadata", and the glossary's containerised text is text "embedded within a
structured file format or a container". This analysis reads a JSON file as one. The very-short-text
exception is written into Sub-measure 1.1.2 and not into 1.1.1, so the file owes the signed metadata and
no watermark. Since Round 86 every download holding generated text carries it.

| Code of Practice | What this system does | What falls short |
|---|---|---|
| 1.1.1: record "whether the content is AI-generated" in the metadata | `ai_provenance` in the file, with `ai_generated: true` and the SHA-256 of the text | Nothing, for the file. Text copied out of it leaves the record behind |
| 1.1.1: "digitally signed and time-stamped" | ML-DSA-65 over the record; `generated_at` and `signed_at` | The time is the visitor's clock, with no time authority |
| 1.1.1: secure signing keys, "except in circumstances where the deployment context does not permit" it, "for example in a local deployment scenario" | A key generated in the tab for one file and discarded after signing: a public page has nowhere to keep a private key | The glossary's digital signature "enables secure verification of authenticity of the provider". This one does not: **it shows the record and the text are unchanged since signing, not who signed them**. Anyone can edit a copy and sign it again under a new key |
| 1.1.2: imperceptible watermark, "with the exception of very short text" | None; under 200 tokens | None today; see the standing constraints |
| 1.2: non-removal of markings | Nothing removes the record | Anyone holding the file can delete it, and a file with no record shows nothing either way |
| 1.3 (optional): system, provider, timestamp, model identifier | All in the record | No model revision number |
| 2.1.1: a public specification or a piece of software, free of charge | The format below; `checkProvenance` in the page, `scripts/verify_ai_provenance.py` in the repository | None |
| 2.1.2: a user interface for the audience, and a result downloadable "in a digitally signed format" | "Check a saved file" on the panel, read in the tab and not uploaded | The result is shown, not offered as a signed download |
| 2.3: results that are clear, say which technique they rest on, and are accessible | A table naming signed metadata as the basis, and saying that no record is no finding | None known; the panel's contrast and control names are in the page audit |
| Commitment 3: robustness | Editing either the text or the record fails the check, in both implementations | Metadata does not survive re-encoding the text, which is why the Code wants a watermark above 200 tokens |

**Detection.** The Guidelines are explicit that marking without detection "will not suffice" (point
70), so the record ships with its check. Neither implementation looks at the text itself: a file with no
record is "no record", not "not AI-generated".

### The record: `qsh/ai-provenance/v1`

The public specification Sub-measure 2.1.1 asks for. `ai_provenance` in the download holds:

| Field | Value |
|---|---|
| `body` | A JSON text: the exact bytes signed, so a verifier needs no canonical form |
| `sig` | The ML-DSA-65 signature (FIPS 204, empty context) over the UTF-8 bytes of `body`, in hex |
| `alg` | `ML-DSA-65` |
| `public_key` | The verifying key, in hex |

`body` holds `spec` (`qsh/ai-provenance/v1`), `ai_generated`, `ai_system`, `provider`, `source`,
`generator` (`engine`, `model`, `runtime`), `generated_at` and `signed_at` (ISO 8601), `content_field`
(`run.text`), and `content_sha256`, the SHA-256 of the UTF-8 bytes of `run.text`, in hex.

To check a file: verify `sig` over `body` under `public_key`, parse `body`, and compare `content_sha256`
with the hash of `run.text`. The two implementations are `web/src/lib/aiProvenance.ts` and
`scripts/verify_ai_provenance.py`; `tests/fixtures/ai_provenance_sample.json`, signed by the first, is
checked by both in CI. The record carries no prompt and nothing about the visitor.

## Standing constraints - what would change this answer

- **Exceeding 200 tokens.** Raising `web.tfjs_max_new_tokens` or the WebLLM limit past 200 makes
  watermarking required. The implementable route is a Kirchenbauer green-list logits processor on
  the WebLLM `LogitProcessor` hook or the Transformers.js `logits_processor` option; both runtimes
  expose the hook. SynthID-Text is not viable here - it replaces the sampler, which neither runtime
  exposes, and its own paper evaluates nothing smaller than Gemma 2B.
- **The 200 moving.** The glossary states it "with the expectation that this threshold will decrease as
  new methods become available", and a Marking & Detection signatory task force has met since 30
  September 2026. Re-read it each round.
- **A second file format.** A PDF, DOCX or HTML download would need the same record inside that format,
  where a C2PA manifest is the Code's natural fit, rather than beside it.
- **C2PA.** Specification 2.5 (October 2026) added embedding and binding for append-only text, and
  c2pa-rs 0.91.0 (2026-09-21) added a plain-text asset handler. On the Conforming Products List of
  2026-11-02, 292 entries (220 on 2.2 only, 67 on 2.4 only, 5 on both), none lists `text/plain` and one
  2.2-only entry lists `txt`. A JSON file is not a format C2PA binds, so this is no route for the panel's
  download today. Read 2026-11-03.

## Open question for the operator

Article 111(4) grants until **2 December 2026** only to systems "placed on the market before
2 August 2026". Whether this demo was placed on the market or put into service before that date is a
factual question about when the public URL first became reachable, and it is not recorded anywhere in
this repository. It does not change the conclusion above, since the download has been marked since
Round 86 and the date of that is before the transition ends, but it should be established and written
down, because it determines whether the download was compliant from 2 August 2026 or only by 2 December.

Scope is also worth stating plainly: the free and open-source exclusion in Art. 2(12) is expressly
disapplied to Article 50, and "putting into service" in Art. 3(11) carries no commercial-activity
qualifier. This system should be treated as in scope.
