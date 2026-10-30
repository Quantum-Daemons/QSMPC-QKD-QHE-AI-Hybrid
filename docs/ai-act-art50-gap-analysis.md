# EU AI Act Article 50 gap analysis

Non-signatories to the Commission's Code of Practice are "expected to demonstrate how they have
complied through other adequate means" and "should carry out a gap analysis that compares the
measures they have implemented with the measures set out by a code of practice that is assessed as
adequate" (Guidelines on Article 50, C(2026) 5054 final, 20 July 2026, point 148). This is that
document. It is short by design: the analysis is short. The Article 50 section of
[`docs/pqc-roadmap.md`](pqc-roadmap.md) points here and carries no analysis of its own.

## The instruments

| Instrument | Status | Date |
|---|---|---|
| Regulation (EU) 2024/1689 (AI Act), Article 50 | applicable | 2 August 2026 |
| Code of Practice on Transparency of AI-generated Content | final, assessed **adequate** 8-9 July 2026 | 10 June 2026 |
| Guidelines on Article 50, C(2026) 5054 final | adopted | 20 July 2026 |
| Regulation (EU) 2026/1744 (Digital Omnibus on AI), new Art. 111(4) | in force | 27 July 2026 |

## What this system does

One AI system is in scope: the Agentic AI panel, which runs a small language model **in the
visitor's browser** (Transformers.js or WebLLM) and displays short free-form text.

## Article 50(1) - interaction disclosure: MET

Implemented since Round 21, shown unconditionally rather than behind a disclosure toggle, because
the requirement is that the information reach the person, not that it be available to them.
Measured contrast ratio 12.42:1, comfortably above WCAG 2.2 AA.

**No transitional relief applies to 50(1)** and none is claimed - the Guidelines (point 153) deny the
Art. 111(4) transition to the interactive half of a hybrid system.

## Article 50(2) - marking and detection: NOT APPLICABLE, and here is the chain

The Code of Practice supplies the reasoning, and each step is quoted from it:

1. The output is **free-form text** - the glossary's own example is "the text that is shown on a
   website or within a chat".
2. Free-form text cannot carry metadata: "given that free-form text cannot transport metadata, a
   single-layer of marking as described in Sub-measure 1.1.2 is considered sufficient". So
   watermarking would be the only available layer.
3. Sub-measure 1.1.2 excepts short text: watermarking applies "with the exception of very short
   text", and the glossary defines very short text as "text shorter than 200 tokens".
4. **This system generates 96 tokens (Transformers.js) or 160 (WebLLM).** Both are under 200.

Therefore no machine-readable marking layer is required. This is a Code provision rather than a
statutory one, and the underlying justification is sound independently: marking short text is not
merely inconvenient but provably limited - Christ, Gunn and Zamir (COLT 2024) prove that "low
empirical entropy outputs are not watermarked in any undetectable watermarking scheme for any
model", and Kirchenbauer et al.'s own Appendix G reports z of 0.25-0.40 for short greedy output
against a detection threshold of 4.

**Detection.** The Guidelines are explicit that marking without detection "will not suffice" (point
70). Since no marking layer applies, no detection obligation is triggered. This is also why the
downloaded artifact's provenance block is labelled as voluntary metadata rather than as an
Article 50(2) marking layer: claiming a marking layer would create a detection obligation that is
not otherwise owed.

## Standing constraints - what would change this answer

- **Exceeding 200 tokens.** Raising `web.tfjs_max_new_tokens` or the WebLLM limit past 200 makes
  watermarking required. The implementable route is a Kirchenbauer green-list logits processor on
  the WebLLM `LogitProcessor` hook or the Transformers.js `logits_processor` option; both runtimes
  expose the hook. SynthID-Text is not viable here - it replaces the sampler, which neither runtime
  exposes, and its own paper evaluates nothing smaller than Gemma 2B.
- **Shipping generated text inside a file.** Text in a container - PDF, DOCX, HTML - is grouped with
  images and video and attracts the **two-layer** rule. A "download as PDF" button would change the
  obligation.
- **C2PA.** Specification 2.4 (2026-04-01) does define text bindings, but no released library
  implements them and no product on the Conforming Products List validates them - all 157 entries
  are 2.2. Not a viable route today.

## Open question for the operator

Article 111(4) grants until **2 December 2026** only to systems "placed on the market before
2 August 2026". Whether this demo was placed on the market or put into service before that date is a
factual question about when the public URL first became reachable, and it is not recorded anywhere in
this repository. It does not change the conclusion above - no marking is owed either way - but it
should be established and written down, because it determines whether any future 50(2) obligation
would have had a transition.

Scope is also worth stating plainly: the free and open-source exclusion in Art. 2(12) is expressly
disapplied to Article 50, and "putting into service" in Art. 3(11) carries no commercial-activity
qualifier. This system should be treated as in scope.
