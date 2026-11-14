# A run's timeline and its per-party view

Read on 2026-11-14 for the use-case pages' replay controls and "Who holds what, and what crossed"
(`web/src/lib/runTimeline.ts`, `web/src/components/TimelineBar.tsx`, `web/src/components/PartyLedger.tsx`), the
owner's choice of that day among four ways to explain a run rather than add cryptography. Each source was opened as an
HTML page and read for the point named; nothing was downloaded.

## What the explainers that work do

- The Garbled Circuit Visualizer (stringhandler, no date; read: html) moves through its protocol with Reset, Back,
  Forward, Run All and Next, counts the steps, shows the phase in a progress bar (Garble, Send GC, Alice Labels, OT,
  Evaluate, Done), and gives each party its own panel: Alice holds both labels of every wire, Bob only the one per wire
  he receives, in a "Known Wire Labels" table. The replay controls and the per-party panel follow it.
- Mitchell and Pearce, *How Federated Learning Protects Privacy*, PAIR Explorables, November 2022 (read: html), shows
  secure aggregation as each device's update masked with shared random numbers, the server seeing only the sum, which
  stays the same while what each device sends changes. The ledger's federated rows say the same thing in words: the
  hospitals send masked updates, the aggregator holds the sum and no hospital's own update.
- Hohman, Conlen, Heer and Chau, *Communicating with Interactive Articles*, Distill, 2020 (read: html), argue for
  details on demand. The ledger is a card the reader opens by scrolling to it, and the canvas stays the summary.

## What this project does with them, and where it stops

- **One record.** The page already kept every event of a run in order for the live log and the Events (JSON) export.
  The timeline folds that log to a position with the rules the live page applies, so a replayed position shows what the
  page showed at that moment; nothing runs again and no generator is re-seeded. `tests/test_r86_run_timeline.py` holds
  the fold at the end of a run to the run's own final metrics, on three pages.
- **The parties' messages are the run's own.** Each stage logs what crossed between the sides and what each side now
  holds, with the size it measured: the handshake's 1,184 + 32 and 1,088 + 32 bytes at ML-KEM-768, the OPRF PSI's sealed
  transcript, the CKKS input ciphertext, the garbled circuit's traffic, the lending checks' own ledger. Where a stage does
  not measure a size (the federated updates, the simulated quantum channel's classical messages) the ledger says "not
  measured" rather than a figure of its own.
- **Nothing a party keeps.** Hex is only the first eight bytes of something sent in the clear (a public key, a KEM
  ciphertext, a signature). The log is exported and copied into the results bundle as it is, so no key, seed, share or
  set member may enter it; the test seeds a session-key leak to show the check fails.
- **The animation from the record.** The GIF and WebM of a finished run are drawn by replaying the log to each stage
  boundary and capturing the figure once it has been painted, one frame per boundary. They were captured live, inside
  the event handler, before the page had drawn the event.
- **Where it stops.** The cards built from a run's artifacts (the agreement scatter, the comparison table, the lending
  panels) arrive whole at the end of the run and are not replayed. All parties run in the visitor's tab, so the
  separation the ledger draws is notional, as its badge says. The full deployment's backend does not emit the per-party
  events, so over it the timeline replays stages and metrics and the ledger stays empty.
- **The same messages as a sequence.** The owner's fourth choice of the same day: one participant per party, the
  messages numbered in order with their sizes, each stage a shaded block, a dotted arrow for a simulated channel, and a
  note for what a party keeps, as Mermaid's `sequenceDiagram` draws them (participants, `autonumber`, `rect` blocks,
  notes; https://mermaid.js.org/syntax/sequenceDiagram.html, read: html). The page draws it itself as SVG and offers the
  same diagram as Mermaid text to copy or download. Mermaid is not loaded on the page: a lazy import of Mermaid 12.2.0
  added 57 chunk files, 5.1 MB, to the client-only build (103.8 MB to 108.8 MB, measured 2026-11-14), every diagram type
  among them, the ELK layout (elkjs, EPL-2.0 or GPL-3.0) included, for one diagram the page can draw in a few kB. The
  generated text of three runs renders with Mermaid 12.2.0 in the repository's own headless check.
