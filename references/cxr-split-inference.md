# Split inference on a chest X-ray: where published work draws the line, and where the card stops

Read on 2026-11-15 for the medical page's "Split inference on a chest X-ray" card
(`web/src/components/CxrSplitInference.tsx`, `web/src/lib/cxrSplit.ts`, `scripts/export_cxr_head.py`), the owner's
fourth choice of 2026-11-14. Each source was opened as an HTML page (an abstract page, a conference page or the HTML
rendering of the paper) and read for the point named; nothing was downloaded.

## Where published work draws the line between the parties

- **Kim, Jo, Ra and Park**, *Privacy Preserving Chest X-ray Classification in Latent Space with Homomorphically
  Encrypted Neural Inference*, MICCAI 2025, LNCS 15973, pp. 485-495, doi:10.1007/978-3-032-05185-1_47; arXiv:2506.15258v2
  (read: html). The client compresses the film with a VQGAN encoder (a $`32 \times 32 \times 4`$ latent at $`f = 8`$),
  encrypts the latent under CKKS (TenSEAL on SEAL) and keeps the key. The server runs a whole CNN on the ciphertext, with
  ReLU replaced by a quadratic and the sigmoid-like steps of the squeeze-and-excitation block by a cubic, and returns
  encrypted logits, which the client decrypts and passes through a sigmoid. On CheXpert with ResNet20 (Table 3), micro
  AUROC 0.918 in the clear and 0.912 encrypted, at 76,913.72 ms against 6.23 ms. For NIH's films they report the model in
  the clear only (micro AUROC 0.862, Table 1).
- **Li, Wu and Miao**, *Decision-Aware Quadratic ReLU Replacement for HE-Friendly Inference*, arXiv:2605.22237v2, May
  2026 (read: html). A frozen extractor runs on the client (DINOv2 for images, Qwen3-Embedding-0.6B for text), its
  embedding is encrypted, and the server evaluates a one-hidden-layer MLP head whose ReLU is replaced, without retraining,
  by a quadratic chosen on a calibration set. SEAL 4.1, leveled CKKS without bootstrapping, $`N = 2^{14}`$ or $`2^{15}`$,
  depth 4 for the quadratic; 46.81 ms a sample, amortised, on DINOv2 features of FGVC-Aircraft. They note that without
  extraction on the client, FHE protects only the head's computation on the features it is given.
- **Galanopoulos, Provatas and Georgakopoulos-Soares**, *Feasibility of Homomorphic Inference for a Genomic Foundation
  Model*, arXiv:2609.16211v1, 14 September 2026 (read: html, the abstract). Client-assisted approximate HE: the provider
  evaluates the linear algebra on ciphertexts and the key-holding data owner evaluates normalisation, causal softmax and
  the activations at fixed boundaries. One input at full prompt length took 6,683 s on one accelerator, and the private
  lookup of the token embeddings is left open.
- **Pereteanu, Alansary and Passerat-Palmbach**, *Split HE: Fast Secure Inference Combining Split Learning and
  Homomorphic Encryption*, PPAI'22 (the AAAI workshop), arXiv:2202.13351 (read: html, the abstract). A subset of the
  weights runs in the clear on the client, as in SplitNN, and the rest under HE, with SEAL through TenSEAL.
- **Lee, Lee, Kim, Shin and Lee**, *HETAL: Efficient Privacy-preserving Transfer Learning with Homomorphic Encryption*,
  ICML 2023, PMLR 202:19010-19035 (read: html, the abstract). The training counterpart: the classification layers added
  to a pretrained model are fine-tuned under CKKS on the client's encrypted data.

## What the card does with them

- **The cheapest end of that range.** The card draws the line after the pooling. The network up to its 512 features
  runs in the clear on the data owner's side, and only the last layer, $`z = w^\top x + b`$ with 512 weights, is
  evaluated under CKKS. No activation is evaluated under encryption, so nothing is approximated but the scheme's own
  noise. The sigmoid and the operating point are applied after decryption, as Kim et al.'s client applies its sigmoid.
  Measured in Chrome on a local client-only build (the log's entry for PR #55): one ciphertext of 393,329 bytes, the
  layer in 52 ms with 9 rotations ($`\log_2 512`$), and the decrypted logit $`1.23 \times 10^{-9}`$ from the one in the
  clear. Kim et al.'s encrypted ResNet20 takes 76,913.72 ms in their Table 3 (the table does not say per how many
  films); Li et al.'s one-hidden-layer head needs a quadratic and depth 4.
- **What that line asks of the data owner.** The owner runs the backbone, so it needs the backbone's weights, which this
  model publishes on the Hub. Galanopoulos et al. keep the model with the provider and pay with a round trip to the owner
  at every nonlinearity.
- **What is not comparable.** Kim et al.'s figures are multi-label AUROC on CheXpert for a CNN trained on VQGAN latents.
  The card's model is a single-label effusion classifier on NIH's films (FedAvg ResNet-18), and the card's figure is the
  agreement of the encrypted logit with the clear one, not an accuracy. The model's own accuracy is on its card and in the
  medical page's comparison table.

## What this split does not protect

- **The holder's last layer, from the owner.** The owner decrypts the logit of a linear map on features it chooses, so
  513 linearly independent feature vectors give 513 equations in the 512 weights and the bias, which then follow to
  within the scheme's noise (this project's arithmetic). Tramer, Zhang, Juels, Reiter and Ristenpart, *Stealing Machine
  Learning Models via Prediction APIs*, USENIX Security 2016, pp. 601-618 (read: html, the abstract), extract logistic
  regression and other models with near-perfect fidelity where confidence values are returned. On this page the last
  layer is published, and the card says that the split would not keep a secret one from the owner.
- **The key, if the decrypted logit went back.** The owner keeps what it decrypts. Were the logit returned to the holder,
  a CKKS decryption leaks its error and with it the key (Li and Micciancio, EUROCRYPT 2021, ePrint 2020/1533; see
  `references/threshold-noise-flooding.md`), so a deployment that returns it needs noise flooding first.
- **The separation itself.** Both parties run in the visitor's tab, so the separation is notional, as the card's badge
  says.
