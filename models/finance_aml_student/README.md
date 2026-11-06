---
license: cdla-sharing-1.0
library_name: onnx
pipeline_tag: tabular-classification
tags:
  - quantum-safe
  - privacy-preserving
  - finance-aml
  - onnx
  - onnxruntime-web
---

# CKKS-linear fraud student

logistic regression distilled from the gradient-boosting teacher

Part of [QSMPC-QKD-QHE-AI-Hybrid](https://github.com/Quantum-Daemons/QSMPC-QKD-QHE-AI-Hybrid),
a quantum-safe orchestration demo.
This is the **encrypted-tier** model for the `finance_aml` use case: a
distilled student small enough to evaluate under CKKS. Whether the public demo
encrypts THIS model differs by use case; the notes below say what actually runs.
It is not the plaintext teacher, and its numbers should be read against that
teacher rather than against the published literature. Its agreement with the
teacher was measured offline, in plaintext.

## Measured performance

| metric | value |
|---|---|
| `agreement` | 0.982943 |
| `disagreement_on_positives` | 0.361913 |
| `metric_delta` | 0.090554 |
| `metric_delta_pp` | 9.0554 |
| `n_eval` | 1523504 |
| `student_metric` | 0.061394 |
| `student_params` | 33 |
| `teacher_metric` | 0.151948 |
| `teacher_params` | 141154 |

## Published baselines this is measured against

- **Target metric**: minority-class F1 (laundering)
- **Baseline to beat**: 0.6323 - GFP+XGBoost, IBM AMLworld HI-Small, NeurIPS 2023 D&B (63.23 +/- 0.17). Same metric and corpus, different protocol: that figure uses a 60/20/20 split by transaction order and IBM's Graph Feature Preprocessor updated over the test window; this project tests on the last 30 percent of transactions (fit 56, validation 14) with its own graph features, whose structural block is fitted on the training window.
- **Published ceiling**: 0.7581 - Multi-FraudGT, the best AML Small HI row of Table 16 ('Minority class F1 scores (%) for six AML datasets') in Bilgi and Atasu, 'MEGA: Message Passing Neural Networks for Multigraphs with EdGe Attributes', TMLR (08/2026), arXiv:2412.00241v3: 75.81 +/- 0.75. In the same column the paper's own MEGA-PNA scores 74.01 +/- 1.55 and MEGA-GenAgg 74.48 +/- 0.84, and Multi-PNA 67.35 +/- 2.89; Small HI is the one AML dataset of the six where Multi-FraudGT is ahead of MEGA-PNA.

## Calibration map

The demo page reads this model's score \\(q\\) as a risk through this map before it compares it with a threshold probability, as its decision curve does. Fitted on the validation split and scored on the held-out test split.

$$p = \frac{1}{1 + \exp\!\big(a\,\operatorname{logit}\big(\operatorname{clip}(q,\ \varepsilon,\ 1-\varepsilon)\big) + b\big)}$$

$$\operatorname{logit}(q) = \log\frac{q}{1-q},\qquad \varepsilon = 10^{-12}$$

Here \\(q\\) is the student's P(laundering), \\(q = \sigma(w^\top z + c)\\), where \\(w^\top z + c\\) is the decision logit the encrypted path decrypts, \\(z\\) the z-scored features and \\(c\\) the student's intercept (`b` in its weights file). The tuned threshold is folded into \\(c\\), so the operating point is \\(q = 0.5\\), a logit of 0.

| quantity | value |
|---|---|
| `a` | -0.659609731534434 |
| `b` | 4.30071908149813 |
| fitted on | 710969 validation rows, 801 positive |
| scored on | 1523504 test rows, 2321 positive |
| Brier score, raw to calibrated | 0.01605 to 0.00154002 |
| log-loss, raw to calibrated | 0.05974498 to 0.00864886 |
| CORP miscalibration (MCB), raw to calibrated | 0.01455503 to 0.00004505 |
| base-rate Brier score (CORP UNC) | 0.00152114 |

Brier skill against the base rate, \\(1 - \mathrm{BS}/\mathrm{UNC}\\), is -0.0124: negative, so as a probability forecast the calibrated score does worse than giving every row the window's rate. A decision curve asks a different question, what acting at each threshold pays.

Raw is the student's own probability, sigmoid of the decision logit. The folded threshold puts 0.5 at the operating point, so it is not a risk.

Why Platt and not isotonic: Measured on this student's windows, not carried from the teacher. Isotonic's Brier 0.00149696 is below Platt's 0.00154002 and its log-loss 0.00837343 is below 0.00864886; lower is better on both. Platt is kept as a choice, not as the better score: it is strictly monotone, so the curve ranks rows as the student does (AUPRC 0.02410110 against the raw 0.02410110, with 168 test rows tied at the logit clip; isotonic's 0.02459877 comes from the test window's scores tied at 109 values, 207,340 of them exactly zero), and it is the teacher's method, so the page's two decision curves differ by the model and not by the map. scripts/measure_decision_curve.py records the student's window curve under both maps.

## Training data

- **Dataset**: IBM AMLworld HI-Small
- **Licence**: CDLA-Sharing-1.0
- **Source**: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml (licence read 2026-08-03)
- **Attribution (CDLA-Sharing-1.0)**: Altman et al., 'Realistic Synthetic Financial Transactions for Anti-Money Laundering Models', NeurIPS 2023 Datasets and Benchmarks. Data: IBM Transactions for Anti-Money Laundering (AML), CDLA-Sharing-1.0.

5M transactions, 515K accounts, ~1 laundering transaction in 981. NeurIPS 2023 D&B. SYNTHETIC: IBM generates it with a multi-agent virtual-world model and states 'Everything is synthetic'; the paper is titled 'Realistic Synthetic Financial Transactions for AML Models'. Chosen for its calibration to real transaction statistics and its CDLA-Sharing terms, not because it is real data - no commercial-OK corpus of real laundering transactions exists.

## Notes and limitations

Encrypted on the public page since Round 84: its private-inference figure scores this 33-parameter AMLworld student under CKKS on the 8,192 test-window rows of finance_aml.gbm's sample (64 ciphertexts), against the same model in plaintext. The page's one-row CKKS step, its MPC argmax, the reported class and the SHAP bars still use the four-feature linear model bundled with the PaySim sample (log amount, a TRANSFER or CASH_OUT flag, a night-hour flag and a first-seen-payee flag; no feature built from PaySim's balance columns), and the full deployment's finance pipeline encrypts its own PaySim-trained surrogate over the same four. Its agreement with the teacher (`agreement`) is measured offline, at training time, in plaintext; each browser run measures `encrypted_agreement`, CKKS against plaintext for the model it encrypts.

## Honest scope

This model is published as part of a research proof of concept, not as a production system.
Numbers above are what this repository measured on the split described, with the code in
`scripts/train/`. Where a figure is carried from the literature rather than measured here,
it is labelled as such.
