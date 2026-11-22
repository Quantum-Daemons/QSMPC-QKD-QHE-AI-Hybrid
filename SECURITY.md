# Security policy

QSH is a research demonstrator, not a product. Nothing here is FIPS 140-3 validated, the QKD and QSDC layers are
simulated, and the MPC parties run in one browser tab. Please do not use it to protect real data. What each layer
defends against, and what it does not, is in [`docs/threat-model.md`](docs/threat-model.md).

## Reporting a vulnerability

Please report suspected vulnerabilities privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**. Do not open a public issue for a security problem.

Helpful details: the page or file, the release (or the date you used the public demo), the steps to reproduce, and
what you expected to happen.

## What to expect

This is a one-person project. I aim to acknowledge a report within 7 days and to agree a disclosure date with you.
Fixes land on `main` and in the next release; the release notes credit the reporter unless you prefer otherwise.

## Scope

In scope: the code in this repository and the public demo at <https://qsmpc-qkd-qhe-ai-hybrid.daemons.jp/>.

Out of scope: vulnerabilities in third-party dependencies (please report them upstream, for example to Microsoft SEAL,
node-seal, @noble/post-quantum, ONNX Runtime, Lattigo (the threshold FHE module) or Polytune (the garbled circuits)),
the licences and content of the public datasets, and the physics of the simulated quantum channel.
