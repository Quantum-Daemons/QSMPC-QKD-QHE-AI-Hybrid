# Contributing

Thank you for your interest. QSH is a one-person research demonstrator; issues and small pull requests are welcome.

- **Issues**: name the page or file, what you saw and how to reproduce it. For security problems, follow
  [`SECURITY.md`](SECURITY.md) instead.
- **Pull requests**: keep each change small and focused, and say what the change verifies or fixes. Before opening
  one, run the test suite with the slow tests deselected (`make test`, which is `pytest -m "not slow"`) and, for a
  change under `web/`, the build (`bash scripts/build_wasm.sh`, then `npm --prefix web run build`). The setup is the
  README's [Quickstart](README.md#quickstart). `scripts/ci_local.sh` is the full gate that CI mirrors; it takes about
  40 minutes and is not required of contributors.
- **Published numbers**: a change that alters a published number (a metric, a timing, a model revision) should say
  where the new number comes from.
- **Labels on the pages**: every layer is labelled real, simulated or notional, and each check says what it was
  checked against. Please keep those labels accurate.
- **Licence**: contributions are accepted under the Apache License 2.0, the licence of this repository.
