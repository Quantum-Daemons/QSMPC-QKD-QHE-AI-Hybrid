"""Console entry point for the ``qsh`` command.

``pyproject.toml`` has declared ``qsh = "qsh.cli:main"`` since the first commit, but this module did
not exist, so ``pip install -e .`` produced a script that raised ``ModuleNotFoundError`` on every
invocation. This is the missing half.

Deliberately thin: it dispatches to the modules that already own the work rather than duplicating
their argument handling.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qsh",
        description="Quantum-safe hybrid orchestration: QKD -> QSMPC -> QHE/FHE -> AI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run an end-to-end use-case pipeline")
    p_run.add_argument("--usecase", default="finance_aml",
                       help="finance_aml | medical_fl | genomics | telecom_ids | satellite_comms | demo")
    p_run.add_argument("--out", default="results/", help="artifact directory")

    p_bench = sub.add_parser("benchmark", help="MPC-FHE vs QSMPC-FHE benchmark")
    p_bench.add_argument("--workload", default="linear", choices=["linear", "cnn", "transformer"])
    p_bench.add_argument("--model-size", default="medium",
                         choices=["lightweight", "medium", "large"])

    sub.add_parser("models", help="list the model registry and what has been trained")
    sub.add_parser("config", help="print the config the SPA is served")
    sub.add_parser("serve", help="run the FastAPI backend (uvicorn)")

    args = parser.parse_args(argv)

    if args.command == "run":
        from .orchestration.pipeline import main as pipeline_main

        return int(pipeline_main(["--usecase", args.usecase, "--out", args.out]) or 0)

    if args.command == "benchmark":
        from .benchmark.runner import run_benchmark

        result = run_benchmark(workload=args.workload, model_size=args.model_size)
        _dump(result)
        return 0

    if args.command == "models":
        from .models.registry import achieved, specs

        for s in specs():
            state = "trained" if achieved(s.key) else "not trained"
            repo = s.hf_repo_id or "-"
            print(f"{s.key:32s} {s.tier:12s} {state:12s} {repo}")
        return 0

    if args.command == "config":
        from .config import frontend_config

        _dump(frontend_config())
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run("api.main:app", host="127.0.0.1", port=8000)
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


def _dump(obj) -> None:
    import json

    json.dump(obj, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
