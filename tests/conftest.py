"""Suite-level guards for skips that would otherwise read as passes.

A skipped test is reported in the same green run as a passing one. That is fine when the reason
is real ("this host has no openfhe") and a lie when it is not ("bootstrap not available",
emitted because the backend swallowed the exception that said otherwise). The OpenFHE
bootstrapping test spent its whole life in the second state: it was never observed to run on any
host, yet nothing ever went red.

So: if the bootstrap test skips on a host where ``openfhe`` imports perfectly well, the skip is
not a fact about the host and the session fails. Extend ``_SKIP_MUST_BE_JUSTIFIED`` when another
test earns the same suspicion.
"""

from __future__ import annotations

import pytest

#: test-name fragment -> (importable module that makes a skip illegitimate, how to read it)
_SKIP_MUST_BE_JUSTIFIED = {
    "test_openfhe_bootstrap": (
        "openfhe",
        "openfhe imports on this host, so bootstrapping is expected to work. The only "
        "legitimate reason this test may skip is an openfhe that will not import at all.",
    ),
}

_suspicious_skips: list[tuple[str, str, str, str]] = []


def _skip_reason(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr) if longrepr else "<no reason recorded>"


def pytest_runtest_logreport(report) -> None:
    if not report.skipped:
        return
    for fragment, (module, explanation) in _SKIP_MUST_BE_JUSTIFIED.items():
        if fragment in report.nodeid:
            _suspicious_skips.append((report.nodeid, module, explanation, _skip_reason(report)))


def pytest_sessionfinish(session, exitstatus) -> None:
    if not _suspicious_skips:
        return
    unjustified = []
    for nodeid, module, explanation, reason in _suspicious_skips:
        try:
            __import__(module)
        except ImportError:
            continue                      # the skip is telling the truth; leave it alone
        unjustified.append((nodeid, explanation, reason))
    if not unjustified:
        return
    lines = ["", "=" * 72, "UNJUSTIFIED SKIP - failing the session", "=" * 72]
    for nodeid, explanation, reason in unjustified:
        lines += [f"  {nodeid}", f"    skipped with: {reason}", f"    but: {explanation}", ""]
    print("\n".join(lines))
    session.exitstatus = 1


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter) -> None:
    if _suspicious_skips:
        terminalreporter.write_line(
            f"skip-guard: inspected {len(_suspicious_skips)} skip(s) that require justification",
            yellow=True,
        )
