"""The repository's LICENSE is the Apache License 2.0 text, byte for byte.

WHY THIS EXISTS (Round 84). The LICENSE file was a shortened copy: parts of sections 4(d) and 5 to 9
were missing, including the limitation of liability and the indemnity wording, so GitHub reported the
licence as NOASSERTION while README.md, pyproject.toml and DATA_LICENSES.md all said Apache-2.0. The
file is now the official text from https://www.apache.org/licenses/LICENSE-2.0.txt, whose SHA-256 is
pinned here so a later edit cannot shorten it again unnoticed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

APACHE_2_0_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def test_the_licence_file_is_the_official_apache_2_0_text() -> None:
    data = (Path(__file__).resolve().parents[1] / "LICENSE").read_bytes()
    assert hashlib.sha256(data).hexdigest() == APACHE_2_0_SHA256, (
        "LICENSE is not the official Apache-2.0 text (https://www.apache.org/licenses/LICENSE-2.0.txt)")
