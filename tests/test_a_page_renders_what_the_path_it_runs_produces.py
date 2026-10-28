"""A page must read the keys its own code path emits, and must not promise controls that do not exist.

WHY THIS EXISTS

Round 44 renamed the per-pulse metric `skr` to `skr_per_pulse` so one name could not carry two
dimensions. It changed three producers - `src/qsh/orchestration/stages.py`,
`src/qsh/usecases/satellite_comms/__init__.py` and `api/inference_service.py` - and it changed the
reader at `web/src/pages/Validations.tsx`. It missed the ONE producer the deployed site executes.

`web/src/lib/api.ts`'s `demoQkd` has two branches. The server branch calls `/api/demo/qkd`, which was
renamed. The client branch builds the payload in the browser and still returns `skr:`. The public
demo is built with `CLIENT_ONLY=1` (`docker/docker-compose.demo.yml`), so `isClientDemo()` is
unconditionally true and the client branch is the ONLY one that ever runs there. `Validations.tsx`
reads `qkd.skr_per_pulse`, gets `undefined`, and React renders an empty string: the public
`/validations` page shows a blank cell under "secret key rate (per pulse)".

Nothing caught it. `SW3` looks for the literal token `undefined` on the page, and an `undefined`
React child renders as nothing at all - the one value that leaves no trace. The comment sitting
directly above the changed line still reads "The payload key is still `skr`; renaming it crosses into
`api/inference_service.py` and is conditioned rather than done here", which Round 44 falsified in the
same commit that broke the page, and which remained true only of the branch it forgot.

THE SECOND CHECK, WHICH IS THE SAME MISTAKE POINTING THE OTHER WAY

`SdnControlPlanePanel.tsx` tells the visitor, twice, that the source repetition rate "is a control in
the Parameters panel" and that the figures "scale exactly with the Source rate control". No such
control exists: `ParameterPanel.tsx` renders `qkd_pulses`, `qkd_distance_km`, `qkd_eve`,
`qkd_misalignment` and the free-space group, and nothing named for the source rate. The rate is read
from baked config at `clientCompute.ts`.

This project has found SIX inert controls - a control the page renders and nothing reads. This is the
inverse and it is worse: prose that asserts a control which was never rendered at all. An inert
control disappoints; an affirmative false statement about the UI misinforms.

THE INVARIANT

A page may only read keys the code path it actually executes emits, and may only describe controls
that are actually rendered. Both halves are about the same gap - between what a surface SAYS and what
the path behind it DOES - and both are checked here so a fix to one cannot be mistaken for a fix to
the other.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "web" / "src"
API = WEB / "lib" / "api.ts"
VALIDATIONS = WEB / "pages" / "Validations.tsx"
PANEL = WEB / "components" / "SdnControlPlanePanel.tsx"
PARAMS = WEB / "components" / "ParameterPanel.tsx"

_COMMENT = re.compile(r"/\*[\s\S]*?\*/|//[^\n]*|\{/\*[\s\S]*?\*/\}")


def _code_only(text: str) -> str:
    """Comments blanked, line count preserved.

    The comment above the broken line quotes the old key while explaining the decision not to rename
    it, so a scan that read comments would find `skr` in the prose describing the defect and call the
    file consistent with itself.
    """
    return _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def _client_payload_keys_of_demo_qkd() -> set[str]:
    """The keys of the object the CLIENT branch of `demoQkd` resolves with.

    Scoped twice over, and both scopings matter:

    - To the client branch, after the `isClientDemo()` early return. The server branch is one line
      and WAS renamed correctly; letting it into the set would allow the correct half to satisfy a
      check about the broken half, which is precisely the defect shape this file exists to catch.
    - To the `Promise.resolve({...})` literal, not to every `word:` in the branch. The channel is
      built inline as `{ ...FIBER, distanceKm: km, eveFraction: eve }`, so a looser scan reports
      `distanceKm` and `eveFraction` as payload keys - and would then quietly accept a page reading
      `qkd.distanceKm`, which resolves to `undefined` exactly like the bug being tested for.
    """
    src = _code_only(API.read_text())
    body = src[src.index("demoQkd:"):]
    body = body[body.index("isClientDemo()"):]
    payload = re.search(r"Promise\.resolve\(\{(.*?)\}\s*\)\s*;", body, re.S)
    assert payload, "the client `demoQkd` no longer resolves an object literal; the shape changed"
    # Top-level keys only: nested initialisers such as `{ ...FIBER, distanceKm: km }` are values.
    depth, keys, buf = 0, set(), payload.group(1)
    for m in re.finditer(r"[{}]|(\w+)\s*:", buf):
        if m.group(0) == "{":
            depth += 1
        elif m.group(0) == "}":
            depth -= 1
        elif depth == 0 and m.group(1):
            keys.add(m.group(1))
    return keys


def test_validations_reads_only_keys_the_client_path_emits() -> None:
    """The deployed page and the only producer it runs, compared key by key."""
    emitted = _client_payload_keys_of_demo_qkd()
    assert "qber" in emitted and "secure_key_bits" in emitted, (
        f"the client `demoQkd` payload no longer looks like a QKD result (found {sorted(emitted)}); "
        "this check has lost its subject rather than passed"
    )
    read = set(re.findall(r"\bqkd\.(\w+)", _code_only(VALIDATIONS.read_text())))
    assert read, "Validations.tsx no longer reads any `qkd.*` key; the table was removed or renamed"
    missing = sorted(read - emitted)
    assert not missing, (
        f"/validations reads {missing} but the CLIENT branch of `demoQkd` emits {sorted(emitted)}.\n\n"
        "The public demo is built CLIENT_ONLY, so that branch is the only one it ever executes: "
        "every key above renders as an empty cell on the deployed site. `SW3` cannot see this, "
        "because an `undefined` React child renders as nothing rather than as the token 'undefined'.\n\n"
        "Rename the key in the client branch too - the server branch alone is half a rename."
    )


def test_prose_naming_a_control_names_one_that_is_rendered() -> None:
    """Any '<X> control in the Parameters panel' must be a control the Parameters panel renders."""
    rendered_labels = set(re.findall(r'\[\s*"([^"]+)"\s*,\s*"[a-z0-9_]+"', PARAMS.read_text()))
    assert len(rendered_labels) >= 5, (
        f"only {len(rendered_labels)} field labels parsed out of ParameterPanel.tsx; the field "
        "tuple shape changed and this check would pass vacuously"
    )
    # `sorted(set(...))`: the panel is both named explicitly and reached by the rglob, and a gate
    # that reports one offender twice reads as two defects.
    files = sorted({*(WEB / "components").rglob("*.tsx"), *(WEB / "pages").rglob("*.tsx"), PANEL})
    offenders: list[str] = []
    for f in files:
        text = f.read_text()
        for m in re.finditer(r"([A-Z][A-Za-z ]{1,24}?) control in the Parameters panel", text):
            named = m.group(1).strip()
            if not any(named.lower() in lab.lower() for lab in rendered_labels):
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(f"{f.relative_to(REPO)}:{line}: promises a {named!r} control")
    assert not offenders, (
        "these rendered strings tell the visitor a control exists that the Parameters panel does "
        "not render:\n  " + "\n  ".join(offenders)
        + f"\n\nRendered labels are: {sorted(rendered_labels)}.\n\n"
        "This project has found six INERT controls - rendered and read by nothing. This is the "
        "inverse: a control asserted in prose and never rendered at all. Either add the control or "
        "delete the sentence; a visitor who goes looking will not find it."
    )


def test_every_rendered_control_is_read_somewhere() -> None:
    """The inert-control direction, which nothing has ever asserted.

    `tests/test_run_config_keys.py` checks that every key the browser READS is emitted. The reverse -
    a control the panel renders that nothing consumes - has been found by hand SIX times in this
    project and has never had a gate. A visitor adjusts it, the run produces an identical number, and
    the page has quietly told them their input does not matter.

    It earned its place immediately. The first version of `sourceRateHz` in `clientCompute.ts` took
    the config object and a KEY NAME and read `config?.[key]` internally, so the two controls added
    in this very round were read at runtime and invisible to any static reader - this check reported
    both as inert. The call sites now pass `config?.qkd_source_rate_mhz` directly. A gate that fails
    on the change that introduced it is the gate working.
    """
    keys = re.findall(r'\[\s*"[^"]+"\s*,\s*"([a-z0-9_]+)"', PARAMS.read_text())
    assert len(keys) >= 15, (
        f"only {len(keys)} controls parsed from ParameterPanel.tsx; the field tuple shape changed "
        "and this check would pass by finding nothing"
    )
    # `_code_only`, like the two checks above it. Without it a key mentioned in a JSDoc counts
    # as a read - and `clientCompute.ts` carries `config?.qkd_source_rate_mhz` inside the very
    # comment Round 45 wrote to explain wiring it, so deleting the real read left this green.
    # The file defines `_code_only` and warns about exactly this; this test did not call it.
    body = "\n".join(
        _code_only(p.read_text()) for p in sorted({*WEB.rglob("*.ts"), *WEB.rglob("*.tsx")})
        if p.name != "ParameterPanel.tsx"
    )
    # A READ is a `config.k` / `config?.k` / `config["k"]` / `params.k` shaped expression. A bare
    # mention of the name in prose is not evidence that anything consumes it.
    inert = [k for k in keys
             if not re.search(rf"""(?:config|params|cfg|p)\s*\??\s*[.\[]\s*["']?{k}\b""", body)]
    assert not inert, (
        f"these controls are rendered by the Parameters panel and read nowhere: {inert}\n\n"
        "A control that changes no number is a promise the page does not keep. Either wire it to the "
        "model or remove it - and if it is read through an indirection the scan cannot see, move the "
        "read to the call site so a reader can find it too."
    )


def test_a_url_that_names_a_use_case_shows_it_or_says_why_not() -> None:
    """`/orchestration?usecase=` was ignored, and silence is not a fix for it either.

    The page held its use case in `useState` and never read the query string, so
    `/orchestration?usecase=demo` rendered `finance_aml`. Nothing linked to that form, which is why
    it survived: it was carried from Round 42, came due in Round 44, and was re-dated to Round 46
    rather than discharged. The register's gate treats moving a date as not discharging it.

    Both halves are asserted, because reading the parameter and then silently substituting a default
    would reproduce the original behaviour with intent added. A rejected value has to be NAMED on
    the page - that is what distinguishes a typo from a stale bookmark for whoever followed the link.
    """
    src = (WEB / "pages" / "Orchestration.tsx").read_text()
    assert "useSearchParams" in src, (
        "/orchestration no longer reads the query string, so `?usecase=` is ignored again and a URL "
        "that names a use case can show a different one"
    )
    assert re.search(r"USECASES\.some\(|USECASES\.find\(", src), (
        "the query parameter is read but not validated against the known use cases; an unknown "
        "value would be passed to the pipeline instead of refused"
    )
    assert "rejected" in src, (
        "nothing reports a rejected `?usecase=` value. Falling back silently is what the page "
        "already did before it read the parameter at all."
    )


def test_the_panel_prose_and_its_tooltip_make_the_same_claim() -> None:
    """One caveat, two surfaces - and they must not drift apart.

    The tooltip and the paragraph below it both describe how the rate is derived. When only one is
    edited the page carries two different accounts of the same number, and whichever the reader
    happens to hover decides which one they believe.
    """
    src = PANEL.read_text()
    # A function since Round 78, `rateCellTitle`, because its ratio to the measured rate is computed
    # from the run; it was the constant RATE_CELL_TITLE, whose "3.8x" no run had matched since Round 76.
    tooltip = re.search(r"function rateCellTitle\(.*?\{(.*?)\n\}", src, re.S)
    assert tooltip, "rateCellTitle is gone; the rate cells have lost their hover explanation"
    assert "title={rateCellTitle(" in src, "the rate cells no longer use the shared tooltip"
    claims_control = "control in the Parameters panel" in tooltip.group(1)
    prose_claims_control = "control in the Parameters panel" in src.replace(tooltip.group(1), "")
    assert claims_control == prose_claims_control, (
        "the tooltip and the panel paragraph disagree about whether a Source rate control exists "
        f"(tooltip says {claims_control}, prose says {prose_claims_control}). They describe one "
        "number and must be changed together."
    )
