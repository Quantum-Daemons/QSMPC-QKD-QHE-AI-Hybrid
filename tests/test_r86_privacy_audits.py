"""The privacy audits are recorded as they came out, and the plan note quotes the records (2026-11-23).

WHY THIS EXISTS

The CPU phase of the audits the owner chose on 2026-11-22 (references/pets-evaluation-2026-11.md) produced records whose
numbers a later edit could round in a flattering direction: a membership-inference rate, a reconstruction share. These
hold the note to the records, and each record to the internal checks its script makes.

WHAT IS ASSERTED

- Membership inference: each record pools as many member as non-member scores; RMIA here agrees with ML Privacy Meter's
  run_rmia to within one population comparison; a rate the ROC cannot reach is recorded as absent, with its reason. The
  BreastMNIST students' record names the published teacher it distilled from, at the pinned sha256.
- Reconstruction: route A recovers every record within 0.1 percent; the student solved to the optimum satisfies its
  stationarity condition (residual under 1e-3) and recovers more than the trainer's solver point does.
- The note quotes each record's figures at the precision it prints them.
- Seeded: a note that rounds a rate up, and a record that calls an unreachable rate zero, are refused.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
M = REPO / "docs" / "measurements"
NOTE = REPO / "references" / "pets-evaluation-2026-11.md"


def _load(name: str) -> dict:
    return json.loads((M / name).read_text(encoding="utf-8"))


def mia_problems(record: dict) -> list[str]:
    out = []
    population = record["design"]["population_records"]
    for model, res in record["results"].items():
        for attack in ("lira_online", "rmia_offline"):
            r = res[attack]
            if r["members"] != r["non_members"]:
                out.append(f"{model} {attack}: {r['members']} members against {r['non_members']} non-members")
            for key, value in r.items():
                if key.startswith("tpr_at_fpr_") and value is None and attack == "rmia_offline":
                    if not any(key.removeprefix("tpr_at_fpr_") in n for n in r.get("not_resolvable", [])):
                        out.append(f"{model} {attack}: {key} is absent with no reason")
                if key.startswith("tpr_at_fpr_") and value == 0.0:
                    out.append(f"{model} {attack}: {key} is recorded as zero; an unreachable rate is absent, not zero")
        if res["rmia_matches_ml_privacy_meter_max_abs_diff"] > 1.0 / population + 1e-12:
            out.append(f"{model}: RMIA differs from ML Privacy Meter by more than one population comparison")
    return out


def _pct(x: float) -> str:
    return f"{100 * x:.2f} percent"


def note_problems(note: str) -> list[str]:
    flat = re.sub(r"\s+", " ", note)
    ops, tel, rec = _load("mia-opssat-2026-11-23.json"), _load("mia-telecom-qml-2026-11-23.json"), \
        _load("reconstruction-opssat-student-2026-11-23.json")
    bm = _load("mia-breastmnist-students-2026-11-23.json")["results"]
    t, s, f = ops["results"]["teacher"], ops["results"]["student"], tel["results"]["forest"]
    cnn, sq = bm["student"], bm["square_student"]
    C = rec["routes"]["C_weights_solved_to_the_optimum"]
    want = [f"LiRA AUC {t['lira_online']['auc']:.3f}", _pct(t["lira_online"]["tpr_at_fpr_0.001"]),
            f"LiRA AUC {s['lira_online']['auc']:.3f}", _pct(s["lira_online"]["tpr_at_fpr_0.001"]),
            f"RMIA AUC {f['rmia_offline']['auc']:.3f}", _pct(f["rmia_offline"]["tpr_at_fpr_0.01"]),
            f"{_pct(C['share_below_0.1'])} within 10 percent",
            f"RMIA agrees (AUC {cnn['rmia_offline']['auc']:.3f} and {sq['rmia_offline']['auc']:.3f})"]
    # Each student's figures as one phrase, so a count cannot be matched inside another number ("2" in "562").
    exposed = cnn["records_exposed_in_most_targets_at_fpr_1e-3"]
    want.append(f"the CNN student, LiRA AUC {cnn['lira_online']['auc']:.3f} and "
                f"{_pct(cnn['lira_online']['tpr_at_fpr_0.001'])} at a false-positive rate of 0.1 percent, "
                f"{'none' if exposed == 0 else exposed} of {cnn['audit_records']} records exposed")
    want.append(f"the square-activation student, LiRA AUC {sq['lira_online']['auc']:.3f} and "
                f"{_pct(sq['lira_online']['tpr_at_fpr_0.001'])}, "
                f"{sq['records_exposed_in_most_targets_at_fpr_1e-3']} records exposed")
    pipe = _load("mia-breastmnist-pipeline-2026-11-23.json")["results"]
    tl = pipe["teacher"]["lira_online"]
    want.append(f"The teacher leaks: LiRA AUC {tl['auc']:.3f}, {_pct(tl['tpr_at_fpr_0.001'])[:-8]} percent of members found")
    want.append(f"{100 * tl['tpr_at_fpr_0.01']:.2f} at 1 and {100 * tl['tpr_at_fpr_0.1']:.2f} at 10")
    want.append(f"{pipe['teacher']['records_exposed_in_most_targets_at_fpr_1e-3']} of {pipe['teacher']['audit_records']} records")
    for key, label in (("student", "the CNN student"), ("square_student", "the square student")):
        lr = pipe[key]["lira_online"]
        want.append(f"{label}, LiRA AUC {lr['auc']:.3f}")
        want.append(f"{_pct(lr['tpr_at_fpr_0.001'])[:-8]}")
    tel = _load("reconstruction-telecom-student-2026-11-24.json")["routes"]
    heart = _load("reconstruction-fedheart-2026-11-24.json")["routes"]
    want.append(f"{_pct(tel['C_weights_solved_to_the_optimum']['share_below_0.001'])[:-8]} percent within 0.1 and "
                f"{_pct(tel['C_weights_solved_to_the_optimum']['share_below_0.01'])[:-8]} percent within 1")
    want.append(f"({tel['B_weights_at_the_optimum']['solver_iterations']} iterations), "
                f"{_pct(tel['B_weights_at_the_optimum']['share_below_0.1'])[:-8]} percent within 10")
    want.append(f"(Adam, 50 epochs), {_pct(heart['B_pooled_as_the_page_trains_it']['share_below_0.1'])[:-8]} percent within 10")
    if heart["B_fedavg_as_the_page_trains_it"]["share_below_0.1"] == 0:
        want.append("from the FedAvg model as the page trains it, none within 10")
    return [f"the note does not quote {w!r}" for w in want if w not in flat]


def test_the_gradient_inversion_record_is_tied_and_quoted() -> None:
    rec = _load("gradient-inversion-cxr-2026-11-23.json")
    tie = rec["model"]["tied_to_the_published_graph"]
    assert tie["tied"] and tie["classes_equal"] and tie["rows"] == 16
    assert tie["max_abs_margin_difference_vs_int8_graph"] <= 0.03
    r1, r4 = rec["results"]["batch_1"], rec["results"]["batch_4"]
    assert r1["gradients"] == r1["films"] and r4["gradients"] * 4 == r4["films"]
    flat = re.sub(r"\s+", " ", NOTE.read_text(encoding="utf-8"))
    want = [f"{rec['settings']['iterations']:,} iterations",
            f"median SSIM {r1['ssim_median']:.3f} with one film per gradient and {r4['ssim_median']:.3f} with four",
            f"below a flat grey image ({r1['baseline_flat_mean_ssim_median']:.3f})",
            f"below the most similar other film ({r1['baseline_best_other_film_ssim_median']:.3f})",
            f"none of the {r1['films']} films"]
    assert [w for w in want if w not in flat] == []
    assert r1["films_above_best_other_film_ssim"] == 0 and r4["films_above_best_other_film_ssim"] == 0


def test_the_membership_records_hold_their_own_checks() -> None:
    for name in ("mia-opssat-2026-11-23.json", "mia-telecom-qml-2026-11-23.json", "mia-breastmnist-students-2026-11-23.json",
                 "mia-fedheart-2026-11-23.json", "mia-breastmnist-pipeline-2026-11-23.json"):
        assert mia_problems(_load(name)) == [], name


def test_the_students_were_distilled_from_the_published_teacher() -> None:
    pin = json.loads((REPO / "models" / "hub_file_hashes.json").read_text(encoding="utf-8"))["medical_fl.resnet18"]
    teacher = _load("mia-breastmnist-students-2026-11-23.json")["teacher"]
    assert teacher["sha256"] == pin["files"]["onnx/model.onnx"]
    assert teacher["hf_revision"] == pin["revision"]


def test_the_reconstruction_record_holds_its_own_checks() -> None:
    routes = _load("reconstruction-opssat-student-2026-11-23.json")["routes"]
    A, B, C = (routes[k] for k in ("A_published_preprocessing_mean", "B_weights_at_the_optimum",
                                   "C_weights_solved_to_the_optimum"))
    assert A["share_below_0.001"] == 1.0
    assert C["stationarity_residual"] < 1e-3
    assert C["share_below_0.1"] > B["share_below_0.1"]



def test_the_reconstruction_records_of_2026_11_24_hold_their_own_checks() -> None:
    """The QKD-attack student and the heart federation (the owner's choice of 2026-11-24): a published training mean
    gives every record back; the weights do only at the optimum the closed form assumes."""
    tel = _load("reconstruction-telecom-student-2026-11-24.json")["routes"]
    assert tel["A_published_preprocessing_mean"]["share_below_0.001"] == 1.0
    assert tel["B_weights_at_the_optimum"]["published_weights_max_abs_diff"] < 1e-3      # the retrained student is the published one
    assert tel["C_weights_solved_to_the_optimum"]["stationarity_residual"] < 1e-3
    assert tel["C_weights_solved_to_the_optimum"]["share_below_0.01"] > 0.95
    assert tel["C_weights_solved_to_the_optimum"]["share_below_0.1"] > tel["B_weights_at_the_optimum"]["share_below_0.1"]
    heart = _load("reconstruction-fedheart-2026-11-24.json")["routes"]
    assert heart["A_published_preprocessing_mean"]["share_below_0.001"] == 1.0
    assert heart["C_pooled_solved_to_the_optimum"]["share_below_0.001"] == 1.0
    assert heart["C_pooled_solved_to_the_optimum"]["stationarity_residual"] < 1e-6
    assert heart["B_pooled_as_the_page_trains_it"]["share_below_0.1"] < 0.5
    assert heart["B_fedavg_as_the_page_trains_it"]["share_below_0.1"] <= heart["B_pooled_as_the_page_trains_it"]["share_below_0.1"]

def test_the_note_quotes_the_records() -> None:
    assert note_problems(NOTE.read_text(encoding="utf-8")) == []


def test_the_checks_refuse_a_rounded_note_and_a_zeroed_rate() -> None:
    ops = _load("mia-opssat-2026-11-23.json")
    rate = ops["results"]["teacher"]["lira_online"]["tpr_at_fpr_0.001"]
    note = NOTE.read_text(encoding="utf-8")
    assert note_problems(note.replace(f"{100 * rate:.2f} percent", "5.00 percent"))
    assert note_problems(note.replace("0.43 percent, 2 records exposed", "0.43 percent, no records exposed"))
    zeroed = json.loads(json.dumps(ops))
    zeroed["results"]["teacher"]["rmia_offline"]["tpr_at_fpr_0.001"] = 0.0
    zeroed["results"]["teacher"]["rmia_offline"].pop("not_resolvable", None)
    assert mia_problems(zeroed)
