"""The LIDC stream keeps a patient's crops and HU cubes in files of their own.

On 2026-11-24 `Path.with_suffix` turned both 'P.crops28' and 'P.hu64' into 'P.npy', so the 64 mm HU cubes overwrote
the 28 mm crops of every training patient, and the training patients were fetched again
(references/medical-nodule3d-square-student-preregistration-addendum.md, section 3). These tests hold the two kinds
apart, through `merge`, and check that the renaming of the first run's parts sorts them by what they hold.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _stream():
    spec = importlib.util.spec_from_file_location("lidc_stream_under_test", REPO / "scripts" / "data" / "lidc_stream.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meta(patient: str, cluster: int = 0) -> dict:
    return {"patient": patient, "series": "1.2.3", "cluster": cluster}


def test_crops_and_hu_cubes_get_files_of_their_own(tmp_path):
    st = _stream()
    parts = tmp_path / "parts"
    parts.mkdir()
    crop = np.full((28, 28, 28), 7, np.uint8)
    hu = np.full((64, 64, 64), -500, np.float16)
    st.save_parts(parts / "LIDC-IDRI-0001.crops28", [(_meta("LIDC-IDRI-0001"), crop)])
    st.save_parts(parts / "LIDC-IDRI-0001.hu64", [(_meta("LIDC-IDRI-0001"), hu)])
    assert sorted(p.name for p in parts.iterdir()) == [
        "LIDC-IDRI-0001.crops28.json", "LIDC-IDRI-0001.crops28.npy", "LIDC-IDRI-0001.hu64.json", "LIDC-IDRI-0001.hu64.npy"]
    st.merge(SimpleNamespace(out=tmp_path))
    crops = np.load(tmp_path / "crops_28mm.npz")["crops"]
    cubes = np.load(tmp_path / "crops_64mm_hu.npz")["crops"]
    assert crops.dtype == np.uint8 and crops.shape == (1, 28, 28, 28) and int(crops[0, 0, 0, 0]) == 7
    assert cubes.dtype == np.float16 and cubes.shape == (1, 64, 64, 64) and float(cubes[0, 0, 0, 0]) == -500.0
    assert json.loads((tmp_path / "crops_28mm.json").read_text())["nodules"] == 1


def test_the_first_runs_parts_are_renamed_by_what_they_hold(tmp_path):
    st = _stream()
    parts = tmp_path / "parts"
    parts.mkdir()
    np.save(parts / "LIDC-IDRI-0002.npy", np.zeros((2, 28, 28, 28), np.uint8))
    (parts / "LIDC-IDRI-0002.json").write_text(json.dumps([_meta("LIDC-IDRI-0002", 0), _meta("LIDC-IDRI-0002", 1)]))
    np.save(parts / "LIDC-IDRI-0003.npy", np.zeros((1, 64, 64, 64), np.float16))
    (parts / "LIDC-IDRI-0003.json").write_text(json.dumps([_meta("LIDC-IDRI-0003")]))
    (parts / "LIDC-IDRI-0003.centres.json").write_text("[]")
    assert st.rename_old_parts(parts) == {"hu64": 1, "crops28": 1}
    assert sorted(p.name for p in parts.iterdir()) == [
        "LIDC-IDRI-0002.crops28.json", "LIDC-IDRI-0002.crops28.npy", "LIDC-IDRI-0003.centres.json",
        "LIDC-IDRI-0003.hu64.json", "LIDC-IDRI-0003.hu64.npy"]
    assert st.rename_old_parts(parts) == {"hu64": 0, "crops28": 0}


def test_a_part_of_neither_kind_stops_the_renaming(tmp_path):
    st = _stream()
    parts = tmp_path / "parts"
    parts.mkdir()
    np.save(parts / "LIDC-IDRI-0004.npy", np.zeros((1, 32, 32, 32), np.float32))
    (parts / "LIDC-IDRI-0004.json").write_text("[]")
    with pytest.raises(SystemExit):
        st.rename_old_parts(parts)
