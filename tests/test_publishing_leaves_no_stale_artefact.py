"""A publish refuses to leave files at the head of a Hub repo that the local folder no longer produces.

WHY THIS EXISTS

Round 71. `upload_folder` adds and replaces and never deletes. ResNet-50's int8 graph was withheld by
the export gate after a retrain - it was built, measured at a confident agreement of 0.9865 against
the 0.99 bar, and deleted locally - but the copy uploaded on 2026-08-09, quantised from the weights
that retrain replaced, stayed at the head of the repo through six later publishes. The
export_report.json beside it said int8 had been withheld and fp32 only was served. The browser reads
that report and never fetched the file; anyone downloading from the Hub got another model's int8.

WHAT IS ASSERTED, through `publish()` itself with a fake Hub client

- A stale file at the head stops the publish before anything is uploaded, and names the file.
- With delete_stale, the same commit removes exactly that file and nothing else.
- A dry run reports the stale file and uploads nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _publish_module():
    spec = importlib.util.spec_from_file_location("publish_hf", REPO / "scripts" / "publish_hf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeApi:
    def __init__(self, remote: set[str]) -> None:
        self.remote = remote
        self.uploads: list[dict] = []

    def list_repo_files(self, repo_id, repo_type="model"):
        return sorted(self.remote)

    def create_repo(self, *a, **k):
        return None

    def update_repo_settings(self, *a, **k):
        return None

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)

    def model_info(self, repo_id):
        class Info:
            sha = "0" * 40
        return Info()


@pytest.fixture
def staged(tmp_path, monkeypatch):
    mod = _publish_module()
    key = "medical_fl.resnet50"
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"fp32")
    (tmp_path / "export_report.json").write_text("{}")
    monkeypatch.setitem(mod.ARTIFACT_DIRS, key, str(tmp_path))
    return mod, mod.registry.get(key)


def test_a_stale_file_stops_the_publish_and_is_named(staged) -> None:
    mod, spec = staged
    api = FakeApi({".gitattributes", "README.md", "export_report.json", "onnx/model.onnx",
                   "onnx/model_int8.onnx"})
    with pytest.raises(SystemExit, match=r"onnx/model_int8\.onnx"):
        mod.publish(spec, private=True, dry_run=False, api=api)
    assert api.uploads == [], "nothing may be uploaded before the stale file is dealt with"


def test_delete_stale_removes_exactly_that_file_in_the_same_commit(staged) -> None:
    mod, spec = staged
    api = FakeApi({"README.md", "export_report.json", "onnx/model.onnx", "onnx/model_int8.onnx"})
    mod.publish(spec, private=True, dry_run=False, api=api, delete_stale=True)
    assert len(api.uploads) == 1
    assert api.uploads[0]["delete_patterns"] == ["onnx/model_int8.onnx"]


def test_a_dry_run_reports_and_uploads_nothing(staged, capsys) -> None:
    mod, spec = staged
    api = FakeApi({"README.md", "onnx/model.onnx", "onnx/model_int8.onnx"})
    assert mod.publish(spec, private=True, dry_run=True, api=api) is None
    assert "[stale on the Hub] onnx/model_int8.onnx" in capsys.readouterr().out
    assert api.uploads == []


def test_a_clean_head_publishes_with_nothing_deleted(staged) -> None:
    mod, spec = staged
    api = FakeApi({"README.md", "export_report.json", "onnx/model.onnx"})
    mod.publish(spec, private=True, dry_run=False, api=api)
    assert api.uploads[0]["delete_patterns"] is None
