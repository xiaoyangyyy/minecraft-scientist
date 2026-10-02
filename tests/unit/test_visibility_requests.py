import importlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL.Image", reason="optional RGB replay")


@pytest.fixture
def requests(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    module = importlib.import_module("probe_visibility_requests")
    from test_visibility_acquisition import tracks

    timeline = [
        {
            "start_frame": 0,
            "end_frame": 8,
            "scheduled": {"kind": "move", "motion_handle_index": 7},
        },
        {
            "start_frame": 8,
            "end_frame": 16,
            "scheduled": {"kind": "move", "motion_handle_index": 91},
        },
    ]
    monkeypatch.setattr(
        module.active,
        "validate_acquisition",
        lambda route: (None, {"timeline": timeline}, {"public": True}),
    )
    monkeypatch.setattr(
        module,
        "track_frame_chain",
        lambda frames, config: tracks(len(frames) - 1, shift=(12, 12)),
    )
    return module


def test_replay_stops_at_intervention_without_counterfactual_frames(
    requests, monkeypatch
):
    read = []

    def rgb(path):
        slot = int(path.stem.split("-")[-1])
        assert slot in (0, 1, 2, 8, 9, 10), "read images after requested new control"
        read.append(slot)
        return np.zeros((224, 224, 3), dtype=np.uint8)

    monkeypatch.setattr(requests, "rgb", rgb)
    records, checks = requests.replay(Path("unused"), requests.VisibilityPolicy())
    assert checks == {"public": True} and len(records) == 2
    assert read == [0, 1, 2, 8, 9, 10]
    assert all(item["status"] == "request_different_control" for item in records)
    assert [item["handles"] for item in records] == [[7, 91], [91, 7]]
    assert [item["last_frame_read"] for item in records] == [2, 10]


def test_partial_current_trial_is_retained(requests, monkeypatch):
    records = []
    monkeypatch.setattr(
        requests, "rgb", lambda path: np.zeros((224, 224, 3), dtype=np.uint8)
    )
    monkeypatch.setattr(
        requests,
        "track_frame_chain",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("lost tracker")),
    )
    with pytest.raises(RuntimeError, match="lost tracker"):
        requests.replay(Path("unused"), requests.VisibilityPolicy(), records=records)
    assert len(records) == 1 and records[0]["last_frame_read"] == 1
    assert records[0]["prefixes"] == []


def test_overwrite_refuses_before_reading_inputs(requests, tmp_path, monkeypatch):
    monkeypatch.setattr(
        requests.original,
        "inputs",
        lambda *a: pytest.fail("read inputs before overwrite refusal"),
    )
    with pytest.raises(FileExistsError):
        requests.run(Path("missing"), Path("missing"), tmp_path)


def test_source_ledger_includes_input_provider_not_only_controller(requests):
    sources = requests.sources()
    assert sources["pose_competition_common.py"] == Path(requests.original.__file__)
    assert "visibility_acquisition.py" in sources
    assert "probe_visibility_requests.py" in sources
