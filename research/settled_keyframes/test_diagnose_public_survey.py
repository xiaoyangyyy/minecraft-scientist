import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
Image = pytest.importorskip("PIL.Image")

import diagnose_public_survey as survey
from test_public_view_quality import texture
from test_safe_bridge_probe import bind_public, save
from test_safe_bridge_probe import public_capture as public_capture  # noqa: PLC0414


@pytest.fixture
def mock_survey(public_capture):
    """Mock public ledger; not native Minecraft or independent audit evidence."""
    config = json.loads((public_capture / "config.json").read_bytes())
    config.update(
        schema=next(iter(sorted(survey.SCHEMAS))),
        reference_indices=survey.ORDINALS,
        schedule=survey.SCHEDULE,
    )
    save(public_capture / "config.json", config)
    for slot in survey.ORDINALS:
        Image.fromarray(texture(448)).save(
            public_capture / "frames" / f"frame-{slot:05d}.png"
        )
    bind_public(public_capture)
    return public_capture


def test_exact_past_views_rank_without_future_or_private_reads(
    mock_survey, tmp_path, monkeypatch
):
    read = Path.read_bytes

    def guarded(path):
        if (
            path.name.startswith("evaluator-")
            or path.parent.name == "frames"
            and int(path.stem.split("-")[-1]) > 48
        ):
            raise AssertionError("ranking consumed future or evaluator input")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    output = tmp_path / "ranked"
    result = survey.diagnose(mock_survey, output)
    assert all(result["checks"].values())
    assert [view["absolute_frame"] for view in result["views"]] == survey.ORDINALS
    assert result["selected_frame"] == 15
    assert not result["executed_reorientation"] and not result["navigation_accepted"]
    for name, value in result["config"]["sources_sha256"].items():
        assert survey.digest(output / name) == value
    with pytest.raises(FileExistsError):
        survey.diagnose(mock_survey, output)


@pytest.mark.parametrize(
    "case",
    [
        "future_ordinal",
        "schedule",
        "schema",
        "audit",
        "source",
        "snapshot",
        "events",
        "rgb",
    ],
)
def test_invalid_inputs_cannot_get_a_successful_ranking(mock_survey, tmp_path, case):
    path = mock_survey / "config.json"
    config = json.loads(path.read_bytes())
    if case == "future_ordinal":
        config["reference_indices"][-1] = 51
    elif case == "schedule":
        config["schedule"][0] = ["move", 0]
    elif case == "schema":
        config["schema"] = "unregistered"
    elif case in ("source", "snapshot"):
        key = next(iter(config["source_sha256"]))
        if case == "source":
            config["source_sha256"][key] = "0" * 64
        else:
            (mock_survey / "source-snapshot" / key).write_bytes(b"changed source")
    save(path, config)
    bind_public(mock_survey)
    if case == "audit":
        audit = json.loads((mock_survey / "audit.json").read_bytes())
        audit["valid"] = False
        save(mock_survey / "audit.json", audit)
    elif case == "events":
        (mock_survey / "events.jsonl").write_bytes(b"changed events")
    elif case == "rgb":
        Image.fromarray(np.zeros((448, 448, 3), dtype=np.uint8)).save(
            mock_survey / "frames/frame-00015.png"
        )
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="audited|dependency|ledger"):
        survey.diagnose(mock_survey, output)
    assert not (output / "report.json").exists()


def test_late_input_mutation_cannot_leave_a_success_claim(
    mock_survey, tmp_path, monkeypatch
):
    rank = survey.quality.rank_views

    def mutate(*args):
        result = rank(*args)
        path = mock_survey / "frames/frame-00015.png"
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(survey.quality, "rank_views", mutate)
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="inputs changed"):
        survey.diagnose(mock_survey, output)
    assert (output / "probe.source.py").exists() and not (
        output / "report.json"
    ).exists()
