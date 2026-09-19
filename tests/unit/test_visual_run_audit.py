import json

from minecraft_scientist.minecraft.visual_evidence import successful_select_count


def test_successful_select_count_requires_action_compliance_and_event_success(
    tmp_path,
) -> None:
    events = [
        {
            "executed_action": {"kind": "select"},
            "intervention_compliance": True,
            "event": {"success": True},
        },
        {
            "executed_action": {"kind": "select"},
            "intervention_compliance": False,
            "event": {"success": True},
        },
        {
            "executed_action": {"kind": "interact"},
            "intervention_compliance": True,
            "event": {"success": True},
        },
    ]
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    assert successful_select_count(path) == 1


def test_successful_select_count_fails_closed_without_log(tmp_path) -> None:
    assert successful_select_count(tmp_path / "missing.jsonl") == 0
