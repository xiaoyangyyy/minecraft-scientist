import json
import tarfile

import archive_pytest_scratch as archive
import pytest


@pytest.fixture
def scratch(tmp_path):
    parent = tmp_path / "runs"
    source = parent / "shared-anchor-full-temp-windows-v1"
    (source / "test_case").mkdir(parents=True)
    (source / "test_case" / "mock.bin").write_bytes(b"mock fixture bytes" * 100)
    (source / "test_case" / "config.json").write_text("{}\n", encoding="utf-8")
    return parent, source, tmp_path / "archive" / "scratch.tar.gz"


def test_exact_mock_bytes_verified_and_source_not_removed(scratch):
    parent, source, output = scratch
    before = archive.inventory(source)
    report = archive.archive(source, parent=parent, output=output)
    assert report["verified"] and report["source_unchanged"]
    assert not report["source_removed"] and archive.inventory(source) == before
    assert report["file_count"] == 2
    saved = json.loads(output.with_suffix(".json").read_bytes())
    assert saved == report
    with tarfile.open(output) as result:
        assert {entry.name for entry in result} == set(before["files"])
    with pytest.raises(FileExistsError):
        archive.archive(source, parent=parent, output=output)


@pytest.mark.parametrize("case", ["name", "parent", "non_mock", "nested_output"])
def test_non_scoped_inputs_cannot_be_archived(scratch, case, tmp_path):
    parent, source, output = scratch
    if case == "name":
        changed = parent / "native-experiment"
        source.rename(changed)
        source = changed
    elif case == "parent":
        parent = tmp_path
    elif case == "non_mock":
        (source / "report.json").write_text("{}", encoding="utf-8")
    else:
        output = source / "test_case" / "archive.tar.gz"
    with pytest.raises((ValueError, FileExistsError)):
        archive.archive(source, parent=parent, output=output)
    assert not output.exists()


def test_source_changes_discard_verification_not_the_source(scratch, monkeypatch):
    parent, source, output = scratch
    original = archive.inventory
    calls = 0

    def inventory(root):
        nonlocal calls
        calls += 1
        if calls == 2:
            (source / "test_case" / "mock.bin").write_bytes(b"late changed fixture")
        return original(root)

    monkeypatch.setattr(archive, "inventory", inventory)
    with pytest.raises(ValueError, match="changed"):
        archive.archive(source, parent=parent, output=output)
    assert source.exists() and output.exists()
    assert not output.with_suffix(".json").exists()


def test_escape_alias_is_refused_before_any_archive(scratch, tmp_path):
    parent, source, output = scratch
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"must not read")
    try:
        (source / "test_case" / "escape.bin").symlink_to(outside)
    except OSError:
        pytest.skip("host does not permit test symlinks")
    with pytest.raises(ValueError, match="escapes"):
        archive.archive(source, parent=parent, output=output)
    assert not output.exists()
