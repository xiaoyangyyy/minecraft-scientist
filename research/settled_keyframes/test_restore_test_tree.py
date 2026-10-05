import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from restore_test_tree import restore


def fixture(tmp_path, name="src/example.py", kind=None):
    folder = tmp_path / "evidence"
    folder.mkdir()
    data = b"print('source fixture')\n"
    archive_path = folder / "source-test-fixtures.tar.gz"
    with tarfile.open(archive_path, "x:gz") as archive:
        member = tarfile.TarInfo(name)
        if kind == "link":
            member.type, member.linkname = tarfile.SYMTYPE, "/outside"
        else:
            member.size = len(data)
        archive.addfile(member, None if kind == "link" else io.BytesIO(data))
    manifest = folder / "report.json"
    manifest.write_text(
        json.dumps({"files_sha256": {name: hashlib.sha256(data).hexdigest()}})
    )
    (folder / "archive.json").write_text(
        json.dumps(
            {
                "all_member_bytes_verified": True,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            }
        )
    )
    return manifest, data


def test_restore_creates_fresh_byte_identical_tree_without_changing_evidence(tmp_path):
    manifest, data = fixture(tmp_path)
    parent = tmp_path / "native-persistent"
    result = restore(manifest, parent, tmp_path / "restored")
    assert all(result["checks"].values())
    assert parent in Path(result["snapshot_root"]).parents
    assert (Path(result["snapshot_root"]) / "src/example.py").read_bytes() == data
    with pytest.raises(FileExistsError):
        restore(manifest, parent, tmp_path / "restored")


@pytest.mark.parametrize(
    "name,kind",
    [("../escape", None), ("/absolute", None), ("src/link", "link"), (".env", None)],
)
def test_archive_escape_link_and_credentials_reject_before_allocation(
    tmp_path, name, kind
):
    manifest, _ = fixture(tmp_path, name, kind)
    parent = tmp_path / "native-persistent"
    with pytest.raises(ValueError, match="links, credentials or escaping"):
        restore(manifest, parent, tmp_path / "restored")
    assert not parent.exists()


def test_changed_archive_identity_is_not_restored(tmp_path):
    manifest, _ = fixture(tmp_path)
    with (manifest.parent / "source-test-fixtures.tar.gz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        restore(manifest, tmp_path / "native", tmp_path / "restored")
