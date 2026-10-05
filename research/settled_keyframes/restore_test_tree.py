"""Restore verified source bytes into a fresh persistent native test directory.

Never extract arbitrary archive paths, overwrite a checkout, remove prior evidence
or change the host's temporary-file service. Runtime work directories stay outside
the volatile /tmp tree on this host.
"""

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(manifest, parent, output):
    manifest, output = Path(manifest).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing an existing restore report directory")
    raw_parent = Path(parent)
    if raw_parent.is_symlink():
        raise ValueError("restore requires an explicitly scoped persistent parent")
    parent = raw_parent.resolve()
    if parent == Path(parent.anchor) or parent == Path("/tmp"):
        raise ValueError("restore requires an explicitly scoped persistent parent")
    declaration = json.loads(manifest.read_bytes())
    info_path, archive_path = [
        manifest.parent / name
        for name in ("archive.json", "source-test-fixtures.tar.gz")
    ]
    info = json.loads(info_path.read_bytes())
    if (
        info["manifest_sha256"] != digest(manifest)
        or info["archive_sha256"] != digest(archive_path)
        or info["all_member_bytes_verified"] is not True
    ):
        raise ValueError("durable archive/manifest identity differs")
    expected = declaration["files_sha256"]
    if not 1 <= len(expected) <= 3000:
        raise ValueError("source archive exceeds its file-count budget")
    verified = {}
    size = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(
            expected
        ):
            raise ValueError("archive membership differs from the frozen ledger")
        for member in members:
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or any(part.startswith(".env") for part in name.parts)
            ):
                raise ValueError(
                    "archive cannot contain links, credentials or escaping paths"
                )
            size += member.size
            if size > 64 * 1024 * 1024:
                raise ValueError("source archive exceeds its byte budget")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("archive member is not readable")
            data = stream.read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise ValueError("archive member differs from its original byte hash")
            verified[member.name] = (data, member.mode & 0o777)
    # Only now allocate a fresh tree; tarfile.extract/extractall are never used.
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="verified-source-", dir=parent))
    for name, (data, mode) in verified.items():
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("restored source escaped its fresh directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
        path.chmod(mode)
    if not all(digest(root / name) == value for name, value in expected.items()):
        raise ValueError("restored source bytes differ from the manifest")
    result = {
        "schema": "persistent-exact-test-tree-restoration-v1",
        "snapshot_root": str(root),
        "original_manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "archive_sha256": digest(archive_path),
        "files": len(expected),
        "bytes": size,
        "checks": {
            "all_restored_member_bytes_match": True,
            "original_manifest_unchanged": info["manifest_sha256"] == digest(manifest),
            "archive_unchanged": info["archive_sha256"] == digest(archive_path),
        },
        "scope": "Restored exact frozen V29 inputs, not test success or current workspace identity",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output / "restore.source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    restore(arguments.manifest, arguments.parent, arguments.output)
