"""Retain an exact test snapshot archive after verifying every declared member."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persist(manifest):
    report_path = manifest.resolve(strict=True)
    report = json.loads(report_path.read_bytes())
    root = Path(report["snapshot_root"]).resolve(strict=True)
    archive_path = report_path.parent / "source-test-fixtures.tar.gz"
    info_path = report_path.parent / "archive.json"
    if archive_path.exists() or info_path.exists():
        raise FileExistsError("refusing to overwrite a frozen source archive")
    for name, expected in report["files_sha256"].items():
        path = root / name
        if root not in path.resolve(strict=True).parents or path.is_symlink():
            raise ValueError("archive cannot contain external paths or symlinks")
        if digest(path) != expected:
            raise ValueError("declared frozen source changed before persistence")
    with tarfile.open(archive_path, "x:gz") as archive:
        for name in sorted(report["files_sha256"]):
            archive.add(root / name, arcname=name, recursive=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        if set(archive.getnames()) != set(report["files_sha256"]):
            raise ValueError("archive membership differs from the frozen manifest")
        for name, expected in report["files_sha256"].items():
            member = archive.extractfile(name)
            if member is None or hashlib.sha256(member.read()).hexdigest() != expected:
                raise ValueError("archived bytes differ from the verified source")
    result = {
        "schema": "verified-source-archive-v1",
        "files": len(report["files_sha256"]),
        "manifest_sha256": digest(report_path),
        "archive_sha256": digest(archive_path),
        "all_member_bytes_verified": True,
        "scope": "Durable exact source/test/checkpoint archive; not test success",
    }
    info_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    persist(parser.parse_args().manifest)
