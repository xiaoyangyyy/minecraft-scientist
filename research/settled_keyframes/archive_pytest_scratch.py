"""Losslessly archive ONLY named pytest mock scratch; never delete anything.

Formal native experiments, reports, source copies and frozen archives are not
eligible. An explicit caller may remove the original scratch only AFTER the
archive's complete byte verification and source recheck have succeeded.
"""

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path

ALLOWED = {
    "shared-anchor-full-temp-windows-v1",
    "shared-endpoint-full-temp-windows-v1",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root):
    files, links = {}, {}

    def walk(folder):
        for entry in sorted(os.scandir(folder), key=lambda entry: entry.name):
            path = Path(entry.path)
            name = path.relative_to(root).as_posix()
            metadata = entry.stat(follow_symlinks=False)
            reparse = getattr(metadata, "st_file_attributes", 0) & 1024
            if entry.is_symlink() or reparse:
                target = path.resolve(strict=True)
                if not target.is_relative_to(root):
                    raise ValueError("scratch alias escapes the exact source directory")
                links[name] = target.relative_to(root).as_posix()
            elif entry.is_dir(follow_symlinks=False):
                walk(path)
            elif entry.is_file(follow_symlinks=False):
                files[name] = {"sha256": digest(path), "bytes": metadata.st_size}
            else:
                raise ValueError("unsupported scratch filesystem object")

    walk(root)
    return {"files": files, "aliases": links}


def archive(source, *, parent, output):
    source, parent, output = Path(source), Path(parent), Path(output)
    if source.is_symlink() or parent.is_symlink():
        raise ValueError("scratch source/parent cannot be aliases")
    source, parent, output = (
        source.resolve(strict=True),
        parent.resolve(strict=True),
        output.resolve(),
    )
    if (
        source.name not in ALLOWED
        or source.parent != parent
        or parent == Path(parent.anchor)
    ):
        raise ValueError(
            "only explicitly named pytest scratch directories are eligible"
        )
    if (
        output.exists()
        or output.with_suffix(".json").exists()
        or output.is_relative_to(source)
    ):
        raise FileExistsError("archive output must be new and outside source")
    if any(not item.name.startswith("test_") for item in source.iterdir()):
        raise ValueError("scratch root contains something other than pytest mock cases")
    declared = inventory(source)
    if not declared["files"]:
        raise ValueError("cannot archive empty scratch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(
        output, "x:gz", dereference=True, format=tarfile.PAX_FORMAT
    ) as target:
        for name in declared["files"]:
            target.add(source / name, arcname=name, recursive=False)
    seen = {}
    with tarfile.open(output, "r:gz") as saved:
        for member in saved:
            if (
                not member.isfile()
                or member.name not in declared["files"]
                or member.name in seen
            ):
                raise ValueError("archive membership changed")
            stream = saved.extractfile(member)
            if stream is None:
                raise ValueError("archive member unreadable")
            seen[member.name] = {
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                "bytes": member.size,
            }
    if seen != declared["files"] or inventory(source) != declared:
        raise ValueError("archive bytes or source changed during verification")
    result = {
        "schema": "verified-pytest-scratch-archive-v1",
        "source": str(source),
        "archive": str(output),
        "archive_sha256": digest(output),
        "verified": True,
        "source_unchanged": True,
        "source_removed": False,
        "file_count": len(seen),
        "bytes": sum(item["bytes"] for item in seen.values()),
        "inventory": declared,
        "scope": "Only pytest mock scratch. Every real file byte verified; internal current aliases are metadata, not duplicate content. No native capture, report, source tree or original frozen archive is removed.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = archive(args.source, parent=args.parent, output=args.output)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "inventory"},
            sort_keys=True,
        )
    )
