"""Freeze exact source/test bytes on native disk without deleting prior evidence."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(output, ledger_path=None):
    output = output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite test-tree verification")
    ledger_path = (
        Path(ledger_path).resolve(strict=True)
        if ledger_path is not None
        else ROOT / "runs/core-release-seeds0-29-wsl-v28/config.json"
    )
    ledger = json.loads(ledger_path.read_bytes())
    names = set(
        subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        )
        .stdout.decode()
        .split("\0")
    ) - {""}
    # Existing fixture checkpoints are ignored but must not disappear from tests.
    names.update(
        str(path.relative_to(ROOT))
        for path in (ROOT / "checkpoints").iterdir()
        if path.is_file()
    )
    if len(names) > 3000:
        raise ValueError("test source copy exceeds its declared file-count budget")
    sources = {}
    size = 0
    for name in sorted(names):
        source = ROOT / name
        if source.is_symlink() or ROOT not in source.resolve(strict=True).parents:
            raise ValueError("test snapshot cannot follow external paths or symlinks")
        if any(part.startswith(".env") for part in source.relative_to(ROOT).parts):
            raise ValueError("test snapshot must not copy credential files")
        size += source.stat().st_size
        if size > 64 * 1024 * 1024:
            raise ValueError("test source/fixture copy exceeds its byte budget")
        sources[name] = digest(source)
    snapshot = Path(tempfile.mkdtemp(prefix="msc-source-frozen-", dir="/tmp"))
    for name in sources:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name, value in sources.items():
        if digest(ROOT / name) != value or digest(snapshot / name) != value:
            raise ValueError("source changed while creating immutable test snapshot")
    for name, value in ledger["source_sha256"].items():
        relative = (
            "src/minecraft_scientist/" + name[8:]
            if name.startswith("package/")
            else "scripts/verify_core_release.py"
            if name == "verify_core_release.py"
            else name
        )
        if digest(snapshot / relative) != value:
            raise ValueError("copied package/config differs from declared core ledger")
    result = {
        "schema": "exact-test-tree-snapshot-v1",
        "source_root": str(ROOT),
        "snapshot_root": str(snapshot),
        "files_sha256": sources,
        "bytes": size,
        "core_ledger_sha256": digest(ledger_path),
        "core_ledger_path": str(ledger_path),
        "checks": {
            "all_copied_bytes_match": True,
            "all_sources_unchanged": True,
            "package_ledger_matches": True,
        },
        "scope": "complete tracked/unignored test tree plus existing checkpoint fixtures; not test success or clean-Git evidence",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output / "freeze.source.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                "snapshot_root": str(snapshot),
                "files": len(sources),
                "bytes": size,
                "checks": result["checks"],
            },
            sort_keys=True,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    arguments = parser.parse_args()
    freeze(arguments.output, arguments.ledger)
