"""Pinned corpus capability: selection reads no sealed split fixture files."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from research.structural_holdouts.audit_corpus import validate_public


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class DevelopmentCorpus:
    directory: Path
    manifest_sha256: str
    hashes: dict
    fixture_ids: tuple
    groups: dict
    available_groups: int

    def read(self, relative):
        # A caller cannot turn this into a loader for tuning/evaluation truth.
        if relative not in ("registration.json", "partitions.json", "report.json"):
            parts = Path(relative).parts
            if (
                not parts
                or parts[0] not in self.fixture_ids
                or relative
                not in {
                    f"{parts[0]}/public.json",
                    f"{parts[0]}/metadata.json",
                    f"{parts[0]}/evaluator/world.json",
                }
            ):
                raise ValueError("sealed split or unregistered corpus access")
        path = self.directory / relative
        if relative not in self.hashes or not path.resolve().is_relative_to(
            self.directory.resolve()
        ):
            raise ValueError("pinned corpus artifact drift/escape")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.hashes[relative]:
            raise ValueError("pinned corpus artifact drift/escape")
        return json.loads(data)


def load_development(directory, *, manifest_sha256, max_groups=None):
    """Pin certified construction bytes, then select complete development groups.

    The pin must come from the independently verified construction receipt.
    This does not recompute private leakage identities or certify arbitrary data.
    Those operations belong to the construction audit before methods are frozen.
    """
    if (
        type(manifest_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None
        or (max_groups is not None and (type(max_groups) is not int or max_groups < 1))
    ):
        raise ValueError("invalid development selection/pin")
    directory = Path(directory)
    path = directory / "manifest.json"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest_sha256:
        raise ValueError("corpus manifest does not match construction pin")
    manifest = json.loads(data)
    if (
        type(manifest) is not dict
        or set(manifest) != {"schema", "sha256", "nondeterministic_artifacts"}
        or manifest["schema"] != "new-structure-corpus-manifest-v1"
        or type(manifest["sha256"]) is not dict
        or manifest["nondeterministic_artifacts"] != ["timing.json"]
    ):
        raise ValueError("invalid pinned construction manifest")
    hashes = manifest["sha256"]
    shell = DevelopmentCorpus(directory, manifest_sha256, hashes, (), {}, 0)
    registration = shell.read("registration.json")
    partition = shell.read("partitions.json")
    report = shell.read("report.json")
    if (
        registration.get("schema") != "new-structure-corpus-registration-v1"
        or report.get("schema") != "new-structure-corpus-report-v1"
        or type(partition) is not dict
        or type(partition.get("assignments")) is not dict
        or type(partition.get("groups")) is not dict
        or set(partition["assignments"]) != set(partition["groups"])
        or any(
            re.fullmatch(r"[0-9a-f]{24}", fid) is None
            or split not in ("development", "tuning", "evaluation")
            for fid, split in partition["assignments"].items()
        )
    ):
        raise ValueError("invalid pinned development partition")
    group_splits = {}
    for fid, split in partition["assignments"].items():
        group = partition["groups"][fid]
        if type(group) is not str or group_splits.setdefault(group, split) != split:
            raise ValueError("pinned group crosses splits")
    eligible = sorted(g for g, split in group_splits.items() if split == "development")
    if not eligible or (max_groups is not None and max_groups > len(eligible)):
        raise ValueError("no/sufficient requested development groups")
    chosen = set(eligible if max_groups is None else eligible[:max_groups])
    fixture_ids = tuple(
        sorted(fid for fid, group in partition["groups"].items() if group in chosen)
    )
    if report["summary"]["development"]["independent_groups"] != len(
        eligible
    ) or report["summary"]["development"]["fixtures"] != sum(
        v == "development" for v in partition["assignments"].values()
    ):
        raise ValueError("pinned development denominators disagree")
    selected = DevelopmentCorpus(
        directory,
        manifest_sha256,
        hashes,
        fixture_ids,
        {fid: partition["groups"][fid] for fid in fixture_ids},
        len(eligible),
    )
    # Every selected public entry is checked before opening ANY selected truth.
    for fid in fixture_ids:
        validate_public(selected.read(f"{fid}/public.json"))
    return selected
