"""Canonical JSON in bounded deterministic-header gzip, with two hash layers.

All logical data are retained, not sampled or shortened. Payload hashes define
scientific replay; stored-byte hashes detect corruption. Different zlib versions
may encode identical payloads differently, which is not a scientific difference.
"""

import gzip
import hashlib
import io
import json
import re
from pathlib import Path

CODEC = {"name": "canonical-json-gzip-v1", "level": 6, "mtime": 0, "filename": ""}
MAX_PAYLOAD = 256 * 1024 * 1024
MAX_MANIFEST = 16 * 1024 * 1024
MAX_ENTRIES = 30_000
CHUNK_SIZE = 64 * 1024


def canonical_chunks(value):
    """Same JSON bytes as canonical(), without joining the entire document.

    iterencode may still allocate a single escaped scalar; subsequent writes are
    bounded. This avoids document-sized copies, not the input object's memory.
    """
    encoder = json.JSONEncoder(sort_keys=True, indent=2, allow_nan=False)
    pending = bytearray()
    for fragment in encoder.iterencode(value):
        offset = 0
        while offset < len(fragment):
            count = min(CHUNK_SIZE - len(pending), len(fragment) - offset)
            pending.extend(fragment[offset : offset + count].encode("utf-8"))
            offset += count
            if len(pending) == CHUNK_SIZE:
                yield bytes(pending)
                pending.clear()
    pending.extend(b"\n")
    yield bytes(pending)


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_matches(value, payload):
    offset = 0
    for chunk in canonical_chunks(value):
        if payload[offset : offset + len(chunk)] != chunk:
            return False
        offset += len(chunk)
    return offset == len(payload)


class _HashingSink:
    """Hash/count bytes as gzip writes them; never buffer a compressed document."""

    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, data):
        if self.size + len(data) > MAX_PAYLOAD:
            raise ValueError("stored artifact exceeds bound; never truncate evidence")
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("incomplete artifact write")
        self.digest.update(data)
        self.size += written
        return written

    def flush(self):
        self.stream.flush()


def relative_name(name):
    if (
        type(name) is not str
        or len(name) > 240
        or re.fullmatch(r"[a-z0-9_./-]+\.json", name) is None
        or any(part in ("", ".", "..") for part in name.split("/"))
        or any(
            part.split(".")[0].upper()
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }
            for part in name.split("/")
        )
    ):
        raise ValueError("invalid logical artifact path")
    return name


def path_in(directory, name):
    path = directory / (relative_name(name) + ".gz")
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("artifact path escape")
    return path


def encoded(data):
    sink = io.BytesIO()
    with gzip.GzipFile(
        filename="", fileobj=sink, mode="wb", compresslevel=6, mtime=0
    ) as archive:
        archive.write(data)
    return sink.getvalue()


def decoded(data, limit):
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as archive:
            payload = archive.read(limit + 1)
        if len(payload) > limit:
            raise ValueError("gzip logical payload exceeds bound")
        value = json.loads(payload)
        if not canonical_matches(value, payload):
            raise ValueError("artifact is not canonical JSON")
        return payload, value
    except (
        OSError,
        EOFError,
        UnicodeError,
        RecursionError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError("invalid compressed JSON artifact") from error


class Writer:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.artifacts, self.nondeterministic = {}, []
        self.closed = False
        self.failed = False

    def write(self, name, value, *, nondeterministic=False):
        relative_name(name)
        if self.closed or self.failed or name in self.artifacts or name in self.nondeterministic:
            raise ValueError("closed/failed store or duplicate artifact")
        path = path_in(self.directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload_digest, payload_bytes = hashlib.sha256(), 0
        try:
            with path.open("xb") as stream:
                sink = _HashingSink(stream)
                with gzip.GzipFile(
                    filename="", fileobj=sink, mode="wb", compresslevel=6, mtime=0
                ) as archive:
                    for chunk in canonical_chunks(value):
                        if payload_bytes + len(chunk) > MAX_PAYLOAD:
                            raise ValueError(
                                "logical artifact exceeds storage bound; never truncate evidence"
                            )
                        archive.write(chunk)
                        payload_digest.update(chunk)
                        payload_bytes += len(chunk)
        except BaseException:
            # Preserve incomplete bytes for diagnosis, but never certify a store
            # containing an unregistered failed write, even if the caller catches it.
            self.failed = True
            raise
        record = {
            "path": name + ".gz",
            "sha256": sink.digest.hexdigest(),
            "stored_bytes": sink.size,
            "payload_sha256": payload_digest.hexdigest(),
            "payload_bytes": payload_bytes,
        }
        if nondeterministic:
            self.nondeterministic.append(name)
        else:
            self.artifacts[name] = record
        return record

    def reference(self, name):
        # A lazy, pinned read_text capability for the existing audit API. It
        # opens no private bytes until that API explicitly requests them.
        return Reference(self.directory, name, self.artifacts[name])

    def read(self, name):
        return read_pinned(self.directory, name, self.artifacts[name])[1]

    def finish(self):
        if self.closed or self.failed:
            raise ValueError("store already complete or contains a failed write")
        manifest = {
            "schema": "compressed-development-artifacts-v1",
            "codec": CODEC,
            "artifacts": self.artifacts,
            "nondeterministic_artifacts": sorted(self.nondeterministic),
        }
        data = canonical(manifest)
        if len(data) > MAX_MANIFEST or not 1 <= len(self.artifacts) <= MAX_ENTRIES:
            raise ValueError("manifest exceeds bound; retain partial files")
        with (self.directory / "manifest.json").open("xb") as stream:
            stream.write(data)
        self.closed = True
        return manifest


def read_pinned(directory, name, record):
    path = path_in(directory, name)
    if path.stat().st_size != record["stored_bytes"]:
        raise ValueError("stored artifact length mismatch")
    data = path.read_bytes()
    if sha(data) != record["sha256"]:
        raise ValueError("stored artifact hash mismatch")
    payload, value = decoded(data, record["payload_bytes"])
    if (
        len(payload) != record["payload_bytes"]
        or sha(payload) != record["payload_sha256"]
    ):
        raise ValueError("logical artifact hash/length mismatch")
    return payload, value


class Reference:
    def __init__(self, directory, name, record):
        self.directory, self.name, self.record = Path(directory), name, dict(record)

    def read_text(self, *, encoding="utf-8"):
        if encoding != "utf-8":
            raise ValueError("compressed reference requires UTF-8")
        return read_pinned(self.directory, self.name, self.record)[0].decode("utf-8")


class Reader:
    def __init__(self, directory, *, manifest_sha256=None):
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        if path.stat().st_size > MAX_MANIFEST:
            raise ValueError("manifest exceeds bound")
        data = path.read_bytes()
        if manifest_sha256 is not None and sha(data) != manifest_sha256:
            raise ValueError("compressed manifest pin mismatch")
        manifest = json.loads(data)
        if (
            type(manifest) is not dict
            or set(manifest)
            != {"schema", "codec", "artifacts", "nondeterministic_artifacts"}
            or manifest["schema"] != "compressed-development-artifacts-v1"
            or manifest["codec"] != CODEC
            or canonical(manifest) != data
            or type(manifest["artifacts"]) is not dict
            or not 1 <= len(manifest["artifacts"]) <= MAX_ENTRIES
            or type(manifest["nondeterministic_artifacts"]) is not list
        ):
            raise ValueError("invalid compressed manifest")
        for name, record in manifest["artifacts"].items():
            relative_name(name)
            if (
                type(record) is not dict
                or set(record)
                != {"path", "sha256", "stored_bytes", "payload_sha256", "payload_bytes"}
                or record["path"] != name + ".gz"
                or any(
                    type(record[k]) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", record[k]) is None
                    for k in ("sha256", "payload_sha256")
                )
                or any(
                    type(record[k]) is not int or not 1 <= record[k] <= MAX_PAYLOAD
                    for k in ("stored_bytes", "payload_bytes")
                )
            ):
                raise ValueError("invalid compressed artifact descriptor")
        nondet = manifest["nondeterministic_artifacts"]
        for name in nondet:
            relative_name(name)
        if sorted(set(nondet)) != nondet or set(nondet) & set(manifest["artifacts"]):
            raise ValueError("invalid nondeterministic artifact list")
        self.manifest, self.manifest_sha256 = manifest, sha(data)

    def read(self, name):
        relative_name(name)
        return read_pinned(self.directory, name, self.manifest["artifacts"][name])[1]

    def nondeterministic(self, name):
        if name not in self.manifest["nondeterministic_artifacts"]:
            raise ValueError("unregistered nondeterministic artifact")
        path = path_in(self.directory, name)
        if path.stat().st_size > MAX_PAYLOAD:
            raise ValueError("oversized nondeterministic artifact")
        return decoded(path.read_bytes(), MAX_PAYLOAD)[1]

    def reference(self, name):
        return Reference(self.directory, name, self.manifest["artifacts"][name])

    def coverage(self, expected, *, nondeterministic=("timing.json",)):
        if set(self.manifest["artifacts"]) != set(expected) or self.manifest[
            "nondeterministic_artifacts"
        ] != list(nondeterministic):
            raise ValueError("logical artifact/denominator coverage mismatch")
        physical = {name + ".gz" for name in set(expected) | set(nondeterministic)} | {
            "manifest.json"
        }
        if {
            p.relative_to(self.directory).as_posix()
            for p in self.directory.rglob("*")
            if p.is_file()
        } != physical:
            raise ValueError("physical artifact coverage mismatch")

    def payload_hashes(self):
        return {
            name: record["payload_sha256"]
            for name, record in self.manifest["artifacts"].items()
        }
