"""Lossless storage, bounded decompression, lazy truth and corruption negatives."""

import gzip
import json

import pytest

from .store import Reader, Writer, canonical, decoded, encoded, sha


@pytest.mark.parametrize(
    "value",
    (
        None,
        True,
        19,
        0.125,
        "你好",
        [1, {"z": "line\n", "a": []}],
        {"events": [{"kind": "failed", "cost": 9}] * 200},
    ),
)
def test_lossless_canonical_payload_and_deterministic_header(tmp_path, value):
    writer = Writer(tmp_path / "store")
    record = writer.write("run.json", value)
    writer.finish()
    reader = Reader(writer.directory)
    reader.coverage({"run.json"}, nondeterministic=())
    assert reader.read("run.json") == value
    data = (writer.directory / "run.json.gz").read_bytes()
    assert data == encoded(canonical(value)) == encoded(canonical(value))
    assert data[4:8] == b"\0\0\0\0" and data[3] == 0
    assert gzip.decompress(data) == canonical(value)
    assert reader.payload_hashes() == {"run.json": sha(canonical(value))}
    assert record["payload_bytes"] == len(canonical(value))


@pytest.mark.parametrize(
    "name",
    (
        "../run.json",
        "a/../run.json",
        "/run.json",
        "a\\run.json",
        "C:/run.json",
        "CON.json",
        "a/nul.json",
        "a//run.json",
        "a/./run.json",
        "Run.json",
        "run.json.gz",
        "a\0.json",
        "",
    ),
)
def test_no_path_escape_or_cross_platform_alias(tmp_path, name):
    writer = Writer(tmp_path / "store")
    with pytest.raises(ValueError):
        writer.write(name, {"secret": "must not write"})
    assert not list(writer.directory.rglob("*"))


def test_existing_store_duplicates_and_closed_store_never_overwrite(tmp_path):
    writer = Writer(tmp_path / "store")
    writer.write("run.json", [1])
    with pytest.raises(FileExistsError):
        Writer(writer.directory)
    with pytest.raises(ValueError):
        writer.write("run.json", [2])
    writer.finish()
    with pytest.raises(ValueError):
        writer.write("later.json", [3])
    with pytest.raises(ValueError):
        writer.finish()
    assert Reader(writer.directory).read("run.json") == [1]


def test_partial_and_empty_stores_cannot_be_mistaken_for_complete(tmp_path):
    writer = Writer(tmp_path / "store")
    with pytest.raises(ValueError):
        writer.finish()
    writer.write("run.json", {"failed": True})
    assert not (writer.directory / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        Reader(writer.directory)


@pytest.mark.parametrize(
    "tamper", ("flip", "truncate", "length", "payload_hash", "rehash_different_value")
)
def test_corruption_detected_at_both_hash_layers(tmp_path, tamper):
    writer = Writer(tmp_path / "store")
    writer.write("run.json", {"spent": 42})
    manifest = writer.finish()
    path = writer.directory / "run.json.gz"
    record = manifest["artifacts"]["run.json"]
    if tamper == "flip":
        data = bytearray(path.read_bytes())
        data[-5] ^= 1
        path.write_bytes(data)
    elif tamper == "truncate":
        path.write_bytes(path.read_bytes()[:-1])
    elif tamper == "length":
        record["payload_bytes"] -= 1
    elif tamper == "payload_hash":
        record["payload_sha256"] = "0" * 64
    else:
        data = encoded(canonical({"spent": 99}))
        path.write_bytes(data)
        record["sha256"], record["stored_bytes"] = sha(data), len(data)
    (writer.directory / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError):
        Reader(writer.directory).read("run.json")


@pytest.mark.parametrize(
    "payload", (b'{"a":1}\n', b'{"a":1,"a":2}\n', b"NaN\n", b"not json", b'"\xff"\n')
)
def test_noncanonical_or_invalid_json_rejected(payload):
    with pytest.raises(ValueError):
        decoded(encoded(payload), 100)


def test_decompression_limit_stops_bomb_and_bad_crc():
    payload = canonical("x" * 100000)
    with pytest.raises(ValueError, match="bound"):
        decoded(encoded(payload), 20)
    data = bytearray(encoded(canonical({"a": 1})))
    data[-8] ^= 1
    with pytest.raises(ValueError):
        decoded(bytes(data), 100)


def test_private_reference_is_lazy_and_pinned(tmp_path, monkeypatch):
    from pathlib import Path

    writer = Writer(tmp_path / "store")
    value = {"private": "not used by planner"}
    writer.write("evaluator/world.json", value)
    reads = []
    original = Path.read_bytes

    def tracked(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    reference = writer.reference("evaluator/world.json")
    assert not reads
    assert json.loads(reference.read_text(encoding="utf-8")) == value
    assert len(reads) == 1


def test_payload_manifest_pin_and_exact_physical_coverage(tmp_path):
    writer = Writer(tmp_path / "store")
    writer.write("run.json", [1, 2])
    writer.write("timing.json", {"seconds": 1.1}, nondeterministic=True)
    writer.finish()
    pin = sha((writer.directory / "manifest.json").read_bytes())
    reader = Reader(writer.directory, manifest_sha256=pin)
    reader.coverage({"run.json"})
    assert reader.nondeterministic("timing.json") == {"seconds": 1.1}
    with pytest.raises(ValueError):
        Reader(writer.directory, manifest_sha256="0" * 64)
    (writer.directory / "unregistered.json.gz").write_bytes(encoded(canonical([3])))
    with pytest.raises(ValueError, match="coverage"):
        reader.coverage({"run.json"})


@pytest.mark.parametrize("bad", (True, -1, "7", 0))
def test_manifest_integer_bounds_are_exact(tmp_path, bad):
    writer = Writer(tmp_path / "store")
    writer.write("run.json", [])
    manifest = writer.finish()
    manifest["artifacts"]["run.json"]["payload_bytes"] = bad
    (writer.directory / "manifest.json").write_bytes(canonical(manifest))
    with pytest.raises(ValueError):
        Reader(writer.directory)


def test_streamed_large_document_matches_legacy_bytes_with_bounded_writes(tmp_path, monkeypatch):
    from . import store

    value = {"records": [{"action": "a", "failed": False, "receipt": i} for i in range(12000)], "unicode": "你" * 20000}
    payload = canonical(value)
    sizes = []
    original = gzip.GzipFile.write

    def track_write(archive, data):
        sizes.append(len(data))
        return original(archive, data)

    def forbidden(*args, **kwargs):
        raise AssertionError("whole-document serialization/compression is forbidden")

    writer = Writer(tmp_path / "streamed")
    with monkeypatch.context() as context:
        context.setattr(store, "canonical", forbidden)
        context.setattr(store, "encoded", forbidden)
        context.setattr(json, "dumps", forbidden)
        context.setattr(gzip.GzipFile, "write", track_write)
        record = writer.write("run.json", value)
    data = (writer.directory / "run.json.gz").read_bytes()
    assert len(sizes) > 10 and max(sizes) <= store.CHUNK_SIZE
    assert data == encoded(payload)
    assert record == {"path": "run.json.gz", "sha256": sha(data), "stored_bytes": len(data), "payload_sha256": sha(payload), "payload_bytes": len(payload)}
    writer.finish()
    assert Reader(writer.directory).read("run.json") == value


def test_decoder_checks_canonical_bytes_without_full_document_reencoding(monkeypatch):
    from . import store

    value = {"records": list(range(20000))}
    payload = canonical(value)
    data = encoded(payload)

    def forbidden(*args, **kwargs):
        raise AssertionError("joined canonical copy is forbidden")

    monkeypatch.setattr(store, "canonical", forbidden)
    monkeypatch.setattr(json, "dumps", forbidden)
    assert decoded(data, len(payload)) == (payload, value)


@pytest.mark.parametrize("failure", ("payload_bound", "nan", "io", "stored_bound"))
def test_failed_stream_is_retained_but_cannot_be_certified(tmp_path, monkeypatch, failure):
    from . import store

    writer = Writer(tmp_path / "partial")
    writer.write("previous.json", [1])
    previous = (writer.directory / "previous.json.gz").read_bytes()
    value = list(range(100))
    if failure == "payload_bound":
        monkeypatch.setattr(store, "MAX_PAYLOAD", 80)
    elif failure == "nan":
        value = [float("nan")]
    elif failure == "io":
        def broken(*args):
            raise OSError("injected stream failure")
        monkeypatch.setattr(gzip.GzipFile, "write", broken)
    else:
        monkeypatch.setattr(store, "MAX_PAYLOAD", 1)
    with pytest.raises((ValueError, OSError)):
        writer.write("failed.json", value)
    assert writer.failed
    assert (writer.directory / "failed.json.gz").exists()
    assert (writer.directory / "previous.json.gz").read_bytes() == previous
    assert "failed.json" not in writer.artifacts
    with pytest.raises(ValueError, match="failed"):
        writer.finish()
    with pytest.raises(ValueError, match="failed"):
        writer.write("later.json", [])
    assert not (writer.directory / "manifest.json").exists()


def test_existing_artifact_path_is_never_overwritten_after_stream_failure(tmp_path):
    writer = Writer(tmp_path / "partial")
    path = writer.directory / "run.json.gz"
    path.write_bytes(b"retained partial evidence")
    with pytest.raises(FileExistsError):
        writer.write("run.json", {})
    assert path.read_bytes() == b"retained partial evidence"
    with pytest.raises(ValueError, match="failed"):
        writer.finish()
