"""Embedder artifact pin — the anti-auto-update contract.

The model artifact is a dependency of every stored vector: if fastembed
silently pulls a refreshed artifact (its cache lives in macOS-purged $TMPDIR,
re-downloading whatever HF `main` serves), new embeddings land in a different
space than the ~8.4k stored ones. Operator ruling 2026-08-07: embedders must
never auto-update. These tests pin the enforcement mechanics WITHOUT loading
a model — pure filesystem contracts.
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "servers"))
import embedder  # noqa: E402

GOOD_REV = "e" * 40
GOOD_SHA = "b" * 64


def _fake_model(model_dir):
    """Mimic fastembed's TextEmbedding.model._model_dir shape."""
    inner = types.SimpleNamespace(_model_dir=str(model_dir))
    return types.SimpleNamespace(model=inner)


def _hub_snapshot(tmp_path, rev=GOOD_REV, blob_sha=GOOD_SHA):
    """Build a minimal HF-hub cache layout: snapshots/<rev>/onnx/x.onnx -> blobs/<sha>."""
    repo = tmp_path / "models--org--repo"
    blobs = repo / "blobs"
    snap = repo / "snapshots" / rev / "onnx"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    blob = blobs / blob_sha
    blob.write_bytes(b"onnx-bytes")
    (snap / "model_quantized.onnx").symlink_to(blob)
    return repo / "snapshots" / rev


class TestVerifyArtifactPin:
    def test_matching_pins_pass(self, tmp_path):
        snap = _hub_snapshot(tmp_path)
        rev, sha = embedder._verify_artifact_pin(
            _fake_model(snap),
            {"pinned_revision": GOOD_REV, "pinned_onnx_sha256": GOOD_SHA})
        assert rev == GOOD_REV and sha == GOOD_SHA

    def test_revision_mismatch_raises(self, tmp_path):
        snap = _hub_snapshot(tmp_path)
        with pytest.raises(RuntimeError, match="PIN MISMATCH"):
            embedder._verify_artifact_pin(
                _fake_model(snap),
                {"pinned_revision": "f" * 40, "pinned_onnx_sha256": GOOD_SHA})

    def test_sha_mismatch_raises(self, tmp_path):
        snap = _hub_snapshot(tmp_path)
        with pytest.raises(RuntimeError, match="PIN MISMATCH"):
            embedder._verify_artifact_pin(
                _fake_model(snap),
                {"pinned_revision": GOOD_REV, "pinned_onnx_sha256": "a" * 64})

    def test_no_pins_never_raises(self, tmp_path):
        snap = _hub_snapshot(tmp_path)
        rev, sha = embedder._verify_artifact_pin(_fake_model(snap), {})
        assert rev == GOOD_REV and sha == GOOD_SHA  # still reported for stats

    def test_non_hub_layout_hashes_bytes(self, tmp_path):
        # plain dir, real file (no symlink, non-hex name) → sha256 of bytes
        import hashlib
        d = tmp_path / "local-model"
        d.mkdir()
        (d / "model.onnx").write_bytes(b"raw-model-bytes")
        expected = hashlib.sha256(b"raw-model-bytes").hexdigest()
        rev, sha = embedder._verify_artifact_pin(
            _fake_model(d), {"pinned_onnx_sha256": expected})
        assert sha == expected
        assert rev is None  # dir name isn't a 40-hex revision

    def test_missing_model_dir_with_pin_raises(self):
        broken = types.SimpleNamespace(model=types.SimpleNamespace())
        with pytest.raises(RuntimeError, match="cannot locate"):
            embedder._verify_artifact_pin(broken, {"pinned_revision": GOOD_REV})


class TestSeedCacheFromTmp:
    def test_seeds_q_variant_from_base_repo(self, tmp_path, monkeypatch):
        # nomic-Q resolves to the un-suffixed repo dir — seed must find it
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        src = tmp_cache / "models--org--repo"
        (src / "blobs").mkdir(parents=True)
        (src / "blobs" / "x").write_bytes(b"blob")
        monkeypatch.setattr(embedder.tempfile, "gettempdir",
                            lambda: str(tmp_path / "tmpdir"))
        durable = tmp_path / "durable"
        embedder._seed_cache_from_tmp("org/repo-Q", str(durable))
        assert (durable / "models--org--repo" / "blobs" / "x").exists()

    def test_existing_durable_copy_untouched(self, tmp_path, monkeypatch):
        tmp_cache = tmp_path / "tmpdir" / "fastembed_cache"
        (tmp_cache / "models--org--repo" / "blobs").mkdir(parents=True)
        monkeypatch.setattr(embedder.tempfile, "gettempdir",
                            lambda: str(tmp_path / "tmpdir"))
        durable = tmp_path / "durable"
        marker = durable / "models--org--repo" / "KEEP"
        marker.parent.mkdir(parents=True)
        marker.write_text("existing")
        embedder._seed_cache_from_tmp("org/repo", str(durable))
        assert marker.read_text() == "existing"  # no clobber

    def test_seed_failure_is_nonfatal(self, monkeypatch):
        monkeypatch.setattr(embedder.tempfile, "gettempdir",
                            lambda: (_ for _ in ()).throw(OSError("boom")))
        embedder._seed_cache_from_tmp("org/repo", "/nonexistent/durable")  # no raise
