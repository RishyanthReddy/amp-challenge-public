"""Offline regression checks for the pinned training-source downloader."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fetch_training_sources as fetch


class SourceFetchChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = b"exact retained input\n"
        self.item = {"id": "fixture", "path": "data/input.csv", "bytes": len(self.data),
                     "sha256": hashlib.sha256(self.data).hexdigest(), "bundled": False,
                     "download_urls": ["https://example.org/input"]}
        self.target = self.root / self.item["path"]

    def install_fixture(self, item, path, timeout):
        path.write_bytes(self.data)
        fetch.verify(path, item)

    def restore(self, entries=None, verify_only=False, timeout=10):
        return fetch.restore(self.root, entries or [self.item],
                             verify_only=verify_only, timeout=timeout)

    def test_verified_download(self):
        with patch.object(fetch, "download", side_effect=self.install_fixture):
            self.assertEqual(self.restore()["downloaded_inputs"], 1)
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_cache_without_network(self):
        self.target.parent.mkdir()
        self.target.write_bytes(self.data)
        with patch.object(fetch, "download", side_effect=AssertionError("network used")):
            self.assertEqual(self.restore(verify_only=True)["downloaded_inputs"], 0)

    def test_corrupt_cache_preserved(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"different input")
        with self.assertRaises(ValueError):
            self.restore()
        self.assertEqual(self.target.read_bytes(), b"different input")

    def test_missing_verification_input(self):
        with self.assertRaises(FileNotFoundError):
            self.restore(verify_only=True)

    def test_missing_retained_only_input(self):
        with self.assertRaises(FileNotFoundError):
            self.restore([dict(self.item, download_urls=[])])

    def test_failed_batch_installs_nothing(self):
        second = dict(self.item, id="second", path="data/second.csv")

        def download(item, path, timeout):
            if item["id"] == "second":
                raise RuntimeError("upstream unavailable")
            self.install_fixture(item, path, timeout)

        with patch.object(fetch, "download", side_effect=download):
            with self.assertRaises(RuntimeError):
                self.restore([self.item, second])
        self.assertFalse(self.target.exists())
        self.assertFalse((self.root / second["path"]).exists())

    def test_concurrent_file_preserved(self):
        def download(item, path, timeout):
            self.install_fixture(item, path, timeout)
            self.target.parent.mkdir()
            self.target.write_bytes(b"concurrent file")

        with patch.object(fetch, "download", side_effect=download):
            with self.assertRaises(ValueError):
                self.restore()
        self.assertEqual(self.target.read_bytes(), b"concurrent file")

    def test_path_escape_rejected(self):
        for path in ("../outside", "/outside"):
            with self.assertRaises(ValueError):
                fetch.destination(self.root, path)

    def test_nonfinite_timeout_rejected(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.restore(timeout=timeout)

    def test_duplicate_manifest_id_rejected(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps({"schema_version": 1, "inputs": [self.item, self.item]}))
        with self.assertRaises(ValueError):
            fetch.read_manifest(path)

    def test_changed_upstream_rejected(self):
        class Response:
            content = b"changed export"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def geturl(self):
                return "https://example.org/input"

            def read(self, size):
                data, self.content = self.content, b""
                return data

        with patch.object(fetch, "urlopen", return_value=Response()):
            with self.assertRaises(RuntimeError):
                fetch.download(self.item, self.root / "staged", 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
