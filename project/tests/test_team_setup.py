"""Offline download integrity, repair, and signing tests; no cloud credentials."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from test_text_store import write_fixture


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "team-setup"))
import _common
import create_download_links
import download_text
from example_database import query_comments


class TeamDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        write_fixture(self.source)
        self.output = self.root / "download"
        self.manifest = (self.source / "manifest.json").read_bytes()
        self.digest = hashlib.sha256(self.manifest).hexdigest()
        self.enterContext(patch.object(_common, "MANIFEST_SHA256", self.digest))
        self.enterContext(patch.object(download_text, "MANIFEST_SHA256", self.digest))
        entries = _common.read_manifest(self.manifest)[1]
        self.paths = ["manifest.json", *(entry["path"] for entry in entries)]
        self.urls = {relative: f"{_common.PROJECT_URL}/storage/v1/object/sign/"
                     f"{_common.BUCKET}/{_common.PREFIX}/{relative}?token=test"
                     for relative in self.paths}
        self.links = self.root / "links.json"
        self.links.write_text(json.dumps({"version": "v1", "urls": self.urls}))

    def remote_response(self, url, timeout):
        relative = next(relative for relative, signed in self.urls.items() if signed == url)
        return io.BytesIO((self.source / relative).read_bytes())

    def download(self):
        return download_text.download_release(self.links, self.output)

    def test_complete_download_preserves_layout_validates_and_skips_repeat_transfer(self):
        with patch.object(download_text, "urlopen", side_effect=self.remote_response) as remote:
            summary = self.download()
            self.assertEqual(remote.call_count, len(self.paths))
        self.assertEqual(summary["post_rows"], 4)
        self.assertEqual(summary["comment_rows"], 5)
        for relative in self.paths:
            self.assertEqual((self.source / relative).read_bytes(), (self.output / relative).read_bytes())
        with patch.object(download_text, "urlopen", side_effect=AssertionError("unnecessary transfer")):
            self.download()

    def test_rerun_repairs_only_corrupt_file(self):
        with patch.object(download_text, "urlopen", side_effect=self.remote_response):
            self.download()
        relative = "comments/split=train/part-0000.parquet"
        (self.output / relative).write_bytes(b"broken")
        with patch.object(download_text, "urlopen", side_effect=self.remote_response) as remote:
            self.download()
            remote.assert_called_once_with(self.urls[relative], timeout=60)

    def test_failed_checksum_keeps_previous_file_and_removes_temporary(self):
        relative = "comments/split=train/part-0000.parquet"
        entry = next(entry for entry in _common.read_manifest(self.manifest)[1] if entry["path"] == relative)
        target = self.output / relative
        target.parent.mkdir(parents=True)
        target.write_bytes(b"previous file")
        with patch.object(download_text, "urlopen", return_value=io.BytesIO(b"corrupt transfer")):
            with self.assertRaisesRegex(ValueError, "checksum/size"):
                download_text.download_file(self.output, relative, entry, self.urls)
        self.assertEqual(target.read_bytes(), b"previous file")
        self.assertEqual(list(self.output.rglob("*.part")), [])

    def test_manifest_pin_rejects_different_release(self):
        self.assertRaisesRegex(ValueError, "pinned", _common.read_manifest, b"{}")

    def test_rejects_traversal_and_symlink_escape(self):
        for relative in ("../outside", "/absolute", "posts/../../outside", "posts\\outside", "posts//file"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                _common.local_path(self.output, relative)
        self.output.mkdir()
        (self.output / "posts").symlink_to(self.source, target_is_directory=True)
        self.assertRaises(ValueError, _common.local_path, self.output, "posts/file.parquet")

    def test_rejects_wrong_host_or_object_before_network_request(self):
        for url in ("https://example.com/file?token=test",
                    self.urls["manifest.json"].replace("https://", "http://"),
                    self.urls["manifest.json"].replace("manifest.json", "other.json")):
            with self.subTest(url=url), patch.object(download_text, "urlopen") as remote:
                with self.assertRaisesRegex(ValueError, "signed download URL"):
                    download_text.download_file(self.output, "manifest.json", {"sha256": self.digest},
                                                {"manifest.json": url})
                remote.assert_not_called()

    def test_expired_link_error_does_not_expose_signed_token(self):
        error = HTTPError(self.urls["manifest.json"], 403, "Denied", {}, io.BytesIO(b"denied"))
        with patch.object(download_text, "urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "HTTP 403") as caught:
                self.download()
        self.assertNotIn("token=", str(caught.exception))
        self.assertEqual(list(self.output.rglob("*.part")), [])

    def test_signing_covers_every_required_object_without_distributing_secret(self):
        response = [{"path": f"{_common.PREFIX}/{relative}", "error": None,
                     "signedURL": f"/object/sign/{_common.BUCKET}/{_common.PREFIX}/{relative}?token=test"}
                    for relative in self.paths]
        with patch.object(create_download_links, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as api:
            bundle = create_download_links.create_links(self.source / "manifest.json", "sb_secret_fixture")
        self.assertEqual(set(bundle["urls"]), set(self.paths))
        self.assertNotIn("sb_secret_fixture", json.dumps(bundle))
        request = api.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Apikey"), "sb_secret_fixture")
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(set(json.loads(request.data)["paths"]), {f"{_common.PREFIX}/{p}" for p in self.paths})

    def test_signing_rejects_partial_results(self):
        with patch.object(create_download_links, "urlopen", return_value=io.BytesIO(b"[]")):
            self.assertRaisesRegex(ValueError, "omitted", create_download_links.create_links,
                                   self.source / "manifest.json", "sb_secret_fixture")

    def test_database_starter_binds_parameters_and_returns_textstore_keys(self):
        from unittest.mock import Mock
        db = Mock()
        db.execute.return_value = [("t1_a1", "t3_a", None, "author", 9)]
        rows = query_comments(db, "t3_a", limit=3)
        sql, params = db.execute.call_args.args
        self.assertIn("AND content_available", sql)
        self.assertEqual(params, ("t3_a", 3))
        self.assertEqual(rows, [{"comment_id": "t1_a1", "post_id": "t3_a",
                                 "parent_comment_id": None, "author": "author", "score": 9}])


if __name__ == "__main__":
    unittest.main()
