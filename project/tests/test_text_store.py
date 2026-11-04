"""Offline TextStore behavior and optional complete local-release validation."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "database"))
from TextStore import TextStore


POST_COLUMNS = ["post_id", "title", "body"]
COMMENT_COLUMNS = ["comment_id", "post_id", "body"]


def file_info(root, relative):
    path = root / relative
    data = path.read_bytes()
    return {"path": relative, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def write_fixture(root):
    posts = {
        "posts/split=train/part-0000.parquet": [
            {"post_id": "t3_a", "title": "Café 🌱", "body": "First line\nΔ second line"},
            {"post_id": "t3_b", "title": "Another post", "body": ""},
            {"post_id": "t3_empty", "title": "No comments", "body": None},
        ],
        "posts/split=heldout/part-0000.parquet": [
            {"post_id": "t3_h", "title": "Heldout", "body": "Heldout body"},
        ],
    }
    comments = {
        "comments/split=train/part-0000.parquet": [
            {"comment_id": "t1_a1", "post_id": "t3_a", "body": 'Unicode Δ 🌱\nA "quote"'},
            {"comment_id": "t1_a2", "post_id": "t3_a", "body": None},
            {"comment_id": "t1_b1", "post_id": "t3_b", "body": ""},
            {"comment_id": "t1_b2", "post_id": "t3_b", "body": "Same file, different post"},
        ],
        "comments/split=heldout/part-0000.parquet": [
            {"comment_id": "t1_h1", "post_id": "t3_h", "body": "Heldout response"},
        ],
    }
    files = []
    for relative, rows in {**posts, **comments}.items():
        columns = POST_COLUMNS if relative.startswith("posts/") else COMMENT_COLUMNS
        schema = pa.schema([(column, pa.string()) for column in columns])
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
        files.append({**file_info(root, relative), "rows": len(rows)})

    index = {}
    for post_id, split, count in (
        ("t3_a", "train", 2), ("t3_b", "train", 2),
        ("t3_empty", "train", 0), ("t3_h", "heldout", 1),
    ):
        index[post_id] = {
            "split": split,
            "post_file": f"posts/split={split}/part-0000.parquet",
            "comment_file": f"comments/split={split}/part-0000.parquet" if count else None,
            "comment_count": count,
        }
    (root / "discussion_index.json").write_text(json.dumps(index), encoding="utf-8")
    manifest = {
        "version": "v1", "files": files,
        "discussion_index": file_info(root, "discussion_index.json"),
        "sources": [
            {"split": "train", "posts": 3, "comments": 4},
            {"split": "heldout", "posts": 1, "comments": 1},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TextStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_fixture(self.root)
        self.store = TextStore(self.root)

    def test_complete_fixture_validation(self):
        self.assertEqual(self.store.validate(), {
            "version": "v1", "parquet_files": 4,
            "post_rows": 4, "comment_rows": 5, "index_posts": 4,
            "index_checksum_verified": True,
        })

    def test_version_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Expected dataset v2"):
            TextStore(self.root, expected_version="v2")

    def test_missing_manifest(self):
        (self.root / "manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            TextStore(self.root)

    def test_malformed_index(self):
        (self.root / "discussion_index.json").write_text("not JSON")
        with self.assertRaises(json.JSONDecodeError):
            TextStore(self.root)

    def test_post_preserves_unicode_and_newlines(self):
        self.assertEqual(self.store.get_post("t3_a"), {
            "post_id": "t3_a", "title": "Café 🌱", "body": "First line\nΔ second line",
        })

    def test_unknown_post(self):
        for method in (self.store.get_post, self.store.get_comments):
            with self.subTest(method=method.__name__):
                with self.assertRaisesRegex(KeyError, "Unknown post ID"):
                    method("t3_missing")

    def test_comments_exclude_other_posts_in_same_file(self):
        result = self.store.get_comments("t3_a")
        self.assertEqual({r["comment_id"] for r in result}, {"t1_a1", "t1_a2"})
        self.assertTrue(all(r["post_id"] == "t3_a" for r in result))
        by_id = {r["comment_id"]: r["body"] for r in result}
        self.assertEqual(by_id["t1_a1"], 'Unicode Δ 🌱\nA "quote"')
        self.assertIsNone(by_id["t1_a2"])

    def test_empty_discussion_does_not_read_parquet(self):
        with patch("TextStore.pq.read_table") as read:
            self.assertEqual(self.store.get_comments("t3_empty"), [])
            read.assert_not_called()

    def test_missing_comment_file_for_nonempty_post(self):
        self.store.index["t3_a"]["comment_file"] = None
        with self.assertRaisesRegex(ValueError, "Missing comment file"):
            self.store.get_comments("t3_a")

    def test_comment_count_mismatch(self):
        self.store.index["t3_a"]["comment_count"] += 1
        with self.assertRaisesRegex(ValueError, "Comment count mismatch"):
            self.store.get_comments("t3_a")

    def test_duplicate_post_rows(self):
        relative = self.store.index["t3_a"]["post_file"]
        table = pq.read_table(self.root / relative, columns=POST_COLUMNS)
        duplicate = table.filter(pa.compute.equal(table["post_id"], "t3_a"))
        pq.write_table(pa.concat_tables([table, duplicate]), self.root / relative)
        with self.assertRaisesRegex(ValueError, "Expected one post"):
            self.store.get_post("t3_a")

    def test_batch_preserves_order_duplicates_metadata_and_inputs(self):
        rows = [
            {"comment_id": "t1_h1", "post_id": "t3_h", "score": 7},
            {"comment_id": "t1_b2", "post_id": "t3_b", "score": 2},
            {"comment_id": "t1_a1", "post_id": "t3_a", "score": 4},
            {"comment_id": "t1_h1", "post_id": "t3_h", "score": 8},
        ]
        original = copy.deepcopy(rows)
        read_table = pq.read_table
        with patch("TextStore.pq.read_table", wraps=read_table) as read:
            result = self.store.attach_comment_text(rows)
            self.assertEqual(read.call_count, 2, "Each of two files should be read once")
        self.assertEqual(rows, original)
        self.assertEqual([r["comment_id"] for r in result], [r["comment_id"] for r in rows])
        self.assertEqual([r["score"] for r in result], [7, 2, 4, 8])
        self.assertEqual(result[0]["body"], "Heldout response")
        self.assertEqual(result[2]["body"], 'Unicode Δ 🌱\nA "quote"')
        self.assertTrue(all(a is not b for a, b in zip(rows, result)))

    def test_batch_accepts_generator_and_null_or_empty_text(self):
        rows = ({"comment_id": c, "post_id": p} for c, p in (
            ("t1_a2", "t3_a"), ("t1_b1", "t3_b"),
        ))
        result = self.store.attach_comment_text(rows)
        self.assertEqual([r["body"] for r in result], [None, ""])

    def test_empty_batch_does_not_read_parquet(self):
        with patch("TextStore.pq.read_table") as read:
            self.assertEqual(self.store.attach_comment_text([]), [])
            read.assert_not_called()

    def test_unknown_batch_post(self):
        with self.assertRaisesRegex(KeyError, "Unknown post ID"):
            self.store.attach_comment_text([{"post_id": "t3_missing", "comment_id": "t1_a1"}])

    def test_unknown_comment_and_wrong_post_pair(self):
        for post_id, comment_id in (("t3_a", "t1_missing"), ("t3_b", "t1_a1")):
            with self.subTest(post_id=post_id, comment_id=comment_id):
                with self.assertRaisesRegex(KeyError, "Comment text not found"):
                    self.store.attach_comment_text([{"post_id": post_id, "comment_id": comment_id}])

    def test_batch_request_for_empty_discussion(self):
        with self.assertRaisesRegex(KeyError, "No comment text available"):
            self.store.attach_comment_text([{"post_id": "t3_empty", "comment_id": "t1_missing"}])

    def test_missing_parquet_file_has_context(self):
        path = self.root / self.store.index["t3_a"]["comment_file"]
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing Parquet file.*part-0000"):
            self.store.get_comments("t3_a")
        with self.assertRaisesRegex(FileNotFoundError, "Missing dataset file"):
            self.store.validate()

    def test_corrupt_parquet_file_has_context(self):
        path = self.root / self.store.index["t3_a"]["post_file"]
        path.write_bytes(b"not parquet")
        with self.assertRaisesRegex(ValueError, "Cannot read Parquet file.*part-0000"):
            self.store.get_post("t3_a")

    def test_validation_detects_size_mismatch(self):
        path = self.root / self.store.index["t3_a"]["post_file"]
        with path.open("ab") as stream:
            stream.write(b"extra bytes")
        with self.assertRaisesRegex(ValueError, "File size mismatch"):
            self.store.validate()

    def test_validation_detects_same_size_corruption(self):
        path = self.root / self.store.index["t3_a"]["post_file"]
        data = bytearray(path.read_bytes())
        data[10] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            self.store.validate()

    def test_validation_detects_modified_index(self):
        path = self.root / "discussion_index.json"
        path.write_text(path.read_text().replace('"t3_a"', '"t3_z"', 1))
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            self.store.validate()

    def test_validation_detects_modified_in_memory_index(self):
        self.store.index["t3_a"]["comment_count"] = 99
        with self.assertRaisesRegex(ValueError, "In-memory discussion index"):
            self.store.validate()

    def test_validation_detects_manifest_row_count_mismatch(self):
        self.store.manifest["files"][0]["rows"] += 1
        with self.assertRaisesRegex(ValueError, "Parquet row count mismatch"):
            self.store.validate()

    def test_validation_detects_source_count_mismatch(self):
        self.store.manifest["sources"][0]["posts"] += 1
        with self.assertRaisesRegex(ValueError, "Source/Parquet count mismatch"):
            self.store.validate()


class LocalReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = PROJECT_DIR / "data/processed/cmv-text/v1"
        if not (root / "manifest.json").exists():
            raise unittest.SkipTest("Local v1 dataset is not downloaded")
        cls.store = TextStore(root)

    def test_all_local_files_and_index_match_manifest(self):
        result = self.store.validate()
        self.assertEqual(result["parquet_files"], 33)
        self.assertEqual(result["post_rows"], 20626)
        self.assertEqual(result["comment_rows"], 1260266)
        self.assertEqual(result["index_posts"], 20626)

    def test_sample_post_and_all_comment_texts(self):
        post_id = "t3_37i8f6"
        self.assertEqual(self.store.get_post(post_id)["post_id"], post_id)
        comments = self.store.get_comments(post_id)
        self.assertEqual(len(comments), 1367)
        self.assertEqual(len({c["comment_id"] for c in comments}), 1367)
        self.assertTrue(all(c["post_id"] == post_id for c in comments))
        refs = [
            {"post_id": post_id, "comment_id": comments[-1]["comment_id"], "score": 1},
            {"post_id": post_id, "comment_id": comments[0]["comment_id"], "score": 2},
        ]
        result = self.store.attach_comment_text(refs)
        self.assertEqual([r["body"] for r in result], [comments[-1]["body"], comments[0]["body"]])


if __name__ == "__main__":
    unittest.main()
