"""Tiny-archive export, indexed lookup, and release-integrity checks."""

import bz2
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
from cmv_data.MetadataRunner import MetadataRunner
from cmv_data.parquet_loaders.metadata_parquet import MetadataParquetWriter, validate_metadata_dataset
from cmv_data.parquet_loaders.metadata_staging import TABLE_SCHEMAS
from test_metadata_runner import SOURCE_MAP, sample_records


class MetadataParquetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "cmv.tar.bz2"
        self.stage = self.root / "staging.sqlite"
        self.output = self.root / "v1"
        self.partial = self.root / "v1.partial"
        self.write_archive()

    def write_archive(self, *, empty=False, records=None, extra_train_post=False):
        with tarfile.open(self.archive, "w:bz2") as archive:
            for member, record in (sample_records() if records is None else records).items():
                raw = b"" if empty else b"\n" + json.dumps(record).encode() + b"\n"
                if extra_train_post and member == "all/train_period_data.jsonlist.bz2":
                    raw += json.dumps({"name": "t3_extra", "title": "Another discussion", "comments": []}).encode() + b"\n"
                data = bz2.compress(raw)
                info = tarfile.TarInfo(member)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))

    def run_export(self, **options):
        return MetadataRunner(self.archive, SOURCE_MAP, self.stage, output_path=self.output,
                              batch_size=2, rows_per_file=options.get("rows_per_file", 1)).run()

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def save_manifest(self, manifest):
        (self.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def save_index(self, index, manifest):
        path = self.output / "discussion_index.json"
        path.write_text(json.dumps(index), encoding="utf-8")
        manifest["discussion_index"].update(bytes=path.stat().st_size,
                                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.save_manifest(manifest)

    def update_file_info(self, info):
        path = self.output / info["path"]
        info.update(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_export_manifest_shards_schemas_compression_and_checksums(self):
        report = self.run_export()
        self.assertEqual(report["status"], "exported_metadata")
        self.assertEqual(report["export"]["discussions"], 2)
        self.assertEqual(report["export"]["validation"], report["validation"])
        self.assertFalse(self.partial.exists())
        manifest = self.manifest()
        self.assertEqual(manifest["source_map"], SOURCE_MAP)
        self.assertEqual(manifest["raw_archive"]["sha256"], hashlib.sha256(self.archive.read_bytes()).hexdigest())
        self.assertEqual(manifest["raw_archive"]["bytes"], self.archive.stat().st_size)
        self.assertEqual(manifest["input_by_source"], report["input_by_source"])
        self.assertEqual(set(manifest["schemas"]), set(TABLE_SCHEMAS))
        for info in manifest["files"]:
            with self.subTest(path=info["path"]):
                self.assertLessEqual(info["rows"], 6 if info["table"] == "sources" else 1)
                with pq.ParquetFile(self.output / info["path"]) as parquet:
                    self.assertEqual(parquet.schema_arrow, TABLE_SCHEMAS[info["table"]])
                    if info["rows"]:
                        self.assertEqual(parquet.metadata.row_group(0).column(0).compression, "ZSTD")
        self.assertEqual(validate_metadata_dataset(self.output, batch_size=1)["parquet_files"],
                         report["export"]["parquet_files"])

    def test_discussion_spanning_shards_indexes_all_files_and_retrieves_exact_rows(self):
        self.run_export()
        index = json.loads((self.output / "discussion_index.json").read_text())
        location = index["t3_train"]
        self.assertEqual(location["split"], "train")
        self.assertEqual(location["rows"]["comments"], 4)
        self.assertEqual(len(location["files"]["comments"]), 4)
        comments = []
        for relative in location["files"]["comments"]:
            comments.extend(pq.read_table(self.output / relative, columns=TABLE_SCHEMAS["comments"].names,
                                          filters=[("post_id", "=", "t3_train")]).to_pylist())
        self.assertEqual({r["comment_id"] for r in comments},
                         {"t1_train_root", "t1_train_parent", "t1_train_child", "t1_train_negative"})
        root = next(r for r in comments if r["comment_id"] == "t1_train_root")
        self.assertEqual(root["score"], 0)
        self.assertEqual(root["source_id"], 2)
        self.assertIsNone(root["parent_comment_id"])
        self.assertEqual(root["author"], "Café 🌱")
        self.assertTrue(all(r["content_available"] is True for r in comments))

    def test_small_batches_can_share_larger_files_without_losing_index_counts(self):
        self.run_export(rows_per_file=3)
        index = json.loads((self.output / "discussion_index.json").read_text())
        self.assertEqual(len(index["t3_train"]["files"]["comments"]), 2)
        self.assertEqual(index["t3_train"]["rows"]["comments"], 4)
        validate_metadata_dataset(self.output)

    def test_empty_partitions_have_explicit_schema_files(self):
        self.write_archive(empty=True)
        report = self.run_export()
        self.assertEqual(report["export"]["discussions"], 0)
        self.assertEqual(report["export"]["parquet_files"], 13)
        self.assertEqual(json.loads((self.output / "discussion_index.json").read_text()), {})
        self.assertEqual(report["validation"]["tables"]["posts"], 0)
        validate_metadata_dataset(self.output)

    def test_multiple_discussions_can_share_one_indexed_shard(self):
        self.write_archive(extra_train_post=True)
        self.run_export(rows_per_file=10)
        index = json.loads((self.output / "discussion_index.json").read_text())
        self.assertEqual(index["t3_train"]["files"]["posts"], index["t3_extra"]["files"]["posts"])
        relative = index["t3_extra"]["files"]["posts"][0]
        selected = pq.read_table(self.output / relative, columns=TABLE_SCHEMAS["posts"].names,
                                 filters=[("post_id", "=", "t3_extra")]).to_pylist()
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["title"], "Another discussion")
        self.assertEqual(index["t3_extra"]["files"]["comments"], [])
        self.assertEqual(validate_metadata_dataset(self.output)["discussions"], 3)

    def test_missing_parent_placeholder_is_preserved_and_counted_separately(self):
        records = sample_records()
        records["all/train_period_data.jsonlist.bz2"]["comments"].append({
            "name": "t1_extra", "parent_id": "t1_missing",
        })
        self.write_archive(records=records)
        report = self.run_export()
        self.assertEqual(report["validation"]["placeholders"]["comments"], 1)
        index = json.loads((self.output / "discussion_index.json").read_text())
        found = []
        for relative in index["t3_train"]["files"]["comments"]:
            found.extend(pq.read_table(self.output / relative, columns=TABLE_SCHEMAS["comments"].names,
                                       filters=[("comment_id", "=", "t1_missing")]).to_pylist())
        self.assertEqual(len(found), 1)
        self.assertIs(found[0]["content_available"], False)
        self.assertIsNone(found[0]["original_parent_id"])
        self.assertIsNone(found[0]["source_id"])
        validate_metadata_dataset(self.output)

    def test_downloaded_copy_validates_without_staging_or_raw_archive(self):
        self.run_export()
        downloaded = self.root / "downloaded"
        shutil.copytree(self.output, downloaded)
        self.stage.unlink()
        self.archive.unlink()
        with mock.patch("cmv_data.MetadataRunner.ArchiveReader", side_effect=AssertionError("No archive reads")):
            self.assertEqual(validate_metadata_dataset(downloaded)["discussions"], 2)

    def test_existing_release_or_partial_refuses_before_archive_processing(self):
        for path in (self.output, self.partial):
            with self.subTest(path=path):
                path.mkdir()
                with mock.patch("cmv_data.MetadataRunner.ArchiveReader", side_effect=AssertionError("Unexpected scan")):
                    with self.assertRaises(FileExistsError):
                        self.run_export()
                self.assertFalse(self.stage.exists())
                path.rmdir()

    def test_write_failure_retains_partial_and_does_not_publish(self):
        original = MetadataParquetWriter._write_file

        def fail(writer, name, *args):
            if name == "comments":
                raise OSError("simulated disk failure")
            return original(writer, name, *args)

        with mock.patch.object(MetadataParquetWriter, "_write_file", fail):
            with self.assertRaisesRegex(OSError, "simulated disk failure"):
                self.run_export()
        self.assertTrue(self.stage.exists())
        self.assertTrue((self.partial / "sources.parquet").exists())
        self.assertFalse(self.output.exists())

    def test_verification_failure_keeps_release_unpublished(self):
        with mock.patch("cmv_data.parquet_loaders.metadata_parquet.validate_metadata_dataset",
                        side_effect=ValueError("verification failed")):
            with self.assertRaisesRegex(ValueError, "verification failed"):
                self.run_export()
        self.assertTrue((self.partial / "manifest.json").exists())
        self.assertFalse(self.output.exists())

    def test_corrupted_parquet_bytes_are_rejected(self):
        self.run_export()
        info = self.manifest()["files"][0]
        with (self.output / info["path"]).open("ab") as stream:
            stream.write(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
            validate_metadata_dataset(self.output)

    def test_logical_rows_must_match_staging_even_if_byte_checksum_is_updated(self):
        self.run_export()
        manifest = self.manifest()
        info = next(f for f in manifest["files"] if f["table"] == "comments")
        path = self.output / info["path"]
        with pq.ParquetFile(path) as parquet:
            rows = parquet.read().to_pylist()
        rows[0]["score"] = 999
        pq.write_table(pa.Table.from_pylist(rows, schema=TABLE_SCHEMAS["comments"]), path)
        self.update_file_info(info)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "row round-trip mismatch"):
            validate_metadata_dataset(self.output)

    def test_wrong_file_schema_or_count_is_rejected(self):
        self.run_export()
        manifest = self.manifest()
        info = next(f for f in manifest["files"] if f["table"] == "posts")
        info["rows"] += 1
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "schema/count mismatch"):
            validate_metadata_dataset(self.output)
        info["rows"] -= 1
        pq.write_table(pa.table({"wrong_column": ["value"]}), self.output / info["path"])
        self.update_file_info(info)
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "schema/count mismatch"):
            validate_metadata_dataset(self.output)

    def test_index_omission_is_detected_after_valid_index_checksum(self):
        self.run_export()
        manifest = self.manifest()
        index = json.loads((self.output / "discussion_index.json").read_text())
        index["t3_train"]["files"]["comments"].pop()
        self.save_index(index, manifest)
        with self.assertRaisesRegex(ValueError, "Discussion index does not match"):
            validate_metadata_dataset(self.output)

    def test_table_and_placeholder_totals_are_checked(self):
        self.run_export()
        manifest = self.manifest()
        manifest["validation"]["tables"]["comments"] += 1
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "table totals"):
            validate_metadata_dataset(self.output)
        manifest["validation"]["tables"]["comments"] -= 1
        manifest["validation"]["placeholders"]["comments"] += 1
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "placeholder totals"):
            validate_metadata_dataset(self.output)

    def test_unlisted_or_duplicate_parquet_files_are_rejected(self):
        self.run_export()
        manifest = self.manifest()
        extra = self.output / "extra.parquet"
        shutil.copyfile(self.output / "sources.parquet", extra)
        with self.assertRaisesRegex(ValueError, "manifest inventory"):
            validate_metadata_dataset(self.output)
        extra.unlink()
        manifest["files"].append(dict(manifest["files"][0]))
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "Duplicate manifest file"):
            validate_metadata_dataset(self.output)

    def test_source_snapshot_must_match_exported_source_ids(self):
        self.run_export()
        manifest = self.manifest()
        manifest["source_map"]["pair_task/train_pair_data.jsonlist.bz2"]["source_id"] = 999
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "source IDs"):
            validate_metadata_dataset(self.output)

    def test_manifest_contract_and_paths_are_validated(self):
        self.run_export()
        original = self.manifest()
        variants = [
            ({**original, "format_version": 999}, "Unsupported"),
            ({**original, "schemas": {}}, "schemas/primary keys"),
            ({**original, "files": [{**original["files"][0], "path": "../outside.parquet"}]}, "manifest path"),
        ]
        for manifest, message in variants:
            with self.subTest(message=message):
                self.save_manifest(manifest)
                with self.assertRaisesRegex(ValueError, message):
                    validate_metadata_dataset(self.output)

    def test_invalid_writer_and_validator_batch_sizes_are_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    MetadataParquetWriter(self.output, rows_per_file=value)
                with self.assertRaises(ValueError):
                    MetadataParquetWriter(self.output, batch_size=value)
                with self.assertRaises(ValueError):
                    validate_metadata_dataset(self.output, batch_size=value)

    def test_cli_can_publish_export_from_tiny_archive(self):
        snapshot = self.root / "source_map.json"
        snapshot.write_text(json.dumps(SOURCE_MAP))
        result = subprocess.run([
            sys.executable, str(PROJECT_DIR / "database" / "MetadataRunner.py"),
            "--archive", str(self.archive), "--source-map", str(snapshot),
            "--staging", str(self.stage), "--output", str(self.output),
            "--rows-per-file", "2",
        ], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(result.stdout)["status"], "exported_metadata")
        self.assertTrue((self.output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
