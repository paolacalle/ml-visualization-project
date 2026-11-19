"""All-table staging, source fidelity, and cross-table validation checks."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
from cmv_data.parquet_loaders.comments import map_comment
from cmv_data.parquet_loaders.metadata_staging import MetadataStaging, TABLE_SCHEMAS
from cmv_data.parquet_loaders.op_labels import map_op_label
from cmv_data.parquet_loaders.pairs import map_pair
from cmv_data.parquet_loaders.posts import map_post, map_task_post
from cmv_data.parquet_loaders.source_map import SOURCE_SCHEMA, map_sources


SOURCE_MAP = {
    "all/heldout_period_data.jsonlist.bz2": {"source_id": 1, "task": "all", "original_split": "heldout"},
    "all/train_period_data.jsonlist.bz2": {"source_id": 2, "task": "all", "original_split": "train"},
    "op_task/heldout_op_data.jsonlist.bz2": {"source_id": 3, "task": "op_task", "original_split": "heldout"},
    "op_task/train_op_data.jsonlist.bz2": {"source_id": 4, "task": "op_task", "original_split": "train"},
    "pair_task/heldout_pair_data.jsonlist.bz2": {"source_id": 6, "task": "pair_task", "original_split": "heldout"},
    "pair_task/train_pair_data.jsonlist.bz2": {"source_id": 5, "task": "pair_task", "original_split": "train"},
}
ALL = SOURCE_MAP["all/train_period_data.jsonlist.bz2"]
OP = SOURCE_MAP["op_task/train_op_data.jsonlist.bz2"]
MEMBER = "pair_task/train_pair_data.jsonlist.bz2"
PAIR = SOURCE_MAP[MEMBER]
POST_ID = "t3_example"


def post_record():
    return {"name": POST_ID, "title": "Full title", "author": None, "score": 0,
            "over_18": False, "edited": False}


def comment_record(name="t1_root", parent=POST_ID):
    return {"name": name, "parent_id": parent, "link_id": POST_ID,
            "author": "writer", "score": 7}


def pair_record():
    return {
        "op_name": POST_ID, "op_title": "Pair title", "op_author": "pair_author",
        "positive": {"ancestor": "t1_root", "comments": [
            comment_record(), comment_record("t1_child", "t1_root"),
        ]},
        "negative": {"comments": [comment_record("t1_negative")]},
    }


class SourceMappingTests(unittest.TestCase):
    def test_keeps_snapshot_ids_and_types_including_nonsequential_bigints(self):
        snapshot = copy.deepcopy(SOURCE_MAP)
        snapshot[MEMBER]["source_id"] = 2**40
        original = copy.deepcopy(snapshot)
        rows = map_sources(snapshot)
        self.assertEqual(snapshot, original)
        self.assertEqual({r["archive_member"]: r["source_id"] for r in rows},
                         {name: source["source_id"] for name, source in snapshot.items()})
        self.assertEqual(pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA).schema, SOURCE_SCHEMA)

    def test_rejects_missing_extra_duplicate_and_mismatched_sources(self):
        variants = []
        missing = copy.deepcopy(SOURCE_MAP)
        del missing[MEMBER]
        variants.append(missing)
        variants.append({**SOURCE_MAP, "unknown.bz2": PAIR})
        for field, value in (("source_id", 2), ("task", "all"), ("original_split", "heldout"),
                             ("source_id", True), ("source_id", 0), ("source_id", 2**63)):
            snapshot = copy.deepcopy(SOURCE_MAP)
            snapshot[MEMBER][field] = value
            variants.append(snapshot)
        for snapshot in variants:
            with self.subTest(snapshot=snapshot):
                with self.assertRaises(ValueError):
                    map_sources(snapshot)

    def test_task_post_mapping_preserves_available_metadata_without_text(self):
        op = map_task_post({"name": POST_ID, "title": "OP title", "edited": 1441132000,
                            "selftext": "Body", "delta_label": False}, OP)
        pair = map_task_post(pair_record(), PAIR)
        self.assertEqual(op["source_id"], 4)
        self.assertEqual(op["edited_utc"], 1441132000)
        self.assertEqual(pair["source_id"], 5)
        self.assertEqual(pair["author"], "pair_author")
        self.assertIsNone(pair["score"])
        self.assertTrue({"body", "selftext", "op_text", "delta_label"}.isdisjoint(op))
        with self.assertRaisesRegex(ValueError, "expects"):
            map_task_post(post_record(), ALL)


class MetadataStagingTests(unittest.TestCase):
    def test_membership_comment_reference_lookup_uses_an_index(self):
        plan = self.stage.connection.execute(
            "EXPLAIN QUERY PLAN SELECT rowid FROM thread_comments WHERE comment_id = ? AND post_id = ?",
            ("t1_root", POST_ID),
        ).fetchall()
        description = " ".join(row[3] for row in plan)
        self.assertIn("SEARCH", description)
        self.assertIn("thread_comments_comment_idx", description)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stage = MetadataStaging(self.root / "metadata.sqlite", SOURCE_MAP)
        self.addCleanup(self.stage.close)

    def rows(self, table, stage=None, **options):
        return [r for batch in (stage or self.stage).iter_table_batches(table, **options) for r in batch]

    def add_complete(self, stage=None, reverse=False):
        stage = stage or self.stage
        operations = [
            lambda: stage.add_posts([map_post(post_record(), ALL)]),
            lambda: stage.add_comments([map_comment(comment_record(), POST_ID, ALL)], task="all", split="train"),
            lambda: stage.add_op_labels([map_op_label({"name": POST_ID, "delta_label": False}, OP)]),
            lambda: stage.add_pair(map_pair(pair_record(), PAIR, MEMBER, 2)),
            lambda: stage.add_posts([map_task_post(pair_record(), PAIR)], task="pair_task"),
        ]
        for operation in reversed(operations) if reverse else operations:
            operation()

    def test_empty_stage_exports_six_sources_and_valid_empty_tables(self):
        report = self.stage.validate()
        self.assertEqual(report["tables"], {name: 6 if name == "sources" else 0 for name in TABLE_SCHEMAS})
        self.assertEqual(report["placeholders"], {"posts": 0, "comments": 0})
        self.assertEqual(self.rows("sources"), sorted(map_sources(SOURCE_MAP), key=lambda r: r["source_id"]))

    def test_complete_discussion_counts_and_idempotent_reload(self):
        self.add_complete()
        report = self.stage.validate()
        self.assertEqual(report, {"tables": {"sources": 6, "posts": 1, "comments": 3,
                                            "op_labels": 1, "pairs": 1, "threads": 2, "thread_comments": 3},
                                  "placeholders": {"posts": 0, "comments": 0}})
        self.add_complete()
        self.assertEqual(self.stage.validate(), report)
        self.assertIs(self.rows("op_labels")[0]["delta_label"], False)

    def test_source_order_does_not_change_any_exported_table(self):
        self.add_complete()
        with MetadataStaging(self.root / "reverse.sqlite", SOURCE_MAP) as reverse:
            self.add_complete(reverse, reverse=True)
            reverse.validate()
            for name in TABLE_SCHEMAS:
                with self.subTest(table=name):
                    self.assertEqual(self.rows(name), self.rows(name, reverse))

    def test_post_precedence_null_fill_and_edit_fields_are_consistent(self):
        full = map_post(post_record(), ALL)
        op = map_task_post({"name": POST_ID, "title": "OP title", "edited": 1441132000}, OP)
        pair = map_task_post(pair_record(), PAIR)
        for index, candidates in enumerate(([(full, "all"), (op, "op_task"), (pair, "pair_task")],
                                            [(pair, "pair_task"), (op, "op_task"), (full, "all")])):
            with MetadataStaging(self.root / f"precedence-{index}.sqlite", SOURCE_MAP) as stage:
                for row, task in candidates:
                    stage.add_posts([row], task=task)
                post = self.rows("posts", stage)[0]
                self.assertEqual(post["title"], "Full title")
                self.assertEqual(post["author"], "pair_author")
                self.assertEqual(post["source_id"], 2)
                self.assertEqual(post["score"], 0)
                self.assertIs(post["over_18"], False)
                self.assertIs(post["is_edited"], False)
                self.assertIsNone(post["edited_utc"])

    def test_label_reference_creates_post_placeholder_then_actual_post_enriches_it(self):
        self.stage.add_op_labels([map_op_label({"name": POST_ID, "delta_label": True}, OP)])
        placeholder = self.rows("posts")[0]
        self.assertIs(placeholder["content_available"], False)
        self.assertEqual(placeholder["original_split"], "train")
        self.assertIsNone(placeholder["title"])
        self.assertIsNone(placeholder["source_id"])
        self.assertEqual(self.stage.validate()["placeholders"]["posts"], 1)
        self.stage.add_posts([map_post(post_record(), ALL)])
        self.assertIs(self.rows("posts")[0]["content_available"], True)

    def test_conflicting_post_and_label_repeats_are_rejected(self):
        post = map_post(post_record(), ALL)
        self.stage.add_posts([post])
        with self.assertRaisesRegex(ValueError, "Conflicting all title"):
            self.stage.add_posts([{**post, "title": "Different"}])
        label = map_op_label({"name": POST_ID, "delta_label": False}, OP)
        self.stage.add_op_labels([label])
        with self.assertRaisesRegex(ValueError, "Conflicting op_labels"):
            self.stage.add_op_labels([{**label, "delta_label": True}])
        self.assertIs(self.rows("op_labels")[0]["delta_label"], False)

    def test_unknown_wrong_task_and_wrong_split_sources_are_rejected(self):
        post = map_post(post_record(), ALL)
        for row in ({**post, "source_id": 999}, {**post, "source_id": 4},
                    {**post, "original_split": "heldout"}):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    self.stage.add_posts([row])
        comment = map_comment(comment_record(), POST_ID, ALL)
        with self.assertRaises(ValueError):
            self.stage.add_comments([{**comment, "source_id": 1}], task="all", split="train")
        with self.assertRaises(ValueError):
            self.stage.add_op_labels([{"post_id": POST_ID, "delta_label": False, "source_id": 2}])
        self.assertEqual(self.rows("posts"), [])

    def test_post_and_comment_types_and_ranges_are_checked_before_sqlite(self):
        post = map_post(post_record(), ALL)
        for row in ({**post, "content_available": 1}, {**post, "source_id": True},
                    {**post, "score": 2**31}, {**post, "edited_utc": 1441132000}):
            with self.subTest(row=row):
                with self.assertRaises((ValueError, pa.ArrowException)):
                    self.stage.add_posts([row])
        self.assertEqual(self.rows("posts"), [])

    def test_pair_failure_rolls_back_comments_placeholders_and_relational_rows(self):
        mapped = map_pair(pair_record(), PAIR, MEMBER, 2)
        mapped["thread_comments"][1]["position"] = 0
        with self.assertRaisesRegex(ValueError, "Invalid thread_comments"):
            self.stage.add_pair(mapped)
        report = self.stage.validate()
        self.assertEqual(report["tables"]["sources"], 6)
        self.assertTrue(all(count == 0 for name, count in report["tables"].items() if name != "sources"))

    def test_outer_transaction_can_roll_back_nested_posts_labels_and_comments(self):
        with self.assertRaisesRegex(ValueError, "abort"):
            with self.stage.transaction():
                self.add_complete()
                raise ValueError("abort")
        self.assertEqual(self.rows("posts"), [])
        self.assertEqual(self.rows("comments"), [])
        self.assertEqual(self.rows("op_labels"), [])

    def test_pair_rows_must_belong_to_their_declared_pair_and_post(self):
        for section, field, value in (("threads", "post_id", "t3_other"),
                                      ("thread_comments", "thread_id", "other_pair:positive"),
                                      ("comments", "source_id", 6)):
            with self.subTest(section=section):
                mapped = map_pair(pair_record(), PAIR, MEMBER, 2)
                mapped[section][0][field] = value
                with self.assertRaises(ValueError):
                    self.stage.add_pair(mapped)
        self.assertEqual(self.rows("pairs"), [])

    def test_pair_and_thread_ids_must_match_source_line_and_outcome(self):
        mapped = map_pair(pair_record(), PAIR, MEMBER, 2)
        with self.assertRaisesRegex(ValueError, "Pair ID"):
            self.stage.add_pairs([{**mapped["pair"], "pair_id": "guessed_id"}])
        self.stage.add_pairs([mapped["pair"]])
        with self.assertRaisesRegex(ValueError, "Thread ID"):
            self.stage.add_threads([{**mapped["threads"][0], "thread_id": "wrong_thread"}])

    def test_membership_references_create_comment_placeholders_and_enrich_later(self):
        mapped = map_pair(pair_record(), PAIR, MEMBER, 2)
        self.stage.add_pairs([mapped["pair"]])
        self.stage.add_threads(mapped["threads"])
        self.stage.add_thread_comments(mapped["thread_comments"])
        self.assertEqual(self.stage.validate()["placeholders"]["comments"], 3)
        self.stage.add_pair(mapped)
        self.assertEqual(self.stage.validate()["placeholders"]["comments"], 0)

    def test_missing_thread_and_cross_post_membership_are_rejected(self):
        mapped = map_pair(pair_record(), PAIR, MEMBER, 2)
        with self.assertRaisesRegex(ValueError, "existing thread"):
            self.stage.add_thread_comments(mapped["thread_comments"])
        self.stage.add_pair(mapped)
        with self.assertRaisesRegex(ValueError, "same post"):
            self.stage.add_thread_comments([{**mapped["thread_comments"][0], "post_id": "t3_other"}])

    def test_validation_requires_two_outcome_threads(self):
        mapped = map_pair(pair_record(), PAIR, MEMBER, 2)
        self.stage.add_pairs([mapped["pair"]])
        self.stage.add_threads([mapped["threads"][0]])
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            self.stage.validate()

    def test_validation_rejects_gaps_in_membership_positions(self):
        mapped = map_pair(pair_record(), PAIR, MEMBER, 2)
        mapped["thread_comments"][1]["position"] = 2
        self.stage.add_pair(mapped)
        with self.assertRaisesRegex(ValueError, "Non-contiguous"):
            self.stage.validate()

    def test_validation_rejects_parent_cycles_and_can_be_repeated_after_failure(self):
        rows = [map_comment(comment_record("t1_a", "t1_b"), POST_ID, ALL),
                map_comment(comment_record("t1_b", "t1_a"), POST_ID, ALL)]
        self.stage.add_comments(rows, task="all", split="train")
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "parent cycle"):
                self.stage.validate()

    def test_unknown_parent_placeholder_is_a_valid_acyclic_endpoint(self):
        row = map_comment(comment_record("t1_child", "t1_missing"), POST_ID, ALL)
        self.stage.add_comments([row], task="all", split="train")
        self.assertEqual(self.stage.validate()["placeholders"], {"posts": 1, "comments": 1})

    def test_validation_detects_comment_candidate_source_mismatch(self):
        self.stage.add_comments([map_comment(comment_record(), POST_ID, ALL)], task="all", split="train")
        with self.stage.transaction():
            self.stage.connection.execute("UPDATE candidates SET source_id = 3")
        with self.assertRaisesRegex(ValueError, "Invalid candidates source/split"):
            self.stage.validate()

    def test_all_seven_tables_round_trip_with_their_declared_schemas(self):
        self.add_complete()
        self.stage.validate()
        for name, schema in TABLE_SCHEMAS.items():
            with self.subTest(table=name):
                rows = self.rows(name, batch_size=2)
                path = self.root / f"{name}.parquet"
                pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
                restored = pq.read_table(path)
                self.assertEqual(restored.schema, schema)
                self.assertEqual(restored.to_pylist(), rows)

    def test_split_filtered_exports_keep_references_and_sources_consistent(self):
        self.add_complete()
        heldout_source = SOURCE_MAP["op_task/heldout_op_data.jsonlist.bz2"]
        self.stage.add_op_labels([map_op_label({"name": "t3_heldout", "delta_label": True}, heldout_source)])
        self.assertEqual([r["post_id"] for r in self.rows("posts", split="heldout")], ["t3_heldout"])
        for table in ("comments", "pairs", "threads", "thread_comments"):
            self.assertEqual(self.rows(table, split="heldout"), [])
        self.assertEqual(len(self.rows("sources", split="heldout")), 3)
        self.assertEqual(self.rows("op_labels", split="heldout")[0]["source_id"], 3)

    def test_bounded_generator_post_batches_and_invalid_export_options(self):
        self.stage.add_posts(map_post({"name": f"t3_post_{i}"}, ALL) for i in range(5))
        self.assertEqual([len(batch) for batch in self.stage.iter_table_batches("posts", batch_size=2)], [2, 2, 1])
        for table, options in (("unknown", {}), ("posts", {"batch_size": True}),
                               ("posts", {"split": "test"})):
            with self.subTest(table=table, options=options):
                with self.assertRaises(ValueError):
                    self.rows(table, **options)

    def test_inputs_remain_unchanged(self):
        mapped = map_pair(pair_record(), PAIR, MEMBER, 2)
        before = copy.deepcopy(mapped)
        self.stage.add_pair(mapped)
        self.assertEqual(mapped, before)


if __name__ == "__main__":
    unittest.main()
