"""Build a branch cohort using PostgreSQL's frozen Parquet mirror."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import unescape
from math import isfinite
from pathlib import Path
import json
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


NOTEBOOK_DIR = Path(__file__).resolve().parent.parent
REPO = NOTEBOOK_DIR.parents[2]
DATA = REPO / "project/data/processed"
METADATA = DATA / "cmv-metadata/v1"
TEXT = DATA / "cmv-text/v1"
OUT = DATA / "op-engagement-study"

HOUR = 3_600
DAY = 24 * HOUR
DEFAULT_OBSERVATION_HOURS = 24
FOLLOW_UP_DAYS = 7
# The paper notes that DeltaBot coverage in January 2013 is uncertain.
BOT_PERIOD_START = datetime(2013, 2, 1, tzinfo=timezone.utc).timestamp()
# Use the end of the published training period as a conservative archive boundary.
SPLIT_CUTOFFS = {
    "train": datetime(2015, 5, 8, tzinfo=timezone.utc).timestamp(),
    "heldout": datetime(2015, 9, 2, tzinfo=timezone.utc).timestamp(),
}
TRAIN_CUTOFF = SPLIT_CUTOFFS["train"]
CONFIRMATION = re.compile(
    r"^confirmed\s*(?::|[-–—]+)\s*\d+\s+(?:deltas?|points?)\s+awarded\s+to\b",
    re.I,
)
AWARD_MARKER = re.compile(r"(?<![\w/])!delta\b|(?:^|\s)[Δ∆](?=\s|[.!,:;]|$)", re.I)
UNKNOWN_AUTHORS = {"", "[deleted]", "[removed]"}
BOTS = {"deltabot", "automoderator"}
EDIT = re.compile(r"^\s*[*_#-]*\s*(?:ninja\s+)?(?:edit|update)\b\s*(?:\d+\s*)?[:–—-]", re.I)


def manifest(directory):
    return json.loads((directory / "manifest.json").read_text())


def rows_for(meta, table, columns, split="train"):
    """Load only the needed columns from the selected partition of the frozen release."""
    for file in meta["files"]:
        if file["table"] == table and file["split"] in (split, None):
            yield from pq.ParquetFile(METADATA / file["path"]).read(columns=columns).to_pylist()


def speaker(author):
    return (author or "").casefold()


def timestamp(value):
    return value if value is not None and isfinite(value) else None


def text_rows(text_files, ids, split):
    """Read each text shard once and retain only the requested comment IDs."""
    value_set = pa.array(ids, type=pa.string())
    for entry in text_files:
        path = entry["path"]
        if not path.startswith(f"comments/split={split}/"):
            continue
        table = pq.ParquetFile(TEXT / path).read(columns=["comment_id", "body"])
        yield from table.filter(pc.is_in(table["comment_id"], value_set=value_set)).to_pylist()


def signals_from_text(text_files, ids, split="train"):
    for row in text_rows(text_files, ids, split):
        body = (row["body"] or "").strip()
        confirms = bool(CONFIRMATION.match(body.lstrip("*_").strip()))
        readable = bool(body) and body.casefold() not in {"[deleted]", "[removed]"}
        unquoted = "\n".join(
            line for line in unescape(body).splitlines() if not line.lstrip().startswith(">")
        )
        yield row["comment_id"], (confirms, readable, bool(AWARD_MARKER.search(unquoted)))


def opening_text_lines(body):
    """Yield opening text without quoted lines and marked edit sections."""
    body = (body or "").strip()
    if not body or body.casefold() in {"[deleted]", "[removed]"}:
        return
    for line in unescape(unescape(body)).splitlines():
        if line.lstrip().startswith(">"):
            continue
        if EDIT.match(line):
            break
        yield line


def opening_word_count(body):
    """Count words in available opening text."""
    return sum(len(line.split()) for line in opening_text_lines(body)) or None


def branch_roots(post_id, comments):
    """Return top-level openings and each connected comment's branch and depth."""
    children = defaultdict(list)
    for cid, comment in comments.items():
        children[comment["parent"]].append(cid)
    openings = children[post_id]
    root_of = {}
    depths = {}
    for root in openings:
        stack = [(root, 0)]
        while stack:
            cid, depth = stack.pop()
            if cid not in root_of:
                root_of[cid] = root
                depths[cid] = depth
                stack.extend((child, depth + 1) for child in children[cid])
    return openings, root_of, depths


def reconstruct_awards(post_id, op, comments, root_of, signals):
    """Use the OP award comment time, with DeltaBot confirmation as evidence."""
    verified = {}
    uncertain_by_branch = defaultdict(list)
    uncertain_post = []
    known_non_op_confirmations = 0

    for bot_id, bot in comments.items():
        if bot["author"] != "deltabot":
            continue
        confirms, readable, _ = signals.get(bot_id, (False, False, False))
        award_id = bot["parent"]
        award = comments.get(award_id)
        if award and award["author"] not in UNKNOWN_AUTHORS and award["author"] != op:
            known_non_op_confirmations += int(confirms)
            continue
        if readable and not confirms:
            continue

        recipient_id = award["parent"] if award else None
        branch_id = root_of.get(recipient_id)
        award_time = timestamp(award["time"]) if award else None
        recipient = comments.get(recipient_id)
        recipient_time = timestamp(recipient["time"]) if recipient else None
        bot_time = timestamp(bot["time"])
        if (confirms and op not in UNKNOWN_AUTHORS and award and award["author"] == op and recipient
                and recipient["author"] != op and branch_id and award_time is not None
                and recipient_time is not None and recipient_time <= award_time
                and (bot_time is None or award_time <= bot_time)):
            verified.setdefault(award_id, {
                "post_id": post_id, "branch_id": branch_id, "award_comment_id": award_id,
                "confirmation_comment_id": bot_id, "recipient_comment_id": recipient_id,
                "award_utc": award_time, "confirmation_utc": bot_time,
            })
        elif branch_id:
            uncertain_by_branch[branch_id].append(award_time)
        else:
            uncertain_post.append(award_time)

    # An unconfirmed OP delta marker is evidence of uncertainty, never a positive outcome.
    for cid, row in comments.items():
        if row["author"] != op or cid in verified or not signals.get(cid, (False, False, False))[2]:
            continue
        root = root_of.get(row["parent"])
        if root is not None:
            uncertain_by_branch[root].append(timestamp(row["time"]))
        elif cid in root_of and row["parent"] != post_id:
            uncertain_by_branch[root_of[cid]].append(timestamp(row["time"]))
        elif row["parent"] != post_id:
            uncertain_post.append(timestamp(row["time"]))
    return verified, uncertain_by_branch, uncertain_post, known_non_op_confirmations


def earliest_uncertainty(times):
    """Missing event times make a potential problem relevant to every horizon."""
    if any(time is None for time in times):
        return float("-inf")
    return min(times, default=None)


def analyze_post(post, comments, signals, expected_winners, op_delta_label, *,
                 cutoff=TRAIN_CUTOFF, opening_lengths=None, timing_features=False,
                 observation_hours=DEFAULT_OBSERVATION_HOURS):
    post_id = post["post_id"]
    op = speaker(post["author"])
    openings, root_of, depths = branch_roots(post_id, comments)
    awards, branch_uncertainty, post_uncertainty, non_op = reconstruct_awards(
        post_id, op, comments, root_of, signals
    )
    first_awards = {}
    first_confirmations = {}
    for award in awards.values():
        root = award["branch_id"]
        first_awards[root] = min(first_awards.get(root, float("inf")), award["award_utc"])
        if award["confirmation_utc"] is not None:
            first_confirmations[root] = min(
                first_confirmations.get(root, float("inf")), award["confirmation_utc"]
            )
    missing_published = expected_winners - first_awards.keys()
    for root in missing_published:
        if root in openings:
            branch_uncertainty[root].append(None)
        else:
            post_uncertainty.append(None)
    unresolved_op_label = op_delta_label is True and not awards
    if unresolved_op_label:
        post_uncertainty.append(None)

    contents = defaultdict(list)
    # The traversal records parents before children. Keep that order for early depth.
    for cid, root in root_of.items():
        if cid != root:
            contents[root].append((cid, comments[cid]))
    unlinked = [timestamp(row["time"]) for cid, row in comments.items() if cid not in root_of]
    earliest_unlinked = earliest_uncertainty(unlinked) if timing_features else None

    result = []
    window = f"{observation_hours}h"
    for root in openings:
        opening = comments[root]
        begin = timestamp(opening["time"])
        early_end = begin + observation_hours * HOUR if begin is not None else None
        followup_end = early_end + FOLLOW_UP_DAYS * DAY if early_end is not None else None
        first_award = first_awards.get(root)
        first_confirmation = first_confirmations.get(root)
        early_count = early_op = 0
        invalid_descendant_time = False
        non_op_count = max_depth = 0
        non_op_authors = set()
        early_path = {root}
        for cid, child in contents[root]:
            at = timestamp(child["time"])
            if at is None or (begin is not None and at < begin):
                invalid_descendant_time = True
            elif begin is not None and at <= early_end:
                early_count += 1
                early_op += int(bool(op) and child["author"] == op)
                if opening_lengths is not None and child["parent"] in early_path:
                    early_path.add(cid)
                    author = child["author"]
                    if author != op and author not in BOTS | UNKNOWN_AUTHORS:
                        non_op_count += 1
                        non_op_authors.add(author)
                    max_depth = max(max_depth, depths[cid])

        def affects_window(times):
            return followup_end is not None and any(
                time is None or time <= followup_end for time in times
            )

        # Keep every opening in the table. Ineligible rows get an explanation and no outcome.
        if op in UNKNOWN_AUTHORS:
            reason = "unknown_op_author"
        elif opening["author"] in BOTS or opening["author"] == op:
            reason = "not_challenger_opening"
        elif begin is None or invalid_descendant_time:
            reason = "missing_or_invalid_branch_timestamp"
        elif begin < BOT_PERIOD_START:
            reason = "before_reliable_bot_period"
        elif followup_end > cutoff:
            reason = ("beyond_training_followup_boundary" if cutoff == TRAIN_CUTOFF
                      else "beyond_heldout_followup_boundary")
        elif first_award is not None and first_award <= followup_end and (
            first_confirmation is None
            or first_award <= early_end < first_confirmation
            or first_award <= followup_end < first_confirmation
        ):
            # An award may have been added to an existing comment. Keep rows only when
            # using its creation or its confirmation gives the same outcome window.
            reason = "award_and_confirmation_cross_cutoff"
        elif first_award is not None and first_award <= early_end:
            reason = f"delta_by_{window}"
        elif affects_window(unlinked):
            reason = "incomplete_parent_path"
        elif affects_window(post_uncertainty) or affects_window(branch_uncertainty[root]):
            reason = "uncertain_award_attribution"
        else:
            reason = None
        eligible = reason is None
        record = {
            "post_id": post_id, "branch_id": root, "branch_created_utc": begin,
            "observation_end_utc": early_end, "followup_end_utc": followup_end,
            f"comments_{window}": early_count if begin is not None else None,
            f"op_replies_{window}": early_op if begin is not None else None,
            f"op_engaged_{window}": early_op > 0 if begin is not None else None,
            "first_verified_delta_utc": first_award,
            "first_confirmation_utc": first_confirmation, "eligible": eligible,
            f"delta_after_{window}": int(first_award is not None and early_end < first_award <= followup_end)
            if eligible else None,
            "exclusion_reason": reason,
        }
        if opening_lengths is not None:
            post_time = timestamp(post.get("created_utc"))
            delay = (begin - post_time) / 60 if begin is not None and post_time is not None else None
            record.update({
                f"non_op_comments_{window}": non_op_count if begin is not None else None,
                f"unique_non_op_authors_{window}": len(non_op_authors) if begin is not None else None,
                "opening_delay_minutes": delay if delay is not None and delay >= 0 else None,
                "opening_reply_length": opening_lengths.get(root),
                f"max_depth_{window}": max_depth if begin is not None else None,
            })
        if timing_features:
            record.update({
                "branch_static_valid": (
                    op not in UNKNOWN_AUTHORS and opening["author"] not in BOTS
                    and opening["author"] != op and begin is not None and not invalid_descendant_time
                    and begin >= BOT_PERIOD_START
                ),
                "earliest_unlinked_utc": earliest_unlinked,
                "earliest_unresolved_award_utc": earliest_uncertainty(
                    post_uncertainty + branch_uncertainty[root]
                ),
            })
        result.append(record)

    audit = {
        "published_positive_openings": len(expected_winners),
        "published_positive_openings_with_verified_award": len(expected_winners) - len(missing_published),
        "positive_op_labels_without_verified_award": int(unresolved_op_label),
        "confirmed_non_op_awards_ignored": non_op,
        "posts_with_unlinked_comments": int(bool(unlinked)),
        "unlinked_comments": len(unlinked),
    }
    return result, list(awards.values()), audit


def write_parquet(path, rows):
    staging = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows), staging, compression="zstd")
    staging.replace(path)


def build(split, model_features=False, timing_features=False,
          observation_hours=DEFAULT_OBSERVATION_HOURS):
    if observation_hours <= 0:
        raise ValueError("Observation hours must be positive.")
    if not model_features and observation_hours != DEFAULT_OBSERVATION_HOURS:
        raise ValueError("Custom observation windows require model features.")
    cutoff = SPLIT_CUTOFFS[split]
    window = f"{observation_hours}h"
    if timing_features:
        output = DATA / "delta-timing-analysis" / split
    elif model_features:
        output = DATA / "m0-baseline"
        if observation_hours != DEFAULT_OBSERVATION_HOURS:
            output = output / window
        output = output / split
    else:
        output = OUT if split == "train" else OUT / split
    meta, text = manifest(METADATA), manifest(TEXT)
    if meta["version"] != "v1" or text["version"] != "v1":
        raise ValueError("Expected metadata and text releases v1.")
    if meta["raw_archive"]["sha256"] != text["raw_archive"]["sha256"]:
        raise ValueError("Metadata and text archives differ.")
    full_source = next(
        row["source_id"]
        for row in rows_for(meta, "sources", ["source_id", "task", "original_split"], split)
        if row["task"] == "all" and row["original_split"] == split
    )
    posts = {
        row["post_id"]: row
        for row in rows_for(meta, "posts", ["post_id", "author", "source_id", "created_utc"], split)
        if row["source_id"] == full_source
    }
    print(f"Read {len(posts):,} full {split} discussions.", flush=True)
    comments_by_post = defaultdict(dict)
    relevant_text_ids = set()
    columns = ["post_id", "comment_id", "author", "parent_comment_id", "original_parent_id",
               "created_utc", "source_id", "content_available"]
    for row in rows_for(meta, "comments", columns, split):
        pid, cid = row["post_id"], row["comment_id"]
        if pid not in posts or row["source_id"] != full_source or not row["content_available"]:
            continue
        author = speaker(row["author"])
        comments_by_post[pid][cid] = {
            "author": author, "parent": row["parent_comment_id"] or row["original_parent_id"],
            "time": row["created_utc"],
        }
        if author in {"deltabot", speaker(posts[pid]["author"])}:
            relevant_text_ids.add(cid)

    print(f"Scanning Parquet for {len(relevant_text_ids):,} OP or DeltaBot comments.", flush=True)
    signals = dict(signals_from_text(text["files"], relevant_text_ids, split))
    opening_lengths = None
    if model_features:
        opening_ids = {
            cid for pid, comments in comments_by_post.items() for cid, comment in comments.items()
            if comment["parent"] == pid
        }
        opening_lengths = {
            row["comment_id"]: opening_word_count(row["body"])
            for row in text_rows(text["files"], opening_ids, split)
        }
    labels = {
        row["post_id"]: row["delta_label"]
        for row in rows_for(meta, "op_labels", ["post_id", "delta_label"], split)
    }
    positive_threads = {
        row["thread_id"] for row in rows_for(meta, "threads", ["thread_id", "outcome"], split)
        if row["outcome"] == "positive"
    }
    expected_winners = defaultdict(set)
    columns = ["post_id", "thread_id", "position", "comment_id"]
    for row in rows_for(meta, "thread_comments", columns, split):
        if row["position"] == 0 and row["thread_id"] in positive_threads:
            expected_winners[row["post_id"]].add(row["comment_id"])

    branch_rows, award_rows = [], []
    audit = Counter()
    for pid, post in posts.items():
        rows, awards, diagnostics = analyze_post(
            post, comments_by_post[pid], signals, expected_winners[pid], labels.get(pid),
            cutoff=cutoff, opening_lengths=opening_lengths, timing_features=timing_features,
            observation_hours=observation_hours,
        )
        branch_rows.extend(rows)
        award_rows.extend(awards)
        audit.update(diagnostics)
    split_name = "training" if split == "train" else split
    audit.update({
        f"{split_name}_posts": len(posts),
        f"{split_name}_comments": sum(map(len, comments_by_post.values())),
        "op_or_bot_text_ids_requested": len(relevant_text_ids),
        "op_or_bot_text_ids_found": len(signals),
        "top_level_branches": len(branch_rows),
        "verified_op_awards": len(award_rows),
        "eligible_branches": sum(row["eligible"] for row in branch_rows),
        "later_deltas": sum(row[f"delta_after_{window}"] or 0 for row in branch_rows),
    })
    report = {
        "counts": dict(audit),
        "exclusions": dict(Counter(r["exclusion_reason"] for r in branch_rows if not r["eligible"])),
        "first_bot_period_utc": BOT_PERIOD_START, "training_cutoff_utc": TRAIN_CUTOFF,
        "split": split, "followup_boundary_utc": cutoff, "model_features": model_features,
        "timing_features": timing_features,
        "observation_hours": observation_hours, "followup_days_after_cutoff": FOLLOW_UP_DAYS,
        "raw_archive_sha256": meta["raw_archive"]["sha256"],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_parquet(output / "branches.parquet", branch_rows)
    write_parquet(output / "verified_awards.parquet", award_rows)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLIT_CUTOFFS, default="train")
    features = parser.add_mutually_exclusive_group()
    features.add_argument("--model-features", action="store_true",
                          help="Add the M0 features and write to the M0 dataset directory.")
    features.add_argument("--timing-features", action="store_true",
                          help="Export eligibility inputs for alternative timing windows.")
    parser.add_argument("--observation-hours", type=int, default=DEFAULT_OBSERVATION_HOURS,
                        help="Observation window in hours for model features. Defaults to 24.")
    args = parser.parse_args()
    if args.observation_hours <= 0:
        parser.error("--observation-hours must be positive")
    if args.observation_hours != DEFAULT_OBSERVATION_HOURS and not args.model_features:
        parser.error("--observation-hours requires --model-features for nondefault windows")
    build(args.split, args.model_features, args.timing_features, args.observation_hours)


if __name__ == "__main__":
    main()
