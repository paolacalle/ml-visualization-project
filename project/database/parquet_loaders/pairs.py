"""Map one pair-task record into DB-keyed relational rows."""

import pyarrow as pa


PAIR_SCHEMA = pa.schema([
    pa.field("pair_id", pa.string(), nullable=False),
    pa.field("post_id", pa.string(), nullable=False),
    pa.field("source_id", pa.int64(), nullable=False),
    pa.field("source_line", pa.int32(), nullable=False),
])

THREAD_SCHEMA = pa.schema([
    pa.field("thread_id", pa.string(), nullable=False),
    pa.field("pair_id", pa.string(), nullable=False),
    pa.field("post_id", pa.string(), nullable=False),
    pa.field("outcome", pa.string(), nullable=False),
    pa.field("ancestor_id", pa.string()),
    pa.field("author", pa.string()),
])

THREAD_COMMENT_SCHEMA = pa.schema([
    pa.field("thread_id", pa.string(), nullable=False),
    pa.field("comment_id", pa.string(), nullable=False),
    pa.field("post_id", pa.string(), nullable=False),
    pa.field("position", pa.int32(), nullable=False),
])


def map_pair(record, source, member_name, line_number):
    """Return pair, threads, and thread_comments; reconcile comment rows later.

    IDs use the archive member and actual one-based JSON line number, independent
    of database-generated source IDs. Membership positions remain zero-based.
    """
    def invalid(message):
        raise ValueError(f"{member_name}, line {line_number}: {message}")

    if source.get("task") != "pair_task":
        invalid("map_pair expects 'pair_task' source.")

    split = source.get("original_split")
    if split not in ("train", "heldout"):
        invalid(f"Invalid original split: {split}")
    expected_member = f"pair_task/{split}_pair_data.jsonlist.bz2"
    if member_name != expected_member:
        invalid(f"Source split requires archive member {expected_member}")

    if type(line_number) is not int or not 1 <= line_number <= 2**31 - 1:
        invalid("source_line must be a positive PostgreSQL INTEGER.")

    source_id = source.get("source_id")
    if type(source_id) is not int or not 1 <= source_id <= 2**63 - 1:
        invalid("source_id must be a positive PostgreSQL BIGINT.")

    post_id = record.get("op_name")
    if not isinstance(post_id, str) or not post_id.startswith("t3_") or len(post_id) <= 3:
        invalid(f"Invalid post ID: {post_id}")

    pair_id = f"{member_name}#{line_number}"
    pair = {
        "pair_id": pair_id,
        "post_id": post_id,
        "source_id": source_id,
        "source_line": line_number,
    }
    threads = []
    memberships = []

    for outcome in ("positive", "negative"):
        thread = record.get(outcome)
        if not isinstance(thread, dict):
            invalid(f"Expected a {outcome} thread object.")
        comments = thread.get("comments")
        if not isinstance(comments, list):
            invalid(f"{outcome}: expected a comments list.")

        thread_id = f"{pair_id}:{outcome}"
        # Preserve the source ancestor value; it is not assumed to be an FK.
        threads.append({
            "thread_id": thread_id,
            "pair_id": pair_id,
            "post_id": post_id,
            "outcome": outcome,
            "ancestor_id": thread.get("ancestor"),
            "author": thread.get("author"),
        })

        seen = set()
        for position, comment in enumerate(comments):
            if not isinstance(comment, dict):
                invalid(f"{outcome}, position {position}: expected a comment object.")
            comment_id = comment.get("name")
            if not comment_id:
                raw_id = comment.get("id")
                if not isinstance(raw_id, str) or not raw_id:
                    invalid(f"{outcome}, position {position}: missing comment ID.")
                comment_id = f"t1_{raw_id}"
            if (
                not isinstance(comment_id, str)
                or not comment_id.startswith("t1_")
                or len(comment_id) <= 3
            ):
                invalid(f"{outcome}, position {position}: invalid comment ID {comment_id}")

            link_id = comment.get("link_id")
            if link_id is not None and link_id != post_id:
                invalid(f"{comment_id} belongs to {link_id}, not {post_id}")
            if comment_id in seen:
                invalid(f"{outcome}: duplicate comment ID {comment_id}")
            seen.add(comment_id)

            memberships.append({
                "thread_id": thread_id,
                "comment_id": comment_id,
                "post_id": post_id,
                "position": position,
            })

    return {"pair": pair, "threads": threads, "thread_comments": memberships}
