import pyarrow as pa


COMMENT_SCHEMA = pa.schema([
    pa.field("comment_id", pa.string(), nullable=False),
    pa.field("post_id", pa.string(), nullable=False),
    pa.field("parent_comment_id", pa.string()),
    pa.field("original_parent_id", pa.string()),
    pa.field("author", pa.string()),
    pa.field("created_utc", pa.float64()),
    pa.field("score", pa.int32()),
    pa.field("content_available", pa.bool_(), nullable=False),
    pa.field("source_id", pa.int64()),
])

def map_comment(comment, post_id, source):
    """Map one raw comment to the current comments table."""
    if source["task"] != "all":
        raise ValueError("map_comment expects 'all' source.")

    if not isinstance(post_id, str) or not post_id.startswith("t3_"):
        raise ValueError(f"Invalid post ID: {post_id}")

    comment_id = comment.get("name")

    if not comment_id:
        raw_id = comment.get("id")
        if not isinstance(raw_id, str) or not raw_id:
            raise ValueError("Comment is missing its ID.")
        comment_id = f"t1_{raw_id}"

    if (
        not isinstance(comment_id, str)
        or not comment_id.startswith("t1_")
        or len(comment_id) <= 3
    ):
        raise ValueError(f"Invalid comment ID: {comment_id}")

    # some records explicitly identify their owning post.
    link_id = comment.get("link_id")
    if link_id is not None and link_id != post_id:
        raise ValueError(
            f"{comment_id} belongs to {link_id}, not {post_id}"
        )

    parent = comment.get("parent_id")

    if parent == post_id:
        parent_comment_id = None
    elif isinstance(parent, str) and parent.startswith("t1_") and len(parent) > 3:
        parent_comment_id = parent
    else:
        raise ValueError(f"{comment_id} has invalid parent: {parent}")

    if parent_comment_id == comment_id:
        raise ValueError(f"{comment_id} cannot reply to itself.")

    return {
        "comment_id": comment_id,
        "post_id": post_id,
        "parent_comment_id": parent_comment_id,
        "original_parent_id": parent,
        "author": comment.get("author"),
        "created_utc": comment.get("created_utc"),
        "score": comment.get("score"),
        "content_available": True,
        "source_id": source["source_id"],
    }
