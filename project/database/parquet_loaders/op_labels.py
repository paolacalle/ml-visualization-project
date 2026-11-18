import pyarrow as pa


OP_LABEL_SCHEMA = pa.schema([
    pa.field("post_id", pa.string(), nullable=False),
    pa.field("delta_label", pa.bool_(), nullable=False),
    pa.field("source_id", pa.int64(), nullable=False),
])


def map_op_label(record, source):
    """Map a supplied OP-task label without inventing labels for other posts."""
    if source["task"] != "op_task":
        raise ValueError("map_op_label expects 'op_task' source.")

    post_id = record.get("name")
    if (
        not isinstance(post_id, str)
        or not post_id.startswith("t3_")
        or len(post_id) <= 3
    ):
        raise ValueError(f"Invalid post ID: {post_id}")

    label = record.get("delta_label")
    if not isinstance(label, bool):
        raise ValueError(f"{post_id}: expected a boolean delta_label.")

    return {
        "post_id": post_id,
        "delta_label": label,
        "source_id": source["source_id"],
    }
