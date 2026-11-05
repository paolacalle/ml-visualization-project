import pyarrow as pa


POST_SCHEMA = pa.schema([
    pa.field("post_id", pa.string(), nullable=False),
    pa.field("original_split", pa.string(), nullable=False),
    pa.field("title", pa.string()),
    pa.field("author", pa.string()),
    pa.field("created_utc", pa.float64()),
    pa.field("content_available", pa.bool_(), nullable=False),
    pa.field("source_id", pa.int64()),
    pa.field("score", pa.int32()),
    pa.field("ups", pa.int32()),
    pa.field("downs", pa.int32()),
    pa.field("num_comments", pa.int32()),
    pa.field("gilded", pa.int32()),
    pa.field("permalink", pa.string()),
    pa.field("over_18", pa.bool_()),
    pa.field("archived", pa.bool_()),
    pa.field("stickied", pa.bool_()),
    pa.field("distinguished", pa.string()),
    pa.field("link_flair_text", pa.string()),
    pa.field("author_flair_text", pa.string()),
    pa.field("is_edited", pa.bool_()),
    pa.field("edited_utc", pa.float64()),
])


# fields to extract form the raw fields
POST_FIELDS = (
    "title",
    "author",
    "created_utc",
    "score",
    "ups",
    "downs",
    "num_comments",
    "gilded",
    "permalink",
    "over_18",
    "archived",
    "stickied",
    "distinguished",
    "link_flair_text",
    "author_flair_text",
)

def map_post(record, source):
    """
    Method parses the json into the fields we 
    want to retain within our parquet files and tables.
    """
    if source["task"] != "all":
        raise ValueError("map_post expects 'all' source.")
    
    post_id = record["name"]
    
    if not isinstance(post_id, str) or not post_id.startswith("t3_"):
        raise ValueError(f"Invalid post ID: {post_id}")
    
    edited = record.get("edited")
    
    if isinstance(edited, bool):
        is_edited = edited
        edited_utc = None
    elif isinstance(edited, (int, float)):
        is_edited = True
        edited_utc = edited
    else:
        is_edited = None
        edited_utc = None
        
    return {
        "post_id": post_id, 
        "original_split": source["original_split"],
        **{field: record.get(field) for field in POST_FIELDS},
        "content_available": True,
        "source_id": source["source_id"],
        "is_edited": is_edited,
        "edited_utc": edited_utc,
    }
