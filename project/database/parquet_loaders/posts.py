
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
    if source["task"] != "all":
        raise ValueError("map_post expects 'all' source.")
    
    post_id = record["name"]
    
    if not isinstance(post_id, str) and not post_id.startswith("t3_"):
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