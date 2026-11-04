import bz2
import json
import tarfile

def load_posts(archive_path, db, limit_per_source=100):
    sources = db.execute("""
        SELECT archive_member, source_id, original_split
        FROM cmv.sources
        WHERE task = 'all';
    """)

    source_map = {
        member: (source_id, split)
        for member, source_id, split in sources
    }

    if len(source_map) != 2:
        raise ValueError("Register both 'all' source files first.")


    processed_sources = set()

    with tarfile.open(archive_path, "r|bz2") as archive:
        for member in archive:
            if not member.isfile() or member.name not in source_map:
                continue

            source_id, split = source_map[member.name]
            processed = 0
            inserted = 0

            # Commit this source's batch together.
            with db.connection.transaction():
                with archive.extractfile(member) as compressed:
                    with bz2.BZ2File(compressed, "rb") as lines:
                        for line in lines:
                            if not line.strip():
                                continue

                            post = json.loads(line)
                            edited = post.get("edited")

                            # bool is a subclass of int, so exclude it explicitly.
                            has_edit_timestamp = (
                                isinstance(edited, (int, float))
                                and not isinstance(edited, bool)
                            )

                            params = {
                                "post_id": post["name"],
                                "original_split": split,
                                "title": post.get("title"),
                                "author": post.get("author"),
                                "created_utc": post.get("created_utc"),
                                "source_id": source_id,
                                "score": post.get("score"),
                                "ups": post.get("ups"),
                                "downs": post.get("downs"),
                                "num_comments": post.get("num_comments"),
                                "gilded": post.get("gilded"),
                                "permalink": post.get("permalink"),
                                "over_18": post.get("over_18"),
                                "archived": post.get("archived"),
                                "stickied": post.get("stickied"),
                                "distinguished": post.get("distinguished"),
                                "link_flair_text": post.get("link_flair_text"),
                                "author_flair_text": post.get("author_flair_text"),
                                "is_edited": (
                                    True if has_edit_timestamp
                                    else edited if isinstance(edited, bool)
                                    else None
                                ),
                                "edited_utc": edited if has_edit_timestamp else None,
                            }

                            result = db.execute("insert_post.sql", params)
                            processed += 1
                            inserted += bool(result)

                            if limit_per_source is not None and processed >= limit_per_source:
                                break

            print(
                f"{split}: processed {processed}, "
                f"inserted {inserted}, "
                f"already existed {processed - inserted}"
            )

            processed_sources.add(member.name)

            if processed_sources == set(source_map):
                return

    raise ValueError("Not all expected source files were found.")