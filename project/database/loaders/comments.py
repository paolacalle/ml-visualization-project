import bz2
import json
import tarfile
from pathlib import Path


def load_comments(archive_path, db, post_id):
    sources = db.execute(
        """
        SELECT s.source_id, s.archive_member
        FROM cmv.posts p
        JOIN cmv.sources s ON s.source_id = p.source_id
        WHERE p.post_id = %s AND s.task = 'all';
        """,
        (post_id,),
    )

    if not sources:
        raise ValueError("Load the post from its 'all' source first.")

    source_id, member_name = sources[0]

    sql_path = (
        Path(__file__).resolve().parents[1]
        / "queries"
        / "insert_comment.sql"
    )
    sql = sql_path.read_text(encoding="utf-8")

    with tarfile.open(archive_path, "r|bz2") as archive:
        for member in archive:
            if not member.isfile() or member.name != member_name:
                continue

            with archive.extractfile(member) as compressed:
                with bz2.BZ2File(compressed, "rb") as lines:
                    for line in lines:
                        if not line.strip():
                            continue

                        post = json.loads(line)

                        if post["name"] != post_id:
                            continue

                        comments = post.get("comments", [])
                        ids = {
                            c.get("name") or f"t1_{c['id']}"
                            for c in comments
                        }

                        rows = []

                        for comment in comments:
                            parent = comment.get("parent_id")

                            if parent == post_id:
                                parent_comment_id = None
                            elif parent in ids:
                                parent_comment_id = parent
                            else:
                                raise ValueError(
                                    f"Missing or unexpected parent: {parent}"
                                )

                            rows.append({
                                "comment_id": (
                                    comment.get("name")
                                    or f"t1_{comment['id']}"
                                ),
                                "post_id": post_id,
                                "parent_comment_id": parent_comment_id,
                                "original_parent_id": parent,
                                "author": comment.get("author"),
                                "created_utc": comment.get("created_utc"),
                                "score": comment.get("score"),
                                "source_id": source_id,
                            })

                        with db.connection.transaction():
                            conflicts = db.execute(
                                """
                                SELECT comment_id
                                FROM cmv.comments
                                WHERE comment_id = ANY(%s)
                                  AND post_id <> %s;
                                """,
                                (list(ids), post_id),
                            )

                            if conflicts:
                                raise ValueError(
                                    "A comment belongs to another post."
                                )

                            with db.connection.cursor() as cursor:
                                cursor.executemany(sql, rows)

                        print(
                            f"{post_id}: processed {len(rows)} comments; "
                            f"source reports {post.get('num_comments')}"
                        )
                        return

            break

    raise ValueError(f"Post not found in source: {post_id}")