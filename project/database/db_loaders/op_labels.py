import bz2
import json
import tarfile


def load_op_label(archive_path, db, post_id):
    sources = db.execute(
        """
        SELECT s.source_id, s.archive_member
        FROM cmv.sources s
        JOIN cmv.posts p
          ON p.original_split = s.original_split
        WHERE p.post_id = %s
          AND s.task = 'op_task';
        """,
        (post_id,),
    )

    if len(sources) != 1:
        raise ValueError(
            "Expected an existing post and one matching op_task source."
        )

    source_id, member_name = sources[0]

    with tarfile.open(archive_path, "r|bz2") as archive:
        for member in archive:
            if not member.isfile() or member.name != member_name:
                continue

            with archive.extractfile(member) as compressed:
                with bz2.BZ2File(compressed, "rb") as lines:
                    for line in lines:
                        if not line.strip():
                            continue

                        record = json.loads(line)

                        if record["name"] != post_id:
                            continue

                        label = record["delta_label"]

                        if not isinstance(label, bool):
                            raise ValueError("Expected a boolean delta_label.")

                        result = db.execute(
                            "insert_op_label.sql",
                            (post_id, label, source_id),
                        )

                        status = "Inserted" if result else "Already exists"
                        print(f"{status}: {post_id}, delta_label={label}")
                        return

            print(f"No provided label for {post_id}")
            return

    raise FileNotFoundError(member_name)