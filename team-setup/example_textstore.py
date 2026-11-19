"""Text-only starter: read a post and its comments from the local download."""

import argparse

from _common import TEXT_DIR
from cmv_data import TextStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post-id", default="t3_37i8f6")
    args = parser.parse_args()
    store = TextStore(TEXT_DIR, expected_version="v1")
    post = store.get_post(args.post_id)
    comments = store.get_comments(args.post_id)
    print("Post:", post["post_id"])
    print("Title:", post["title"])
    print("Body preview:", (post["body"] or "")[:200])
    print("Comments:", len(comments))
    if comments:
        print("First comment:", comments[0]["comment_id"])
        print("Comment preview:", (comments[0]["body"] or "")[:200])


if __name__ == "__main__":
    main()
