import argparse
import tarfile
import bz2
import json


def inspect_dataset(archive_path, target_file, n=3):
    with tarfile.open(archive_path, "r:bz2") as tar:
        member = tar.extractfile(target_file)

        if member is None:
            raise FileNotFoundError(target_file)

        # view the structure of the files without opening them
        with bz2.open(member, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break

                record = json.loads(line)
                print(f"\n--- Record {i + 1} ---")
                print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", required=True)
    parser.add_argument(
        "-t", "--target",
        default="pair_task/train_pair_data.jsonlist.bz2"
    )
    parser.add_argument("-n", "--num", type=int, default=3)

    args = parser.parse_args()
    inspect_dataset(args.file, args.target, args.num)


if __name__ == "__main__":
    main()