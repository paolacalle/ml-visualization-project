import json 
from pathlib import Path 
import os
import pyarrow as pa


SOURCE_SCHEMA = pa.schema([
    pa.field("source_id", pa.int64(), nullable=False),
    pa.field("archive_member", pa.string(), nullable=False),
    pa.field("task", pa.string(), nullable=False),
    pa.field("original_split", pa.string(), nullable=False),
])


def map_sources(source_map):
    """Validate all six snapshot entries and retain their actual database IDs."""
    expected = {
        f"{task}/{split}_{kind}_data.jsonlist.bz2": (task, split)
        for task, kind in (("all", "period"), ("op_task", "op"), ("pair_task", "pair"))
        for split in ("train", "heldout")
    }
    if not isinstance(source_map, dict) or set(source_map) != set(expected):
        raise ValueError("Source map must contain exactly the six expected archive members.")
    seen = set()
    rows = []
    for member, (task, split) in sorted(expected.items()):
        source = source_map[member]
        if not isinstance(source, dict):
            raise ValueError(f"Invalid source entry for {member}")
        source_id = source.get("source_id")
        if type(source_id) is not int or not 1 <= source_id <= 2**63 - 1:
            raise ValueError(f"Invalid source_id for {member}: {source_id}")
        if source_id in seen:
            raise ValueError(f"Duplicate source_id: {source_id}")
        if source.get("task") != task or source.get("original_split") != split:
            raise ValueError(f"Source task/split does not match archive member {member}")
        seen.add(source_id)
        rows.append({"source_id": source_id, "archive_member": member,
                     "task": task, "original_split": split})
    return rows

def snapshot_sources(db, output_path):
    output_path = Path(output_path)
    
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            return json.load(file)
        
    rows = db.execute("select_sources.sql")
    
    if len(rows) != 6:
        raise ValueError(f"Expected six sources, found {len(rows)}.")


    # create a simple dict mapper
    source_map = {
        archive_member: {
            "source_id": source_id,
            "task": task,
            "original_split": split,
        }
        for source_id, archive_member, task, split in rows
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # prevent overwrite
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(source_map, file, indent=2)
        file.write("\n")
        
    return source_map
