import json 
from pathlib import Path 

def snapshot_sources(db, output_path):
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
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # prevent overwrite
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(source_map, file, indent=2)
        file.write("\n")
        
    return source_map