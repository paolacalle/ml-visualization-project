import bz2
import json
import tarfile
from pathlib import Path

class ArchiveReader(): 
    """Stream JSON records from the CMV archive"""
    
    def __init__(self, archive_path, source_map):
        self.archive_path = archive_path
        self.source_map = source_map
        
    def iter_records(self, task=None):
        with tarfile.open(self.archive_path, "r|bz2") as archive:
            for member in archive:
                
                if not member.isfile() or member.name not in self.source_map:
                    continue
                
                source = self.source_map[member.name]
                
                if task is not None and source["task"] != task:
                    continue
                
                with archive.extractfile(member) as compressed:
                    with bz2.BZ2File(compressed, "rb") as lines:
                        
                        for line_number, line in enumerate(lines, start=1):
                            if not line.strip():
                                continue
                            
                            try:
                                # unpack the line
                                record = json.loads(line)
                            except (
                                json.JSONDecodeError,
                                UnicodeDecodeError,
                            ) as exc:
                                raise ValueError(
                                    f"{member.name}, line {line_number}: {exc}"
                                ) from exc
                                
                            if not isinstance(record, dict):
                                raise ValueError(
                                    f"{member.name}, line {line_number}: "
                                    "expected a JSON object"
                                )

                            # yield specific record
                            yield member.name, line_number, record