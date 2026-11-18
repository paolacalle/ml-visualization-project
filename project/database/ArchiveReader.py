"""Read nested bzip2 JSON sources sequentially without extracting raw files.

See ``guides/READERS.md`` for an example that closes an iterator after sampling.
"""

import bz2
import json
import tarfile
from pathlib import Path

class ArchiveReader(): 
    """Stream records from the archive members named in a source snapshot.

    This reader decodes JSON but does not map fields, write files, or contact a
    database. One JSON record can include an entire discussion's comments.
    ``seen_members`` records recognized filenames, including empty sources.
    """
    
    def __init__(self, archive_path, source_map):
        """Store the archive path and filename-to-source dictionary; open nothing.

        ``source_map`` entries contain ``source_id``, ``task``, and
        ``original_split``. Validate a full six-source map with ``map_sources``
        before preparing a release; this reader also accepts a smaller map.
        """
        self.archive_path = archive_path
        self.source_map = source_map
        self.seen_members = set()
        
    def iter_records(self, task=None):
        """Yield ``(archive_member, physical_line_number, record_dict)`` tuples.

        Args:
            task: Optional source-task filter: ``all``, ``op_task``, or
                ``pair_task``. ``None`` reads every recognized source.

        Blank lines are skipped but still count toward one-based line numbers.
        The outer tar and each inner bzip2 stream are read sequentially. Duplicate
        recognized members, malformed JSON, and non-object records raise
        ``ValueError`` with source context. ``seen_members`` resets each pass.

        Exhaust or explicitly close this generator to release its file handles.
        A second call starts another archive pass; this is not random access.
        """
        self.seen_members.clear()
        with tarfile.open(self.archive_path, "r|bz2") as archive:
            for member in archive:
                
                if not member.isfile() or member.name not in self.source_map:
                    continue

                if member.name in self.seen_members:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                self.seen_members.add(member.name)
                
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

                            # yield information so controller can use 
                            # this data right away
                            yield member.name, line_number, record
