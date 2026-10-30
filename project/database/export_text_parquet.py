"""Stream CMV post/comment text into versioned Parquet files; no database connection."""
import argparse

import bz2
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_DIR / 'data/raw/cmv.tar.bz2'
DEFAULT_OUTPUT = PROJECT_DIR / 'data/processed/cmv-text/v1'
BUFFER_BYTES = 24 * 1024 * 1024
MAX_FILE_BYTES = 50_000_000

POST_SCHEMA = pa.schema([
    pa.field('post_id', pa.string(), nullable=False),
    pa.field('title', pa.string()),
    pa.field('body', pa.string()),
])
COMMENT_SCHEMA = pa.schema([
    pa.field('comment_id', pa.string(), nullable=False),
    pa.field('post_id', pa.string(), nullable=False),
    pa.field('body', pa.string()),
])


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def rows_checksum(rows, key):
    payload = json.dumps(sorted(rows, key=lambda row: row[key]),
                         ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class Shards:
    def __init__(self, root, kind, split, schema, files):
        self.root = root
        self.folder = Path(kind) / f'split={split}'
        (self.root / self.folder).mkdir(parents=True)
        self.schema = schema
        self.files = files
        self.buffers = []
        self.nbytes = 0
        self.part = 0

    def add(self, rows):
        if not rows:
            return None
        table = pa.Table.from_pylist(rows, schema=self.schema)

        if self.buffers and self.nbytes + table.nbytes > BUFFER_BYTES:
            self.flush()
            
        relative = (self.folder / f'part-{self.part:04d}.parquet').as_posix()
        self.buffers.append(table)
        self.nbytes += table.nbytes
        
        return relative

    def flush(self):
        if not self.buffers:
            return
        table = pa.concat_tables(self.buffers)
        sorting = [('post_id', 'ascending')]
        if 'comment_id' in table.column_names:
            sorting.append(('comment_id', 'ascending'))
        table = table.sort_by(sorting)
        relative = (self.folder / f'part-{self.part:04d}.parquet').as_posix()
        path = self.root / relative
        pq.write_table(table, path, compression='zstd', compression_level=9,
                       row_group_size=4096, use_dictionary=['post_id'],
                       write_statistics=['post_id'])
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f'File exceeds upload limit: {relative} ({size} bytes)')
        self.files.append({'path': relative, 'rows': table.num_rows,
                           'bytes': size, 'sha256': checksum(path)})
        self.part += 1
        self.buffers = []
        self.nbytes = 0


def export_text(archive_path, output, verify_post='t3_37i8f6'):
    archive_path = Path(archive_path).resolve()
    output = Path(output).resolve()
    partial = output.with_name(output.name + '.partial')
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if output.exists() or partial.exists():
        raise FileExistsError(
            f'Refusing to overwrite {output} or {partial}; choose a new --output path.'
        )
    partial.mkdir(parents=True)
    files, index, sources = [], {}, []
    expected_target = None

    with tarfile.open(archive_path, 'r|bz2') as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith('all/'):
                continue
            
            if not member.name.endswith('.jsonlist.bz2'):
                continue
            
            split = member.name.rsplit('/', 1)[-1].split('_', 1)[0]
            
            if split not in ('train', 'heldout'):
                raise ValueError(member.name)
            
            posts = Shards(partial, 'posts', split, POST_SCHEMA, files)
            comments = Shards(partial, 'comments', split, COMMENT_SCHEMA, files)
            post_count = comment_count = 0
            
            print(f'Exporting {member.name}', flush=True)
            
            with archive.extractfile(member) as compressed, bz2.BZ2File(compressed) as lines:
                for line in lines:
                    if not line.strip():
                        continue
                    
                    record = json.loads(line)
                    post_id = record['name']
                    
                    if post_id in index:
                        raise ValueError(f'Duplicate post ID: {post_id}')
                    
                    post_rows = [{'post_id': post_id, 'title': record.get('title'),
                                  'body': record.get('selftext')}]
                    
                    comment_rows = [
                        {'comment_id': c.get('name') or f"t1_{c['id']}",
                         'post_id': post_id, 'body': c.get('body')}
                        for c in record.get('comments', [])
                    ]
                    
                    if len({c['comment_id'] for c in comment_rows}) != len(comment_rows):
                        raise ValueError(f'Duplicate comment IDs in {post_id}')
                    index[post_id] = {
                        'split': split,
                        'post_file': posts.add(post_rows),
                        'comment_file': comments.add(comment_rows),
                        'comment_count': len(comment_rows),
                    }
                    
                    if post_id == verify_post:
                        expected_target = {
                            'post_sha256': rows_checksum(post_rows, 'post_id'),
                            'comments_sha256': rows_checksum(comment_rows, 'comment_id'),
                        }
                        
                    post_count += 1
                    comment_count += len(comment_rows)
                    
                    if post_count % 2000 == 0:
                        print(f'  {post_count:,} posts / {comment_count:,} comments', flush=True)
                        
            posts.flush()
            comments.flush()
            sources.append({'archive_member': member.name, 'split': split,
                            'posts': post_count, 'comments': comment_count})
            print(f'Finished {split}: {post_count:,} posts / {comment_count:,} comments', flush=True)

    if len(sources) != 2 or {s['split'] for s in sources} != {'train', 'heldout'}:
        raise ValueError('Expected both discussion sources')
    if expected_target is None:
        raise ValueError('Retrieval-test discussion not found')

    # Verify metadata for every shard and retrieve the selected discussion using
    # its small sidecar index, reading only the corresponding two files
    
    totals = Counter()
    for file in files:
        metadata = pq.read_metadata(partial / file['path'])
        assert metadata.num_rows == file['rows']
        totals[file['path'].split('/')[0]] += metadata.num_rows
        
    assert totals['posts'] == sum(s['posts'] for s in sources)
    assert totals['comments'] == sum(s['comments'] for s in sources)

    location = index[verify_post]
    actual_post = pq.read_table(partial / location['post_file'],
                               columns=POST_SCHEMA.names,
                               filters=[('post_id', '=', verify_post)]).to_pylist()
    actual_comments = pq.read_table(partial / location['comment_file'],
                                   columns=COMMENT_SCHEMA.names,
                                   filters=[('post_id', '=', verify_post)]).to_pylist() if location['comment_file'] else []
    assert rows_checksum(actual_post, 'post_id') == expected_target['post_sha256']
    assert rows_checksum(actual_comments, 'comment_id') == expected_target['comments_sha256']
    assert len(actual_comments) == location['comment_count']

    index_path = partial / 'discussion_index.json'
    index_path.write_text(json.dumps(index, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {
        'version': output.name, 'compression': 'zstd', 'compression_level': 9,
        'pyarrow_version': pa.__version__,
        'raw_archive': {'name': archive_path.name, 'bytes': archive_path.stat().st_size,
                        'sha256': checksum(archive_path)},
        'schemas': {'posts': str(POST_SCHEMA), 'comments': str(COMMENT_SCHEMA)},
        'sources': sources, 'files': files,
        'discussion_index': {'path': index_path.name, 'bytes': index_path.stat().st_size,
                             'sha256': checksum(index_path)},
        'verification': {'post_rows': totals['posts'], 'comment_rows': totals['comments'],
                         'all_shards_below_50_MB': True,
                         'retrieval_test': {'post_id': verify_post, 'comments': len(actual_comments),
                                            'exact_text_match': True}},
    }
    (partial / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    total_bytes = sum(path.stat().st_size for path in partial.rglob('*') if path.is_file())
    partial.rename(output)
    print(json.dumps({'output': str(output), 'post_rows': totals['posts'],
                      'comment_rows': totals['comments'], 'parquet_files': len(files),
                      'parquet_bytes': sum(f['bytes'] for f in files),
                      'largest_file_bytes': max(f['bytes'] for f in files),
                      'index_bytes': index_path.stat().st_size if index_path.exists()
                                     else (output / index_path.name).stat().st_size,
                      'total_bytes_including_manifest_and_index': total_bytes,
                      'retrieval_test': f'Exact match of post and all {len(actual_comments)} comment texts'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='New output directory; existing directories are never overwritten')
    parser.add_argument('--verify-post', default='t3_37i8f6',
                        help='Post ID whose exported text must match the source exactly')
    args = parser.parse_args()
    export_text(args.archive, args.output, args.verify_post)


if __name__ == '__main__':
    main()
