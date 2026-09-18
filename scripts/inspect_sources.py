"""Print schemas and short metadata only, without dumping images or test cases."""
import json
import gzip
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))
import pyarrow.parquet as pq  # noqa: E402

def short(value):
    if isinstance(value, bytes):
        return f'<{len(value)} bytes>'
    if isinstance(value, dict):
        return {k: short(v) for k, v in value.items()}
    return str(value)[:500]

for name in ['mathvision', 'mmmu', 'mmlu_pro']:
    files = list((ROOT / 'data/sources' / name).rglob('*.parquet'))
    if files:
        table = pq.read_table(files[0])
        print(name, files[0].name, table.num_rows)
        print(json.dumps(short(table.slice(0, 1).to_pylist()[0]), ensure_ascii=False))
path = ROOT / 'data/sources/livecodebench/test6.jsonl'
if path.is_file():
    rows = [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]
    print('livecodebench', len(rows))
    print(json.dumps(short(rows[0]), ensure_ascii=False))
    print('dates', min(r['contest_date'] for r in rows), max(r['contest_date'] for r in rows))
path = ROOT / 'data/sources/multimodalqa/dataset/MMQA_dev.jsonl.gz'
if path.is_file():
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    cross_modal = [row for row in rows if len(set(row['metadata']['modalities'])) >= 2]
    print('multimodalqa', len(rows), 'cross_modal', len(cross_modal))
    print(json.dumps(short(cross_modal[0]), ensure_ascii=False))
