"""Diagnose a stalled public source download without printing signed URLs."""
import json
import time
import urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
info = json.loads((ROOT / 'data/sources/mmmu/repository.json').read_text(encoding='utf-8'))
url = f'https://huggingface.co/datasets/MMMU/MMMU/resolve/{info["sha"]}/Agriculture/validation-00000-of-00001.parquet'
start = time.monotonic()
request = urllib.request.Request(url, headers={'Range': 'bytes=0-1023'})
with urllib.request.urlopen(request, timeout=30) as response:
    print('status', response.status, 'length', response.headers.get('Content-Length'),
          'range', response.headers.get('Content-Range'), flush=True)
    block = response.read(1024)
    print('bytes', len(block), 'seconds', round(time.monotonic() - start, 2), flush=True)
