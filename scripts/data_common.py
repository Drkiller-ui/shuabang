"""Deterministic sampling and portable local benchmark files."""
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/sources'
OUT = ROOT / 'data/mini_eval_v1'
SEED = 20260916

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def rank(namespace, value):
    return hashlib.sha256(f'{SEED}|{namespace}|{value}'.encode()).hexdigest()

def stratified(rows, n, key, identifier, namespace):
    """Proportional largest-remainder allocation; hash ranks independent of row order."""
    assert 0 < n <= len(rows)
    assert len({identifier(r) for r in rows}) == len(rows), 'Duplicate source IDs'
    groups = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    quotas = {k: n * len(v) // len(rows) for k, v in groups.items()}
    remaining = n - sum(quotas.values())
    order = sorted(groups, key=lambda k: (-(n * len(groups[k]) % len(rows)), rank(namespace + ':quota', k)))
    for k in order[:remaining]:
        quotas[k] += 1
    chosen = []
    for k in sorted(groups):
        chosen.extend(sorted(groups[k], key=lambda r: rank(namespace, identifier(r)))[:quotas[k]])
    chosen.sort(key=lambda r: str(identifier(r)))
    assert len(chosen) == n
    report = {'method': 'proportional_largest_remainder_sha256', 'seed': SEED,
              'namespace': namespace, 'population_count': len(rows), 'selected_count': n,
              'population_strata': {k: len(v) for k, v in sorted(groups.items())},
              'selected_strata': dict(sorted(quotas.items()))}
    return chosen, report

def source_info(name, split, config=None):
    info = read_json(SOURCE / name / 'repository.json')
    return {'repo': info['id'], 'revision': info['sha'], 'split': split, 'config': config}

def save_image(raw, relative):
    if isinstance(raw, dict):
        raw = raw['bytes']
    assert raw, relative
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
        fmt = image.format
    extension = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp', 'GIF': '.gif'}.get(fmt)
    assert extension, fmt
    relative = Path(relative).with_suffix(extension)
    path = OUT / relative
    assert path.resolve().is_relative_to(OUT.resolve()), 'Image path escapes output directory'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return relative.as_posix()

def write_suite(name, questions, references, selection, population_ids):
    assert len(questions) == len(references)
    assert [r['id'] for r in questions] == [r['id'] for r in references]
    write_jsonl(OUT / 'questions' / f'{name}.jsonl', questions)
    write_jsonl(OUT / 'references' / f'{name}.jsonl', references)
    selection = dict(selection, source=questions[0]['source'],
                     selected_ids=[q['source_id'] for q in questions])
    write_json(OUT / 'manifests' / f'{name}.json', selection)
    selected_ids = {q['source_id'] for q in questions}
    write_json(OUT / 'manifests' / f'{name}.unused_ids.json', sorted(set(population_ids) - selected_ids))
    print(f'Built {name}: {len(questions)} questions', flush=True)
