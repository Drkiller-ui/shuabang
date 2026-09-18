"""Create portable manifests, a 200-question smoke subset, and file checksums."""
import json
from data_common import ROOT, SOURCE, OUT, SEED, read_json, write_json, write_jsonl, sha256, stratified

COUNTS = {'gpqa': 198, 'mmmu': 200, 'multimodalqa': 100}
# Proportional allocation across the three benchmarks, summing to 200.
SMOKE = {'gpqa': 80, 'mmmu': 80, 'multimodalqa': 40}
PRIVATE_OUTPUTS = {
    'questions/gpqa.jsonl', 'references/gpqa.jsonl',
    'questions/all.jsonl', 'questions/smoke_200.jsonl',
    'manifests/gpqa.json', 'manifests/gpqa.source_README.md',
    'manifests/gpqa.unused_ids.json', 'manifests/smoke_200.ids.json',
}

def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]

def main():
    combined, smoke, summary = [], [], {}
    for name, count in COUNTS.items():
        rows = read_jsonl(OUT / 'questions' / f'{name}.jsonl')
        assert len(rows) == count
        combined.extend(rows)
        def key(row):
            m = row['metadata']
            return m.get('subject', m.get('category', m.get('difficulty', m.get('modality_composition'))))
        chosen, report = stratified(rows, SMOKE[name], key, lambda r: r['id'], 'smoke:' + name)
        smoke.extend(chosen)
        summary[name] = {'count': len(rows), 'smoke_count': len(chosen),
                         'source': rows[0]['source'], 'image_count': sum(len(r['images']) for r in rows),
                         'question_file': f'questions/{name}.jsonl',
                         'reference_file': f'references/{name}.jsonl'}
    write_jsonl(OUT / 'questions' / 'all.jsonl', combined)
    write_jsonl(OUT / 'questions' / 'smoke_200.jsonl', smoke)
    (OUT / 'questions/all.jsonl').chmod(0o600)
    (OUT / 'questions/smoke_200.jsonl').chmod(0o600)
    write_json(OUT / 'manifests/smoke_200.ids.json', [r['id'] for r in smoke])
    write_json(OUT / 'suite.json', {'name': 'mini_eval_v1', 'schema_version': 1, 'seed': SEED,
                                   'total': len(combined), 'purpose': 'fixed development evaluation',
                                   'benchmarks': summary, 'smoke_total': len(smoke)})
    old_lock = read_json(ROOT / 'config/sources.lock.json')
    lock = {name: old_lock[name] for name in COUNTS}
    write_json(ROOT / 'config/sources.lock.json', lock)
    write_json(OUT / 'manifests/sources.lock.json', lock)
    prior_source_checksums = read_json(OUT / 'manifests/source_checksums.json')
    source_checksums = {name: value for name, value in prior_source_checksums.items()
                        if name.split('/', 1)[0] in COUNTS and not name.startswith('gpqa/')}
    for path in sorted(SOURCE.rglob('*')):
        if path.is_file() and path.relative_to(SOURCE).parts[0] in {'mmmu', 'multimodalqa'} \
                and not path.name.endswith(('.part', '.assembled')) \
                and not any(part.endswith('.chunks') for part in path.parts):
            source_checksums[path.relative_to(SOURCE).as_posix()] = {
                'sha256': sha256(path), 'bytes': path.stat().st_size}
    write_json(OUT / 'manifests/source_checksums.json', source_checksums)
    checksums = {}
    for path in sorted(OUT.rglob('*')):
        relative = path.relative_to(OUT).as_posix()
        if path.is_file() and relative not in PRIVATE_OUTPUTS \
                and path.name not in {'checksums.json', 'validation_report.json'}:
            checksums[relative] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    write_json(OUT / 'checksums.json', checksums)
    print(json.dumps({'total': len(combined), 'smoke': len(smoke), 'files_hashed': len(checksums)}, indent=2))

if __name__ == '__main__':
    main()
