"""Create portable manifests, a 200-question smoke subset, and file checksums."""
import json
import shutil
from data_common import ROOT, SOURCE, OUT, SEED, read_json, write_json, write_jsonl, sha256, stratified

COUNTS = {'mathvision': 304, 'mmmu': 200, 'mmlu_pro': 300, 'livecodebench': 100,
          'multimodalqa': 100, 'gpqa': 198}
# Proportional allocation across the six benchmarks, summing to 200.
SMOKE = {'mathvision': 51, 'mmmu': 33, 'mmlu_pro': 50, 'livecodebench': 17,
         'multimodalqa': 16, 'gpqa': 33}

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
        shutil.copyfile(SOURCE / name / 'README.md', OUT / 'manifests' / f'{name}.source_README.md')
    write_jsonl(OUT / 'questions' / 'all.jsonl', combined)
    write_jsonl(OUT / 'questions' / 'smoke_200.jsonl', smoke)
    write_json(OUT / 'manifests/smoke_200.ids.json', [r['id'] for r in smoke])
    write_json(OUT / 'suite.json', {'name': 'mini_eval_v1', 'schema_version': 1, 'seed': SEED,
                                   'total': len(combined), 'purpose': 'fixed development evaluation',
                                   'benchmarks': summary, 'smoke_total': len(smoke)})
    lock = {name: {'repo': read_json(SOURCE / name / 'repository.json')['id'],
                   'revision': read_json(SOURCE / name / 'repository.json')['sha']} for name in COUNTS}
    write_json(ROOT / 'config/sources.lock.json', lock)
    write_json(OUT / 'manifests/sources.lock.json', lock)
    source_checksums = {}
    for path in sorted(SOURCE.rglob('*')):
        if (path.is_file() and not path.name.endswith(('.part', '.assembled'))
                and not any(p.endswith('.chunks') for p in path.parts)):
            source_checksums[path.relative_to(SOURCE).as_posix()] = {
                'sha256': sha256(path), 'bytes': path.stat().st_size}
    write_json(OUT / 'manifests/source_checksums.json', source_checksums)
    checksums = {}
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name not in {'checksums.json', 'validation_report.json'}:
            checksums[path.relative_to(OUT).as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    write_json(OUT / 'checksums.json', checksums)
    print(json.dumps({'total': len(combined), 'smoke': len(smoke), 'files_hashed': len(checksums)}, indent=2))

if __name__ == '__main__':
    main()
