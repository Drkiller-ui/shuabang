"""Offline integrity validation; does not execute any benchmark program."""
import json
import re
from collections import Counter
from PIL import Image
from data_common import OUT, read_json, write_json, sha256
from finalize_mini_eval import COUNTS, read_jsonl

def require(condition, message):
    if not condition:
        raise ValueError(message)

def local_path(relative):
    path = (OUT / relative).resolve()
    require(path.is_relative_to(OUT.resolve()), f'Unsafe path: {relative}')
    require(path.is_file(), f'Missing file: {relative}')
    return path

def main():
    expected_questions = set(COUNTS) | {'all', 'smoke_200'}
    actual_questions = {path.stem for path in (OUT / 'questions').glob('*.jsonl')}
    require(actual_questions == expected_questions, f'Unexpected question files: {actual_questions ^ expected_questions}')
    actual_references = {path.stem for path in (OUT / 'references').glob('*.jsonl')}
    require(actual_references == set(COUNTS), f'Unexpected reference files: {actual_references ^ set(COUNTS)}')
    actual_asset_dirs = {path.name for path in (OUT / 'assets').iterdir() if path.is_dir()}
    require(actual_asset_dirs == {'mmmu', 'multimodalqa'}, f'Unexpected asset directories: {actual_asset_dirs}')
    allowed_manifest_prefixes = set(COUNTS) | {'smoke_200', 'source_checksums', 'sources'}
    require(all(path.name.split('.', 1)[0] in allowed_manifest_prefixes
                for path in (OUT / 'manifests').iterdir() if path.is_file()),
            'Unexpected benchmark manifest')
    totals = Counter()
    image_paths = set()
    all_ids = set()
    report = {'status': 'passed', 'benchmarks': {}}
    for name, expected in COUNTS.items():
        rows = read_jsonl(OUT / 'questions' / f'{name}.jsonl')
        refs = read_jsonl(OUT / 'references' / f'{name}.jsonl')
        require(len(rows) == len(refs) == expected, f'{name}: wrong count')
        require([r['id'] for r in rows] == [r['id'] for r in refs], f'{name}: answer join mismatch')
        selected = read_json(OUT / 'manifests' / f'{name}.json')
        require(selected['selected_ids'] == [r['source_id'] for r in rows], f'{name}: ID manifest mismatch')
        unused = read_json(OUT / 'manifests' / f'{name}.unused_ids.json')
        require(not set(unused).intersection(selected['selected_ids']), f'{name}: unused IDs overlap')
        hashes = set()
        for q, ref in zip(rows, refs):
            require(q['id'] not in all_ids, f'Duplicate ID: {q["id"]}')
            all_ids.add(q['id'])
            require(q['question'].strip(), f'Empty question: {q["id"]}')
            require(isinstance(q['options'], list), f'Options type: {q["id"]}')
            require(not {'answer', 'solution', 'native', 'tests_file', 'cot_content'} & q.keys(), 'Answer leakage')
            require(re.fullmatch(r'[0-9a-f]{40}', q['source']['revision']), 'Unpinned revision')
            fingerprint = (q['question'], tuple(q['options']), tuple(q['images']))
            require(fingerprint not in hashes, f'Duplicate content: {q["id"]}')
            hashes.add(fingerprint)
            image_map = q['metadata'].get('image_map', {})
            for token in re.findall(r'<image\s*\d+>', q['question'] + '\n' + '\n'.join(q['options'])):
                require(token in image_map, f'Unmapped image token {q["id"]}: {token}')
            for relative in q['images']:
                path = local_path(relative)
                with Image.open(path) as image:
                    image.verify()
                image_paths.add(relative)
            if name in {'mmmu', 'multimodalqa'}:
                require(bool(q['images']), f'No image: {q["id"]}')
            if name in {'mmmu', 'gpqa'}:
                require(ref['answer'] is not None and str(ref['answer']).strip(), f'No answer: {q["id"]}')
                if q['options'] and (name != 'mmmu' or q['metadata']['question_type'] == 'multiple-choice'):
                    require(ref['answer'] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(q['options'])], f'Bad MC answer: {q["id"]}')
            if name == 'gpqa':
                require(len(q['options']) == 4, f'GPQA choice count: {q["id"]}')
                require(ord(ref['answer']) - ord('A') == ref['answer_index'], 'GPQA answer mismatch')
                require(q['options'][ref['answer_index']] == ref['answer_text'], 'GPQA shuffled answer mismatch')
            if name == 'multimodalqa':
                require(len(set(q['metadata']['modalities'])) >= 2, f'Not cross-modal: {q["id"]}')
                require(q['metadata']['context_setting'] == 'official_full_context_with_distractors',
                        f'Wrong MMQA context setting: {q["id"]}')
                require(q['metadata']['image_candidate_count'] == len(q['images']) > 0,
                        f'Bad MMQA image candidates: {q["id"]}')
                require(bool(ref['answers']) and all(str(value).strip() for value in ref['answers']),
                        f'No MMQA answer: {q["id"]}')
                serialized = json.dumps(q, ensure_ascii=False)
                for forbidden in ('supporting_context', 'intermediate_answers', 'wiki_entities_in_answers'):
                    require(forbidden not in serialized, f'MMQA reference leakage {forbidden}: {q["id"]}')
        report['benchmarks'][name] = {'questions': len(rows), 'unused_source_ids': len(unused)}
        totals['questions'] += len(rows)
    combined = read_jsonl(OUT / 'questions/all.jsonl')
    smoke = read_jsonl(OUT / 'questions/smoke_200.jsonl')
    require(len(combined) == sum(COUNTS.values()) and {r['id'] for r in combined} == all_ids, 'Combined suite mismatch')
    combined_by_id = {r['id']: r for r in combined}
    require(len(smoke) == 200 and len({r['id'] for r in smoke}) == 200, 'Smoke size mismatch')
    require(all(r == combined_by_id.get(r['id']) for r in smoke), 'Smoke differs from main suite')
    for relative, expected in read_json(OUT / 'checksums.json').items():
        path = local_path(relative)
        require(path.stat().st_size == expected['bytes'] and sha256(path) == expected['sha256'], f'Checksum mismatch: {relative}')
    report.update(dict(totals))
    report['unique_input_images'] = len(image_paths)
    report['smoke_questions'] = len(smoke)
    report['limitations'] = ['No model inference or benchmark scoring was performed.',
                             'Custom subsets are development evaluations, not full leaderboard scores.']
    write_json(OUT / 'validation_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
