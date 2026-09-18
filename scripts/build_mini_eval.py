"""Build the frozen 498-question GPQA, MMMU, MultiModalQA suite."""
import ast
import csv
import hashlib
import gzip
import io
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))
import pyarrow.parquet as pq  # noqa: E402
from data_common import OUT, SOURCE, SEED, rank, save_image, source_info, stratified, write_json, write_suite  # noqa: E402

def parquet_rows(name, pattern):
    files = sorted((SOURCE / name).glob(pattern))
    assert files, (name, pattern)
    return [(path, row) for path in files for row in pq.read_table(path).to_pylist()]

def question(name, source_id, text, options, images, metadata, source):
    return {'id': f'{name}:{source_id}', 'benchmark': name, 'source_id': str(source_id),
            'source': source, 'question': text, 'options': options, 'images': images,
            'metadata': metadata}

def native_json(row):
    return {k: v for k, v in row.items()
            if not (isinstance(v, dict) and 'bytes' in v)}

def build_mmmu():
    entries = parquet_rows('mmmu', '*/validation-*.parquet')
    assert len(entries) == 900 and len({p.parent.name for p, r in entries}) == 30
    rows = [dict(r, _subject=p.parent.name) for p, r in entries]
    chosen, selection = stratified(rows, 200, lambda r: r['_subject'], lambda r: r['id'], 'mmmu')
    questions, references = [], []
    for row in chosen:
        sid = row['id']
        images, image_map = [], {}
        for i in range(1, 8):
            if row[f'image_{i}'] is not None:
                path = save_image(row[f'image_{i}'], f'assets/mmmu/{sid}_{i}')
                images.append(path)
                image_map[f'<image {i}>'] = path
        options = ast.literal_eval(row['options'])
        q = question('mmmu', sid, row['question'], options, images,
                     {'subject': row['_subject'], 'subfield': row['subfield'],
                      'difficulty': row['topic_difficulty'], 'question_type': row['question_type'],
                      'image_types': ast.literal_eval(row['img_type']), 'image_map': image_map},
                     source_info('mmmu', 'validation', row['_subject']))
        questions.append(q)
        references.append({'id': q['id'], 'answer': row['answer'],
                           'explanation': row['explanation'], 'native': native_json(row)})
    write_suite('mmmu', questions, references, selection, [r['id'] for r in rows])

def read_gzip_jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def format_mmqa_table(record):
    table = record['table']
    headers = [str(item['column_name']).replace('|', '\\|') for item in table['header']]
    rows = [[str(cell.get('text', '')).replace('|', '\\|').replace('\n', ' ') for cell in row]
            for row in table['table_rows']]
    lines = [f"Title: {record['title']}", f"Source: {record['url']}",
             '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n'.join(lines)


def build_multimodalqa():
    folder = SOURCE / 'multimodalqa'
    rows = read_gzip_jsonl(folder / 'dataset/MMQA_dev.jsonl.gz')
    # This replacement benchmark intentionally contains only genuine cross-modal
    # questions (two or more distinct required modalities).
    population = [row for row in rows if len(set(row['metadata']['modalities'])) >= 2]

    def stratum(row):
        modalities = '+'.join(sorted(row['metadata']['modalities']))
        return f"{modalities}|{row['metadata']['type']}"

    chosen, selection = stratified(population, 100, stratum, lambda row: row['qid'], 'multimodalqa')
    selection.update({'split': 'dev', 'setting': 'official_full_context_cross_modal',
                      'source_files': ['MMQA_dev.jsonl.gz', 'MMQA_texts.jsonl.gz',
                                       'MMQA_tables.jsonl.gz', 'MMQA_images.jsonl.gz'],
                      'note': 'Inputs include the official per-question full context with distractors; '
                              'supporting context, intermediate answers, and final answers remain reference-only.'})
    texts = {row['id']: row for row in read_gzip_jsonl(folder / 'dataset/MMQA_texts.jsonl.gz')}
    tables = {row['id']: row for row in read_gzip_jsonl(folder / 'dataset/MMQA_tables.jsonl.gz')}
    image_records = {row['id']: row for row in read_gzip_jsonl(folder / 'dataset/MMQA_images.jsonl.gz')}
    questions, references = [], []
    copied_images = {}
    for row in chosen:
        sid = row['qid']
        metadata = row['metadata']
        text_records = [texts[value] for value in metadata['text_doc_ids']]
        table_record = tables[metadata['table_id']]
        candidate_images = [image_records[value] for value in metadata['image_doc_ids']]
        context_sections = ['Official candidate text passages (supporting evidence and distractors are mixed):']
        for index, record in enumerate(text_records, 1):
            context_sections.append(
                f"[Text {index}] {record['title']}\nSource: {record['url']}\n{record['text']}")
        context_sections.append('Official candidate table:\n' + format_mmqa_table(table_record))
        context_sections.append('Official candidate images follow the textual prompt in numbered order:\n' +
                                '\n'.join(f"[Image {index}] {record['title']} — {record['url']}"
                                          for index, record in enumerate(candidate_images, 1)))
        images = []
        image_candidates = []
        for index, record in enumerate(candidate_images, 1):
            source_path = folder / 'images' / record['path']
            assert source_path.is_file(), source_path
            if record['id'] not in copied_images:
                copied_images[record['id']] = save_image(
                    source_path.read_bytes(), f"assets/multimodalqa/{record['id']}")
            images.append(copied_images[record['id']])
            image_candidates.append({'label': f'Image {index}', 'document_id': record['id'],
                                     'title': record['title'], 'url': record['url']})
        modalities = sorted(set(metadata['modalities']))
        q = question(
            'multimodalqa', sid, row['question'], [], images,
            {'question_type': metadata['type'], 'modalities': modalities,
             'modality_composition': '+'.join(modalities),
             'context_setting': 'official_full_context_with_distractors',
             'context_text': '\n\n'.join(context_sections),
             'image_candidates': image_candidates,
             'text_candidate_count': len(text_records), 'image_candidate_count': len(candidate_images),
             'table_document_id': table_record['id']},
            source_info('multimodalqa', 'dev', 'cross_modal_full_context'))
        questions.append(q)
        references.append({
            'id': q['id'], 'answers': [answer['answer'] for answer in row['answers']],
            'answer_records': row['answers'], 'supporting_context': row['supporting_context'],
            'intermediate_answers': metadata['intermediate_answers'],
            'wiki_entities_in_answers': metadata['wiki_entities_in_answers']})
    write_suite('multimodalqa', questions, references, selection, [row['qid'] for row in population])
    print(f'Copied {len(copied_images)} unique MultiModalQA candidate images', flush=True)

def build_gpqa():
    archive = SOURCE / 'gpqa/dataset.zip'
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.endswith('/gpqa_diamond.csv')]
        assert len(members) == 1, f'Expected one gpqa_diamond.csv, found {members}'
        password = os.environ.get('GPQA_ZIP_PASSWORD')
        if not password:
            raise RuntimeError('Set GPQA_ZIP_PASSWORD to rebuild the private GPQA source')
        with bundle.open(members[0], pwd=password.encode('utf-8')) as raw:
            rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline='')))
    assert len(rows) == 198, f'Expected 198 GPQA Diamond questions, found {len(rows)}'
    questions, references = [], []
    for row in rows:
        sid = row.get('Record ID') or hashlib.sha256(row['Question'].encode()).hexdigest()[:16]
        choices = [row['Correct Answer'], row['Incorrect Answer 1'],
                   row['Incorrect Answer 2'], row['Incorrect Answer 3']]
        assert choices.count(row['Correct Answer']) == 1, f'Non-unique GPQA correct answer: {sid}'
        permutation = sorted(range(4), key=lambda index: rank(f'gpqa:choices:{sid}', str(index)))
        shuffled = [choices[index] for index in permutation]
        answer_index = permutation.index(0)
        q = question('gpqa', sid, row['Question'], shuffled, [],
                     {'subset': 'diamond', 'domain': row.get('High-level domain', ''),
                      'subdomain': row.get('Subdomain', ''),
                      'choice_order': 'deterministic_sha256_shuffle',
                      'duplicate_distractor': len(set(choices[1:])) < 3},
                     source_info('gpqa', 'gpqa_diamond', 'diamond'))
        questions.append(q)
        references.append({'id': q['id'], 'answer': 'ABCD'[answer_index],
                           'answer_index': answer_index, 'answer_text': row['Correct Answer'],
                           'explanation': row.get('Explanation', ''),
                           'writer_difficulty': row.get("Writer's Difficulty Estimate", ''),
                           'expert_validator_accuracy': row.get('Expert Validator Accuracy', ''),
                           'non_expert_validator_accuracy': row.get('Non-Expert Validator Accuracy', '')})
    questions_and_refs = sorted(zip(questions, references), key=lambda pair: pair[0]['source_id'])
    questions = [pair[0] for pair in questions_and_refs]
    references = [pair[1] for pair in questions_and_refs]
    selection = {'method': 'official_gpqa_diamond_all', 'seed': SEED,
                 'choice_shuffle': 'per-question SHA256 rank',
                 'population_count': 198, 'selected_count': 198,
                 'population_strata': dict(Counter(q['metadata']['domain'] for q in questions))}
    write_suite('gpqa', questions, references, selection, [q['source_id'] for q in questions])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['gpqa', 'mmmu', 'multimodalqa'])
    args = parser.parse_args()
    builders = {'gpqa': build_gpqa, 'mmmu': build_mmmu, 'multimodalqa': build_multimodalqa}
    if args.only:
        builders[args.only]()
    else:
        for builder in builders.values():
            builder()

if __name__ == '__main__':
    main()
