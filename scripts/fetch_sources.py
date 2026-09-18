"""Download pinned public benchmark sources; never execute dataset code."""
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'sources'
LOCK = ROOT / 'config' / 'sources.lock.json'
REPOS = {
    'mathvision': 'MathLLMs/MathVision',
    'mmmu': 'MMMU/MMMU',
    'mmlu_pro': 'TIGER-Lab/MMLU-Pro',
    'livecodebench': 'livecodebench/code_generation_lite',
    'multimodalqa': 'allenai/multimodalqa',
    'gpqa': 'idavidrein/gpqa',
}

def fetch(url, dest):
    dest = Path(dest)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, part.open('wb') as out:
                expected = response.headers.get('Content-Length')
                while chunk := response.read(64 * 1024):
                    out.write(chunk)
            if expected is not None:
                assert part.stat().st_size == int(expected), f'Incomplete download: {dest.name}'
            part.replace(dest)
            print(f'Downloaded {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)', flush=True)
            return dest
        except Exception as exc:
            print(f'Retry {attempt + 1}: {dest.name}: {exc}', flush=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

def fetch_ranged(url, dest):
    """Use bounded range requests for large HF files; verify length of each chunk."""
    dest = Path(dest)
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        assert response.status == 206
        size = int(response.headers['Content-Range'].split('/')[-1])
        response.read()
    if dest.exists() and dest.stat().st_size == size:
        return dest
    chunks = dest.parent / (dest.name + '.chunks')
    chunks.mkdir(parents=True, exist_ok=True)
    chunk_size = 4 * 1024 * 1024
    def chunk(start):
        end = min(start + chunk_size, size) - 1
        part = chunks / str(start)
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
                    data = response.read()
                assert len(data) == end - start + 1
                part.write_bytes(data)
                print(f'{dest.name}: chunk {start // chunk_size + 1}/{(size + chunk_size - 1) // chunk_size}', flush=True)
                return part
            except Exception as exc:
                print(f'Chunk retry {dest.name} {start}: {exc}', flush=True)
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(chunk, range(0, size, chunk_size)))
    assembled = dest.with_name(dest.name + '.assembled')
    with assembled.open('wb') as out:
        for part in parts:
            with part.open('rb') as source:
                while block := source.read(1024 * 1024):
                    out.write(block)
    assert assembled.stat().st_size == size
    assembled.replace(dest)
    print(f'Downloaded {dest.name}: {size:,} bytes', flush=True)
    return dest

def initialize(name, repo):
    folder = CACHE / name
    lock = json.loads(LOCK.read_text(encoding='utf-8')) if LOCK.exists() else {}
    if name in {'gpqa', 'multimodalqa'}:
        if name not in lock:
            raise RuntimeError(f'{name} GitHub source must be pinned in config/sources.lock.json')
        revision = lock[name]['revision']
        filenames = (('README.md', 'LICENSE', 'dataset.zip') if name == 'gpqa' else
                     ('README.md', 'dataset/MMQA_dev.jsonl.gz', 'dataset/MMQA_images.jsonl.gz',
                      'dataset/MMQA_tables.jsonl.gz', 'dataset/MMQA_texts.jsonl.gz'))
        info = {'id': repo, 'sha': revision,
                'siblings': [{'rfilename': value} for value in filenames]}
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'repository.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        base = f'https://raw.githubusercontent.com/{repo}/{revision}'
        fetch(f'{base}/README.md', folder / 'README.md')
        if name == 'gpqa':
            fetch(f'{base}/LICENSE', folder / 'LICENSE')
        print(json.dumps({'name': name, 'repo': repo, 'revision': revision,
                          'files': [x['rfilename'] for x in info['siblings']] if '--inspect' in __import__('sys').argv else 3}), flush=True)
        return info
    endpoint = f'https://huggingface.co/api/datasets/{repo}'
    if name in lock:
        endpoint += '/revision/' + lock[name]['revision']
    api = fetch(endpoint, folder / 'repository.json')
    info = json.loads(api.read_text(encoding='utf-8'))
    revision = info['sha']
    if name in lock:
        assert revision == lock[name]['revision'], f'{name}: cached revision differs from lock'
    for filename in ['README.md']:
        fetch(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}', folder / filename)
    paths = [x['rfilename'] for x in info['siblings']]
    print(json.dumps({'name': name, 'repo': repo, 'revision': revision,
                      'files': paths if '--inspect' in __import__('sys').argv else len(paths)}, ensure_ascii=False), flush=True)
    return info

def verify_downloads():
    """Compare caches against the suite's original source hashes, when available."""
    manifest = ROOT / 'data/mini_eval_v1/manifests/source_checksums.json'
    if not manifest.exists():
        return
    expected = json.loads(manifest.read_text(encoding='utf-8'))
    for relative, item in expected.items():
        path = CACHE / relative
        assert path.is_file() and path.stat().st_size == item['bytes'], f'Bad source size: {relative}'
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert actual == item['sha256'], f'Bad source checksum: {relative}'
    print('Source caches match frozen SHA-256 checksums.', flush=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--repair-large', action='store_true')
    parser.add_argument('--only', choices=sorted(REPOS))
    args = parser.parse_args()
    selected = {args.only: REPOS[args.only]} if args.only else REPOS
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        infos = dict(zip(selected, pool.map(lambda pair: initialize(*pair), selected.items())))
    if not LOCK.exists():
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps({name: {'repo': REPOS[name], 'revision': info['sha']}
                                    for name, info in infos.items()}, indent=2) + '\n', encoding='utf-8')
    if args.inspect:
        return
    if args.repair_large:
        targets = [('mmmu', 'Agriculture/validation-00000-of-00001.parquet'),
                   ('livecodebench', 'test6.jsonl')]
        for name, path in targets:
            url = f'https://huggingface.co/datasets/{REPOS[name]}/resolve/{infos[name]["sha"]}/{path}'
            fetch_ranged(url, CACHE / name / path)
        verify_downloads()
        return
    jobs = []
    for name, info in infos.items():
        paths = [x['rfilename'] for x in info['siblings']]
        if name == 'mathvision':
            paths = [p for p in paths if 'testmini' in p and p.endswith('.parquet')]
        elif name == 'mmmu':
            paths = [p for p in paths if '/validation-' in p and p.endswith('.parquet')]
        elif name == 'mmlu_pro':
            paths = [p for p in paths if '/test-' in p and p.endswith('.parquet')]
        elif name == 'livecodebench':
            paths = ['code_generation_lite.py', 'test6.jsonl']
        elif name == 'multimodalqa':
            paths = [p for p in paths if p.startswith('dataset/')]
        elif name == 'gpqa':
            paths = ['dataset.zip']
        assert paths, name
        for path in paths:
            if name in {'gpqa', 'multimodalqa'}:
                url = f'https://raw.githubusercontent.com/{REPOS[name]}/{info["sha"]}/{path}'
            else:
                url = f'https://huggingface.co/datasets/{REPOS[name]}/resolve/{info["sha"]}/{urllib.parse.quote(path)}'
            jobs.append((url, CACHE / name / path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        def download(job):
            url, path = job
            if path.name in {'test6.jsonl'} or path.parent.name == 'Agriculture':
                return fetch_ranged(url, path)
            return fetch(url, path)
        for result in pool.map(download, jobs):
            pass
    verify_downloads()
    print('Source download complete.', flush=True)

if __name__ == '__main__':
    main()
