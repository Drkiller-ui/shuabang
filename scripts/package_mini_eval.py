"""Package the validated data and scripts without the large download caches."""
import zipfile
from data_common import ROOT, OUT, read_json, sha256

def main():
    report = read_json(OUT / 'validation_report.json')
    assert report['status'] == 'passed' and report['questions'] == 498
    for relative, entry in read_json(OUT / 'checksums.json').items():
        assert sha256(OUT / relative) == entry['sha256'], relative
    archive = ROOT / 'artifacts/mini_eval_v1.zip'
    archive.parent.mkdir(exist_ok=True)
    files = [p for p in OUT.rglob('*') if p.is_file()]
    files += [p for p in (ROOT / 'scripts').glob('*.py')]
    files += [ROOT / 'README.md', ROOT / 'requirements-data.txt', ROOT / 'config/sources.lock.json']
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(files):
            output.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
    print(f'Package: {archive}')
    print(f'Bytes: {archive.stat().st_size:,}')
    print(f'SHA-256: {sha256(archive)}')

if __name__ == '__main__':
    main()
