"""Create a portable bundle containing data, evaluator, docs, and launch scripts."""
from pathlib import Path
import hashlib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    archive = ROOT / "artifacts/qwen35_mini_eval_bundle.zip"
    archive.parent.mkdir(exist_ok=True)
    files = []
    for directory in ("data/mini_eval_v1", "mmeval", "docker", "tests", "licenses"):
        files.extend(path for path in (ROOT / directory).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    files.extend(path for path in (ROOT / "scripts").glob("*.sh"))
    files.extend(path for path in (ROOT / "scripts").glob("*.py"))
    files.extend([
        ROOT / "README.md", ROOT / "EVALUATION.md", ROOT / "DSPARK_SGLANG_SINGLE_GPU.md",
        ROOT / "requirements-eval.txt",
        ROOT / "requirements-server.txt", ROOT / "config/sources.lock.json",
        ROOT / ".dockerignore", ROOT / "THIRD_PARTY_NOTICES.md",
    ])
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(set(files)):
            output.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
    print(f"Package: {archive}")
    print(f"Bytes: {archive.stat().st_size:,}")
    print(f"SHA-256: {sha256(archive)}")


if __name__ == "__main__":
    main()
