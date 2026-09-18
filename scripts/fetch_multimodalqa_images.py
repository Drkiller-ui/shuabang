"""Range-fetch only the official MultiModalQA images needed by the fixed sample."""
from __future__ import annotations

import gzip
import io
import json
import concurrent.futures
import threading
import struct
import time
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

from data_common import SOURCE, stratified


ARCHIVE_URL = "https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip"
SOURCE_DIR = SOURCE / "multimodalqa"
BLOCK_SIZE = 512 * 1024


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_rows(rows: list[dict]) -> list[dict]:
    rows = [row for row in rows if len(set(row["metadata"]["modalities"])) >= 2]

    def stratum(row: dict) -> str:
        modalities = "+".join(sorted(row["metadata"]["modalities"]))
        return f"{modalities}|{row['metadata']['type']}"

    chosen, _ = stratified(rows, 100, stratum, lambda row: row["qid"], "multimodalqa")
    return chosen


class HTTPRangeReader(io.RawIOBase):
    """Seekable cached HTTP reader sufficient for Python's zipfile module."""

    def __init__(self, url: str, cache_dir: Path, max_memory_blocks: int = 8):
        self.url = url
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=90) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.max_memory_blocks = max_memory_blocks
        self.memory: OrderedDict[int, bytes] = OrderedDict()
        self.lock = threading.Lock()
        self.fetched_blocks = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.size)
        return self.position

    def _block(self, index: int) -> bytes:
        with self.lock:
            if index in self.memory:
                value = self.memory.pop(index)
                self.memory[index] = value
                return value
        start = index * BLOCK_SIZE
        end = min(self.size, start + BLOCK_SIZE) - 1
        path = self.cache_dir / f"{start:012d}-{end:012d}.bin"
        if path.is_file() and path.stat().st_size == end - start + 1:
            value = path.read_bytes()
        else:
            value = b""
            for attempt in range(5):
                try:
                    request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if response.status != 206:
                            raise RuntimeError(f"server ignored ZIP range request: HTTP {response.status}")
                        value = response.read()
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            if len(value) != end - start + 1:
                raise RuntimeError(f"incomplete ZIP range {start}-{end}: {len(value)} bytes")
            temporary = path.with_suffix(".part")
            temporary.write_bytes(value)
            temporary.replace(path)
            with self.lock:
                self.fetched_blocks += 1
                if self.fetched_blocks == 1 or self.fetched_blocks % 50 == 0:
                    print(f"Fetched {self.fetched_blocks} required image ZIP blocks", flush=True)
        with self.lock:
            self.memory[index] = value
            while len(self.memory) > self.max_memory_blocks:
                self.memory.popitem(last=False)
        return value

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        output = bytearray()
        while size:
            block_index = self.position // BLOCK_SIZE
            block_offset = self.position % BLOCK_SIZE
            block = self._block(block_index)
            take = min(size, len(block) - block_offset)
            output.extend(block[block_offset:block_offset + take])
            self.position += take
            size -= take
        return bytes(output)


def main() -> None:
    rows = select_rows(read_jsonl_gz(SOURCE_DIR / "dataset/MMQA_dev.jsonl.gz"))
    image_records = {row["id"]: row for row in read_jsonl_gz(SOURCE_DIR / "dataset/MMQA_images.jsonl.gz")}
    image_ids = sorted({image_id for row in rows for image_id in row["metadata"].get("image_doc_ids", [])})
    missing_metadata = set(image_ids) - set(image_records)
    if missing_metadata:
        raise RuntimeError(f"Missing image metadata for {sorted(missing_metadata)[:5]}")
    output = SOURCE_DIR / "images"
    output.mkdir(parents=True, exist_ok=True)
    pending = {image_records[image_id]["path"] for image_id in image_ids
               if not (output / image_records[image_id]["path"]).is_file()}
    if not pending:
        print(f"MultiModalQA images already present: {len(image_ids)}")
        return
    remote = HTTPRangeReader(ARCHIVE_URL, SOURCE_DIR / ".image_zip_ranges")
    with zipfile.ZipFile(remote) as archive:
        members = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        selected_infos = []
        for filename in pending:
            member = members.get(Path(filename).name)
            if not member:
                raise FileNotFoundError(f"{filename!r} is absent from official image archive")
            selected_infos.append(archive.getinfo(member))
        header_blocks = {info.header_offset // BLOCK_SIZE for info in selected_infos}
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
            list(pool.map(remote._block, sorted(header_blocks)))
        required_blocks = set(header_blocks)
        for info in selected_infos:
            remote.seek(info.header_offset)
            header = remote.read(30)
            signature, _, _, _, _, _, _, _, _, filename_length, extra_length = struct.unpack(
                "<IHHHHHIIIHH", header)
            if signature != 0x04034B50:
                raise RuntimeError(f"bad local ZIP header for {info.filename}")
            data_start = info.header_offset + 30 + filename_length + extra_length
            data_end = data_start + info.compress_size - 1
            required_blocks.update(range(data_start // BLOCK_SIZE, data_end // BLOCK_SIZE + 1))
        print(f"Prefetching {len(required_blocks)} exact ZIP blocks for {len(pending)} images", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(remote._block, sorted(required_blocks)))
        for index, filename in enumerate(sorted(pending), 1):
            member = members.get(Path(filename).name)
            if not member:
                raise FileNotFoundError(f"{filename!r} is absent from official image archive")
            target = output / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            print(f"Extracted MultiModalQA image {index}/{len(pending)}: {filename}", flush=True)
    print(f"MultiModalQA selected image set complete: {len(image_ids)} images", flush=True)


if __name__ == "__main__":
    main()
