#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

RAW_DIR = Path("data/raw")
CHECKSUMS_PATH = Path("configs/dataset_checksums.json")
CORPUS_YAML = Path("configs/corpus.yaml")

DEFAULT_ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHECKSUMS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _parse_corpus_yaml(path: Path) -> Tuple[str, str]:
    """Parse dataset and dataset_version without requiring PyYAML."""
    dataset = ""
    dataset_version = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("dataset:"):
            dataset = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("dataset_version:"):
            dataset_version = line.split(":", 1)[1].strip().strip('"').strip("'")
    if not dataset or not dataset_version:
        raise RuntimeError(f"Could not find dataset/dataset_version in {path}")
    return dataset, dataset_version


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checksums() -> Dict[str, str]:
    if not CHECKSUMS_PATH.exists():
        return {}
    try:
        return json.loads(CHECKSUMS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {CHECKSUMS_PATH}. Delete it or fix JSON."
        ) from exc


def _save_checksums(d: Dict[str, str]) -> None:
    CHECKSUMS_PATH.write_text(
        json.dumps(d, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _download(url: str, dest: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=dest.parent,
        prefix=f"{dest.name}.",
        suffix=".part",
        delete=False,
    ) as temp_file:
        tmp = Path(temp_file.name)

    attempts = 0
    max_attempts = 3
    current_url = url
    try:
        while attempts < max_attempts:
            attempts += 1
            try:
                print(
                    f"Downloading (attempt {attempts}/{max_attempts}):\n"
                    f" URL: {current_url}\n Dest: {dest}"
                )
                request = urllib.request.Request(
                    current_url,
                    headers={
                        "User-Agent": os.getenv(
                            "ENRON_UA",
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                        ),
                        "Accept": "*/*",
                    },
                )
                with urllib.request.urlopen(request, timeout=30) as r, tmp.open(
                    "wb"
                ) as f:
                    total = int(r.headers.get("Content-Length", "0") or 0)
                    downloaded = 0
                    block = 1024 * 1024
                    last_percent = -1
                    while True:
                        chunk = r.read(block)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = int(downloaded * 100 / total)
                            if percent != last_percent:
                                sys.stdout.write(
                                    f"\r {downloaded:,}/{total:,} bytes ({percent:3d}%)"
                                )
                                sys.stdout.flush()
                                last_percent = percent
                if total:
                    print()
                tmp.replace(dest)
                print("Download complete.")
                return
            except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout) as exc:
                print(f"Download failed: {exc}")
                if current_url.startswith("https://") and os.getenv("ENRON_URL") is None:
                    current_url = "http://" + current_url[len("https://") :]
                    print(f"Retrying with HTTP fallback: {current_url}")
                    time.sleep(2)
                    continue
                if attempts < max_attempts:
                    wait = 2 * attempts
                    print(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print("Giving up after retries.")
                    raise
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    _ensure_dirs()
    if not CORPUS_YAML.exists():
        print(
            f"ERROR: {CORPUS_YAML} not found. Did you complete Phase 0 → Task 2?",
            file=sys.stderr,
        )
        return 2

    dataset, dataset_version = _parse_corpus_yaml(CORPUS_YAML)
    if dataset != "enron":
        print(
            f"ERROR: corpus.yaml dataset='{dataset}' (expected 'enron')",
            file=sys.stderr,
        )
        return 2

    url = os.getenv("ENRON_URL", DEFAULT_ENRON_URL)
    dest = RAW_DIR / f"{dataset_version}.tgz"

    checksums = _load_checksums()
    expected = checksums.get(dataset_version)

    if dest.exists():
        print(f"File already present: {dest}")
    else:
        _download(url, dest)

    actual = _sha256_file(dest)
    print(f"SHA256: {actual}")

    if expected:
        if actual != expected:
            print(
                f"ERROR: Checksum mismatch for {dataset_version}!\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}",
                file=sys.stderr,
            )
            return 3
        print("Checksum matches previously recorded value.")
    else:
        print(
            f"No recorded checksum for {dataset_version}. "
            "Recording this one for reproducibility."
        )
        checksums[dataset_version] = actual
        _save_checksums(checksums)
        print(f"Wrote checksum to {CHECKSUMS_PATH}")

    try:
        with tarfile.open(dest, "r:gz") as tar:
            names = [m.name for m in tar.getmembers() if "/" not in m.name]
            print(f"Tar OK. Top-level entries: {len(names)}")
    except tarfile.TarError as exc:
        print(f"ERROR: Tarball appears corrupt: {exc}", file=sys.stderr)
        return 4

    print("enron_download.py finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
