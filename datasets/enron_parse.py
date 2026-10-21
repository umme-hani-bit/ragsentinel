#!/usr/bin/env python3
from __future__ import annotations

import email
import email.policy
from email.utils import getaddresses, parsedate_to_datetime
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
CHECKSUMS_PATH = Path("configs/dataset_checksums.json")
CORPUS_YAML = Path("configs/corpus.yaml")


def _parse_corpus_yaml(path: Path) -> Tuple[str, str]:
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


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_checksums() -> Dict[str, str]:
    if not CHECKSUMS_PATH.exists():
        raise RuntimeError(
            f"Checksum file missing: {CHECKSUMS_PATH}. Run enron_download.py first."
        )
    try:
        return json.loads(CHECKSUMS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {CHECKSUMS_PATH}. Delete it or fix JSON."
        ) from exc


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_tar_if_needed(tar_path: Path, extract_root: Path) -> Path:
    """Extract the archive into a versioned directory and return its path."""
    version_name = tar_path.stem
    out_dir = extract_root / version_name
    maildir = out_dir / "maildir"
    if maildir.exists():
        print(f"Reusing existing extracted directory: {maildir}")
        return out_dir

    print(f"Extracting {tar_path} to {out_dir} (first time, this can take a minute)...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        root = out_dir.resolve()
        members = tar.getmembers()
        for member in members:
            member_path = (out_dir / member.name).resolve()
            try:
                member_path.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe path in tar: {member.name}") from exc
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsupported tar entry: {member.name}")
        tar.extractall(out_dir, members=members)
    print("Extraction complete.")
    return out_dir


def _iter_email_files(maildir_root: Path) -> Iterator[Path]:
    for path in maildir_root.rglob("*"):
        if path.is_file():
            yield path


def _parse_addresses(val: Optional[str]) -> List[str]:
    if not val:
        return []
    return [addr for _, addr in getaddresses([val]) if addr]


def _extract_body(msg: email.message.Message) -> str:
    """Prefer text/plain parts and decode safely, replacing invalid characters."""
    parts: List[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_type() == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    continue
    else:
        try:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                charset = msg.get_content_charset() or "utf-8"
                parts.append(payload.decode(charset, errors="replace"))
            else:
                parts.append(str(msg.get_payload()))
        except Exception:
            pass
    return "\n".join(parts).strip()


def _build_record(root_maildir: Path, file_path: Path) -> Optional[Dict[str, object]]:
    rel = file_path.relative_to(root_maildir).as_posix()
    parts = rel.split("/")
    if len(parts) < 2:
        return None
    employee_id = parts[0]
    folder_path = "/".join(parts[1:-1])
    raw = file_path.read_bytes()
    try:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
    except Exception:
        return None

    sender = (msg.get("From") or "").strip()
    recipients: List[str] = []
    for header in ("To", "Cc", "Bcc"):
        recipients.extend(_parse_addresses(msg.get(header)))
    seen = set()
    recipients = [addr for addr in recipients if not (addr in seen or seen.add(addr))]

    subject = (msg.get("Subject") or "").strip()
    date_iso = ""
    try:
        dt = parsedate_to_datetime(msg.get("Date"))
        if dt is not None:
            date_iso = dt.isoformat()
    except Exception:
        date_iso = ""

    body = _extract_body(msg)
    return {
        "doc_id": rel,
        "employee_id": employee_id,
        "sender": sender,
        "recipients": recipients,
        "date": date_iso,
        "subject": subject,
        "body": body,
        "folder_path": folder_path,
    }


def main() -> int:
    if not CORPUS_YAML.exists():
        print(
            f"ERROR: {CORPUS_YAML} not found. Run Phase 0 → Task 2 first.",
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

    tar_path = RAW_DIR / f"{dataset_version}.tgz"
    if not tar_path.exists():
        print(
            f"ERROR: Tarball missing: {tar_path}. "
            "Run datasets/enron_download.py first.",
            file=sys.stderr,
        )
        return 2

    checksums = _load_checksums()
    expected = checksums.get(dataset_version)
    if not expected:
        print(
            f"ERROR: No recorded checksum for {dataset_version} in "
            f"{CHECKSUMS_PATH}. Run downloader first.",
            file=sys.stderr,
        )
        return 3
    actual = _sha256_file(tar_path)
    if actual != expected:
        print(
            f"ERROR: Checksum mismatch.\n  expected: {expected}\n  actual:   {actual}",
            file=sys.stderr,
        )
        return 4

    out_dir = _extract_tar_if_needed(tar_path, RAW_DIR)
    maildir_root = out_dir / "maildir"
    if not maildir_root.exists():
        print(
            f"ERROR: Expected 'maildir' directory under {out_dir}, not found.",
            file=sys.stderr,
        )
        return 5

    _ensure_dirs()
    out_path = OUT_DIR / "enron_emails.jsonl"
    count = 0
    skipped = 0

    print(f"Parsing emails under: {maildir_root}")
    with out_path.open("w", encoding="utf-8") as out_f:
        for file_path in _iter_email_files(maildir_root):
            record = _build_record(maildir_root, file_path)
            if record is None:
                skipped += 1
                continue
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 5000 == 0:
                print(f"  wrote {count:,} records...")

    print(f"Done. Wrote {count:,} records to {out_path} (skipped {skipped:,}).")
    print("enron_parse.py finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
