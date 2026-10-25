#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


IN_PATH = Path("data/processed/enron_emails.jsonl")
OUT_PATH = Path("data/processed/enron_emails_clean.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Tiny helpers (single-purpose)
# ---------------------------
def _normalize_newlines(text: str) -> str:
    # Unify CRLF/CR to LF; trim trailing spaces
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _strip_quoted_and_signature(text: str) -> str:
    """
    Conservative cleanup:
    - Drop lines starting with '>' (quoted reply).
    - Cut at common forward/reply headers (Original/Forwarded message).
    - Remove signature/disclaimer blocks starting with '--' or known phrases.
    """
    t = _normalize_newlines(text)

    # Cut everything after a common quoted/forwarded marker (first hit wins)
    markers = [
        "-----Original Message-----",
        "Original Message",
        "----- Forwarded by",
        "Forwarded by",
        "Begin forwarded message",
        "Forwarded message",
    ]
    for m in markers:
        idx = t.find(m)
        if idx != -1:
            t = t[:idx]
            break

    # Drop simple quoted lines
    lines = [ln for ln in t.split("\n") if not ln.lstrip().startswith(">")]
    t = "\n".join(lines)

    # Remove signatures/disclaimers from the tail if present
    sig_rx = re.compile(r"(?im)^(--\s?$|__+)$")
    disc_rx = re.compile(
        r"(?is)(confidentiality notice|this message.*confidential|this e-?mail.*confidential)"
    )

    m = None
    for i, ln in enumerate(lines):
        if sig_rx.match(ln):
            m = i
            break

    if m is not None:
        t = "\n".join(lines[:m]).strip()
    else:
        # Fallback: cut at disclaimer phrases if they appear near the end
        dm = disc_rx.search(t)
        if dm and dm.start() > max(0, len(t) - 2000):
            t = t[: dm.start()].strip()

    # Collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _sentence_split(text: str) -> List[str]:
    """
    Very simple sentence splitter: split on . ! ? followed by whitespace.
    Then trim and drop empties; also break very long sentences (>500 chars).
    """
    if not text:
        return []
    cand = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    for s in cand:
        s = s.strip()
        if not s:
            continue
        if len(s) > 500:
            # Soft split long sentences at semicolons/commas/spaces
            parts = re.split(r"([;,:])\s+", s)
            buf = ""
            for p in parts:
                if len(buf) + len(p) + 1 > 500:
                    if buf:
                        out.append(buf.strip())
                    buf = p
                else:
                    buf = (buf + " " + p).strip()
            if buf:
                out.append(buf.strip())
        else:
            out.append(s)
    return out


def _approx_token_count(text: str) -> int:
    # Approximate token count (words + punctuation tokens)
    return len(re.findall(r"\w+|[^\w\s]", text))


@dataclass
class Record:
    doc_id: str
    employee_id: str
    date: str
    subject: str
    raw_body: str
    clean_body: str
    sentences: List[str]
    token_count: int


def _iter_jsonl(path: Path, limit: int = 0) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
            if limit and i >= limit:
                break


def _process_row(row: Dict[str, object]) -> Record:
    doc_id = str(row.get("doc_id", ""))
    employee_id = str(row.get("employee_id", "") or "")
    date = str(row.get("date", "") or "")
    subject = str(row.get("subject", "") or "")
    raw_body = str(row.get("body", "") or "")

    clean = _strip_quoted_and_signature(raw_body)
    sents = _sentence_split(clean)
    toks = _approx_token_count(clean)

    return Record(
        doc_id=doc_id,
        employee_id=employee_id,
        date=date,
        subject=subject,
        raw_body=raw_body,
        clean_body=clean,
        sentences=sents,
        token_count=toks,
    )


def _write_parquet_stream(
    records: Iterable[Record], out_path: Path, batch_size: int = 5000
) -> Tuple[int, int]:
    """
    Stream records to a single Parquet file using ParquetWriter to avoid OOM.
    Returns (written_count, batches).
    """
    schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("employee_id", pa.string()),
            pa.field("date", pa.string()),
            pa.field("subject", pa.string()),
            pa.field("raw_body", pa.string()),
            pa.field("clean_body", pa.string()),
            pa.field("sentences", pa.list_(pa.string())),
            pa.field("token_count", pa.int32()),
        ]
    )

    writer: pq.ParquetWriter | None = None
    written = 0
    batches = 0

    def flush(batch: List[Record]) -> None:
        nonlocal writer, written, batches
        if not batch:
            return
        tbl = pa.table(
            {
                "doc_id": [r.doc_id for r in batch],
                "employee_id": [r.employee_id for r in batch],
                "date": [r.date for r in batch],
                "subject": [r.subject for r in batch],
                "raw_body": [r.raw_body for r in batch],
                "clean_body": [r.clean_body for r in batch],
                "sentences": pa.array(
                    [r.sentences for r in batch], type=pa.list_(pa.string())
                ),
                "token_count": [int(r.token_count) for r in batch],
            },
            schema=schema,
        )
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema=schema, compression="zstd")
        writer.write_table(tbl)
        written += len(batch)
        batches += 1
        print(f"  wrote batch {batches} (+{len(batch)} rows), total={written:,}")

    # Remove existing file to avoid mixing runs
    if out_path.exists():
        out_path.unlink()

    buf: List[Record] = []
    for rec in records:
        buf.append(rec)
        if len(buf) >= batch_size:
            flush(buf)
            buf = []
    if buf:
        flush(buf)
    if writer is not None:
        writer.close()
    return written, batches


def main() -> int:
    if not IN_PATH.exists():
        print(f"ERROR: input JSONL not found: {IN_PATH}. Run enron_parse.py first.")
        return 2

    limit = int(os.getenv("PREP_LIMIT", "0"))
    print(
        f"Starting clean + prep. Input={IN_PATH}, Output={OUT_PATH}, PREP_LIMIT={limit or 'none'}"
    )

    gen = (_process_row(row) for row in _iter_jsonl(IN_PATH, limit=limit))
    written, batches = _write_parquet_stream(gen, OUT_PATH, batch_size=5000)

    print(f"Done. Wrote {written:,} rows in {batches} batches to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
