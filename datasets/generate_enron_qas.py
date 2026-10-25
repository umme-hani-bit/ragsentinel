#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


AUG_INJ = Path("data/processed/enron_emails_aug_inj.parquet")  # preferred (has tenant_id)
CLEAN = Path("data/processed/enron_emails_clean.parquet")      # fallback
TENANTS = Path("data/metadata/doc_tenants.parquet")            # used only with CLEAN fallback
OUT = Path("data/metadata/qas_enron.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)


STOPWORDS = {
    "re", "fw", "fwd", "the", "and", "for", "with", "from", "to", "of", "in", "on", "at",
    "a", "an", "about", "meeting", "update", "regarding", "subject", "info", "information",
    "question", "enron",
}


@dataclass
class QA:
    tenant_id: str
    doc_id: str
    qtype: str
    question: str
    answer: str
    alt_answers: List[str]
    evidence_field: str
    subject: str
    sender: str
    recipients: List[str]
    date: str


def _short_date(date_iso: str) -> str:
    return date_iso[:10] if date_iso else ""


def _uniq(seq: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_email_headers(raw_body: str) -> Tuple[str, List[str]]:
    """
    Extract sender and recipient addresses from raw Enron email headers.
    Falls back to empty values if headers are missing.
    """
    if not raw_body:
        return "", []

    sender = ""
    recipients: List[str] = []

    m = re.search(r"^From:\s*([^\r\n]+)", raw_body, re.MULTILINE | re.IGNORECASE)
    if m:
        sender = m.group(1).strip()

    for field in ("To", "Cc"):
        m = re.search(rf"^{field}:\s*([^\r\n]+)", raw_body, re.MULTILINE | re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            parts = re.split(r"[;,]\s*", value)
            recipients.extend(p.strip() for p in parts if p.strip())

    return sender, _uniq(recipients)


def _keyword_from_subject(subj: str) -> str | None:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", subj or "")
    toks = [t.lower() for t in toks if t.lower() not in STOPWORDS]
    if not toks:
        return None
    return toks[min(len(toks) // 2, len(toks) - 1)]


def _gen_from_row(
    doc_id: str,
    tenant_id: str,
    subject: str,
    sender: str,
    recipients: List[str],
    date_iso: str,
    employee_id: str,
) -> List[QA]:
    out: List[QA] = []
    ymd = _short_date(date_iso)
    rcps = _uniq([r for r in (recipients or []) if r])

    # employee_id is the mailbox owner; use it if the From: header is missing
    effective_sender = sender or employee_id or ""

    base = {
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "subject": subject or "",
        "sender": effective_sender,
        "recipients": rcps,
        "date": date_iso or "",
    }

    # Q1: subject from sender + date
    if subject and effective_sender and ymd:
        q = f"What is the subject of the email sent by {effective_sender} on {ymd}?"
        out.append(
            QA(
                qtype="subject_from_sender_date",
                question=q,
                answer=subject,
                alt_answers=[subject],
                evidence_field="subject",
                **base,
            )
        )

    # Q2: sender from subject
    if subject and effective_sender:
        q = f"Who sent the email titled '{subject}'?"
        out.append(
            QA(
                qtype="sender_from_subject",
                question=q,
                answer=effective_sender,
                alt_answers=[effective_sender],
                evidence_field="sender",
                **base,
            )
        )

    # Q3: a recipient from subject
    if subject and rcps:
        q = f"Name one recipient of the email titled '{subject}'."
        out.append(
            QA(
                qtype="recipient_from_subject",
                question=q,
                answer=rcps[0],
                alt_answers=rcps,
                evidence_field="recipients",
                **base,
            )
        )

    # Q4: sender from subject keyword (+ optional date)
    kw = _keyword_from_subject(subject or "")
    if kw and effective_sender:
        hint = f" about '{kw}'"
        if ymd:
            hint += f" on {ymd}"
        q = f"Who sent the email{hint}?"
        out.append(
            QA(
                qtype="sender_from_keyword",
                question=q,
                answer=effective_sender,
                alt_answers=[effective_sender],
                evidence_field="sender",
                **base,
            )
        )

    return out


def _iter_source() -> Tuple[Dict[str, str], pq.ParquetFile]:
    """
    Returns (doc_to_tenant, parquet_file).
    Prefers the injection-augmented file because it already contains tenant_id.
    """
    if AUG_INJ.exists():
        return {}, pq.ParquetFile(AUG_INJ)

    if not CLEAN.exists() or not TENANTS.exists():
        raise SystemExit(
            "ERROR: Missing inputs. Need enron_emails_aug_inj.parquet "
            "OR (enron_emails_clean.parquet + doc_tenants.parquet)."
        )

    m = pq.read_table(TENANTS, columns=["doc_id", "tenant_id"])
    doc_to_tenant = dict(zip(m["doc_id"].to_pylist(), m["tenant_id"].to_pylist()))
    return doc_to_tenant, pq.ParquetFile(CLEAN)


def main() -> int:
    limit = int(os.getenv("QA_LIMIT", "300"))
    seed = int(os.getenv("QA_SEED", "34567"))

    doc_to_tenant, pf = _iter_source()
    names = set(pf.schema.names)

    required = {"doc_id", "subject", "date", "raw_body", "employee_id"}
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"ERROR: source parquet missing columns: {missing}")

    has_tenant_col = "tenant_id" in names

    # Discover tenants
    tenant_set: Set[str] = set()
    for batch in pf.iter_batches(batch_size=5000):
        if has_tenant_col:
            tenant_set.update(set(batch.column(batch.schema.names.index("tenant_id")).to_pylist()))
        else:
            docs = batch.column(batch.schema.names.index("doc_id")).to_pylist()
            for d in docs:
                t = doc_to_tenant.get(d, "")
                if t:
                    tenant_set.add(t)
        if len(tenant_set) >= 3:
            break

    tenants_sorted = sorted(tenant_set) if tenant_set else ["A", "B", "C"]
    per_tenant_target = max(1, limit // max(1, len(tenants_sorted)))
    target_per_tenant = {t: per_tenant_target for t in tenants_sorted}

    qas: List[Dict[str, object]] = []
    per_tenant_counts: Dict[str, int] = {}

    for batch in pf.iter_batches(batch_size=5000):
        if limit <= 0:
            break

        idx = {n: i for i, n in enumerate(batch.schema.names)}
        docs = batch.column(idx["doc_id"]).to_pylist()
        subs = batch.column(idx["subject"]).to_pylist()
        dts = batch.column(idx["date"]).to_pylist()
        raws = batch.column(idx["raw_body"]).to_pylist()
        emps = batch.column(idx["employee_id"]).to_pylist()
        tenants = (
            batch.column(idx["tenant_id"]).to_pylist()
            if has_tenant_col
            else [doc_to_tenant.get(d, "") for d in docs]
        )

        for d, t, s, dt, raw, emp in zip(docs, tenants, subs, dts, raws, emps):
            if limit <= 0:
                break

            t = t or "A"
            if per_tenant_counts.get(t, 0) >= target_per_tenant.get(t, 0):
                continue

            sender, recipients = _parse_email_headers(raw or "")
            cands = _gen_from_row(
                doc_id=d,
                tenant_id=t,
                subject=s or "",
                sender=sender,
                recipients=recipients,
                date_iso=dt or "",
                employee_id=emp or "",
            )

            if not cands:
                continue

            # Take up to 2 QAs per doc for variety
            taken = 0
            for qa in cands:
                if limit <= 0 or taken >= 2:
                    break

                rec = {
                    "qa_id": "",
                    "tenant_id": qa.tenant_id,
                    "doc_id": qa.doc_id,
                    "qtype": qa.qtype,
                    "question": qa.question,
                    "answer": qa.answer,
                    "alt_answers": qa.alt_answers,
                    "evidence_field": qa.evidence_field,
                    "subject": qa.subject,
                    "sender": qa.sender,
                    "recipients": qa.recipients,
                    "date": qa.date,
                    "noisy_label_note": "Generated via templates on real emails; spot-check ~30 before trusting scores.",
                }
                qas.append(rec)
                per_tenant_counts[t] = per_tenant_counts.get(t, 0) + 1
                limit -= 1
                taken += 1

    # Assign stable QA IDs
    for i, r in enumerate(qas, 1):
        r["qa_id"] = f"QA-{i:04d}"

    # Write Parquet
    schema = pa.schema(
        [
            pa.field("qa_id", pa.string()),
            pa.field("tenant_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("qtype", pa.string()),
            pa.field("question", pa.string()),
            pa.field("answer", pa.string()),
            pa.field("alt_answers", pa.list_(pa.string())),
            pa.field("evidence_field", pa.string()),
            pa.field("subject", pa.string()),
            pa.field("sender", pa.string()),
            pa.field("recipients", pa.list_(pa.string())),
            pa.field("date", pa.string()),
            pa.field("noisy_label_note", pa.string()),
        ]
    )
    tbl = pa.table({k: [r[k] for r in qas] for k in schema.names}, schema=schema)
    pq.write_table(tbl, OUT, compression="zstd")

    print(f"Generated {len(qas)} QA pairs → {OUT}")
    print("Per-tenant counts:", dict(per_tenant_counts))
    print(
        "Types:",
        {
            t: sum(1 for r in qas if r["qtype"] == t)
            for t in sorted(set(r["qtype"] for r in qas))
        },
    )
    print("Note: labels are noisy; manually spot-check ~30 before relying on scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())