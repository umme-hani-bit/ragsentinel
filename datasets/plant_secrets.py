#!/usr/bin/env python3
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


CLEAN_PATH = Path("data/processed/enron_emails_clean.parquet")
TENANTS_PATH = Path("data/metadata/doc_tenants.parquet")
SECRETS_YAML = Path("configs/secrets.yaml")
CORPUS_YAML = Path("configs/corpus.yaml")

AUG_PATH = Path("data/processed/enron_emails_augmented.parquet")
SPANS_PATH = Path("data/metadata/secret_spans.parquet")

AUG_PATH.parent.mkdir(parents=True, exist_ok=True)
SPANS_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------- small helpers ----------------
def _load_yaml(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return data


def _load_doc_tenants(path: Path) -> Dict[str, str]:
    tbl = pq.read_table(path, columns=["doc_id", "tenant_id"])
    doc_ids = tbl.column("doc_id").to_pylist()
    tids = tbl.column("tenant_id").to_pylist()
    return dict(zip(doc_ids, tids))


def _first_sentence_end(text: str) -> int:
    m = re.search(r"[.!?]", text)
    return m.end() if m else len(text)


def _random_alnum(rng: random.Random, n: int, alphabet: str) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


# ---------------- secret generators ----------------
def gen_api_key(tag: str, rng: random.Random) -> str:
    body = _random_alnum(
        rng,
        32,
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    )
    return f"sk_rags_{tag}_{body}"


def gen_jwt_like(tag: str, rng: random.Random) -> str:
    def seg(n: int) -> str:
        return _random_alnum(
            rng,
            n,
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
        )

    return f"rgsjwt_{tag}.{seg(16)}.{seg(16)}.{seg(16)}"


def gen_ssn_fake(tag: str, rng: random.Random) -> str:
    return (
        f"SSN-{tag}-{rng.randint(100, 999)}-{rng.randint(10, 99)}-"
        f"{rng.randint(1000, 9999)}"
    )


def gen_cc_fake(tag: str, rng: random.Random) -> str:
    def block() -> str:
        return f"{rng.randint(0, 9999):04d}"

    return f"RAGCARD-{tag} {block()}-{block()}-{block()}-{block()}"


def gen_codename(tag: str, rng: random.Random) -> str:
    n = rng.randint(5, 8)
    word = _random_alnum(rng, n, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f"PRJ-{tag}-{word}"


GEN_BY_TYPE = {
    "api_key": gen_api_key,
    "jwt_like": gen_jwt_like,
    "ssn_fake": gen_ssn_fake,
    "credit_card_fake": gen_cc_fake,
    "project_codename": gen_codename,
}


# ---------------- placements ----------------
@dataclass
class InsertionPlan:
    secret_type: str
    value: str
    placement: str  # footnote | inline | signature


@dataclass
class SpanRow:
    doc_id: str
    tenant_id: str
    secret_type: str
    value: str
    placement: str
    start: int
    end: int


def apply_insertion(text: str, plan: InsertionPlan) -> Tuple[str, int, int]:
    if plan.placement == "footnote":
        prefix = "\n\nNote: "
        suffix = "\n"
        start = len(text) + len(prefix)
        new_text = text + prefix + plan.value + suffix
        return new_text, start, start + len(plan.value)

    if plan.placement == "inline":
        i = _first_sentence_end(text)
        prefix = " ("
        suffix = ")"
        start = i + len(prefix)
        new_text = text[:i] + prefix + plan.value + suffix + text[i:]
        return new_text, start, start + len(plan.value)

    if plan.placement == "signature":
        prefix = "\n\n-- \n"
        suffix = "\n"
        start = len(text) + len(prefix)
        new_text = text + prefix + plan.value + suffix
        return new_text, start, start + len(plan.value)

    raise ValueError(f"Unknown placement: {plan.placement}")


# ---------------- planning ----------------
def _plan_for_tenant(
    tenant_id: str,
    tenant_docs: List[str],
    quotas: Dict[str, int],
    placements: List[str],
    base_seed: int,
    secrets_cfg: Dict,
) -> Dict[str, List[InsertionPlan]]:
    by_doc: Dict[str, List[InsertionPlan]] = {}
    tenant_docs_sorted = sorted(tenant_docs)

    for s_type, quota in quotas.items():
        if quota <= 0:
            continue

        rng = random.Random(f"{base_seed}|{tenant_id}|{s_type}")
        picks = tenant_docs_sorted[:]
        rng.shuffle(picks)
        picks = picks[: min(quota, len(picks))]

        regex = secrets_cfg["types"][s_type]["detect_regex"]

        for i, doc_id in enumerate(picks):
            placement = placements[i % len(placements)] if placements else "footnote"
            gen = GEN_BY_TYPE[s_type]
            value = gen(tenant_id, rng)

            if not re.search(regex, value):
                raise ValueError(
                    f"Generated {s_type} does not match its detect_regex: {value!r}"
                )

            plan = InsertionPlan(secret_type=s_type, value=value, placement=placement)
            by_doc.setdefault(doc_id, []).append(plan)

    return by_doc


def build_full_plan(
    doc_to_tenant: Dict[str, str],
    quotas_per_tenant: Dict[str, int],
    placements: List[str],
    payload_seed: int,
    secrets_cfg: Dict,
) -> Dict[str, List[InsertionPlan]]:
    per_tenant: Dict[str, List[str]] = {}
    for doc, tid in doc_to_tenant.items():
        per_tenant.setdefault(tid, []).append(doc)

    full: Dict[str, List[InsertionPlan]] = {}
    for tid in sorted(per_tenant.keys()):
        tenant_docs = per_tenant[tid]
        d = _plan_for_tenant(
            tid, tenant_docs, quotas_per_tenant, placements, payload_seed, secrets_cfg
        )
        for doc_id, plans in d.items():
            full.setdefault(doc_id, []).extend(plans)

    return full


# ---------------- streaming augment + write ----------------
def stream_augment_and_write(
    clean_path: Path,
    aug_path: Path,
    doc_to_tenant: Dict[str, str],
    plan_by_doc: Dict[str, List[InsertionPlan]],
    spans_out: Path,
    batch_size: int = 2000,
) -> Tuple[int, int, int]:
    pf = pq.ParquetFile(clean_path)

    out_fields = list(pf.schema_arrow) + [
        pa.field("tenant_id", pa.string()),
        pa.field("augmented_body", pa.string()),
        pa.field("inserted_secrets_count", pa.int32()),
    ]
    out_schema = pa.schema(out_fields)

    if aug_path.exists():
        aug_path.unlink()
    writer = pq.ParquetWriter(aug_path, schema=out_schema, compression="zstd")

    spans_schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("tenant_id", pa.string()),
            pa.field("secret_type", pa.string()),
            pa.field("value", pa.string()),
            pa.field("placement", pa.string()),
            pa.field("start", pa.int32()),
            pa.field("end", pa.int32()),
        ]
    )
    if spans_out.exists():
        spans_out.unlink()
    spans_writer = pq.ParquetWriter(spans_out, schema=spans_schema, compression="zstd")

    rows_written = 0
    docs_modified = 0
    spans_written = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()

        tenant_ids: List[str] = []
        aug_bodies: List[str] = []
        counts: List[int] = []
        spans_buffer: List[SpanRow] = []

        for _, row in df.iterrows():
            doc_id = row["doc_id"]
            t_id = doc_to_tenant.get(doc_id, "")
            text = str(row["clean_body"] or "")
            plans = plan_by_doc.get(doc_id, [])

            if not plans:
                tenant_ids.append(t_id)
                aug_bodies.append(text)
                counts.append(0)
                continue

            cur = text
            for plan in plans:
                cur, start, end = apply_insertion(cur, plan)
                spans_buffer.append(
                    SpanRow(
                        doc_id=doc_id,
                        tenant_id=t_id,
                        secret_type=plan.secret_type,
                        value=plan.value,
                        placement=plan.placement,
                        start=int(start),
                        end=int(end),
                    )
                )

            tenant_ids.append(t_id)
            aug_bodies.append(cur)
            counts.append(len(plans))
            docs_modified += 1

        df["tenant_id"] = tenant_ids
        df["augmented_body"] = aug_bodies
        df["inserted_secrets_count"] = counts

        df = df[[field.name for field in out_fields]]
        writer.write_table(
            pa.Table.from_pandas(df, schema=out_schema, preserve_index=False)
        )

        if spans_buffer:
            spans_tbl = pa.table(
                {
                    "doc_id": [span.doc_id for span in spans_buffer],
                    "tenant_id": [span.tenant_id for span in spans_buffer],
                    "secret_type": [span.secret_type for span in spans_buffer],
                    "value": [span.value for span in spans_buffer],
                    "placement": [span.placement for span in spans_buffer],
                    "start": [span.start for span in spans_buffer],
                    "end": [span.end for span in spans_buffer],
                },
                schema=spans_schema,
            )
            spans_writer.write_table(spans_tbl)
            spans_written += len(spans_buffer)

        rows_written += batch.num_rows
        if rows_written % 50000 == 0:
            print(f"  processed {rows_written:,} rows...")

    writer.close()
    spans_writer.close()
    return rows_written, docs_modified, spans_written


def _resolve_payload_seed(corpus: Dict, secrets_cfg: Dict) -> int:
    # Prefer explicit seed in secrets.yaml
    planting = secrets_cfg.get("planting", {})
    if "seed" in planting:
        return int(planting["seed"])

    # Fallback to corpus.yaml (handles both 'payload' and 'payloads')
    seeds = corpus.get("seeds", {})
    raw = seeds.get("payload", seeds.get("payloads", 23456))
    if isinstance(raw, dict):
        raw = raw.get("tier1", raw.get("default", 23456))
    return int(raw)


def main() -> int:
    if not CLEAN_PATH.exists():
        print(f"ERROR: {CLEAN_PATH} not found. Run clean_and_prep.py first.")
        return 2
    if not TENANTS_PATH.exists():
        print(f"ERROR: {TENANTS_PATH} not found. Ensure tenant partition ran (Phase 2).")
        return 2
    if not SECRETS_YAML.exists():
        print(f"ERROR: {SECRETS_YAML} missing. Run Task 6 first.")
        return 2
    if not CORPUS_YAML.exists():
        print(f"ERROR: {CORPUS_YAML} missing.")
        return 2

    corpus = _load_yaml(CORPUS_YAML)
    secrets_cfg = _load_yaml(SECRETS_YAML)

    payload_seed = _resolve_payload_seed(corpus, secrets_cfg)

    planting = secrets_cfg.get("planting", {})
    quotas_per_tenant: Dict[str, int] = planting.get("quotas_per_tenant", {})
    placements: List[str] = planting.get(
        "placements", ["footnote", "inline", "signature"]
    )

    for s_type in secrets_cfg.get("types", {}).keys():
        if s_type not in GEN_BY_TYPE:
            raise RuntimeError(f"No generator defined for secret type '{s_type}'")

    doc_to_tenant = _load_doc_tenants(TENANTS_PATH)

    print("Planning insertions...")
    plan_by_doc = build_full_plan(
        doc_to_tenant, quotas_per_tenant, placements, payload_seed, secrets_cfg
    )
    total_plans = sum(len(plans) for plans in plan_by_doc.values())
    print(f"  will plant {total_plans} secrets across {len(plan_by_doc)} documents")

    rows, modified, spans = stream_augment_and_write(
        CLEAN_PATH, AUG_PATH, doc_to_tenant, plan_by_doc, SPANS_PATH, batch_size=2000
    )

    print(
        f"\nDone. Augmented rows written: {rows:,}; "
        f"docs modified: {modified:,}; spans written: {spans:,}"
    )
    print(f"Augmented file: {AUG_PATH}")
    print(f"Spans metadata: {SPANS_PATH}")

    spans_df = pd.read_parquet(SPANS_PATH)
    print("\nSecrets per tenant/type:")
    print(spans_df.groupby(["tenant_id", "secret_type"]).size().unstack(fill_value=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
