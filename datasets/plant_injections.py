#!/usr/bin/env python3
from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


# Inputs
AUG_IN_PATH = Path("data/processed/enron_emails_augmented.parquet")
TENANTS_PATH = Path("data/metadata/doc_tenants.parquet")
INJ_CFG_PATH = Path("configs/injections.json")
CORPUS_YAML = Path("configs/corpus.yaml")

# Outputs
AUG_OUT_PATH = Path("data/processed/enron_emails_aug_inj.parquet")
INJ_SPANS_PATH = Path("data/metadata/injection_spans.parquet")
INJ_SPLIT_PATH = Path("data/metadata/injection_doc_splits.parquet")

AUG_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
INJ_SPANS_PATH.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Loading and deterministic RNG
# ----------------------------
def _load_json(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return data


def _load_payload_seed(path: Path) -> int:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")

    seeds = data.get("seeds", {})
    if not isinstance(seeds, dict):
        raise RuntimeError("'seeds' in corpus.yaml must be a mapping.")

    raw = seeds.get("payloads", seeds.get("payload", 23456))

    # Also supports a future nested seed layout.
    if isinstance(raw, dict):
        raw = raw.get("injections", raw.get("default", 23456))

    return int(raw)


def _stable_int(seed: int, *parts: object) -> int:
    value = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _stable_rng(seed: int, *parts: object) -> random.Random:
    return random.Random(_stable_int(seed, *parts))


def _load_tenants_map(path: Path) -> Dict[str, str]:
    table = pq.read_table(path, columns=["doc_id", "tenant_id"])
    doc_ids = table.column("doc_id").to_pylist()
    tenant_ids = table.column("tenant_id").to_pylist()

    if len(doc_ids) != len(set(doc_ids)):
        raise RuntimeError("doc_tenants.parquet contains duplicate doc_id values.")

    mapping = {
        str(doc_id): str(tenant_id)
        for doc_id, tenant_id in zip(doc_ids, tenant_ids)
    }

    if not mapping:
        raise RuntimeError("doc_tenants.parquet is empty.")

    if any(not doc_id or not tenant_id for doc_id, tenant_id in mapping.items()):
        raise RuntimeError("Tenant mapping contains an empty doc_id or tenant_id.")

    return mapping


def _other_tenant(
    this_tenant: str,
    all_tenants: List[str],
    rng: random.Random,
) -> str:
    alternatives = [
        tenant for tenant in all_tenants if tenant != this_tenant
    ]
    return rng.choice(alternatives) if alternatives else this_tenant


# ----------------------------
# Payload loading and rendering
# ----------------------------
@dataclass(frozen=True)
class Payload:
    payload_id: str
    tier: int
    intents: List[str]
    encoding: str
    template: str
    wrapper: str | None
    placeholders: List[str]


def _load_payloads(config_path: Path) -> List[Payload]:
    config = _load_json(config_path)
    payloads: List[Payload] = []
    seen_ids = set()

    for item in config.get("payloads", []):
        payload_id = str(item["id"])

        if payload_id in seen_ids:
            raise RuntimeError(f"Duplicate payload ID: {payload_id}")
        seen_ids.add(payload_id)

        payloads.append(
            Payload(
                payload_id=payload_id,
                tier=int(item["tier"]),
                intents=list(item.get("intent", [])),
                encoding=str(item.get("encoding", "none")),
                template=str(item.get("template", "")),
                wrapper=(
                    str(item["wrapper"])
                    if item.get("wrapper") is not None
                    else None
                ),
                placeholders=list(item.get("placeholders", [])),
            )
        )

    if not payloads:
        raise RuntimeError("No payloads found in configs/injections.json.")

    return payloads


def _replace_placeholders(text: str, substitutions: Dict[str, str]) -> str:
    rendered = text

    for name, value in substitutions.items():
        rendered = rendered.replace(f"{{{name}}}", value)

    unresolved = set(re.findall(r"\{([A-Z_]+)\}", rendered))
    if unresolved:
        raise RuntimeError(
            f"Unresolved placeholder(s): {sorted(unresolved)}"
        )

    return rendered


def _render_payload(
    payload: Payload,
    this_tenant: str,
    other_tenant: str,
) -> str:
    substitutions = {
        "TENANT": this_tenant,
        "THIS_TENANT": this_tenant,
        "OTHER_TENANT": other_tenant,
        "EXFIL_URL": (
            f"https://exfil.ragsentinel.invalid/collect/{this_tenant}"
        ),
        "EXFIL_EMAIL": "audit@ragsentinel.invalid",
    }

    base = _replace_placeholders(payload.template, substitutions)

    if payload.encoding == "base64":
        encoded = base64.b64encode(base.encode("utf-8")).decode("ascii")
        wrapper = payload.wrapper or "b64:{B64}"
        wrapped = wrapper.replace("{B64}", encoded)
        return _replace_placeholders(wrapped, substitutions)

    return base


# ----------------------------
# Planning and placements
# ----------------------------
@dataclass(frozen=True)
class Plan:
    doc_id: str
    tenant_id: str
    target_tenant: str
    payload_id: str
    tier: int
    intents: List[str]
    placement: str
    encoding: str
    text: str


@dataclass
class SpanRow:
    doc_id: str
    tenant_id: str
    target_tenant: str
    payload_id: str
    tier: int
    intents: List[str]
    placement: str
    encoding: str
    block_start: int
    block_end: int
    inserted_block: str


def _apply_placement(
    body: str,
    placement: str,
    injection_text: str,
) -> Tuple[str, int, int, str]:
    """
    Insert an injection and return:
    (new_body, block_start, block_end, exact_inserted_block).

    All placements append to the document. This intentionally preserves the
    existing secret offsets produced by Task 7.
    """
    if placement == "footnote":
        block = f"\n\n[injection]: {injection_text}\n"

    elif placement == "quote":
        quoted = "\n".join(
            f"> {line}" for line in injection_text.splitlines()
        )
        block = f"\n\n{quoted}\n"

    elif placement == "forwarded":
        block = (
            "\n\n----- Forwarded message -----\n"
            f"{injection_text}"
            "\n----- End forwarded message -----\n"
        )

    else:
        raise ValueError(f"Unknown placement: {placement}")

    start = len(body)
    end = start + len(block)
    new_body = body + block

    if new_body[start:end] != block:
        raise RuntimeError("Internal offset validation failed.")

    return new_body, start, end, block


def _plan_docs(
    doc_to_tenant: Dict[str, str],
    tenants: List[str],
    payloads: List[Payload],
    per_tenant_quota: Dict[int, int],
    seed: int,
) -> Dict[str, List[Plan]]:
    by_tier = {
        tier: [payload for payload in payloads if payload.tier == tier]
        for tier in (1, 2, 3)
    }

    for tier in (1, 2, 3):
        if not by_tier[tier]:
            raise RuntimeError(
                f"Tier {tier} has no payloads in injections.json."
            )

    docs_by_tenant: Dict[str, List[str]] = {}
    for doc_id, tenant_id in doc_to_tenant.items():
        docs_by_tenant.setdefault(tenant_id, []).append(doc_id)

    plan_by_doc: Dict[str, List[Plan]] = {}
    placements = ["footnote", "quote", "forwarded"]

    for tenant_id in tenants:
        tenant_docs = sorted(docs_by_tenant.get(tenant_id, []))
        total_needed = sum(per_tenant_quota.values())

        if len(tenant_docs) < total_needed:
            raise RuntimeError(
                f"Tenant {tenant_id} has {len(tenant_docs)} documents, "
                f"but {total_needed} are required."
            )

        used_docs = set()

        for tier in (1, 2, 3):
            quota = per_tenant_quota.get(tier, 0)
            if quota < 0:
                raise RuntimeError(
                    f"Tier {tier} quota cannot be negative."
                )
            if quota == 0:
                continue

            # A separate deterministic ranking prevents run-to-run variation.
            ranked_docs = sorted(
                tenant_docs,
                key=lambda doc_id: _stable_int(
                    seed,
                    "document",
                    tenant_id,
                    tier,
                    doc_id,
                ),
            )

            selected_docs: List[str] = []
            for doc_id in ranked_docs:
                if doc_id in used_docs:
                    continue
                selected_docs.append(doc_id)
                used_docs.add(doc_id)
                if len(selected_docs) == quota:
                    break

            if len(selected_docs) != quota:
                raise RuntimeError(
                    f"Could not select {quota} unique documents for "
                    f"tenant {tenant_id}, tier {tier}."
                )

            # Cycle through a deterministically shuffled payload list so all
            # payload IDs receive approximately equal representation.
            tier_payloads = by_tier[tier][:]
            payload_rng = _stable_rng(
                seed,
                "payload-order",
                tenant_id,
                tier,
            )
            payload_rng.shuffle(tier_payloads)

            for index, doc_id in enumerate(selected_docs):
                payload = tier_payloads[index % len(tier_payloads)]
                other_rng = _stable_rng(
                    seed,
                    "other-tenant",
                    tenant_id,
                    tier,
                    doc_id,
                )
                other_tenant = _other_tenant(
                    tenant_id,
                    tenants,
                    other_rng,
                )

                rendered = _render_payload(
                    payload,
                    this_tenant=tenant_id,
                    other_tenant=other_tenant,
                )

                target_tenant = (
                    other_tenant
                    if "OTHER_TENANT" in payload.placeholders
                    else ""
                )

                plan = Plan(
                    doc_id=doc_id,
                    tenant_id=tenant_id,
                    target_tenant=target_tenant,
                    payload_id=payload.payload_id,
                    tier=tier,
                    intents=payload.intents,
                    placement=placements[index % len(placements)],
                    encoding=payload.encoding,
                    text=rendered,
                )
                plan_by_doc.setdefault(doc_id, []).append(plan)

    return plan_by_doc


# ----------------------------
# Streaming Parquet write
# ----------------------------
def _update_and_write(
    input_path: Path,
    output_path: Path,
    spans_path: Path,
    doc_to_tenant: Dict[str, str],
    plan_by_doc: Dict[str, List[Plan]],
    batch_size: int = 2000,
) -> Tuple[int, int, int]:
    parquet_file = pq.ParquetFile(input_path)
    base_schema = parquet_file.schema_arrow

    required = {"doc_id", "tenant_id", "augmented_body"}
    missing = required - set(base_schema.names)
    if missing:
        raise RuntimeError(
            f"Input Parquet is missing columns: {sorted(missing)}"
        )

    extra_fields = [
        pa.field("inserted_injections_count", pa.int32()),
        pa.field("injection_payload_ids", pa.list_(pa.string())),
        pa.field("injection_tiers", pa.list_(pa.int32())),
        pa.field("injection_placements", pa.list_(pa.string())),
    ]

    collisions = {
        field.name for field in extra_fields
    } & set(base_schema.names)
    if collisions:
        raise RuntimeError(
            f"Input already contains output columns: {sorted(collisions)}"
        )

    output_schema = pa.schema(
        list(base_schema) + extra_fields,
        metadata=base_schema.metadata,
    )

    spans_schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("tenant_id", pa.string()),
            pa.field("target_tenant", pa.string()),
            pa.field("payload_id", pa.string()),
            pa.field("tier", pa.int32()),
            pa.field("intents", pa.list_(pa.string())),
            pa.field("placement", pa.string()),
            pa.field("encoding", pa.string()),
            pa.field("block_start", pa.int32()),
            pa.field("block_end", pa.int32()),
            pa.field("inserted_block", pa.string()),
        ]
    )

    for path in (output_path, spans_path):
        if path.exists():
            path.unlink()

    writer = pq.ParquetWriter(
        output_path,
        schema=output_schema,
        compression="zstd",
    )
    spans_writer = pq.ParquetWriter(
        spans_path,
        schema=spans_schema,
        compression="zstd",
    )

    rows_written = 0
    docs_modified = 0
    spans_written = 0
    seen_planned_docs = set()

    try:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            names = batch.schema.names

            doc_ids = batch.column(names.index("doc_id")).to_pylist()
            tenant_ids = batch.column(
                names.index("tenant_id")
            ).to_pylist()
            bodies = batch.column(
                names.index("augmented_body")
            ).to_pylist()

            new_bodies: List[str] = []
            injection_counts: List[int] = []
            payload_ids_column: List[List[str]] = []
            tiers_column: List[List[int]] = []
            placements_column: List[List[str]] = []
            batch_spans: List[SpanRow] = []

            for row_index, raw_doc_id in enumerate(doc_ids):
                doc_id = str(raw_doc_id)
                tenant_id = str(tenant_ids[row_index] or "")
                expected_tenant = doc_to_tenant.get(doc_id)

                if expected_tenant is None:
                    raise RuntimeError(
                        f"Document has no tenant mapping: {doc_id}"
                    )

                if tenant_id != expected_tenant:
                    raise RuntimeError(
                        f"Tenant mismatch for {doc_id}: "
                        f"input={tenant_id!r}, mapping={expected_tenant!r}"
                    )

                body = str(bodies[row_index] or "")
                plans = plan_by_doc.get(doc_id, [])

                if not plans:
                    new_bodies.append(body)
                    injection_counts.append(0)
                    payload_ids_column.append([])
                    tiers_column.append([])
                    placements_column.append([])
                    continue

                current_body = body
                row_payload_ids: List[str] = []
                row_tiers: List[int] = []
                row_placements: List[str] = []

                for plan in plans:
                    if plan.tenant_id != tenant_id:
                        raise RuntimeError(
                            f"Plan tenant mismatch for {doc_id}."
                        )

                    current_body, start, end, block = _apply_placement(
                        current_body,
                        plan.placement,
                        plan.text,
                    )

                    if current_body[start:end] != block:
                        raise RuntimeError(
                            f"Offset verification failed for {doc_id}."
                        )

                    batch_spans.append(
                        SpanRow(
                            doc_id=doc_id,
                            tenant_id=tenant_id,
                            target_tenant=plan.target_tenant,
                            payload_id=plan.payload_id,
                            tier=plan.tier,
                            intents=plan.intents,
                            placement=plan.placement,
                            encoding=plan.encoding,
                            block_start=start,
                            block_end=end,
                            inserted_block=block,
                        )
                    )

                    row_payload_ids.append(plan.payload_id)
                    row_tiers.append(plan.tier)
                    row_placements.append(plan.placement)

                new_bodies.append(current_body)
                injection_counts.append(len(plans))
                payload_ids_column.append(row_payload_ids)
                tiers_column.append(row_tiers)
                placements_column.append(row_placements)

                docs_modified += 1
                seen_planned_docs.add(doc_id)

            output_arrays = []

            for column_index, name in enumerate(names):
                if name == "augmented_body":
                    output_arrays.append(
                        pa.array(new_bodies, type=pa.string())
                    )
                else:
                    output_arrays.append(batch.column(column_index))

            output_arrays.extend(
                [
                    pa.array(injection_counts, type=pa.int32()),
                    pa.array(
                        payload_ids_column,
                        type=pa.list_(pa.string()),
                    ),
                    pa.array(
                        tiers_column,
                        type=pa.list_(pa.int32()),
                    ),
                    pa.array(
                        placements_column,
                        type=pa.list_(pa.string()),
                    ),
                ]
            )

            output_table = pa.Table.from_arrays(
                output_arrays,
                schema=output_schema,
            )
            writer.write_table(output_table)

            if batch_spans:
                spans_table = pa.table(
                    {
                        "doc_id": [
                            span.doc_id for span in batch_spans
                        ],
                        "tenant_id": [
                            span.tenant_id for span in batch_spans
                        ],
                        "target_tenant": [
                            span.target_tenant for span in batch_spans
                        ],
                        "payload_id": [
                            span.payload_id for span in batch_spans
                        ],
                        "tier": [
                            span.tier for span in batch_spans
                        ],
                        "intents": [
                            span.intents for span in batch_spans
                        ],
                        "placement": [
                            span.placement for span in batch_spans
                        ],
                        "encoding": [
                            span.encoding for span in batch_spans
                        ],
                        "block_start": [
                            span.block_start for span in batch_spans
                        ],
                        "block_end": [
                            span.block_end for span in batch_spans
                        ],
                        "inserted_block": [
                            span.inserted_block for span in batch_spans
                        ],
                    },
                    schema=spans_schema,
                )
                spans_writer.write_table(spans_table)
                spans_written += len(batch_spans)

            rows_written += batch.num_rows

            if rows_written % 50_000 == 0:
                print(f"  processed {rows_written:,} rows...")

    finally:
        writer.close()
        spans_writer.close()

    missing_planned_docs = set(plan_by_doc) - seen_planned_docs
    if missing_planned_docs:
        sample = sorted(missing_planned_docs)[:5]
        raise RuntimeError(
            f"{len(missing_planned_docs)} planned documents were not "
            f"found in the input. Examples: {sample}"
        )

    return rows_written, docs_modified, spans_written


# ----------------------------
# Train/dev/test split
# ----------------------------
def _write_splits(
    spans_path: Path,
    output_path: Path,
    seed: int,
) -> Tuple[int, Counter]:
    table = pq.read_table(
        spans_path,
        columns=["doc_id", "tenant_id", "tier"],
    )

    doc_ids = table.column("doc_id").to_pylist()
    tenant_ids = table.column("tenant_id").to_pylist()
    tiers = table.column("tier").to_pylist()

    rows = {
        "doc_id": [],
        "tenant_id": [],
        "tier": [],
        "split": [],
    }

    seen_docs = set()
    split_counts: Counter = Counter()

    for doc_id, tenant_id, tier in zip(
        doc_ids,
        tenant_ids,
        tiers,
    ):
        if doc_id in seen_docs:
            continue

        seen_docs.add(doc_id)
        bucket = _stable_int(
            seed,
            "split",
            doc_id,
        ) % 20

        if bucket < 14:
            split = "train"
        elif bucket < 17:
            split = "dev"
        else:
            split = "test"

        rows["doc_id"].append(str(doc_id))
        rows["tenant_id"].append(str(tenant_id))
        rows["tier"].append(int(tier))
        rows["split"].append(split)
        split_counts[split] += 1

    split_schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("tenant_id", pa.string()),
            pa.field("tier", pa.int32()),
            pa.field("split", pa.string()),
        ]
    )

    if output_path.exists():
        output_path.unlink()

    pq.write_table(
        pa.table(rows, schema=split_schema),
        output_path,
        compression="zstd",
    )

    return len(seen_docs), split_counts


def _print_injection_summary(spans_path: Path) -> None:
    table = pq.read_table(
        spans_path,
        columns=["tenant_id", "tier", "payload_id"],
    )

    tenants = table.column("tenant_id").to_pylist()
    tiers = table.column("tier").to_pylist()
    payload_ids = table.column("payload_id").to_pylist()

    tenant_tier_counts = Counter(zip(tenants, tiers))
    payload_counts = Counter(payload_ids)

    print("\nPer-tenant summary:")
    for tenant_id in sorted(set(tenants)):
        print(
            f"  Tenant {tenant_id}: "
            f"T1={tenant_tier_counts[(tenant_id, 1)]}, "
            f"T2={tenant_tier_counts[(tenant_id, 2)]}, "
            f"T3={tenant_tier_counts[(tenant_id, 3)]}"
        )

    print("\nPer-payload summary:")
    for payload_id in sorted(payload_counts):
        print(f"  {payload_id}: {payload_counts[payload_id]}")


def main() -> int:
    required_files = [
        AUG_IN_PATH,
        TENANTS_PATH,
        INJ_CFG_PATH,
        CORPUS_YAML,
    ]

    for path in required_files:
        if not path.exists():
            print(f"ERROR: required file not found: {path}")
            return 2

    base_seed = _load_payload_seed(CORPUS_YAML)
    injection_seed = base_seed + 101
    split_seed = base_seed + 202

    payloads = _load_payloads(INJ_CFG_PATH)

    per_tenant_quota = {
        1: int(os.getenv("INJ_T1_PER_TENANT", "30")),
        2: int(os.getenv("INJ_T2_PER_TENANT", "30")),
        3: int(os.getenv("INJ_T3_PER_TENANT", "30")),
    }

    if any(quota < 0 for quota in per_tenant_quota.values()):
        raise RuntimeError("Injection quotas cannot be negative.")

    doc_to_tenant = _load_tenants_map(TENANTS_PATH)
    tenants = sorted(set(doc_to_tenant.values()))

    plan_by_doc = _plan_docs(
        doc_to_tenant=doc_to_tenant,
        tenants=tenants,
        payloads=payloads,
        per_tenant_quota=per_tenant_quota,
        seed=injection_seed,
    )

    total_plans = sum(len(plans) for plans in plan_by_doc.values())
    expected_plans = len(tenants) * sum(per_tenant_quota.values())

    if total_plans != expected_plans:
        raise RuntimeError(
            f"Expected {expected_plans} plans, got {total_plans}."
        )

    quota_text = ", ".join(
        f"T{tier}={per_tenant_quota[tier]} per tenant"
        for tier in (1, 2, 3)
    )

    print(
        f"Planning complete: {total_plans} injections across "
        f"{len(plan_by_doc)} documents ({quota_text})."
    )

    rows, modified, spans = _update_and_write(
        input_path=AUG_IN_PATH,
        output_path=AUG_OUT_PATH,
        spans_path=INJ_SPANS_PATH,
        doc_to_tenant=doc_to_tenant,
        plan_by_doc=plan_by_doc,
        batch_size=2000,
    )

    input_rows = pq.ParquetFile(AUG_IN_PATH).metadata.num_rows

    if rows != input_rows:
        raise RuntimeError(
            f"Output row count mismatch: input={input_rows}, output={rows}."
        )
    if modified != len(plan_by_doc):
        raise RuntimeError(
            f"Modified document mismatch: "
            f"expected={len(plan_by_doc)}, actual={modified}."
        )
    if spans != total_plans:
        raise RuntimeError(
            f"Span count mismatch: expected={total_plans}, actual={spans}."
        )

    split_rows, split_counts = _write_splits(
        INJ_SPANS_PATH,
        INJ_SPLIT_PATH,
        seed=split_seed,
    )

    if split_rows != modified:
        raise RuntimeError(
            f"Split row mismatch: expected={modified}, actual={split_rows}."
        )

    print(
        f"\nDone. Rows copied: {rows:,}; "
        f"docs modified: {modified:,}; spans written: {spans:,}"
    )
    print(f"Augmented-with-injections: {AUG_OUT_PATH}")
    print(f"Injection spans:           {INJ_SPANS_PATH}")
    print(f"Document splits:           {INJ_SPLIT_PATH}")

    _print_injection_summary(INJ_SPANS_PATH)

    print(
        "\nSplit summary: "
        f"train={split_counts['train']}, "
        f"dev={split_counts['dev']}, "
        f"test={split_counts['test']}"
    )
    print("\nValidation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())