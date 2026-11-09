#!/usr/bin/env python3
from __future__ import annotations

import os

import psycopg2
import pyarrow.parquet as pq


DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragsvc:ragsvc@localhost:5432/ragsentinel",
)
QA_PATH = "data/metadata/qas_enron.parquet"
TENANTS = ("A", "B", "C")


conn = psycopg2.connect(DB_URL)
conn.autocommit = True
by_tenant = {}

with conn.cursor() as cur:
    for tenant in TENANTS:
        cur.execute("SELECT set_config('app.tenant_id', %s, false)", (tenant,))
        cur.execute("SELECT doc_id FROM documents")
        by_tenant[tenant] = {str(r[0]) for r in cur.fetchall()}

conn.close()

all_ids = set().union(*by_tenant.values())
qas = pq.read_table(QA_PATH).to_pylist()

covered = sum(1 for qa in qas if str(qa["doc_id"]) in all_ids)
tenant_ok = sum(
    1
    for qa in qas
    if str(qa["doc_id"]) in by_tenant.get(str(qa["tenant_id"]), set())
)

print(
    "DB documents: "
    + f"{len(all_ids):,}  ("
    + ", ".join(f"{t}={len(v):,}" for t, v in sorted(by_tenant.items()))
    + ")"
)
print(f"QA pairs: {len(qas):,}")
print(f"QA doc_ids present in DB: {covered}/{len(qas)}")
print(f"QA doc_ids in correct tenant: {tenant_ok}/{len(qas)}")

assert covered == len(qas), f"Some QA doc_ids are not ingested! ({covered}/{len(qas)})"
assert tenant_ok == len(qas), f"Some QA tenant labels disagree with DB! ({tenant_ok}/{len(qas)})"
print("OK")
