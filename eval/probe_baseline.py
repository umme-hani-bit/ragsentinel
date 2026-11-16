#!/usr/bin/env python3
"""Baseline vs defended probe for the RLS ablation.

Expected results — anything else is a BUG to investigate:
  [2] BASELINE (rls_enabled=False): HTTP 200, citations include tenant-B docs.
  [3] DEFENDED (rls_enabled=True):  HTTP 200, citations are tenant-A only.

A 400 on the baseline means RAGSENTINEL_BENCHMARK_MODE is not active.
That is never the "expected" result for this probe.
"""
import sys
import time

import pandas as pd
import psycopg2
import requests

DB_URL = "postgresql://ragsentinel:ragsentinel@localhost:5432/ragsentinel"
API_URL = "http://localhost:8000"


def wait_for_api(max_wait_s: int = 600) -> None:
    print("[0] Waiting for API + model warmup...")
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{API_URL}/healthz", timeout=5)
            if r.status_code == 200:
                print("    API healthy (models preloaded at startup).")
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    sys.exit("FAIL: API never became healthy — run: docker logs ragsentinel-api --tail 50")


def pick_target_subject() -> str:
    print("[1] Finding an ingested Tenant B document...")
    secrets = pd.read_parquet("data/metadata/secret_spans.parquet")
    b_secret_docs = secrets[secrets.tenant_id == "B"]["doc_id"].tolist()

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(
        "SELECT doc_id, subject FROM documents WHERE tenant_id = 'B' AND doc_id = ANY(%s) LIMIT 1;",
        (b_secret_docs,),
    )
    row = cur.fetchone()
    if row is None:
        print("    (no ingested B secret docs; falling back to any B doc)")
        cur.execute(
            "SELECT doc_id, subject FROM documents WHERE tenant_id = 'B' AND subject <> '' LIMIT 1;"
        )
        row = cur.fetchone()
    conn.close()
    if row is None:
        sys.exit("FAIL: no tenant-B documents ingested at all.")
    print(f"    Target: {row[0]} | subject: {row[1]!r}")
    return row[1]


def probe(label: str, rls_enabled: bool, subject: str) -> dict:
    print(f"\n[{label}] rls_enabled={rls_enabled}, querying as tenant A...")
    payload = {
        "tenant_id": "A",
        "query": subject,
        "top_k": 5,
        "defense_profile": {
            "rls_enabled": rls_enabled,
            "scanner_enabled": False,
            "tool_guard_enabled": False,
        },
        "run_nli_verifier": False,
    }
    r = requests.post(f"{API_URL}/query", json=payload, timeout=600)
    print(f"    Status: {r.status_code}")
    if r.status_code != 200:
        print(f"    Body: {r.text[:300]}")
        return {"ok": False, "cross": False}

    data = r.json()
    tenants = [c.get("tenant_id") for c in data.get("citations", [])]
    cross = any(t != "A" for t in tenants)
    print(f"    Citation tenants: {tenants}")
    print(f"    Leakage: {data.get('leakage')}")
    print(f"    llm_generation_ms: {data.get('timings_ms', {}).get('llm_generation_ms')}")
    return {"ok": True, "cross": cross}


def main() -> None:
    wait_for_api()
    subject = pick_target_subject()

    base = probe("2] BASELINE", rls_enabled=False, subject=subject)
    defended = probe("3] DEFENDED", rls_enabled=True, subject=subject)

    print("\n--- VERDICTS ---")
    b = base["ok"] and base["cross"]
    d = defended["ok"] and not defended["cross"]
    print(f"[{'PASS' if b else 'FAIL'}] Baseline returns 200 AND retrieves cross-tenant chunks")
    print(f"[{'PASS' if d else 'FAIL'}] Defended returns 200 AND retrieves only tenant-A chunks")
    sys.exit(0 if (b and d) else 1)


if __name__ == "__main__":
    main()