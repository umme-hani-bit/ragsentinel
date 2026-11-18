"""Diagnostic: what does the API return for the injection attacks? (baseline profile)
Usage: python eval/diag_injection.py 0 1 4   (attack indices)"""
import json
import sys

import requests

import attack_harness as h

idx = [int(x) for x in sys.argv[1:]] or [0, 1, 4]
attacks = [a for a in h.generate_all_attacks() if a["attack_id"].startswith("INJ-")]

for i in idx:
    a = attacks[i]
    print("=" * 70)
    print(a["attack_id"], "| tenant", a["tenant_id"], "| tier", a["tier"], "| query_version", a["query_version"])
    print("TARGET doc_id:", a["doc_id"])
    print("QUERY:", a["query"][:300])
    payload = {
        "tenant_id": a["tenant_id"],
        "query": a["query"],
        "top_k": h.TOP_K,
        "defense_profile": h.PROFILES["baseline"],
        "run_nli_verifier": False,
    }
    r = requests.post(h.API_URL, json=payload, timeout=120).json()
    ctxs = r.get("contexts") or []
    print("n_contexts:", len(ctxs))
    for c in ctxs:
        if isinstance(c, dict):
            print("  ", {k: str(v)[:90] for k, v in c.items() if k != "embedding"})
        else:
            print("  ", str(c)[:200])
    print("timings:", r.get("timings_ms"))
    print("answer:", (r.get("answer") or "")[:150])