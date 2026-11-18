"""Diagnostic: does the query retrieve the injected document? (3 API calls, baseline profile)"""
import json
import sys

import requests

import attack_harness as h

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
injs = h.load_injection_spans("test")
attacks = [a for a in h.generate_all_attacks() if a["attack_id"].startswith("INJ-")]

for i in range(n):
    inj, a = injs[i], attacks[i]
    print("=" * 70)
    print(a["attack_id"], "| tenant", a["tenant_id"], "| tier", a["tier"], "| payload", inj["payload_id"])
    print("target doc_id:", inj["doc_id"])
    print("query:", a["query"])
    payload = {
        "tenant_id": a["tenant_id"],
        "query": a["query"],
        "top_k": h.TOP_K,
        "defense_profile": h.PROFILES["baseline"],
        "run_nli_verifier": False,
    }
    r = requests.post(h.API_URL, json=payload, timeout=120).json()
    blob = json.dumps(r.get("citations")) + json.dumps(r.get("contexts"))
    print("TARGET DOC RETRIEVED:", inj["doc_id"] in blob)
    print("citations:", json.dumps(r.get("citations"))[:500])
    print("taint_flags:", json.dumps(r.get("taint_flags"))[:300])
    print("answer:", (r.get("answer") or "")[:200])