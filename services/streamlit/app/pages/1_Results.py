import json
import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

EVAL = Path(os.getenv("EVAL_DIR", "/app/eval"))
RES = EVAL / "results"

st.set_page_config(page_title="RAGsentinel results", layout="wide")
st.title("Benchmark results")
st.caption("Frozen 60-attack leakage subset and 38 injection / tool-abuse attacks. Model: openai/gpt-oss-20b.")


def load(name):
    path = RES / name
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def stat(rows):
    valid = [r for r in rows if not r.get("error")]
    hits = sum(1 for r in valid if r["success"])
    return len(valid), hits, len(rows) - len(valid)


def fisher_two_sided(a, b, c, d):
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def p(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)

    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9))


manifest_path = EVAL / "attack_subset_v1.json"
ids = set(json.loads(manifest_path.read_text(encoding="utf-8"))["attack_ids"]) if manifest_path.exists() else set()

weak_base = [r for r in load("attacks_baseline.jsonl")
             if r["attack_id"] in ids and r["attack_class"] == "cross_tenant_leakage"]
weak_rls = load("attacks_rls.jsonl")
strong_base = load("attacks_baseline_leak_v2.jsonl")
strong_rls = load("attacks_rls_leak_v2.jsonl")
inj = load("attacks_baseline_inj_v3.jsonl")

st.header("1. Cross-tenant leakage: RLS off vs on")

scenarios = []
for label, base, rls in [
    ("Weak attacker (subject-line queries)", weak_base, weak_rls),
    ("Strong attacker (quotes email text)", strong_base, strong_rls),
]:
    nb, hb, eb = stat(base)
    nr, hr, er = stat(rls)
    scenarios.append((label, nb, hb, eb, nr, hr, er))

table = []
for label, nb, hb, eb, nr, hr, er in scenarios:
    table.append({
        "Attacker": label,
        "No defense": f"{hb}/{nb} ({100 * hb / nb:.1f}%)" if nb else "n/a",
        "RLS on": (f"{hr}/{nr} ({100 * hr / nr:.1f}%)" if nr and er == 0 else "not run / incomplete"),
        "Fisher p": (f"{fisher_two_sided(hb, nb - hb, hr, nr - hr):.4f}" if nb and nr and er == 0 else "-"),
    })
st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

complete = [s for s in scenarios if s[1] and s[4] and s[6] == 0]
if complete:
    chart = pd.DataFrame(
        {
            "No defense": [100 * s[2] / s[1] for s in complete],
            "RLS on": [100 * s[5] / s[4] for s in complete],
        },
        index=[s[0] for s in complete],
    )
    st.bar_chart(chart)
    st.caption("Attack success rate (%), identical attacks in both arms.")

st.info(
    "A hit means a planted secret owned by another tenant appeared in the answer. "
    "Errors and timeouts are excluded, never counted as blocked attacks."
)

st.header("2. Prompt injection and tool abuse (no defense)")
inj_rows = []
for key, label in [
    (("prompt_injection", 1), "Prompt injection, tier 1"),
    (("prompt_injection", 2), "Prompt injection, tier 2"),
    (("tool_abuse", 3), "Tool abuse, tier 3"),
]:
    group = [r for r in inj if (r["attack_class"], r["tier"]) == key]
    valid = [r for r in group if not r.get("error")]
    exposed = [r for r in valid if r.get("target_retrieved")]
    inj_rows.append({
        "Attack": label,
        "Valid attacks": len(valid),
        "Injected email reached model": len(exposed),
        "Successful": sum(1 for r in valid if r["success"]),
        "Excluded (errors)": len(group) - len(valid),
    })
if inj:
    st.dataframe(pd.DataFrame(inj_rows), use_container_width=True, hide_index=True)
st.write(
    "The model read the injected instructions and ignored them. Because the undefended baseline is already 0%, "
    "the scanner and tool guard are evaluated by detection quality and direct control checks, shown below."
)

st.header("3. Defense evidence")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Injection scanner (DistilBERT)")
    st.dataframe(pd.DataFrame({
        "Evaluation": ["Known families (doc-level test)", "Held-out template families (macro)", "Tier 2 held-out (hardest)"],
        "F1": [1.0, 0.9167, 0.75],
    }), hide_index=True, use_container_width=True)
with c2:
    st.subheader("Capability tool guard")
    st.write("Verified with 7 direct synthetic tool-call checks (eval/check_tool_guard.py), "
             "not through stochastic model behaviour.")

with st.expander("Limitations"):
    st.markdown(
        "- Measured ASR is bounded by retrieval recall: the target email reached the model in only part of the attacks.\n"
        "- One injection attack (INJ-0010) was excluded because its context exceeded the provider's per-request token limit.\n"
        "- Scanner F1 on known families uses the same 12 templates in train and test; the held-out score is more representative.\n"
        "- A daily token cap limited the evaluation to a stratified 98-attack subset."
    )