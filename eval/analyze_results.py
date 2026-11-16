"""Aggregate attack-harness JSONL results into ASR and latency tables.

Usage:
    python eval/analyze_results.py eval/results/attacks_baseline.jsonl
    python eval/analyze_results.py eval/results/attacks_baseline.jsonl eval/results/attacks_rls.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

SECRET_PREFIXES = (
    ("rgsjwt", "jwt"),
    ("sk_rags", "api_key"),
    ("RAGCARD", "card"),
    ("SSN", "ssn"),
    ("PRJ", "project_code"),
)


def secret_type(evidence: str) -> str:
    m = re.search(r"target_secret=([^\s,]+)", evidence or "")
    if not m:
        return "unknown"
    token = m.group(1)
    for prefix, label in SECRET_PREFIXES:
        if token.startswith(prefix):
            return label
    return "other"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rate(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else "0/0"


def profile_name(path: Path) -> str:
    return path.stem.replace("attacks_", "")


def summarize(path: Path) -> dict:
    rows = load_rows(path)
    print(f"\n=== {profile_name(path)} ({path}) ===")
    print(f"rows: {len(rows)}")

    fingerprints = {r.get("config_fingerprint") for r in rows}
    if len(fingerprints) > 1:
        print(f"WARNING: mixed config_fingerprint values: {fingerprints}")
    profiles = {json.dumps(r.get("defense_profile", {}), sort_keys=True) for r in rows}
    if len(profiles) > 1:
        print("WARNING: mixed defense_profile values in one file:")
        for p in sorted(profiles):
            print(f"  {p}")
    else:
        print(f"defense_profile: {next(iter(profiles))}")

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r.get("attack_class", "?")].append(r)

    class_asr = {}
    for cls, cls_rows in sorted(by_class.items()):
        ok = sum(1 for r in cls_rows if r.get("success"))
        class_asr[cls] = (ok, len(cls_rows))
        lat = [float(r.get("latency_ms", 0.0)) for r in cls_rows]
        print(f"\n[{cls}] ASR {rate(ok, len(cls_rows))} | "
              f"latency mean={statistics.fmean(lat):.0f}ms "
              f"median={statistics.median(lat):.0f}ms max={max(lat):.0f}ms")

        if cls == "cross_tenant_leakage":
            for dim, keyfn in (
                ("target_tenant", lambda r: r.get("target_tenant", "?")),
                ("secret_type", lambda r: secret_type(r.get("evidence", ""))),
            ):
                groups: dict[str, list[bool]] = defaultdict(list)
                for r in cls_rows:
                    groups[keyfn(r)].append(bool(r.get("success")))
                parts = ", ".join(f"{k}: {rate(sum(v), len(v))}" for k, v in sorted(groups.items()))
                print(f"  by {dim}: {parts}")
        else:
            tiers: dict[str, list[bool]] = defaultdict(list)
            for r in cls_rows:
                tiers[str(r.get("tier"))].append(bool(r.get("success")))
            if not (len(tiers) == 1 and "None" in tiers):
                parts = ", ".join(f"tier {k}: {rate(sum(v), len(v))}" for k, v in sorted(tiers.items()))
                print(f"  by tier: {parts}")
            if cls == "tool_abuse":
                att = sum(int(r.get("tool_calls_attempted", 0)) for r in cls_rows)
                blk = sum(int(r.get("tool_calls_blocked", 0)) for r in cls_rows)
                print(f"  tool calls: attempted={att} blocked={blk}")

    total_ok = sum(1 for r in rows if r.get("success"))
    print(f"\noverall ASR: {rate(total_ok, len(rows))}")
    return class_asr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()

    per_profile = {}
    for f in args.files:
        per_profile[profile_name(f)] = summarize(f)

    if len(per_profile) > 1:
        classes = sorted({c for asr in per_profile.values() for c in asr})
        print("\n=== ASR comparison across profiles ===")
        print(f"{'attack_class':<24}" + "".join(f"{p:>22}" for p in per_profile))
        for cls in classes:
            line = f"{cls:<24}"
            for p, asr in per_profile.items():
                ok, n = asr.get(cls, (0, 0))
                line += f"{rate(ok, n):>22}"
            print(line)


if __name__ == "__main__":
    main()