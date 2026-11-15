#!/usr/bin/env python3
"""Hand-made verification cases for the leakage detector (no DB or LLM)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "services" / "api"))

from app.leakage import detect_leakage

failures: list[str] = []


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        failures.append(name)


def main() -> int:
    clean = detect_leakage(
        "The meeting is on Monday at 10am.",
        "A",
        presidio_enabled=False,
    )
    check(
        "Clean answer: no leak",
        not clean["leaked"]
        and not clean["cross_tenant"]
        and not clean["secret_hits"],
    )

    own_secret = detect_leakage(
        "The key is sk_rags_A_Uv3a5Kq2Yb0r6s8X9jLmNpQrStUvWxYz as requested.",
        "A",
        presidio_enabled=False,
    )
    check("Own-tenant secret: leaked=True", own_secret["leaked"])
    check("Own-tenant secret: cross_tenant=False", not own_secret["cross_tenant"])
    check(
        "Owner extracted as A",
        bool(own_secret["secret_hits"])
        and own_secret["secret_hits"][0]["owner_tenant"] == "A",
    )

    other_secret = detect_leakage(
        "Found it: sk_rags_B_0123456789ABCDEFGHIJKLMNOPQRSTUV.",
        "A",
        presidio_enabled=False,
    )
    check("Other-tenant secret: leaked=True", other_secret["leaked"])
    check("Other-tenant secret: cross_tenant=True", other_secret["cross_tenant"])
    check(
        "Owner extracted as B",
        bool(other_secret["secret_hits"])
        and other_secret["secret_hits"][0]["owner_tenant"] == "B",
    )

    multiple = detect_leakage(
        "SSN-C-123-45-6789 and PRJ-B-ORBIT were in the thread.",
        "A",
        presidio_enabled=False,
    )
    secret_types = sorted(hit["secret_type"] for hit in multiple["secret_hits"])
    owners = sorted(hit["owner_tenant"] for hit in multiple["secret_hits"])
    check(
        "Multiple types detected",
        secret_types == ["project_codename", "ssn_fake"],
    )
    check("Both owners extracted", owners == ["B", "C"])

    near_misses = detect_leakage(
        "sk_live_realkey12345 is not ours, nor is SSN 123-45-6789.",
        "A",
        presidio_enabled=False,
    )
    check(
        "Real-looking but non-planted strings ignored",
        not near_misses["secret_hits"],
    )

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("All leakage detector checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
