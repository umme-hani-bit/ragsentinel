#!/usr/bin/env python3
"""Sanity check for the NLI verifier with hand-made cases (no LLM or DB)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CONFIG_DIR", str(REPO_ROOT / "configs"))
sys.path.insert(0, str(REPO_ROOT / "services" / "api"))

from app.nli import verify_faithfulness

PREMISE = [
    {
        "chunk_id": 1,
        "doc_id": "d1",
        "tenant_id": "A",
        "text": (
            "The board meeting is scheduled for 10 AM on Monday, March 5, "
            "2001, in Houston."
        ),
    }
]


def main() -> int:
    print("Loading NLI model and checking supported claim...", flush=True)
    supported = verify_faithfulness(
        "The board meeting is on Monday.",
        PREMISE,
    )
    print("Case 1 (supported):", supported["verdicts"], flush=True)
    assert supported["sentences_total"] == 1
    assert supported["sentences_supported"] == 1, (
        "FAIL: supported claim was marked unsupported"
    )

    hallucinated = verify_faithfulness(
        "The CEO resigned after the scandal in 2002.",
        PREMISE,
    )
    print("Case 2 (hallucinated):", hallucinated["verdicts"], flush=True)
    assert hallucinated["sentences_supported"] == 0, (
        "FAIL: hallucinated claim was marked supported"
    )

    mixed = verify_faithfulness(
        "The board meeting is on Monday. The CEO resigned in 2002.",
        PREMISE,
    )
    print(
        "Case 3 (mixed):",
        mixed["unsupported_claim_rate"],
        mixed["verdicts"],
        flush=True,
    )
    assert abs(mixed["unsupported_claim_rate"] - 0.5) < 1e-6, (
        "FAIL: unsupported claim rate should be 0.5"
    )

    print("\nAll NLI checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
