#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

CFG = Path("configs/injections.json")

ALLOWED_INTENTS = {"prompt_override", "cross_tenant_leakage", "tool_exfiltration"}
ALLOWED_ENCODINGS = {"none", "base64"}
ALLOWED_PLACEHOLDERS = {
    "TENANT",
    "THIS_TENANT",
    "OTHER_TENANT",
    "EXFIL_URL",
    "EXFIL_EMAIL",
}


class ConfigError(Exception):
    pass


def load_cfg(path: Path) -> Dict:
    if not path.exists():
        raise ConfigError(f"Missing config: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "payloads" not in data:
        raise ConfigError("Config must be an object with a 'payloads' array.")
    return data


def find_placeholders(text: str) -> Set[str]:
    # Matches {TOKEN}, used to verify declared placeholders
    return set(re.findall(r"{([A-Z_]+)}", text or ""))


def validate_payloads(cfg: Dict) -> Tuple[int, Counter]:
    payloads = cfg["payloads"]
    if not isinstance(payloads, list) or not payloads:
        raise ConfigError("No payloads found.")

    ids: Set[str] = set()
    by_tier: Counter = Counter()

    for p in payloads:
        pid = p.get("id", "")
        tier = int(p.get("tier", 0))
        intents = set(p.get("intent", []))
        enc = p.get("encoding", "none")
        template = p.get("template", "")
        wrapper = p.get("wrapper", "")
        declared = set(p.get("placeholders", []))

        if not pid or pid in ids:
            raise ConfigError(f"Duplicate or missing id: {pid!r}")
        ids.add(pid)

        if tier not in (1, 2, 3):
            raise ConfigError(f"{pid}: invalid tier {tier}")
        by_tier[tier] += 1

        if not intents or not intents.issubset(ALLOWED_INTENTS):
            raise ConfigError(
                f"{pid}: intents must be subset of {sorted(ALLOWED_INTENTS)}"
            )

        if enc not in ALLOWED_ENCODINGS:
            raise ConfigError(f"{pid}: encoding must be one of {sorted(ALLOWED_ENCODINGS)}")

        # Tier 3 must include tool_exfiltration intent
        if tier == 3 and "tool_exfiltration" not in intents:
            raise ConfigError(f"{pid}: tier 3 requires 'tool_exfiltration' intent")

        # Placeholders: union of those found in template+wrapper (minus the special {B64})
        found = find_placeholders(template) | find_placeholders(wrapper)
        found.discard("B64")

        unknown = found - ALLOWED_PLACEHOLDERS
        if unknown:
            raise ConfigError(f"{pid}: unknown placeholder(s): {sorted(unknown)}")

        if not found.issubset(declared):
            missing = sorted(found - declared)
            raise ConfigError(f"{pid}: placeholders not declared: {missing}")

        if declared - ALLOWED_PLACEHOLDERS:
            bad = sorted(declared - ALLOWED_PLACEHOLDERS)
            raise ConfigError(f"{pid}: declared placeholder(s) not allowed: {bad}")

        # If base64, wrapper must contain {B64} token to receive encoded text
        if enc == "base64" and "{B64}" not in (wrapper or ""):
            raise ConfigError(
                f"{pid}: base64 payload requires a 'wrapper' with {{B64}} token"
            )

        # Keep templates shortish so they fit in chunks comfortably
        if len(template) > 600:
            raise ConfigError(f"{pid}: template too long (>600 chars)")

    return len(payloads), by_tier


def main() -> int:
    cfg = load_cfg(CFG)
    total, by_tier = validate_payloads(cfg)
    print(
        f"Injection payload config valid: {total} payloads "
        f"(tier1={by_tier[1]}, tier2={by_tier[2]}, tier3={by_tier[3]})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
