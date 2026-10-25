#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

CFG_PATH = Path("configs/secrets.yaml")


class SecretsConfigError(Exception):
    pass


@dataclass
class SecretType:
    name: str
    regex: re.Pattern[str]
    examples: List[str]


def _load_cfg(path: Path) -> Dict:
    if not path.exists():
        raise SecretsConfigError(f"Missing config file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SecretsConfigError("Config root must be a mapping.")
    return data


def _parse_types(cfg: Dict) -> List[SecretType]:
    types = cfg.get("types", {})
    if not isinstance(types, dict) or not types:
        raise SecretsConfigError("No secret types defined under 'types'.")

    out: List[SecretType] = []
    for name, spec in types.items():
        if not isinstance(spec, dict):
            raise SecretsConfigError(f"Type '{name}' must be a mapping.")

        rx = spec.get("detect_regex")
        ex = spec.get("examples", [])

        if not rx:
            raise SecretsConfigError(f"Type '{name}' must have detect_regex.")
        if not ex:
            raise SecretsConfigError(f"Type '{name}' must have at least one example.")

        try:
            pat = re.compile(rx)
        except re.error as e:
            raise SecretsConfigError(f"Invalid regex for '{name}': {e}") from e

        out.append(SecretType(name=name, regex=pat, examples=list(ex)))

    return out


def _validate_examples(st: SecretType) -> None:
    for s in st.examples:
        if not st.regex.search(s):
            raise SecretsConfigError(
                f"Regex for '{st.name}' did not match example: {s!r}"
            )

    # Negative control: a normal sentence should NOT match
    negative = "this string contains no secrets at all"
    if st.regex.search(negative):
        raise SecretsConfigError(
            f"Regex for '{st.name}' matched a negative control string."
        )


def main() -> int:
    cfg = _load_cfg(CFG_PATH)
    types = _parse_types(cfg)

    for st in types:
        _validate_examples(st)

    total_examples = sum(len(t.examples) for t in types)
    print(f"All secret regex tests passed ({len(types)} types, {total_examples} examples).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
