#!/usr/bin/env python3
"""Direct verification of the capability-based tool guard (no LLM needed)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CONFIG_DIR", str(REPO_ROOT / "configs"))
sys.path.insert(0, str(REPO_ROOT / "services" / "api"))

from app.tool_guard import ToolGuardError, intercept_tool_calls


failures: List[str] = []


def check(name: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        failures.append(name)


def run_check(name: str, test: Callable[[], bool]) -> None:
    try:
        check(name, test())
    except (ToolGuardError, OSError, RuntimeError) as exc:
        print(f"[FAIL] {name} ({type(exc).__name__})")
        failures.append(name)


def first_call_status(tenant_id: str, tool_call: dict, taint_flags=None):
    results, notes = intercept_tool_calls(
        tenant_id,
        [tool_call],
        taint_flags=taint_flags,
    )
    if len(results) != 1:
        raise RuntimeError("Tool guard did not return exactly one result.")
    return results[0], notes


email_call = {
    "tool_name": "email_send",
    "arguments": {
        "to": "x@example.com",
        "subject": "synthetic guard check",
        "body": "stub test only",
    },
}


def check_tenant_a_email() -> bool:
    result, _ = first_call_status("A", email_call)
    return (
        result.status == "allowed"
        and "stubbed only" in result.reason.lower()
        and "no external action" in result.reason.lower()
    )


def check_tenant_b_email() -> bool:
    result, _ = first_call_status("B", email_call)
    return result.status == "blocked"


def check_tenant_c_http_post() -> bool:
    result, _ = first_call_status(
        "C",
        {
            "tool_name": "http_post",
            "arguments": {
                "url": "https://api.internal.example.com/x",
                "payload": {"synthetic": True},
            },
        },
    )
    return (
        result.status == "allowed"
        and "stubbed only" in result.reason.lower()
        and "no external action" in result.reason.lower()
    )


def check_tenant_c_db_query() -> bool:
    result, _ = first_call_status(
        "C",
        {
            "tool_name": "db_query",
            "arguments": {"sql": "SELECT * FROM chunks"},
        },
    )
    return result.status == "blocked"


def check_unknown_tool() -> bool:
    result, _ = first_call_status(
        "A",
        {"tool_name": "shell_exec", "arguments": {}},
    )
    return result.status == "blocked"


def check_taint_revokes_capabilities() -> bool:
    result, notes = first_call_status(
        "A",
        email_call,
        taint_flags=["TAINT_INJECTION_DETECTED: chunk_1"],
    )
    return (
        result.status == "blocked"
        and "revoked" in result.reason.lower()
        and any("revoked" in note.lower() for note in notes)
    )


def check_missing_required_argument() -> bool:
    result, _ = first_call_status(
        "A",
        {
            "tool_name": "email_send",
            "arguments": {"to": "x@example.com"},
        },
    )
    return result.status == "blocked"


def main() -> int:
    run_check("Tenant A email_send allowed (stub only)", check_tenant_a_email)
    run_check("Tenant B email_send blocked", check_tenant_b_email)
    run_check("Tenant C http_post allowed (stub only)", check_tenant_c_http_post)
    run_check("Tenant C db_query blocked", check_tenant_c_db_query)
    run_check("Unknown tool blocked", check_unknown_tool)
    run_check("Taint revokes capabilities", check_taint_revokes_capabilities)
    run_check("Missing required argument blocked", check_missing_required_argument)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("All tool guard checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
