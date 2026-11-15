from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from email.utils import parseaddr
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import psycopg2
import yaml

CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "configs"))
POLICIES_PATH = CONFIG_DIR / "policies.yaml"
TOOLS_PATH = CONFIG_DIR / "tools.yaml"
TOOL_LOGS_TABLE = "tool_logs"
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragsvc:ragsvc@localhost:5432/ragsentinel",
)


class ToolGuardError(Exception):
    """Raised when tool validation or required audit logging fails."""


@dataclass(frozen=True)
class ToolCallResult:
    tool_name: str
    arguments: Dict[str, Any]
    status: str
    reason: str
    execution_time_ms: float


class ToolGuard:
    def __init__(
        self,
        policies_path: Path = POLICIES_PATH,
        tools_path: Path = TOOLS_PATH,
        db_url: str = DB_URL,
    ) -> None:
        self.policies = self._load_yaml(policies_path)
        self.tools = self._load_yaml(tools_path)
        self.db_url = db_url
        self.tool_defs = self.tools.get("tools")
        self.policies_tenants = self.policies.get("tenants")
        self.policies_tools = self.policies.get("tools")
        if not all(
            isinstance(value, dict)
            for value in (self.tool_defs, self.policies_tenants, self.policies_tools)
        ):
            raise ToolGuardError("Tool and policy configurations must define mappings.")

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ToolGuardError(f"Unable to load tool guard config: {path.name}") from exc
        if not isinstance(data, dict):
            raise ToolGuardError(f"Tool guard config root must be a mapping: {path.name}")
        return data

    def _tenant_capabilities(self, tenant_id: str, user: str) -> List[str]:
        tenant = self.policies_tenants.get(tenant_id)
        if not isinstance(tenant, dict):
            return []
        users = tenant.get("users", {})
        user_config = users.get(user, {}) if isinstance(users, dict) else {}
        if not isinstance(user_config, dict):
            return []
        capabilities = user_config.get("capabilities", [])
        if not isinstance(capabilities, list) or any(
            not isinstance(capability, str) for capability in capabilities
        ):
            raise ToolGuardError("Tenant capabilities must be a list of strings.")
        return capabilities

    def _has_capability(
        self,
        tenant_id: str,
        tool_name: str,
        user: str,
    ) -> Tuple[bool, str]:
        capabilities = self._tenant_capabilities(tenant_id, user)
        policy = self.policies_tools.get(tool_name, {})
        if not isinstance(policy, dict):
            return False, "Tool policy is missing or invalid."

        if policy.get("default_capability", False) is not True:
            permitted = any(
                capability == tool_name or capability.startswith(f"{tool_name}:")
                for capability in capabilities
            )
            if not permitted:
                return False, f"No capability grants access to '{tool_name}'."
        return True, f"Capability grants access to '{tool_name}'."

    def _validate_arguments(
        self,
        tool_name: str,
        arguments: Any,
    ) -> Tuple[bool, str]:
        if not isinstance(arguments, dict):
            return False, "Arguments must be an object."

        tool_definition = self.tool_defs.get(tool_name)
        if not isinstance(tool_definition, dict):
            return False, "Tool definition is invalid."
        parameters = tool_definition.get("parameters", {})
        if not isinstance(parameters, dict):
            return False, "Tool parameter definition is invalid."

        unknown = set(arguments) - set(parameters)
        if unknown:
            return False, f"Unknown parameters: {', '.join(sorted(unknown))}."

        for name, spec in parameters.items():
            if not isinstance(spec, dict):
                return False, f"Parameter definition for '{name}' is invalid."
            if spec.get("required", False) and name not in arguments:
                return False, f"Missing required parameter: {name}."
            if name not in arguments:
                continue
            value = arguments[name]
            expected_type = spec.get("type")
            if expected_type == "string" and not isinstance(value, str):
                return False, f"Parameter '{name}' must be a string."
            if expected_type == "object" and not isinstance(value, dict):
                return False, f"Parameter '{name}' must be an object."
        return True, "Arguments are valid."

    def _validate_constraints(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, str]:
        policy = self.policies_tools.get(tool_name, {})
        if tool_name == "email_send":
            address = parseaddr(arguments["to"])[1]
            if address.count("@") != 1:
                return False, "Recipient must be a valid email address."
            domain = address.rsplit("@", 1)[1].rstrip(".").casefold()
            allowed = {
                item.rstrip(".").casefold()
                for item in policy.get("allowed_domains", [])
                if isinstance(item, str)
            }
            if not domain or domain not in allowed:
                return False, "Recipient domain is not allowed."

        elif tool_name == "http_post":
            try:
                parsed = urlsplit(arguments["url"])
                port = parsed.port
            except ValueError:
                return False, "URL is invalid."
            host = (parsed.hostname or "").rstrip(".").casefold()
            allowed = {
                item.rstrip(".").casefold()
                for item in policy.get("allowed_hosts", [])
                if isinstance(item, str)
            }
            if (
                parsed.scheme.casefold() != "https"
                or parsed.username is not None
                or parsed.password is not None
                or port not in (None, 443)
                or host not in allowed
            ):
                return False, "URL must use HTTPS and an explicitly allowed host."

        elif tool_name == "db_query":
            sql = arguments["sql"]
            if (
                not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE)
                or ";" in sql
                or "--" in sql
                or "/*" in sql
                or "*/" in sql
                or re.search(
                    r"\b(INSERT|UPDATE|DELETE|MERGE|DROP|ALTER|CREATE|GRANT|"
                    r"REVOKE|COPY|CALL|DO|TRUNCATE|SET|RESET)\b",
                    sql,
                    re.IGNORECASE,
                )
            ):
                return False, "Only a single read-only SELECT query is permitted."
            tables = {
                table.casefold()
                for table in re.findall(
                    r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_$]*)",
                    sql,
                    re.IGNORECASE,
                )
            }
            allowed_tables = {
                table.casefold()
                for table in policy.get("allowed_tables", [])
                if isinstance(table, str)
            }
            if not tables or not tables <= allowed_tables:
                return False, "Query references a table that is not allowed."

        return True, "Policy constraints passed."

    def _decide_and_audit(
        self,
        tenant_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        user: str,
        tainted: bool,
    ) -> Tuple[str, str]:
        conn = None
        try:
            conn = psycopg2.connect(self.db_url)
            with conn.cursor() as cur:
                cur.execute("SET LOCAL app.tenant_id = %s", (tenant_id,))
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s)::bigint)",
                    (f"{tenant_id}:{tool_name}",),
                )
                cur.execute(
                    f"""
                    SELECT count(*)
                    FROM {TOOL_LOGS_TABLE}
                    WHERE tenant_id = %s
                      AND tool_name = %s
                      AND created_at >= now() - interval '1 minute'
                    """,
                    (tenant_id, tool_name),
                )
                recent_calls = int(cur.fetchone()[0])

                if tainted:
                    allowed, reason = False, "Injection taint revoked tool capabilities."
                elif tool_name not in self.tool_defs:
                    allowed, reason = False, f"Unknown tool '{tool_name}'."
                else:
                    allowed, reason = self._has_capability(tenant_id, tool_name, user)
                    if allowed:
                        valid, validation_reason = self._validate_arguments(
                            tool_name,
                            arguments,
                        )
                        if not valid:
                            allowed, reason = False, validation_reason
                    if allowed:
                        valid, validation_reason = self._validate_constraints(
                            tool_name,
                            arguments,
                        )
                        if not valid:
                            allowed, reason = False, validation_reason

                policy = self.policies_tools.get(tool_name, {})
                limit = policy.get("rate_limit_per_minute", 0) if isinstance(policy, dict) else 0
                if allowed and (not isinstance(limit, int) or limit <= 0):
                    allowed, reason = False, "Tool rate limit is not configured."
                elif allowed and recent_calls >= limit:
                    allowed, reason = False, "Tool rate limit exceeded."

                payload = json.dumps({"argument_names": sorted(arguments)})
                cur.execute(
                    """
                    INSERT INTO tool_logs
                        (tenant_id, tool_name, allowed, reason, payload)
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    """,
                    (tenant_id, tool_name, allowed, reason, payload),
                )
            conn.commit()
        except (psycopg2.Error, TypeError, ValueError, KeyError) as exc:
            if conn is not None:
                conn.rollback()
            raise ToolGuardError(
                "Tool authorization/audit failed; the call was blocked."
            ) from exc
        finally:
            if conn is not None:
                conn.close()
        return ("allowed" if allowed else "blocked"), reason

    def intercept_and_execute(
        self,
        tenant_id: str,
        tool_calls: List[Dict[str, Any]],
        user: str = "default",
        taint_flags: Optional[List[str]] = None,
    ) -> Tuple[List[ToolCallResult], List[str]]:
        results: List[ToolCallResult] = []
        notes: List[str] = []
        tainted = bool(taint_flags)

        for call in tool_calls:
            started = time.perf_counter()
            if not isinstance(call, dict):
                raise ToolGuardError("Malformed tool call received from model.")
            tool_name = call.get("tool_name")
            arguments = call.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name:
                raise ToolGuardError("Tool call is missing a valid tool_name.")
            if not isinstance(arguments, dict):
                arguments = {}

            status, reason = self._decide_and_audit(
                tenant_id,
                tool_name,
                arguments,
                user,
                tainted,
            )
            if status == "allowed":
                reason += " Stubbed only; no external action was performed."
            result = ToolCallResult(
                tool_name=tool_name,
                arguments=arguments,
                status=status,
                reason=reason,
                execution_time_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            results.append(result)
            notes.append(f"{status.upper()}: {tool_name}: {reason}")

        return results, notes


@lru_cache(maxsize=1)
def get_tool_guard() -> ToolGuard:
    return ToolGuard()


def intercept_tool_calls(
    tenant_id: str,
    tool_calls: List[Dict[str, Any]],
    user: str = "default",
    taint_flags: Optional[List[str]] = None,
) -> Tuple[List[ToolCallResult], List[str]]:
    return get_tool_guard().intercept_and_execute(
        tenant_id,
        tool_calls,
        user,
        taint_flags,
    )
