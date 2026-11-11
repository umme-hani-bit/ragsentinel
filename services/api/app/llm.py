from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml


class LLMError(Exception):
    """Raised when the hosted LLM cannot produce a usable response."""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    model: str
    api_key_env: str
    timeout_s: float = 60.0
    max_tokens: int = 1024
    temperature: float = 0.1
    json_mode: bool = False
    max_retries: int = 2


def _candidate_config_paths():
    override = os.getenv("PROVIDER_CONFIG")
    if override:
        yield Path(override)
    yield Path("/app/configs/provider.yaml")
    yield Path("configs/provider.yaml")


def _load_yaml_llm_block() -> Dict[str, Any]:
    for path in _candidate_config_paths():
        try:
            if not path.exists():
                continue
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(raw, dict):
            block = raw.get("llm")
            if isinstance(block, dict):
                return block
    return {}


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def load_llm_settings() -> LLMSettings:
    block = _load_yaml_llm_block()

    base_url = (os.getenv("LLM_BASE_URL") or block.get("base_url") or "").strip()
    model = (os.getenv("LLM_MODEL") or block.get("model") or "").strip()
    api_key_env = (
        os.getenv("LLM_API_KEY_ENV") or block.get("api_key_env") or "LLM_API_KEY"
    ).strip()

    if not base_url:
        raise LLMError(
            "LLM base URL is not configured. Set LLM_BASE_URL or llm.base_url "
            "in configs/provider.yaml."
        )
    if not model:
        raise LLMError(
            "LLM model is not configured. Set LLM_MODEL or llm.model in "
            "configs/provider.yaml."
        )

    json_mode_raw = os.getenv("LLM_JSON_MODE")
    json_mode = (
        _as_bool(json_mode_raw)
        if json_mode_raw is not None
        else _as_bool(block.get("json_mode"), False)
    )

    timeout_value = os.getenv("LLM_TIMEOUT_S")
    if timeout_value is None:
        timeout_value = block.get("timeout_s", block.get("timeout_seconds"))

    return LLMSettings(
        base_url=base_url.rstrip("/"),
        model=model,
        api_key_env=api_key_env,
        timeout_s=_as_float(timeout_value, 60.0),
        max_tokens=_as_int(os.getenv("LLM_MAX_TOKENS", block.get("max_tokens")), 1024),
        temperature=_as_float(
            os.getenv("LLM_TEMPERATURE", block.get("temperature")),
            0.1,
        ),
        json_mode=json_mode,
        max_retries=_as_int(os.getenv("LLM_MAX_RETRIES", block.get("max_retries")), 2),
    )


def api_key_for(settings: LLMSettings) -> str:
    return os.getenv(settings.api_key_env, "") or os.getenv("LLM_API_KEY", "")


def chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def llm_mode() -> str:
    """auto = hosted when a key is present, otherwise stub fallback."""
    mode = (os.getenv("LLM_MODE") or "auto").strip().lower()
    if mode not in {"auto", "hosted", "stub"}:
        raise LLMError(f"Invalid LLM_MODE: {mode!r}. Use auto|hosted|stub.")
    return mode


def llm_settings_summary() -> Dict[str, Any]:
    """Safe, key-free summary for /configz."""
    try:
        settings = load_llm_settings()
    except LLMError as exc:
        return {
            "mode": os.getenv("LLM_MODE", "auto"),
            "configured": False,
            "error": str(exc),
        }

    return {
        "mode": llm_mode(),
        "configured": bool(api_key_for(settings)),
        "base_url": settings.base_url,
        "model": settings.model,
        "json_mode": settings.json_mode,
        "timeout_s": settings.timeout_s,
        "api_key_env": settings.api_key_env,
    }


def _extract_content(data: Any) -> str:
    if not isinstance(data, dict):
        raise LLMError("LLM response root is not a JSON object.")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("LLM response contained no choices.")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None

    if not isinstance(content, str) or not content.strip():
        raise LLMError("LLM response contained empty content.")

    return content


def generate_llm_response(
    prompt: str,
    settings: Optional[LLMSettings] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Call an OpenAI-compatible /chat/completions endpoint and return text."""
    settings = settings or load_llm_settings()

    api_key = api_key_for(settings)
    if not api_key:
        raise LLMError(
            f"Missing LLM API key. Set {settings.api_key_env} (or LLM_API_KEY) "
            "in the API container environment."
        )

    endpoint = chat_endpoint(settings.base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    base_payload: Dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }

    json_mode = settings.json_mode
    last_error = "unknown error"
    attempts = max(0, settings.max_retries) + 1

    for attempt in range(1, attempts + 1):
        payload = dict(base_payload)
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=settings.timeout_s,
            )
        except requests.exceptions.RequestException as exc:
            last_error = f"network error: {type(exc).__name__}: {exc}"
            if attempt < attempts:
                time.sleep(1.5 * attempt)
                continue
            raise LLMError(last_error) from exc

        if response.status_code == 200:
            try:
                return _extract_content(response.json())
            except ValueError as exc:
                raise LLMError("LLM returned non-JSON body.") from exc

        body_preview = (response.text or "")[:300]

        # Some providers reject response_format. Drop it once and retry.
        if (
            response.status_code in (400, 422)
            and json_mode
            and "response_format" in body_preview.lower()
        ):
            json_mode = False
            continue

        if response.status_code in (401, 403):
            raise LLMError(
                f"LLM authentication failed [{response.status_code}]. "
                f"Check {settings.api_key_env}: {body_preview}"
            )

        last_error = f"HTTP {response.status_code}: {body_preview}"

        if response.status_code == 429 or response.status_code >= 500:
            if attempt < attempts:
                retry_after = response.headers.get("Retry-After")
                delay = 2.0 * attempt
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                time.sleep(delay)
                continue
            raise LLMError(f"LLM unavailable after retries: {last_error}")

        raise LLMError(f"LLM request failed: {last_error}")

    raise LLMError(f"LLM failed after retries: {last_error}")
