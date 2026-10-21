from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import yaml


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Seeds:
    partition: int
    payloads: int
    qa: int


@dataclass(frozen=True)
class CorpusConfig:
    dataset: Literal["enron", "wikipedia", "ragtruth"]
    dataset_version: str
    tenants: int
    seeds: Seeds


@dataclass(frozen=True)
class DefensesConfig:
    rls_enabled: bool
    injection_scanner_enabled: bool
    scanner_threshold: float
    tool_guard_enabled: bool


@dataclass(frozen=True)
class ProviderConfig:
    llm_base_url: str
    llm_model: str
    llm_api_key_env: str
    embeddings_model: str
    reranker_model: str
    nli_model: str


@dataclass(frozen=True)
class ConfigBundle:
    corpus: CorpusConfig
    defenses: DefensesConfig
    policies: Dict[str, Any]
    provider: ProviderConfig
    fingerprint: str
    config_dir: Path


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping at top-level: {path}")
    return data


def _sha256_of_obj(obj: Dict[str, Any]) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_keys(d: Dict[str, Any], keys: list[str], label: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError(f"Missing keys in {label}: {missing}")


def load_bundle(config_dir: str | os.PathLike[str]) -> ConfigBundle:
    base = Path(config_dir)

    corpus_raw = _load_yaml_file(base / "corpus.yaml")
    defenses_raw = _load_yaml_file(base / "defenses.yaml")
    policies_raw = _load_yaml_file(base / "policies.yaml")
    provider_raw = _load_yaml_file(base / "provider.yaml")

    _require_keys(corpus_raw, ["dataset", "dataset_version", "tenants", "seeds"], "corpus.yaml")
    _require_keys(corpus_raw["seeds"], ["partition", "payloads", "qa"], "corpus.yaml:seeds")

    seeds = Seeds(
        partition=int(corpus_raw["seeds"]["partition"]),
        payloads=int(corpus_raw["seeds"]["payloads"]),
        qa=int(corpus_raw["seeds"]["qa"]),
    )
    corpus = CorpusConfig(
        dataset=str(corpus_raw["dataset"]),
        dataset_version=str(corpus_raw["dataset_version"]),
        tenants=int(corpus_raw["tenants"]),
        seeds=seeds,
    )

    _require_keys(defenses_raw, ["rls", "injection_scanner", "tool_guard"], "defenses.yaml")
    defenses = DefensesConfig(
        rls_enabled=bool(defenses_raw["rls"].get("enabled", False)),
        injection_scanner_enabled=bool(defenses_raw["injection_scanner"].get("enabled", False)),
        scanner_threshold=float(defenses_raw["injection_scanner"].get("threshold", 0.5)),
        tool_guard_enabled=bool(defenses_raw["tool_guard"].get("enabled", False)),
    )

    _require_keys(provider_raw, ["llm", "embeddings", "reranker", "nli"], "provider.yaml")
    llm = provider_raw["llm"]
    provider = ProviderConfig(
        llm_base_url=str(llm.get("base_url", "")),
        llm_model=str(llm.get("model", "")),
        llm_api_key_env=str(llm.get("api_key_env", "LLM_API_KEY")),
        embeddings_model=str(provider_raw["embeddings"].get("model", "")),
        reranker_model=str(provider_raw["reranker"].get("model", "")),
        nli_model=str(provider_raw["nli"].get("model", "")),
    )

    merged_for_hash = {
        "corpus": corpus_raw,
        "defenses": defenses_raw,
        "policies": policies_raw,
        "provider": provider_raw,
    }
    fingerprint = _sha256_of_obj(merged_for_hash)

    return ConfigBundle(
        corpus=corpus,
        defenses=defenses,
        policies=policies_raw,
        provider=provider,
        fingerprint=fingerprint,
        config_dir=base,
    )
