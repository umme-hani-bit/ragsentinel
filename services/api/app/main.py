import os

from app.config import ConfigError, load_bundle
from fastapi import FastAPI

app = FastAPI(title="RAGsentinel API")

CONFIG_DIR = os.getenv("CONFIG_DIR", "/app/configs")


@app.on_event("startup")
def load_config() -> None:
    try:
        app.state.config_bundle = load_bundle(CONFIG_DIR)
    except ConfigError as exc:
        raise RuntimeError(f"Config load failed: {exc}") from exc


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "ragsentinel-api", "status": "ok"}


@app.get("/configz")
def configz():
    bundle = app.state.config_bundle
    return {
        "dataset": bundle.corpus.dataset,
        "dataset_version": bundle.corpus.dataset_version,
        "tenants": bundle.corpus.tenants,
        "seeds": {
            "partition": bundle.corpus.seeds.partition,
            "payloads": bundle.corpus.seeds.payloads,
            "qa": bundle.corpus.seeds.qa,
        },
        "defenses": {
            "rls_enabled": bundle.defenses.rls_enabled,
            "injection_scanner_enabled": bundle.defenses.injection_scanner_enabled,
            "scanner_threshold": bundle.defenses.scanner_threshold,
            "tool_guard_enabled": bundle.defenses.tool_guard_enabled,
        },
        "fingerprint": bundle.fingerprint,
    }