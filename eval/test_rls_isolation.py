from __future__ import annotations

import os

import psycopg2
import pytest

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragsvc:ragsvc@localhost:5432/ragsentinel",
)


@pytest.fixture
def db_conn():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def test_rls_default_deny_when_unset(db_conn):
    """Without a tenant context, RLS must expose no chunk rows."""
    with db_conn.cursor() as cur:
        cur.execute("RESET app.tenant_id;")
        cur.execute("SELECT count(*) FROM chunks;")
        count = cur.fetchone()[0]

    assert count == 0, (
        f"SECURITY VIOLATION: unset tenant_id returned {count} rows; expected 0."
    )


def test_rls_tenant_isolation_boundary(db_conn):
    """Tenant A must never see chunks belonging to another tenant."""
    with db_conn.cursor() as cur:
        cur.execute("SET LOCAL app.tenant_id = %s;", ("A",))
        cur.execute("SELECT DISTINCT tenant_id FROM chunks;")
        tenants = {row[0] for row in cur.fetchall()}

    assert tenants <= {"A"}, (
        f"SECURITY VIOLATION: tenant A saw rows belonging to {sorted(tenants - {'A'})}."
    )


def test_rls_sql_injection_resilience(db_conn):
    """A parameterized injection-shaped tenant value must not reveal rows."""
    malicious_tenant = "' OR '1'='1"
    with db_conn.cursor() as cur:
        cur.execute("SET LOCAL app.tenant_id = %s;", (malicious_tenant,))
        cur.execute("SELECT count(*) FROM chunks;")
        count = cur.fetchone()[0]

    assert count == 0, (
        f"SECURITY VIOLATION: injection-shaped tenant_id returned {count} rows."
    )
