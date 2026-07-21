"""Tests for interim-build schema-stamp detection and guided remediation (fix #7).

A database touched by an interim development build can carry a numeric
``schema_version`` ahead of this build's ladder while its actual schema is the
v5 shape plus named feature markers. These tests cover classification of that
condition, the refusal-message guidance, and the explicit backup-first repair.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.command import (
    _doctor_repair_schema_stamp_apply_text,
    _doctor_repair_schema_stamp_text,
)
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.db_bootstrap import (
    SchemaVersionTooNewError,
    classify_version_mismatch,
    remediate_interim_schema_stamp,
)
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.rollup_store import RollupStore
from hermes_lcm.store import MessageStore
from hermes_lcm.vector_store import VectorStore
from hermes_lcm.sqlite_writer import get_writer_coordinator


def _build_v5_db(path: Path, *, with_features: bool = False) -> None:
    """Materialize a genuine v5-shaped DB (core tables + both FTS indexes)."""
    store = MessageStore(path)
    store.close()
    dag = SummaryDAG(path)
    dag.close()
    if with_features:
        rollups = RollupStore(path)
        rollups.close()
        conn = sqlite3.connect(path)
        try:
            db_bootstrap.ensure_embedding_tables(conn)
            conn.commit()
        finally:
            conn.close()


def _add_early_feature_tables(path: Path) -> None:
    """Create EARLY-variant feature tables (missing later-added columns/tables).

    Mirrors the real interim operator DB: family-prefixed tables that predate
    later schema additions (lcm_rollups without generation/lease_nonce/failed_at
    and no lcm_rollup_invalidations; lcm_embedding_profile keyed on model_name
    without identity_hash/data_version). These fail the final-shape verifiers.
    """
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE lcm_rollups (
                rollup_id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_kind TEXT, period_start TEXT, scope TEXT,
                summary TEXT, token_count INTEGER, status TEXT,
                built_at TEXT, source_fingerprint TEXT, error TEXT
            );
            CREATE TABLE lcm_rollup_sources (
                rollup_id INTEGER, node_id INTEGER,
                PRIMARY KEY(rollup_id, node_id)
            );
            CREATE TABLE lcm_rollup_state (
                period_kind TEXT PRIMARY KEY,
                last_build_cursor TEXT, last_built_at TEXT
            );
            CREATE TABLE lcm_embedding_profile (
                model_name TEXT PRIMARY KEY, provider TEXT, dim INTEGER,
                registered_at TEXT, active INTEGER DEFAULT 1, archived_at TEXT
            );
            CREATE TABLE lcm_embedding_meta (
                embedded_id TEXT, embedded_kind TEXT, model_name TEXT,
                embedded_at TEXT, PRIMARY KEY(embedded_id, embedded_kind)
            );
            CREATE TABLE lcm_embedding_vectors (
                embedded_id TEXT PRIMARY KEY, vec BLOB
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _add_early_chunk_tables(path: Path) -> None:
    """Create an EARLY-variant chunk schema (missing later-added columns/indexes).

    Mirrors a DB whose chunk corpus predates char-span columns and the required
    partial index — it fails ``verify_chunk_schema`` with missing-object /
    malformed-table findings (never an unexpected-column one).
    """
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE lcm_chunk_meta (
                chunk_id TEXT, identity_hash TEXT, store_id INTEGER,
                chunk_index INTEGER, embedded_at TEXT, archived INTEGER DEFAULT 0,
                PRIMARY KEY(chunk_id, identity_hash)
            );
            CREATE TABLE lcm_chunk_vectors (
                chunk_id TEXT, identity_hash TEXT, vec BLOB NOT NULL,
                PRIMARY KEY(chunk_id, identity_hash)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def _table_names(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
    finally:
        conn.close()


def _stamp(path: Path, version: int) -> None:
    conn = sqlite3.connect(path)
    try:
        db_bootstrap.set_schema_version(conn, version)
        conn.commit()
    finally:
        conn.close()


def _stored_version(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return db_bootstrap.read_existing_schema_version(conn)
    finally:
        conn.close()


def _schema_state(path: Path) -> tuple[tuple[tuple[str, str, str], ...], tuple[tuple[str, str], ...]]:
    conn = sqlite3.connect(path)
    try:
        objects = tuple(
            (str(row[0]), str(row[1]), str(row[2] or ""))
            for row in conn.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
        )
        metadata = ()
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
        ).fetchone():
            metadata = tuple(
                (str(row[0]), str(row[1]))
                for row in conn.execute(
                    "SELECT key, value FROM metadata ORDER BY key"
                ).fetchall()
            )
        return objects, metadata
    finally:
        conn.close()


class _InjectFutureSchemaBeforeFirstTransaction:
    """Install a future schema after preflight but before constructor DDL."""

    def __init__(self, path: Path):
        self._path = path
        self._inner = get_writer_coordinator(path)
        self._injected = False
        self.injected_state = None

    def bind_owner(self):
        return self._inner.bind_owner()

    def close_owner(self, *args, **kwargs):
        return self._inner.close_owner(*args, **kwargs)

    def write_region(self, *args, **kwargs):
        return self._inner.write_region(*args, **kwargs)

    @contextmanager
    def transaction(self, connection, **kwargs):
        if not self._injected:
            self._injected = True
            other = sqlite3.connect(self._path)
            try:
                other.execute("CREATE TABLE lcm_future_widgets (id INTEGER PRIMARY KEY)")
                db_bootstrap.set_schema_version(
                    other, db_bootstrap.SCHEMA_VERSION + 1
                )
                other.commit()
            finally:
                other.close()
            self.injected_state = _schema_state(self._path)
        with self._inner.transaction(connection, **kwargs) as admitted:
            yield admitted


def _remove_v6_v7_core_shape(path: Path) -> None:
    """Leave the durable core at v5 while preserving its numeric stamp."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("DROP TABLE IF EXISTS lcm_prepared_summary_nodes")
        conn.execute("DROP TABLE IF EXISTS lcm_prepared_compactions")
        conn.execute("DROP INDEX IF EXISTS idx_summary_nodes_coverage_key_unique")
        conn.execute("ALTER TABLE summary_nodes DROP COLUMN coverage_key")
        conn.commit()
    finally:
        conn.close()


# --- classification --------------------------------------------------------


def test_classify_interim_stamp_on_v5_shape(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert classify_version_mismatch(conn) == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
    finally:
        conn.close()


def test_classify_interim_stamp_with_feature_marker_tables(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path, with_features=True)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        # temporal-rollup + embedding tables are known feature markers, so the
        # DB is still classified as an interim stamp, not a genuinely newer DB.
        assert classify_version_mismatch(conn) == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
    finally:
        conn.close()


def test_classify_genuinely_newer_on_unknown_table(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE lcm_future_widgets (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert classify_version_mismatch(conn) == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
    finally:
        conn.close()


def test_classify_genuinely_newer_on_extra_feature_family_column(tmp_path):
    """An EXTRA column on a feature-family table is a newer-build signature.

    Reproduces F2-schema-stamp-drops-newer-data: a future release adds a column
    to ``lcm_rollups``. The old classifier ignored feature-table internal shape
    and called this an interim stamp, so remediation DROPPED the table and its
    siblings. It must classify ``genuinely_newer`` instead — an unexpected
    (extra) column is never an early-variant signature.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path, with_features=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE lcm_rollups ADD COLUMN future_col INTEGER DEFAULT 0")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
        )
    finally:
        conn.close()


def test_remediate_apply_refuses_and_preserves_extra_column_family(tmp_path):
    """Remediation must NOT drop a family table that carries an extra column.

    The data-destruction guard: with an extra ``lcm_rollups`` column present,
    ``remediate_interim_schema_stamp(apply=True)`` refuses and leaves every
    feature table (and the stamp) untouched.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path, with_features=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE lcm_rollups ADD COLUMN future_col INTEGER DEFAULT 0")
        conn.commit()
    finally:
        conn.close()
    stamped = db_bootstrap.SCHEMA_VERSION + 1
    _stamp(db_path, stamped)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "refused"
    assert result["classification"] == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
    assert result["applied"] is False
    assert result["dropped_tables"] == []
    # Nothing dropped, stamp untouched — real data survives.
    assert _stored_version(db_path) == stamped
    assert "lcm_rollups" in _table_names(db_path)


def test_classify_interim_stamp_on_missing_feature_family_column(tmp_path):
    """A feature table only MISSING a later-added column stays an interim stamp.

    The counterpart to the extra-column case: an early variant omits pieces (no
    ``generation``/``lease_nonce``/``failed_at`` on ``lcm_rollups``) and must
    still be classified interim so remediation can drop-and-rebuild it.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _add_early_feature_tables(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        )
    finally:
        conn.close()


def test_classify_genuinely_newer_on_unknown_core_column(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN future_flag INTEGER DEFAULT 0")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        assert classify_version_mismatch(conn) == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
    finally:
        conn.close()


# --- refusal-message guidance ---------------------------------------------


def test_refuse_message_points_at_remediation_for_interim_stamp(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    with pytest.raises(SchemaVersionTooNewError) as excinfo:
        MessageStore(db_path)
    message = str(excinfo.value)
    assert "schema-stamp" in message
    assert "do NOT upgrade" in message


@pytest.mark.parametrize("stamped_version", [6, 7])
def test_supported_numeric_stamp_ahead_of_actual_core_shape_requires_repair(
    tmp_path,
    stamped_version,
):
    db_path = tmp_path / f"interim-v{stamped_version}.db"
    _build_v5_db(db_path)
    _remove_v6_v7_core_shape(db_path)
    _stamp(db_path, stamped_version)

    with pytest.raises(SchemaVersionTooNewError, match="schema-stamp"):
        MessageStore(db_path)

    conn = sqlite3.connect(db_path)
    try:
        dry_run = remediate_interim_schema_stamp(conn, apply=False)
    finally:
        conn.close()
    assert dry_run["status"] == "dry-run"
    assert dry_run["classification"] == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
    assert _stored_version(db_path) == stamped_version

    conn = sqlite3.connect(db_path)
    try:
        applied = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert applied["status"] == "ok"
    assert applied["applied"] is True

    store = MessageStore(db_path)
    store.close()
    dag = SummaryDAG(db_path)
    try:
        summary_columns = {
            str(row[1])
            for row in dag.connection.execute("PRAGMA table_info(summary_nodes)")
        }
        tables = _table_names(db_path)
        assert "coverage_key" in summary_columns
        assert {
            "lcm_prepared_compactions",
            "lcm_prepared_summary_nodes",
        } <= tables
        assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION
    finally:
        dag.close()


@pytest.mark.parametrize(
    ("stamped_version", "partial_shape"),
    [
        (7, "missing_prepared_summary"),
        (7, "prepared_without_coverage"),
        (6, "malformed_prepared_compaction"),
    ],
)
def test_supported_stamp_with_partial_or_out_of_order_core_shape_fails_closed(
    tmp_path,
    stamped_version,
    partial_shape,
):
    db_path = tmp_path / f"partial-{partial_shape}.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        if partial_shape == "missing_prepared_summary":
            conn.execute("DROP TABLE lcm_prepared_summary_nodes")
        elif partial_shape == "prepared_without_coverage":
            conn.execute("DROP INDEX idx_summary_nodes_coverage_key_unique")
            conn.execute("ALTER TABLE summary_nodes DROP COLUMN coverage_key")
        else:
            conn.execute(
                "ALTER TABLE lcm_prepared_compactions "
                "ADD COLUMN future_partial_state TEXT"
            )
        db_bootstrap.set_schema_version(conn, stamped_version)
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(SchemaVersionTooNewError, match="exact compatible core shape"):
        MessageStore(db_path)

    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "refused"
    assert result["classification"] == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
    assert result["applied"] is False
    assert _stored_version(db_path) == stamped_version


def test_summary_only_v7_without_coverage_fails_closed_without_mutation(tmp_path):
    db_path = tmp_path / "summary-only-missing-coverage.db"
    dag = SummaryDAG(db_path)
    dag.close()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP INDEX idx_summary_nodes_coverage_key_unique")
        conn.execute("ALTER TABLE summary_nodes DROP COLUMN coverage_key")
        conn.commit()
    finally:
        conn.close()
    before = _schema_state(db_path)

    with pytest.raises(SchemaVersionTooNewError, match="exact compatible core shape"):
        SummaryDAG(db_path)

    assert _schema_state(db_path) == before


def test_v7_prepared_table_with_name_only_shape_fails_closed_without_mutation(
    tmp_path,
):
    db_path = tmp_path / "malformed-prepared-semantics.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        columns = [
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(lcm_prepared_summary_nodes)"
            ).fetchall()
        ]
        conn.execute("DROP TABLE lcm_prepared_summary_nodes")
        conn.execute(
            "CREATE TABLE lcm_prepared_summary_nodes ("
            + ", ".join(f'"{column}" TEXT' for column in columns)
            + ")"
        )
        conn.commit()
    finally:
        conn.close()
    before = _schema_state(db_path)

    with pytest.raises(SchemaVersionTooNewError, match="exact compatible core shape"):
        MessageStore(db_path)

    assert _schema_state(db_path) == before


def test_v7_prepared_parent_with_name_only_shape_fails_closed_without_mutation(
    tmp_path,
):
    db_path = tmp_path / "malformed-prepared-parent-semantics.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        columns = [
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(lcm_prepared_compactions)"
            ).fetchall()
        ]
        child_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='lcm_prepared_summary_nodes'"
        ).fetchone()[0]
        index_sql = [
            str(row[0])
            for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' "
                "AND name LIKE 'idx_lcm_prepared_%' AND sql IS NOT NULL "
                "ORDER BY name"
            ).fetchall()
        ]
        conn.execute("DROP TABLE lcm_prepared_summary_nodes")
        conn.execute("DROP TABLE lcm_prepared_compactions")
        conn.execute(
            "CREATE TABLE lcm_prepared_compactions ("
            + ", ".join(f'"{column}" TEXT' for column in columns)
            + ")"
        )
        conn.execute(child_sql)
        for statement in index_sql:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()
    before = _schema_state(db_path)

    with pytest.raises(SchemaVersionTooNewError, match="exact compatible core shape"):
        MessageStore(db_path)

    assert _schema_state(db_path) == before


@pytest.mark.parametrize(
    ("label", "old", "new"),
    [
        ("wrong_type", "token_count INTEGER NOT NULL", "token_count TEXT NOT NULL"),
        ("nullable", "summary TEXT NOT NULL", "summary TEXT"),
        (
            "wrong_default",
            "previous_pending_ids TEXT NOT NULL DEFAULT '[]'",
            "previous_pending_ids TEXT NOT NULL DEFAULT '{}'",
        ),
        ("missing_primary_key", "pending_id TEXT PRIMARY KEY", "pending_id TEXT"),
        ("wrong_check", "CHECK (depth = 0)", "CHECK (depth >= 0)"),
        ("missing_unique", "batch_id TEXT NOT NULL UNIQUE", "batch_id TEXT NOT NULL"),
        (
            "wrong_fk_target",
            "REFERENCES lcm_prepared_compactions(batch_id)",
            "REFERENCES lcm_prepared_compactions(coverage_key)",
        ),
        ("missing_cascade", " ON DELETE CASCADE", ""),
    ],
)
def test_each_prepared_summary_semantic_mismatch_fails_closed_without_mutation(
    tmp_path,
    label,
    old,
    new,
):
    db_path = tmp_path / f"prepared-{label}.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='lcm_prepared_summary_nodes'"
        ).fetchone()[0]
        assert old in sql
        conn.execute("DROP TABLE lcm_prepared_summary_nodes")
        conn.execute(str(sql).replace(old, new, 1))
        conn.execute(
            "CREATE INDEX idx_lcm_prepared_node_batch "
            "ON lcm_prepared_summary_nodes(batch_id)"
        )
        conn.commit()
    finally:
        conn.close()
    before = _schema_state(db_path)

    with pytest.raises(SchemaVersionTooNewError, match="exact compatible core shape"):
        MessageStore(db_path)

    assert _schema_state(db_path) == before


def test_prepared_summary_extra_unique_constraint_fails_closed_without_mutation(
    tmp_path,
):
    db_path = tmp_path / "prepared-extra-unique.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        sql = str(
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' "
                "AND name='lcm_prepared_summary_nodes'"
            ).fetchone()[0]
        )
        conn.execute("DROP TABLE lcm_prepared_summary_nodes")
        conn.execute(sql.rsplit(")", 1)[0] + ", UNIQUE(session_id))")
        conn.execute(
            "CREATE INDEX idx_lcm_prepared_node_batch "
            "ON lcm_prepared_summary_nodes(batch_id)"
        )
        conn.commit()
    finally:
        conn.close()
    before = _schema_state(db_path)

    with pytest.raises(SchemaVersionTooNewError, match="exact compatible core shape"):
        MessageStore(db_path)

    assert _schema_state(db_path) == before


@pytest.mark.parametrize(
    "replacement",
    [
        "CREATE INDEX idx_summary_nodes_coverage_key_unique "
        "ON summary_nodes(coverage_key) WHERE coverage_key IS NOT NULL",
        "CREATE UNIQUE INDEX idx_summary_nodes_coverage_key_unique "
        "ON summary_nodes(session_id) WHERE coverage_key IS NOT NULL",
        "CREATE UNIQUE INDEX idx_summary_nodes_coverage_key_unique "
        "ON summary_nodes(coverage_key)",
    ],
)
def test_each_coverage_index_semantic_mismatch_fails_closed_without_mutation(
    tmp_path,
    replacement,
):
    db_path = tmp_path / "malformed-coverage-index.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP INDEX idx_summary_nodes_coverage_key_unique")
        conn.execute(replacement)
        conn.commit()
    finally:
        conn.close()
    before = _schema_state(db_path)

    with pytest.raises(SchemaVersionTooNewError, match="exact compatible core shape"):
        MessageStore(db_path)

    assert _schema_state(db_path) == before


@pytest.mark.parametrize("stamped_version", [6, 7])
def test_stamped_metadata_only_shape_fails_closed_without_mutation(
    tmp_path,
    stamped_version,
):
    db_path = tmp_path / f"metadata-only-v{stamped_version}.db"
    conn = sqlite3.connect(db_path)
    try:
        db_bootstrap.ensure_metadata_table(conn)
        db_bootstrap.set_schema_version(conn, stamped_version)
        conn.commit()
    finally:
        conn.close()
    before = _schema_state(db_path)

    with pytest.raises(SchemaVersionTooNewError, match="exact compatible core shape"):
        MessageStore(db_path)

    assert _schema_state(db_path) == before


@pytest.mark.parametrize(
    ("constructor", "forbidden_table"),
    [
        (lambda path, coordinator: MessageStore(path, writer_coordinator=coordinator), "messages"),
        (lambda path, coordinator: SummaryDAG(path, writer_coordinator=coordinator), "summary_nodes"),
    ],
)
def test_helper_rechecks_after_sqlite_writer_admission_before_any_ddl(
    tmp_path,
    constructor,
    forbidden_table,
):
    db_path = tmp_path / f"toctou-{forbidden_table}.db"
    coordinator = _InjectFutureSchemaBeforeFirstTransaction(db_path)

    with pytest.raises(SchemaVersionTooNewError):
        constructor(db_path, coordinator)

    assert coordinator.injected_state is not None
    assert _schema_state(db_path) == coordinator.injected_state
    assert forbidden_table not in _table_names(db_path)


@pytest.mark.parametrize(
    ("module", "constructor"),
    [
        (
            "hermes_lcm.store",
            lambda path: MessageStore(path),
        ),
        (
            "hermes_lcm.dag",
            lambda path: SummaryDAG(path),
        ),
    ],
)
def test_late_bootstrap_failure_rolls_back_owner_schema(
    tmp_path,
    monkeypatch,
    module,
    constructor,
):
    db_path = tmp_path / f"rollback-{module.rsplit('.', 1)[-1]}.db"

    def fail_after_owner_ddl(_conn, _spec, **kwargs):
        if kwargs.get("structural_only"):
            raise RuntimeError("forced late bootstrap failure")

    monkeypatch.setattr(f"{module}.ensure_external_content_fts", fail_after_owner_ddl)

    with pytest.raises(RuntimeError, match="forced late bootstrap failure"):
        constructor(db_path)

    assert _schema_state(db_path) == ((), ())


def test_known_single_store_bootstrap_shapes_remain_openable(tmp_path):
    message_db = tmp_path / "message-first.db"
    messages = MessageStore(message_db)
    messages.close()
    dag = SummaryDAG(message_db)
    dag.close()

    summary_db = tmp_path / "summary-first.db"
    dag = SummaryDAG(summary_db)
    dag.close()
    messages = MessageStore(summary_db)
    messages.close()

    shared_db = tmp_path / "shared-first.db"
    lifecycle = LifecycleStateStore(shared_db)
    lifecycle.close()
    lifecycle = LifecycleStateStore(shared_db)
    lifecycle.close()


def test_refuse_message_stays_generic_for_genuinely_newer(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE lcm_future_widgets (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    with pytest.raises(SchemaVersionTooNewError) as excinfo:
        MessageStore(db_path)
    message = str(excinfo.value)
    assert "restore a pre-upgrade backup" in message
    assert "schema-stamp" not in message


# --- remediation helper ----------------------------------------------------


def test_remediate_dry_run_reports_without_mutating(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    stamped = db_bootstrap.SCHEMA_VERSION + 1
    _stamp(db_path, stamped)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=False)
    finally:
        conn.close()
    assert result["status"] == "dry-run"
    assert result["classification"] == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
    assert result["applied"] is False
    assert _stored_version(db_path) == stamped


def test_remediate_apply_resets_stamp_and_db_reopens(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "ok"
    assert result["applied"] is True
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION
    # After the reset the store opens again without refusing.
    store = MessageStore(db_path)
    store.close()


def test_remediate_apply_rechecks_after_writer_admission_without_mutation(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "remediation-toctou.db"
    _build_v5_db(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)
    original_write_transaction = db_bootstrap._write_transaction
    injected_state = None

    @contextmanager
    def inject_future_schema_before_admission(connection, **kwargs):
        nonlocal injected_state
        if injected_state is None:
            other = sqlite3.connect(db_path)
            try:
                other.execute(
                    "CREATE TABLE lcm_future_durable (id INTEGER PRIMARY KEY)"
                )
                db_bootstrap.set_schema_version(
                    other, db_bootstrap.SCHEMA_VERSION + 2
                )
                other.commit()
            finally:
                other.close()
            injected_state = _schema_state(db_path)
        with original_write_transaction(connection, **kwargs):
            yield

    monkeypatch.setattr(
        db_bootstrap, "_write_transaction", inject_future_schema_before_admission
    )
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()

    assert result["status"] == "refused"
    assert result["classification"] == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
    assert result["applied"] is False
    assert result["dropped_tables"] == []
    assert injected_state is not None
    assert _schema_state(db_path) == injected_state


def test_remediate_refuses_genuinely_newer(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE lcm_future_widgets (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    stamped = db_bootstrap.SCHEMA_VERSION + 1
    _stamp(db_path, stamped)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "refused"
    assert result["classification"] == db_bootstrap.VERSION_MISMATCH_GENUINELY_NEWER
    assert result["applied"] is False
    assert _stored_version(db_path) == stamped


def test_early_variant_feature_tables_remediate_end_to_end(tmp_path):
    """Early-variant feature tables classify as interim and recover after apply.

    This is the real-operator-DB shape: clean v5 core plus family-prefixed
    tables that are EARLY variants failing the final-shape verifiers.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    _add_early_feature_tables(db_path)
    stamped = db_bootstrap.SCHEMA_VERSION + 1
    _stamp(db_path, stamped)

    # Classification ignores feature-table internal shape → interim_stamp.
    conn = sqlite3.connect(db_path)
    try:
        assert classify_version_mismatch(conn) == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        dry = remediate_interim_schema_stamp(conn, apply=False)
    finally:
        conn.close()
    assert dry["status"] == "dry-run"
    would_drop = {t for fam in dry["drop_plan"] for t in fam["tables"]}
    assert {"lcm_rollups", "lcm_embedding_profile"} <= would_drop
    # Dry-run mutates nothing.
    assert _stored_version(db_path) == stamped
    assert "lcm_rollups" in _table_names(db_path)

    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "ok"
    dropped = set(result["dropped_tables"])
    assert {"lcm_rollups", "lcm_rollup_sources", "lcm_rollup_state"} <= dropped
    assert {"lcm_embedding_profile", "lcm_embedding_meta", "lcm_embedding_vectors"} <= dropped
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION
    # Early feature tables are gone; core tables remain untouched.
    remaining = _table_names(db_path)
    assert not any(t.startswith(("lcm_rollup", "lcm_embedding")) for t in remaining)
    assert {"messages", "summary_nodes"} <= remaining

    # refuse now passes, and each feature store reconstructs the final shape.
    conn = sqlite3.connect(db_path)
    try:
        db_bootstrap.refuse_schema_version_too_new(conn)  # must not raise
    finally:
        conn.close()
    rollups = RollupStore(db_path)
    try:
        assert db_bootstrap.verify_temporal_rollup_schema(rollups.connection) == []
    finally:
        rollups.close()
    vectors = VectorStore(db_path)
    try:
        assert db_bootstrap.verify_embedding_schema(vectors._conn) == []
    finally:
        vectors.close()


def test_early_variant_chunk_family_remediates(tmp_path):
    """A broken chunk schema is dropped by remediation, not silently kept.

    Reproduces F2-schema-stamp-chunk-family-missing / F4-chunk-family-verifier-
    missing: with no ``lcm_chunk`` entry in the interim feature families the
    remediator reported ``status: ok, dropped_tables: []`` while leaving a broken
    ``lcm_chunk_meta``/``lcm_chunk_vectors`` in place. The family must now be
    verified and dropped so its marker-gated init rebuilds the final shape.
    """
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path, with_features=True)
    _add_early_chunk_tables(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)

    conn = sqlite3.connect(db_path)
    try:
        assert (
            classify_version_mismatch(conn)
            == db_bootstrap.VERSION_MISMATCH_INTERIM_STAMP
        )
        # The broken chunk schema fails its verifier (missing pieces, not extra).
        assert db_bootstrap.verify_chunk_schema(conn) != []
        assert not db_bootstrap._family_reports_newer_shape(
            db_bootstrap.verify_chunk_schema(conn)
        )
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "ok"
    dropped = set(result["dropped_tables"])
    assert {"lcm_chunk_meta", "lcm_chunk_vectors"} <= dropped
    assert not any(t.startswith("lcm_chunk") for t in _table_names(db_path))

    # The chunk feature's own init recreates the final, verifier-clean shape.
    conn = sqlite3.connect(db_path)
    try:
        db_bootstrap.ensure_embedding_tables(conn)
        db_bootstrap.ensure_chunk_tables(conn)
        conn.commit()
        assert db_bootstrap.verify_chunk_schema(conn) == []
    finally:
        conn.close()


def test_remediate_noop_when_version_supported(tmp_path):
    db_path = tmp_path / "lcm.db"
    _build_v5_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        result = remediate_interim_schema_stamp(conn, apply=True)
    finally:
        conn.close()
    assert result["status"] == "noop"
    assert result["applied"] is False


# --- /lcm doctor repair schema-stamp command path --------------------------


def _healthy_engine(tmp_path: Path) -> LCMEngine:
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    return LCMEngine(config=config, hermes_home=str(tmp_path / "home"))


def test_doctor_repair_schema_stamp_dry_run_and_apply(tmp_path):
    engine = _healthy_engine(tmp_path)
    db_path = Path(engine._store.db_path)
    # Add early-variant feature tables + stamp ahead of the ladder to simulate
    # an interim build, then drive the operator-facing command path.
    _add_early_feature_tables(db_path)
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)

    dry = _doctor_repair_schema_stamp_text(engine)
    assert "status: repair-needed" in dry
    assert "classification: interim_stamp" in dry
    assert "would_drop: lcm_rollups" in dry
    assert "/lcm rollups rebuild" in dry
    assert "would_drop: lcm_embedding_profile" in dry
    assert "no schema changes were made" in dry
    # Dry-run must not mutate the stamp or drop anything.
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION + 1
    assert "lcm_rollups" in _table_names(db_path)

    applied = _doctor_repair_schema_stamp_apply_text(engine)
    assert "status: ok" in applied
    assert "backup_path:" in applied
    assert f"schema_version_reset_to: {db_bootstrap.SCHEMA_VERSION}" in applied
    assert "dropped: lcm_rollups" in applied
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION
    assert not any(
        t.startswith(("lcm_rollup", "lcm_embedding")) for t in _table_names(db_path)
    )


def test_doctor_repair_schema_stamp_apply_refuses_genuinely_newer(tmp_path):
    engine = _healthy_engine(tmp_path)
    db_path = Path(engine._store.db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE lcm_future_widgets (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    _stamp(db_path, db_bootstrap.SCHEMA_VERSION + 1)

    applied = _doctor_repair_schema_stamp_apply_text(engine)
    assert "status: refused" in applied
    assert "classification: genuinely_newer" in applied
    # No backup and no mutation on the refused path.
    assert "backup_path:" not in applied
    assert _stored_version(db_path) == db_bootstrap.SCHEMA_VERSION + 1
