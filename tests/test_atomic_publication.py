"""Atomic foreground leaf-publication acceptance tests (MUW 3A).

The summarizer is intentionally outside the durable boundary.  These tests
exercise the much smaller publication contract: immutable captured source
identity goes into one ``BEGIN IMMEDIATE`` transaction that either publishes a
canonical leaf and advances the lifecycle frontier together, recognizes a
retry, or leaves every durable and active-context surface unchanged.
"""

from __future__ import annotations

import copy
import math
import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.db_bootstrap import (
    SCHEMA_VERSION,
    SchemaVersionTooNewError,
    run_versioned_migrations,
)
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.publication import (
    AtomicPublicationStore,
    PublicationCaptureError,
    PublicationIntent,
    PublicationResult,
)
from hermes_lcm.sqlite_writer import get_writer_coordinator
from hermes_lcm.store import MessageStore
from hermes_lcm.tokens import count_messages_tokens, count_tokens


SESSION_ID = "session:atomic"
CONVERSATION_ID = "conversation:atomic"


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "alpha canary"},
        {"role": "assistant", "content": "beta answer"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"gamma"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "gamma result"},
    ]


class AtomicFixture:
    def __init__(self, db_path: Path, *, after_insert=None) -> None:
        self.db_path = db_path
        self.coordinator = get_writer_coordinator(db_path)
        self.store = MessageStore(db_path, writer_coordinator=self.coordinator)
        self.dag = SummaryDAG(db_path, writer_coordinator=self.coordinator)
        self.lifecycle = LifecycleStateStore(
            db_path,
            writer_coordinator=self.coordinator,
        )
        self.lifecycle.bind_session(SESSION_ID, conversation_id=CONVERSATION_ID)
        self.source_ids = self.store.append_batch(
            SESSION_ID,
            _messages(),
            conversation_id=CONVERSATION_ID,
        )
        self.publisher = AtomicPublicationStore(
            db_path,
            writer_coordinator=self.coordinator,
            after_insert=after_insert,
        )

    def intent(self, *, summary: str = "canonical leaf summary") -> PublicationIntent:
        captured = self.publisher.capture_leaf_intent(
            conversation_id=CONVERSATION_ID,
            session_id=SESSION_ID,
            expected_frontier_store_id=0,
            new_frontier_store_id=max(self.source_ids),
            source_store_ids=self.source_ids,
        )
        return captured.with_summary(
            summary=summary,
            token_count=count_tokens(summary),
            source_token_count=count_messages_tokens(_messages()),
            created_at=1234.5,
            earliest_at=1.0,
            latest_at=4.0,
            expand_hint="alpha, beta, gamma",
        )

    def close(self) -> None:
        self.publisher.close()
        self.lifecycle.close()
        self.dag.close()
        self.store.close()


@pytest.fixture()
def atomic(tmp_path: Path):
    fixture = AtomicFixture(tmp_path / "atomic.db")
    try:
        yield fixture
    finally:
        fixture.close()


def _node_count(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0])


def _fts_count(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM nodes_fts_docsize").fetchone()[0])


def _frontier(path: Path, conversation_id: str = CONVERSATION_ID) -> int:
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT current_frontier_store_id FROM lcm_lifecycle_state "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return int(row[0])


def _compress_once(
    engine: LCMEngine,
    messages: list[dict],
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict]:
    def summarize(chunk, focus_topic=None):
        return (
            list(chunk),
            count_messages_tokens(chunk),
            "lifecycle boundary summary\n[Expand for details: lifecycle-boundary]",
            1,
            1,
        )

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    return engine.compress(messages, force=True)


def _lifecycle_boundary_messages() -> list[dict]:
    return [
        {"role": "system", "content": "stable system prompt"},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "fresh request"},
    ]


def _lifecycle_boundary_engine(tmp_path: Path, name: str) -> LCMEngine:
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / f"{name}.db"),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            dynamic_leaf_chunk_enabled=False,
            incremental_max_depth=0,
        ),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:same-id-boundary",
        conversation_id="conversation:same-id-boundary",
        platform="discord",
        context_length=10_000,
    )
    return engine


def test_session_end_exactly_matching_last_ingested_snapshot_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    engine = _lifecycle_boundary_engine(tmp_path, "same-id-end-idempotent")
    messages = _lifecycle_boundary_messages()
    try:
        compressed = _compress_once(engine, messages, monkeypatch)
        assert len(compressed) < len(messages)
        durable_count = engine._store.get_session_count("session:same-id-boundary")

        engine.on_session_end("session:same-id-boundary", messages)

        assert engine._store.get_session_count("session:same-id-boundary") == durable_count
    finally:
        engine.shutdown()


def test_same_id_compression_boundary_preserves_published_state_and_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    engine = _lifecycle_boundary_engine(tmp_path, "same-id-start-preserves")
    messages = _lifecycle_boundary_messages()
    try:
        compressed = _compress_once(engine, messages, monkeypatch)
        durable_count = engine._store.get_session_count("session:same-id-boundary")
        node_ids = [
            node.node_id
            for node in engine._dag.get_session_nodes("session:same-id-boundary")
        ]
        frontier = engine._last_compacted_store_id
        assert node_ids
        assert frontier > 0
        assert engine._ingest_cursor == len(compressed)

        # Reproduce the legacy host ordering too: a redundant session-end
        # callback can finalize the lifecycle row immediately before the
        # same-id compression boundary reopens the logical continuation.
        engine.on_session_end("session:same-id-boundary", messages)

        engine.on_session_start(
            "session:same-id-boundary",
            boundary_reason="compression",
            old_session_id="session:same-id-boundary",
            conversation_id="conversation:same-id-boundary",
            platform="discord",
            context_length=10_000,
        )

        assert engine._ingest_cursor == len(compressed)
        assert engine._last_compacted_store_id == frontier
        assert [
            node.node_id
            for node in engine._dag.get_session_nodes("session:same-id-boundary")
        ] == node_ids

        new_message = {"role": "assistant", "content": "fresh answer after boundary"}
        engine.ingest([*compressed, new_message])
        assert engine._store.get_session_count("session:same-id-boundary") == durable_count + 1
    finally:
        engine.shutdown()


def test_session_end_persists_only_genuine_new_tail(
    tmp_path: Path,
):
    engine = _lifecycle_boundary_engine(tmp_path, "same-id-end-tail")
    messages = _lifecycle_boundary_messages()
    try:
        engine.ingest(messages)
        durable_count = engine._store.get_session_count("session:same-id-boundary")
        new_message = {"role": "assistant", "content": "late final answer"}

        engine.on_session_end(
            "session:same-id-boundary",
            [*messages, new_message],
        )

        rows = engine._store.get_session_messages("session:same-id-boundary")
        assert len(rows) == durable_count + 1
        assert rows[-1]["content"] == "late final answer"
    finally:
        engine.shutdown()


def test_ambiguous_reconciliation_warning_applies_only_to_its_insert(
    tmp_path: Path,
    caplog,
):
    db_path = tmp_path / "reconciliation-warning-scope.db"
    config = LCMConfig(database_path=str(db_path))
    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    first.on_session_start(
        "session:warning-scope",
        conversation_id="conversation:warning-scope",
        platform="discord",
    )
    first.ingest([{"role": "user", "content": "durable history"}])
    first.shutdown()

    restarted = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    restarted.on_session_start(
        "session:warning-scope",
        conversation_id="conversation:warning-scope",
        platform="discord",
    )
    first_delta = {"role": "user", "content": "ambiguous new branch"}
    second_delta = {"role": "assistant", "content": "unambiguously appended reply"}
    try:
        restarted.ingest([first_delta])
        restarted.ingest([first_delta, second_delta])

        assert caplog.text.count("LCM persisted ambiguous ingest delta") == 1
    finally:
        restarted.shutdown()


def test_ambiguous_reconciliation_without_insert_does_not_label_later_append(
    tmp_path: Path,
    caplog,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "reconciliation-warning-no-write.db"
    config = LCMConfig(database_path=str(db_path))
    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    first.on_session_start(
        "session:warning-no-write",
        conversation_id="conversation:warning-no-write",
        platform="discord",
    )
    first.ingest([{"role": "user", "content": "durable history"}])
    first.shutdown()

    restarted = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    restarted.on_session_start(
        "session:warning-no-write",
        conversation_id="conversation:warning-no-write",
        platform="discord",
    )
    dropped = {"role": "user", "content": "DROP_THIS_RESTART_BATCH"}
    real_append = {"role": "assistant", "content": "later real append"}
    restarted._compiled_ignore_message_patterns = [object()]
    monkeypatch.setattr(
        restarted,
        "_matches_ignore_message_patterns",
        lambda message, *, stored_row=False: (
            not stored_row
            and message.get("content") == "DROP_THIS_RESTART_BATCH"
        ),
    )

    def ambiguous_reconciliation(messages):
        restarted._record_ingest_reconciliation(
            action="persisted batch",
            reason="persisted ambiguous delta",
            cursor=0,
            incoming=len(messages),
            session_count=1,
            stored_tail_count=1,
        )
        return 0

    monkeypatch.setattr(
        restarted,
        "_reconcile_ingest_cursor_from_store",
        ambiguous_reconciliation,
    )
    try:
        restarted.ingest([dropped])
        assert (
            restarted._store.get_session_count("session:warning-no-write")
            == 1
        )
        caplog.clear()

        restarted.ingest([dropped, real_append])

        assert "LCM persisted ambiguous ingest delta" not in caplog.text
        rows = restarted._store.get_session_messages("session:warning-no-write")
        assert rows[-1]["content"] == "later real append"
    finally:
        restarted.shutdown()


def test_publication_values_are_immutable_and_capture_is_bounded(atomic: AtomicFixture):
    intent = atomic.intent()
    result = PublicationResult(
        status="stale",
        node_id=None,
        frontier_store_id=0,
        reason="test",
    )

    with pytest.raises(FrozenInstanceError):
        intent.summary = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.status = "published"  # type: ignore[misc]

    assert intent.source_store_ids == tuple(atomic.source_ids)
    assert len(intent.source_identities) == len(atomic.source_ids)
    assert len(intent.coverage_key) == 64
    assert intent.expected_frontier_store_id == 0
    assert intent.new_frontier_store_id == max(atomic.source_ids)


def test_schema_v6_adds_nullable_coverage_key_and_unique_partial_index(
    tmp_path: Path,
):
    db_path = tmp_path / "v5.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO metadata(key, value) VALUES('schema_version', '5');
        CREATE TABLE messages(
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            conversation_id TEXT DEFAULT '',
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL,
            token_estimate INTEGER DEFAULT 0,
            pinned INTEGER DEFAULT 0
        );
        CREATE TABLE summary_nodes(
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            depth INTEGER NOT NULL DEFAULT 0,
            summary TEXT NOT NULL,
            token_count INTEGER DEFAULT 0,
            source_token_count INTEGER DEFAULT 0,
            source_ids TEXT NOT NULL DEFAULT '[]',
            source_type TEXT NOT NULL DEFAULT 'messages',
            created_at REAL NOT NULL,
            earliest_at REAL,
            latest_at REAL,
            expand_hint TEXT DEFAULT ''
        );
        INSERT INTO summary_nodes(session_id, summary, created_at)
        VALUES('legacy', 'legacy node remains valid', 1.0);
        """
    )
    conn.commit()

    run_versioned_migrations(conn)
    run_versioned_migrations(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(summary_nodes)")}
    indexes = {
        row[1]: row
        for row in conn.execute("PRAGMA index_list(summary_nodes)").fetchall()
    }
    legacy_key = conn.execute(
        "SELECT coverage_key FROM summary_nodes WHERE session_id = 'legacy'"
    ).fetchone()[0]
    version = int(
        conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]
    )
    conn.close()

    assert version == SCHEMA_VERSION
    assert "coverage_key" in columns
    assert legacy_key is None
    assert indexes["idx_summary_nodes_coverage_key_unique"][2] == 1
    assert indexes["idx_summary_nodes_coverage_key_unique"][4] == 1


@pytest.mark.parametrize("starting_version", [4, 5])
def test_v4_v5_to_v6_migration_is_concurrent_and_idempotent(
    tmp_path: Path,
    starting_version: int,
):
    db_path = tmp_path / f"v{starting_version}-race.db"
    seed = sqlite3.connect(db_path)
    conversation_column = "conversation_id TEXT DEFAULT ''," if starting_version >= 5 else ""
    seed.executescript(
        f"""
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO metadata(key, value) VALUES('schema_version', '{starting_version}');
        CREATE TABLE messages(
            store_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            source TEXT DEFAULT '',
            {conversation_column}
            role TEXT NOT NULL,
            content TEXT,
            timestamp REAL NOT NULL
        );
        CREATE TABLE summary_nodes(
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            depth INTEGER NOT NULL DEFAULT 0,
            summary TEXT NOT NULL,
            source_ids TEXT NOT NULL DEFAULT '[]',
            source_type TEXT NOT NULL DEFAULT 'messages',
            created_at REAL NOT NULL
        );
        """
    )
    seed.commit()
    seed.close()

    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def migrate() -> None:
        conn = sqlite3.connect(db_path, timeout=30.0)
        try:
            barrier.wait()
            run_versioned_migrations(conn)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _index: migrate(), range(8)))

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0]
        index_count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' "
            "AND name='idx_summary_nodes_coverage_key_unique'"
        ).fetchone()[0]
        columns = {row[1] for row in conn.execute("PRAGMA table_info(summary_nodes)")}
    assert errors == []
    assert version == str(SCHEMA_VERSION)
    assert index_count == 1
    assert "coverage_key" in columns


def test_response_lost_retry_returns_same_node_without_duplicate(
    atomic: AtomicFixture,
):
    intent = atomic.intent()

    first = atomic.publisher.publish_leaf(intent)
    retry = atomic.publisher.publish_leaf(intent)

    assert first.status == "published"
    assert retry.status == "already_published"
    assert retry.node_id == first.node_id
    assert _node_count(atomic.db_path) == 1
    assert _frontier(atomic.db_path) == intent.new_frontier_store_id


def test_coverage_key_includes_ordered_source_lineage_subset(
    atomic: AtomicFixture,
):
    first_source, second_source = atomic.source_ids[:2]
    first = atomic.publisher.capture_leaf_intent(
        conversation_id=CONVERSATION_ID,
        session_id=SESSION_ID,
        expected_frontier_store_id=0,
        new_frontier_store_id=max(atomic.source_ids),
        source_store_ids=[first_source],
        validation_store_ids=atomic.source_ids,
    ).with_summary(
        summary="first lineage",
        token_count=2,
        source_token_count=2,
    )
    second = atomic.publisher.capture_leaf_intent(
        conversation_id=CONVERSATION_ID,
        session_id=SESSION_ID,
        expected_frontier_store_id=0,
        new_frontier_store_id=max(atomic.source_ids),
        source_store_ids=[second_source],
        validation_store_ids=atomic.source_ids,
    ).with_summary(
        summary="second lineage",
        token_count=2,
        source_token_count=2,
    )

    assert first.coverage_key != second.coverage_key
    assert atomic.publisher.publish_leaf(first).status == "published"
    assert atomic.publisher.publish_leaf(second).status == "stale"
    assert _node_count(atomic.db_path) == 1


def test_two_clone_barrier_same_range_race_has_one_canonical_node(
    atomic: AtomicFixture,
):
    intent = atomic.intent()
    publishers = [
        AtomicPublicationStore(
            atomic.db_path,
            writer_coordinator=atomic.coordinator,
        )
        for _ in range(2)
    ]
    barrier = threading.Barrier(2)

    def publish(publisher: AtomicPublicationStore) -> PublicationResult:
        barrier.wait()
        return publisher.publish_leaf(intent)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(publish, publishers))
    finally:
        for publisher in publishers:
            publisher.close()

    assert sorted(result.status for result in results) == [
        "already_published",
        "published",
    ]
    assert len({result.node_id for result in results}) == 1
    assert _node_count(atomic.db_path) == 1
    assert _fts_count(atomic.db_path) == 1
    assert _frontier(atomic.db_path) == intent.new_frontier_store_id


def test_source_mutation_mismatch_does_not_publish_or_advance(
    atomic: AtomicFixture,
):
    intent = atomic.intent()
    source_id = atomic.source_ids[1]
    with atomic.coordinator.transaction(
        atomic.store.connection,
        begin_immediate=True,
    ):
        atomic.store.connection.execute(
            "UPDATE messages SET content='mutated after capture' WHERE store_id=?",
            (source_id,),
        )

    result = atomic.publisher.publish_leaf(intent)

    assert result.status == "source_mismatch"
    assert result.frontier_store_id == 0
    assert _node_count(atomic.db_path) == 0
    assert _fts_count(atomic.db_path) == 0
    assert _frontier(atomic.db_path) == 0


def test_stale_frontier_does_not_publish_or_regress(
    atomic: AtomicFixture,
):
    intent = atomic.intent()
    atomic.lifecycle.advance_frontier(CONVERSATION_ID, SESSION_ID, 1)

    result = atomic.publisher.publish_leaf(intent)

    assert result.status == "stale"
    assert result.frontier_store_id == 1
    assert _node_count(atomic.db_path) == 0
    assert _frontier(atomic.db_path) == 1


def test_failure_after_insert_rolls_back_node_frontier_and_fts(tmp_path: Path):
    def fail_after_insert(_connection, _intent, _node_id):
        raise RuntimeError("injected after insert")

    fixture = AtomicFixture(tmp_path / "rollback.db", after_insert=fail_after_insert)
    try:
        intent = fixture.intent()
        with pytest.raises(RuntimeError, match="injected after insert"):
            fixture.publisher.publish_leaf(intent)

        assert _node_count(fixture.db_path) == 0
        assert _fts_count(fixture.db_path) == 0
        assert _frontier(fixture.db_path) == 0
    finally:
        fixture.close()


def test_publication_uses_one_begin_immediate_and_one_commit(
    atomic: AtomicFixture,
):
    intent = atomic.intent()
    statements: list[str] = []
    atomic.publisher.connection.set_trace_callback(statements.append)

    result = atomic.publisher.publish_leaf(intent)

    normalized = [statement.strip().upper() for statement in statements]
    assert result.status == "published"
    assert sum(statement == "BEGIN IMMEDIATE" for statement in normalized) == 1
    assert sum(statement == "COMMIT" for statement in normalized) == 1
    assert sum(statement == "ROLLBACK" for statement in normalized) == 0
    # sqlite3's trace hook repeats the outer INSERT while executing FTS
    # triggers; every callback carries the same one application statement.
    node_inserts = {
        statement for statement in normalized if "INSERT INTO SUMMARY_NODES" in statement
    }
    assert len(node_inserts) == 1
    assert sum("UPDATE LCM_LIFECYCLE_STATE" in statement for statement in normalized) == 1


def test_summarizer_runs_without_writer_permit_and_capture_precedes_it(
    tmp_path: Path,
    monkeypatch,
):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "summarizer.db"),
            fresh_tail_count=2,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        )
    )
    messages = [{"role": "system", "content": "system canary"}, *_messages()]
    observed: dict[str, object] = {}
    engine.on_session_start(
        SESSION_ID,
        conversation_id=CONVERSATION_ID,
        platform="cli",
        context_length=10_000,
    )

    original_capture = engine._publication.capture_leaf_intent

    def capture(**kwargs):
        observed["captured"] = True
        return original_capture(**kwargs)

    def summarize(chunk, focus_topic=None):
        observed["summarized_after_capture"] = observed.get("captured", False)
        observed["active_writers"] = engine._store.writer_coordinator.metrics_snapshot()[
            "active_writers"
        ]
        text = "leaf keeps alpha canary and tool relation\n[Expand for details: alpha, gamma]"
        return list(chunk), count_messages_tokens(chunk), text, 1, 1

    monkeypatch.setattr(engine._publication, "capture_leaf_intent", capture)
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    try:
        compressed = engine.compress(messages, force=True)

        assert observed == {
            "captured": True,
            "summarized_after_capture": True,
            "active_writers": 0,
        }
        assert engine._last_compacted_store_id > 0
        assert engine._lifecycle.get_by_conversation(
            CONVERSATION_ID
        ).current_frontier_store_id == engine._last_compacted_store_id
        assert "alpha canary" in "\n".join(str(msg.get("content", "")) for msg in compressed)
    finally:
        engine.shutdown()


def test_success_preserves_canary_fresh_tail_and_tool_pair_invariants(
    tmp_path: Path,
    monkeypatch,
):
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(tmp_path / "context.db"),
            fresh_tail_count=2,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        )
    )
    messages = [
        {"role": "system", "content": "system canary"},
        {"role": "user", "content": "old alpha canary"},
        {"role": "assistant", "content": "old beta"},
        *_messages()[2:],
    ]
    engine.on_session_start(
        SESSION_ID,
        conversation_id=CONVERSATION_ID,
        platform="cli",
        context_length=10_000,
    )

    def summarize(chunk, focus_topic=None):
        text = "old alpha canary preserved\n[Expand for details: old alpha]"
        return list(chunk), count_messages_tokens(chunk), text, 1, 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    try:
        compressed = engine.compress(messages, force=True)
        content = "\n".join(str(message.get("content") or "") for message in compressed)
        roles = [message.get("role") for message in compressed]

        assert compressed[0]["role"] == "system"
        assert "system canary" in str(compressed[0]["content"])
        assert "old alpha canary" in content
        assert compressed[-2:] == messages[-2:]
        assert roles[-2:] == ["assistant", "tool"]
        assert compressed[-1]["tool_call_id"] == compressed[-2]["tool_calls"][0]["id"]
    finally:
        engine.shutdown()


def test_cross_session_rollover_maps_more_than_one_thousand_lineage_rows(
    tmp_path: Path,
    monkeypatch,
):
    db_path = tmp_path / "rollover-1005.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            dynamic_leaf_chunk_enabled=False,
            incremental_max_depth=0,
        )
    )
    conversation_id = "conversation:rollover-1005"
    new_session_id = "session:new-1005"
    source_messages = [
        {"role": "user", "content": f"old-lineage-{index:04d}"}
        for index in range(1005)
    ]
    fresh_canary = {"role": "assistant", "content": "fresh-tail-canary-1005"}
    source_ids = engine._store.append_batch(
        "session:old-1005",
        [*source_messages, fresh_canary],
        conversation_id=conversation_id,
    )
    active = [
        {"role": "system", "content": "system-canary-1005"},
        *source_messages,
        fresh_canary,
    ]
    original_active = copy.deepcopy(active)
    engine.on_session_start(
        new_session_id,
        conversation_id=conversation_id,
        platform="cli",
        context_length=100_000,
    )
    engine._ingest_cursor = len(active)
    engine._ingest_cursor_needs_reconcile = False

    def summarize(chunk, focus_topic=None):
        assert len(chunk) == len(source_messages)
        text = "all rollover lineage retained\n[Expand for details: old-lineage]"
        return list(chunk), count_messages_tokens(chunk), text, 1, 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    try:
        compressed = engine.compress(active, force=True)
        nodes = engine._dag.get_session_nodes(new_session_id)

        assert active == original_active
        assert len(nodes) == 1
        assert nodes[0].source_ids == source_ids[:-1]
        assert len(nodes[0].source_ids) == 1005
        assert _frontier(db_path, conversation_id) == source_ids[-2]
        assert str(compressed[0]["content"]).startswith("system-canary-1005")
        assert compressed[-1] == fresh_canary
    finally:
        engine.shutdown()


def test_sparse_rollover_gap_maps_between_exact_neighbors_in_large_conversation(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    """A small rollover gap must not depend on exhausting unrelated later rows."""

    db_path = tmp_path / "rollover-sparse-gap.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            dynamic_leaf_chunk_enabled=False,
            incremental_max_depth=0,
        )
    )
    conversation_id = "conversation:rollover-sparse-gap"
    current_session_id = "session:current-sparse-gap"
    old_session_id = "session:old-sparse-gap"
    engine.on_session_start(
        current_session_id,
        conversation_id=conversation_id,
        platform="discord",
        context_length=10_000,
    )
    first = {"role": "user", "content": "exact-current-before-gap"}
    carried = {"role": "assistant", "content": "carried-old-session-gap"}
    following = {"role": "user", "content": "exact-current-after-gap"}
    fresh = {"role": "assistant", "content": "fresh-tail-canary-sparse-gap"}
    first_id = engine._store.append(
        current_session_id,
        first,
        conversation_id=conversation_id,
    )
    carried_id = engine._store.append(
        old_session_id,
        carried,
        conversation_id=conversation_id,
    )
    following_id = engine._store.append(
        current_session_id,
        following,
        conversation_id=conversation_id,
    )
    fresh_id = engine._store.append(
        current_session_id,
        fresh,
        conversation_id=conversation_id,
    )
    engine._store.append_batch(
        old_session_id,
        [
            {"role": "assistant", "content": f"unrelated-later-row-{index}"}
            for index in range(80)
        ],
        conversation_id=conversation_id,
    )
    active = [
        {"role": "system", "content": "system-canary-sparse-gap"},
        first,
        carried,
        following,
        fresh,
    ]
    engine._remember_active_replay_messages(
        active,
        active,
        [None, first_id, None, following_id, fresh_id],
    )
    engine._ingest_cursor = len(active)
    engine._ingest_cursor_needs_reconcile = False

    def summarize(chunk, focus_topic=None):
        assert chunk == active[1:4]
        text = "sparse gap retained\n[Expand for details: sparse-gap]"
        return list(chunk), count_messages_tokens(chunk), text, 1, 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    try:
        compressed = engine.compress(active, force=True)
        nodes = engine._dag.get_session_nodes(current_session_id)

        assert len(nodes) == 1
        assert nodes[0].source_ids == [first_id, carried_id, following_id]
        assert _frontier(db_path, conversation_id) == following_id
        assert compressed[-1] == fresh
        assert "LCM publication mapping incomplete" not in caplog.text
    finally:
        engine.shutdown()


def test_missing_rollover_mapping_fails_safe_without_active_context_mutation(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    db_path = tmp_path / "rollover-missing.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        )
    )
    conversation_id = "conversation:rollover-missing"
    mapped = {"role": "user", "content": "mapped-old-row"}
    missing = {"role": "assistant", "content": "missing-old-row"}
    fresh = {"role": "user", "content": "fresh-canary-missing"}
    engine._store.append_batch(
        "session:old-missing",
        [mapped, fresh],
        conversation_id=conversation_id,
    )
    active = [{"role": "system", "content": "system-canary"}, mapped, missing, fresh]
    original_active = copy.deepcopy(active)
    engine.on_session_start(
        "session:new-missing",
        conversation_id=conversation_id,
        platform="cli",
        context_length=10_000,
    )
    engine._ingest_cursor = len(active)
    engine._ingest_cursor_needs_reconcile = False

    def forbidden_summary(*_args, **_kwargs):
        raise AssertionError("partial source mapping must fail before summarization")

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", forbidden_summary)
    try:
        compressed = engine.compress(active, force=True)

        assert active == original_active
        assert compressed == original_active
        assert _node_count(db_path) == 0
        assert _frontier(db_path, conversation_id) == 0
        assert "LCM publication mapping incomplete" in caplog.text
        assert "validation=2 mapped=1 missing=1" in caplog.text
        assert "frontier=0" in caplog.text
        assert engine._last_compression_status == "noop"
        assert "selected leaf chunk source mapping" in engine._last_compression_noop_reason
        assert engine._compression_boundary_cooldown_active() is True
    finally:
        engine.shutdown()


def test_missing_mapping_stays_noop_when_scaffold_cleanup_changes_context(
    tmp_path: Path,
    monkeypatch,
):
    """Cleanup must not disguise a failed publication capture as progress."""

    db_path = tmp_path / "missing-with-scaffold-cleanup.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        )
    )
    conversation_id = "conversation:missing-with-scaffold-cleanup"
    scaffold = {
        "role": "user",
        "content": (
            "[Durable Summary (d2, node 42)]\n"
            "Derived active context.\n"
            "[Expand for details: prior-history]"
        ),
    }
    mapped = {"role": "user", "content": "mapped-old-row"}
    missing = {"role": "assistant", "content": "missing-old-row"}
    fresh = {"role": "user", "content": "fresh-canary"}
    engine._store.append_batch(
        "session:old-with-scaffold",
        [mapped, fresh],
        conversation_id=conversation_id,
    )
    active = [scaffold, mapped, missing, fresh]
    engine.on_session_start(
        "session:new-with-scaffold",
        conversation_id=conversation_id,
        platform="discord",
        context_length=10_000,
    )
    engine._ingest_cursor = len(active)
    engine._ingest_cursor_needs_reconcile = False

    def forbidden_summary(*_args, **_kwargs):
        raise AssertionError("partial source mapping must fail before summarization")

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", forbidden_summary)
    try:
        compressed = engine.compress(active, force=True)

        assert compressed == [mapped, missing, fresh]
        assert engine._last_compression_status == "noop"
        assert "selected leaf chunk source mapping" in engine._last_compression_noop_reason
        assert engine._compression_boundary_cooldown_active() is True
        assert _node_count(db_path) == 0
        assert _frontier(db_path, conversation_id) == 0
    finally:
        engine.shutdown()


def test_rescue_capture_failure_routes_to_noop_and_cooldown(
    tmp_path: Path,
    monkeypatch,
):
    """A later rescue capture failure must not escape or leave running state."""

    db_path = tmp_path / "rescue-capture-failure.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        )
    )
    messages = [
        {"role": "user", "content": "old-source-one"},
        {"role": "assistant", "content": "old-source-two"},
        {"role": "user", "content": "fresh-canary"},
    ]
    engine.on_session_start(
        "session:rescue-capture-failure",
        conversation_id="conversation:rescue-capture-failure",
        platform="discord",
        context_length=10_000,
    )
    capture_calls = 0
    original_mapper = engine._get_publication_store_id_map

    def fail_second_capture(chunk):
        nonlocal capture_calls
        capture_calls += 1
        if capture_calls == 2:
            raise PublicationCaptureError(
                "rescue publication mapping failed",
                frontier_store_id=engine._last_compacted_store_id,
            )
        return original_mapper(chunk)

    def summarize(chunk, focus_topic=None):
        engine._thread_context.leaf_publication_capture_callback(chunk)
        raise AssertionError("failed rescue capture must stop summarization")

    monkeypatch.setattr(engine, "_get_publication_store_id_map", fail_second_capture)
    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    try:
        compressed = engine.compress(messages, force=True)

        assert capture_calls == 2
        assert compressed == messages
        assert engine._last_compression_status == "noop"
        assert engine._last_compression_noop_reason == "rescue publication mapping failed"
        assert engine._compression_boundary_cooldown_active() is True
        assert _node_count(db_path) == 0
        assert _frontier(
            db_path,
            "conversation:rescue-capture-failure",
        ) == 0
    finally:
        engine.shutdown()


def test_fresh_restart_ingest_keeps_exact_lineage_for_missing_persisted_output(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    """Rows inserted by this pass remain publishable without content guessing.

    A host ``<persisted-output>`` marker can become unrecoverable after its
    temporary backing file disappears.  Restart reconciliation must remain
    conservative, but if the current compression pass inserts that snapshot it
    has exact row identity and must not immediately lose it to replay-identity
    normalization.
    """

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    db_path = tmp_path / "fresh-ingest-lineage.db"
    results = tmp_path / "hermes-results"
    results.mkdir()
    full_output = "FULL_CANARY:" + ("x" * 100)
    output_path = results / "call_raw.txt"
    output_path.write_text(full_output, encoding="utf-8")
    marker = (
        "<persisted-output>\n"
        f"This tool result was too large ({len(full_output):,} characters, 0.1 KB).\n"
        f"Full output saved to: {output_path}\n"
        "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
        "Preview (first 30 chars):\n"
        f"{full_output[:30]}\n...\n"
        "</persisted-output>"
    )
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old request A"},
        {
            "role": "assistant",
            "content": "calling",
            "tool_calls": [
                {
                    "id": "call_raw",
                    "type": "function",
                    "function": {"name": "dump", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_raw", "content": marker},
        {"role": "assistant", "content": "old answer B"},
        {"role": "user", "content": "fresh tail"},
    ]
    original_messages = copy.deepcopy(messages)
    config = LCMConfig(
        database_path=str(db_path),
        fresh_tail_count=1,
        leaf_chunk_tokens=1,
        dynamic_leaf_chunk_enabled=False,
        incremental_max_depth=0,
    )
    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    first.on_session_start(
        "session:persisted-output",
        conversation_id="conversation:persisted-output",
        platform="cli",
        context_length=10_000,
    )
    first.ingest(messages)
    old_rows = first._store.get_session_messages("session:persisted-output")
    old_count = len(old_rows)
    old_max_store_id = max(int(row["store_id"]) for row in old_rows)
    first.shutdown()
    output_path.unlink()

    restarted = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes"))
    restarted.on_session_start(
        "session:persisted-output",
        conversation_id="conversation:persisted-output",
        platform="cli",
        context_length=10_000,
    )
    summary_calls: list[list[dict]] = []

    def summarize(chunk, focus_topic=None):
        summary_calls.append(list(chunk))
        return (
            list(chunk),
            count_messages_tokens(chunk),
            "persisted output retained\n[Expand for details: persisted-output]",
            1,
            1,
        )

    monkeypatch.setattr(
        restarted,
        "_summarize_leaf_chunk_with_rescue",
        summarize,
    )
    try:
        compressed = restarted.compress(messages, force=True)
        nodes = restarted._dag.get_session_nodes("session:persisted-output")

        assert messages == original_messages
        assert summary_calls
        assert len(nodes) == 1
        assert nodes[0].source_ids
        assert min(nodes[0].source_ids) > old_max_store_id
        assert _frontier(db_path, "conversation:persisted-output") > old_max_store_id
        assert (
            restarted._store.get_session_count("session:persisted-output")
            == old_count + len(messages)
        )
        assert compressed[-1] == messages[-1]
        assert len(compressed) < len(messages)
        assert "LCM persisted ambiguous ingest delta" in caplog.text
        assert "incoming=6 cursor_before=0 inserted=6" in caplog.text
    finally:
        restarted.shutdown()


def test_exact_runtime_lineage_never_falls_back_past_out_of_order_store_id(
    tmp_path: Path,
):
    """Known provenance fails closed instead of rebinding to a later duplicate."""

    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "exact-order.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:exact-order",
        conversation_id="conversation:exact-order",
        platform="cli",
        context_length=10_000,
    )
    first = {"role": "user", "content": "duplicate"}
    second = {"role": "user", "content": "duplicate"}
    ids = engine._store.append_batch(
        "session:exact-order",
        [first, second, {"role": "user", "content": "duplicate"}],
        conversation_id="conversation:exact-order",
    )
    active = [dict(first), dict(second)]
    engine._remember_active_replay_messages(
        active,
        active,
        [ids[1], ids[0]],
    )
    try:
        mapped = engine._get_publication_store_id_map(active)

        assert mapped.get(id(active[0])) == ids[1]
        assert id(active[1]) not in mapped
        assert ids[2] not in mapped.values()
    finally:
        engine.shutdown()


def test_suffix_runtime_lineage_preserves_active_store_id_order(
    tmp_path: Path,
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "suffix-exact-order.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:suffix-exact-order",
        conversation_id="conversation:suffix-exact-order",
        platform="discord",
        context_length=10_000,
    )
    first = {"role": "user", "content": "duplicate-durable"}
    second = {"role": "user", "content": "duplicate-durable"}
    ids = engine._store.append_batch(
        engine._session_id,
        [first, second],
        conversation_id=engine._conversation_id,
    )
    active = [dict(first), dict(second)]
    engine._remember_active_replay_messages(
        active,
        active,
        [ids[1], ids[0]],
    )
    suffix = (
        "\n\n[Your active task list was preserved across context compression]\n"
        "- [>] v4-review. Verify ordered provenance (in_progress)"
    )
    active[0]["content"] += suffix
    active[1]["content"] += suffix
    try:
        mapped = engine._get_publication_store_id_map(active)

        assert mapped.get(id(active[0])) == ids[1]
        assert id(active[1]) not in mapped
    finally:
        engine.shutdown()


def test_suffix_exact_runtime_lineage_does_not_require_bounded_scan_exhaustion(
    tmp_path: Path,
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "suffix-exact-bounded.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:suffix-exact-bounded",
        conversation_id="conversation:suffix-exact-bounded",
        platform="discord",
        context_length=10_000,
    )
    durable = {"role": "user", "content": "proven-durable-prefix"}
    store_id = engine._store.append(
        engine._session_id,
        durable,
        conversation_id=engine._conversation_id,
    )
    active = dict(durable)
    engine._remember_active_replay_messages([active], [active], [store_id])
    engine._store.append_batch(
        engine._session_id,
        [
            {"role": "assistant", "content": f"later-row-{index}"}
            for index in range(80)
        ],
        conversation_id=engine._conversation_id,
    )
    active["content"] += (
        "\n\n[Your active task list was preserved across context compression]\n"
        "- [>] v4-review. Verify bounded exact lineage (in_progress)"
    )
    try:
        assert engine._get_publication_store_id_map([active]) == {
            id(active): store_id
        }
    finally:
        engine.shutdown()


def test_exact_runtime_lineage_rejects_mutated_associated_message(
    tmp_path: Path,
):
    """Object identity alone cannot bind changed active content to an old row."""

    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "exact-mutation.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:exact-mutation",
        conversation_id="conversation:exact-mutation",
        platform="cli",
        context_length=10_000,
    )
    active = {"role": "user", "content": "before"}
    before_id = engine._store.append(
        "session:exact-mutation",
        active,
        conversation_id="conversation:exact-mutation",
    )
    engine._remember_active_replay_messages([active], [active], [before_id])
    active["content"] = "after"
    after_id = engine._store.append(
        "session:exact-mutation",
        active,
        conversation_id="conversation:exact-mutation",
    )
    try:
        mapped = engine._get_publication_store_id_map([active])

        assert mapped == {}
        assert after_id not in mapped.values()
    finally:
        engine.shutdown()


def test_publication_lineage_accepts_only_trailing_hermes_task_list_suffix(
    tmp_path: Path,
):
    """Hermes may append its synthetic task-list block after LCM ingestion."""

    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "todo-suffix.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:todo-suffix",
        conversation_id="conversation:todo-suffix",
        platform="discord",
        context_length=10_000,
    )
    durable = {
        "role": "user",
        "content": "[Async delegation result]\nWorker completed the bounded task.",
    }
    store_id = engine._store.append(
        "session:todo-suffix",
        durable,
        conversation_id="conversation:todo-suffix",
    )
    active = dict(durable)
    engine._remember_active_replay_messages([active], [active], [store_id])
    active["content"] += (
        "\n\n[Your active task list was preserved across context compression]\n"
        "- [>] v4-review. Verify the worker result across\n"
        "multiple generated lines (in_progress)\n"
        "- [ ] report-final. Report the verification outcome (pending)"
    )
    try:
        assert engine._get_publication_store_id_map([active]) == {
            id(active): store_id
        }

        changed = dict(active)
        changed["content"] = str(changed["content"]).replace(
            "Worker completed",
            "Worker did not complete",
            1,
        )
        assert engine._get_publication_store_id_map([changed]) == {}
    finally:
        engine.shutdown()


def test_compaction_publishes_original_row_after_hermes_task_list_suffix(
    tmp_path: Path,
    monkeypatch,
):
    db_path = tmp_path / "todo-suffix-publication.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        ),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:todo-suffix-publication",
        conversation_id="conversation:todo-suffix-publication",
        platform="discord",
        context_length=10_000,
    )
    messages = [
        {
            "role": "user",
            "content": "[Async delegation result]\nWorker completed the bounded task.",
        },
        {"role": "assistant", "content": "I will verify the worker output."},
        {"role": "user", "content": "fresh-canary"},
    ]
    source_ids = engine._store.append_batch(
        engine._session_id,
        messages,
        conversation_id=engine._conversation_id,
    )
    engine._remember_active_replay_messages(messages, messages, source_ids)
    engine._ingest_cursor = len(messages)
    engine._ingest_cursor_needs_reconcile = False
    messages[0]["content"] += (
        "\n\n[Your active task list was preserved across context compression]\n"
        "- [>] v4-review. Verify the worker result (in_progress)"
    )

    def summarize(chunk, focus_topic=None):
        text = "worker result retained\n[Expand for details: worker-result]"
        return list(chunk), count_messages_tokens(chunk), text, 1, 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)
    try:
        compressed = engine.compress(messages, force=True)
        nodes = engine._dag.get_session_nodes(engine._session_id)

        assert len(nodes) == 1
        assert nodes[0].source_ids == source_ids[:2]
        assert engine._last_compression_status == "compacted"
        assert compressed[-1] == messages[-1]
    finally:
        engine.shutdown()


@pytest.mark.parametrize(
    "invalid_suffix",
    [
        "\n\n[Your active task list was preserved across context compression]",
        (
            "\n\n[Your active task list was preserved across context compression]\n"
            "arbitrary payload"
        ),
        (
            "\n\n[Your active task list was preserved across context compression]\n"
            "- [x] 1. Verify the worker result (pending)"
        ),
        (
            "\n\n[Your active task list was preserved across context compression]\n"
            "- [>] 1. Verify the worker result (pending)"
        ),
        (
            "\n\n[Your active task list was preserved across context compression]\n"
            "- [ ] 1. Verify the worker result (in_progress)"
        ),
        (
            "\n\n[Your active task list was preserved across context compression]\n"
            "- [ ] 1. Verify the worker result (pending)\n"
            "trailing text"
        ),
        (
            "\n\n[Your active task list was preserved across context compression]\n"
            "- [>] . Verify the worker result (in_progress)"
        ),
        (
            "\n\n[Your active task list was preserved across context compression]\n"
            "- [>] 1.    (in_progress)"
        ),
        (
            "\n\n[Your active task list was preserved across context compression]\n"
            "- [>] 1. Verify the worker result (in_progress)\n"
        ),
        (
            "\n\n[Your active task list was preserved across context compression]\n"
            "- [>] 1. Verify the worker result (in_progress)\n"
            "arbitrary text after a valid task line"
        ),
    ],
)
def test_publication_lineage_rejects_malformed_hermes_task_list_suffix(
    tmp_path: Path,
    invalid_suffix: str,
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "invalid-todo-suffix.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:invalid-todo-suffix",
        conversation_id="conversation:invalid-todo-suffix",
        platform="discord",
        context_length=10_000,
    )
    durable = {"role": "user", "content": "durable-prefix"}
    engine._store.append(
        engine._session_id,
        durable,
        conversation_id=engine._conversation_id,
    )
    active = dict(durable)
    active["content"] += invalid_suffix
    try:
        assert engine._get_publication_store_id_map([active]) == {}
    finally:
        engine.shutdown()


def test_publication_lineage_rejects_ambiguous_duplicate_task_list_prefix(
    tmp_path: Path,
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "ambiguous-todo-suffix.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:ambiguous-todo-suffix",
        conversation_id="conversation:ambiguous-todo-suffix",
        platform="discord",
        context_length=10_000,
    )
    durable = {"role": "user", "content": "duplicate-durable-prefix"}
    engine._store.append_batch(
        engine._session_id,
        [durable, durable],
        conversation_id=engine._conversation_id,
    )
    active = dict(durable)
    active["content"] += (
        "\n\n[Your active task list was preserved across context compression]\n"
        "- [ ] 1. Verify the worker result (pending)"
    )
    try:
        assert engine._get_publication_store_id_map([active]) == {}
    finally:
        engine.shutdown()


def test_publication_suffix_fallback_rejects_reordered_distinct_rows(
    tmp_path: Path,
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "reordered-todo-suffix.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:reordered-todo-suffix",
        conversation_id="conversation:reordered-todo-suffix",
        platform="discord",
        context_length=10_000,
    )
    durable_a = {"role": "user", "content": "durable-A"}
    durable_b = {"role": "assistant", "content": "durable-B"}
    ids = engine._store.append_batch(
        engine._session_id,
        [durable_a, durable_b],
        conversation_id=engine._conversation_id,
    )
    suffix = (
        "\n\n[Your active task list was preserved across context compression]\n"
        "- [>] v4-review. Verify fallback ordering (in_progress)"
    )
    active_b = dict(durable_b)
    active_a = dict(durable_a)
    active_b["content"] += suffix
    active_a["content"] += suffix
    try:
        mapped = engine._get_publication_store_id_map([active_b, active_a])

        assert not (
            mapped.get(id(active_b)) == ids[1]
            and mapped.get(id(active_a)) == ids[0]
        )
        assert len(mapped) < 2
    finally:
        engine.shutdown()


def test_publication_suffix_fallback_rejects_reorder_before_base_mapping(
    tmp_path: Path,
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "mixed-reordered-todo-suffix.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:mixed-reordered-todo-suffix",
        conversation_id="conversation:mixed-reordered-todo-suffix",
        platform="discord",
        context_length=10_000,
    )
    durable_a = {"role": "user", "content": "durable-A"}
    durable_b = {"role": "assistant", "content": "durable-B"}
    ids = engine._store.append_batch(
        engine._session_id,
        [durable_a, durable_b],
        conversation_id=engine._conversation_id,
    )
    active_b = dict(durable_b)
    active_b["content"] += (
        "\n\n[Your active task list was preserved across context compression]\n"
        "- [>] v4-review. Verify mixed fallback ordering (in_progress)"
    )
    active_a = dict(durable_a)
    try:
        mapped = engine._get_publication_store_id_map([active_b, active_a])

        assert mapped.get(id(active_a)) == ids[0]
        assert id(active_b) not in mapped
        assert len(mapped) < 2
    finally:
        engine.shutdown()


def test_exact_runtime_lineage_survives_cached_copy_and_clears_on_reset(
    tmp_path: Path,
):
    engine = LCMEngine(
        config=LCMConfig(database_path=str(tmp_path / "exact-cache.db")),
        hermes_home=str(tmp_path / "hermes"),
    )
    engine.on_session_start(
        "session:exact-cache",
        conversation_id="conversation:exact-cache",
        platform="cli",
        context_length=10_000,
    )
    source = [{"role": "user", "content": "cached"}]
    store_id = engine._store.append(
        "session:exact-cache",
        source[0],
        conversation_id="conversation:exact-cache",
    )
    engine._remember_active_replay_messages(source, source, [store_id])
    try:
        copied = engine._cached_active_replay_messages(source)
        assert copied is not None
        assert engine._get_publication_store_id_map(copied) == {
            id(copied[0]): store_id
        }

        engine._reset_session_scoped_runtime_state()
        assert engine._current_active_replay_store_associations_by_message_id == {}
    finally:
        engine.shutdown()


def test_ambiguous_rollover_mapping_fails_safe_without_publication(
    tmp_path: Path,
    monkeypatch,
):
    db_path = tmp_path / "rollover-ambiguous.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        )
    )
    conversation_id = "conversation:rollover-ambiguous"
    duplicate = {"role": "user", "content": "identical-old-row"}
    fresh = {"role": "assistant", "content": "fresh-canary-ambiguous"}
    engine._store.append_batch(
        "session:old-ambiguous",
        [duplicate, dict(duplicate), fresh],
        conversation_id=conversation_id,
    )
    active = [{"role": "system", "content": "system-canary"}, duplicate, fresh]
    original_active = copy.deepcopy(active)
    engine.on_session_start(
        "session:new-ambiguous",
        conversation_id=conversation_id,
        platform="cli",
        context_length=10_000,
    )
    engine._ingest_cursor = len(active)
    engine._ingest_cursor_needs_reconcile = False

    def forbidden_summary(*_args, **_kwargs):
        raise AssertionError("ambiguous source mapping must fail before summarization")

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", forbidden_summary)
    try:
        compressed = engine.compress(active, force=True)

        assert active == original_active
        assert compressed == original_active
        assert _node_count(db_path) == 0
        assert _frontier(db_path, conversation_id) == 0
    finally:
        engine.shutdown()


def test_cas_loser_reloads_canonical_context_without_gc_or_input_mutation(
    tmp_path: Path,
    monkeypatch,
):
    db_path = tmp_path / "cas-loser-context.db"
    engine = LCMEngine(
        config=LCMConfig(
            database_path=str(db_path),
            fresh_tail_count=2,
            leaf_chunk_tokens=1,
            incremental_max_depth=0,
        )
    )
    messages = [{"role": "system", "content": "system canary"}, *_messages()]
    original_messages = copy.deepcopy(messages)
    competitor = AtomicPublicationStore(
        db_path,
        writer_coordinator=engine._store.writer_coordinator,
    )
    engine.on_session_start(
        SESSION_ID,
        conversation_id=CONVERSATION_ID,
        platform="cli",
        context_length=10_000,
    )
    competed = False

    def summarize(chunk, focus_topic=None):
        nonlocal competed
        if not competed:
            source_row = engine._store.connection.execute(
                "SELECT store_id FROM messages WHERE content='alpha canary'"
            ).fetchone()
            source_id = int(source_row[0])
            intent = competitor.capture_leaf_intent(
                conversation_id=CONVERSATION_ID,
                session_id=SESSION_ID,
                expected_frontier_store_id=0,
                new_frontier_store_id=source_id,
                source_store_ids=[source_id],
            ).with_summary(
                summary="competitor canonical canary\n[Expand for details: alpha]",
                token_count=8,
                source_token_count=8,
                created_at=10.0,
                expand_hint="alpha",
            )
            assert competitor.publish_leaf(intent).status == "published"
            competed = True
        text = "loser summary must not publish\n[Expand for details: loser]"
        return list(chunk), count_messages_tokens(chunk), text, 1, 1

    monkeypatch.setattr(engine, "_summarize_leaf_chunk_with_rescue", summarize)

    def forbidden_gc(*_args, **_kwargs):
        raise AssertionError("CAS loser must not GC transcript rows")

    monkeypatch.setattr(engine, "_maybe_gc_compacted_tool_results", forbidden_gc)
    try:
        compressed = engine.compress(messages, force=True)
        content = "\n".join(str(message.get("content") or "") for message in compressed)

        assert messages == original_messages
        assert "competitor canonical canary" in content
        assert "loser summary must not publish" not in content
        assert _node_count(db_path) == 1
        assert engine._last_compacted_store_id == _frontier(db_path)
    finally:
        competitor.close()
        engine.shutdown()


def test_eleven_engine_clone_bounded_publication_stress(tmp_path: Path):
    fixture = AtomicFixture(tmp_path / "eleven.db")
    intent = fixture.intent()
    engines = [
        LCMEngine(
            config=LCMConfig(database_path=str(fixture.db_path)),
        )
        for _ in range(11)
    ]
    barrier = threading.Barrier(11)

    def publish(engine: LCMEngine) -> tuple[PublicationResult, float]:
        barrier.wait()
        started = time.perf_counter()
        result = engine._publication.publish_leaf(intent)
        return result, time.perf_counter() - started

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=11) as pool:
            outcomes = list(pool.map(publish, engines))
    finally:
        elapsed = time.perf_counter() - started
        for engine in engines:
            engine.shutdown()
        fixture.close()

    results = [result for result, _latency in outcomes]
    latencies = sorted(latency for _result, latency in outcomes)
    p95 = latencies[math.ceil(len(latencies) * 0.95) - 1]
    p99 = latencies[math.ceil(len(latencies) * 0.99) - 1]
    assert [result.status for result in results].count("published") == 1
    assert [result.status for result in results].count("already_published") == 10
    assert len({result.node_id for result in results}) == 1
    assert p95 <= p99 < 10.0
    assert elapsed < 10.0


def test_atomic_publication_refuses_too_new_schema_before_configuration(
    tmp_path: Path,
):
    db_path = tmp_path / "publication-future.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION + 1),),
        )
        conn.commit()

    publisher = AtomicPublicationStore(db_path)
    try:
        with pytest.raises(SchemaVersionTooNewError):
            _ = publisher.connection
    finally:
        publisher.close()

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert tables == {"metadata"}
    assert journal_mode == "delete"
