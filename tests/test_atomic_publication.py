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

    assert SCHEMA_VERSION == 6
    assert version == 6
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
    assert version == "6"
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


def test_missing_rollover_mapping_fails_safe_without_active_context_mutation(
    tmp_path: Path,
    monkeypatch,
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
