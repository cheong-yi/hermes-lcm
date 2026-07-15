"""Process-wide SQLite writer coordination acceptance tests.

The production target is eleven independently cloned LCM engines sharing one
database.  SQLite still has one writer slot, so helpers must coordinate before
opening a write transaction without serializing reads or expensive ingest
preparation.
"""

from __future__ import annotations

import gc
import sqlite3
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryNode, build_nodes_fts_spec
from hermes_lcm.db_bootstrap import (
    _write_transaction,
    check_external_content_fts_integrity,
)
from hermes_lcm.sqlite_writer import get_writer_coordinator
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.store import MessageStore, build_message_fts_spec


def test_canonical_path_aliases_share_one_coordinator(tmp_path: Path, monkeypatch):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(real_dir, target_is_directory=True)

    monkeypatch.chdir(tmp_path)
    absolute = get_writer_coordinator(real_dir / "lcm.db")
    symlinked = get_writer_coordinator(alias_dir / "lcm.db")
    relative = get_writer_coordinator(Path("real") / "lcm.db")

    assert absolute is symlinked is relative
    assert get_writer_coordinator(real_dir / "other.db") is not absolute


def test_transaction_admission_is_reentrant_and_rolls_back(tmp_path: Path):
    db_path = tmp_path / "reentrant.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE entries(value TEXT)")
    conn.commit()
    coordinator = get_writer_coordinator(db_path)

    with coordinator.transaction(conn, begin_immediate=True):
        conn.execute("INSERT INTO entries(value) VALUES ('outer')")
        with coordinator.transaction(conn):
            conn.execute("INSERT INTO entries(value) VALUES ('inner')")

    try:
        with coordinator.transaction(conn, begin_immediate=True):
            conn.execute("INSERT INTO entries(value) VALUES ('rolled-back')")
            raise RuntimeError("inject failure")
    except RuntimeError:
        pass

    assert conn.execute("SELECT value FROM entries ORDER BY rowid").fetchall() == [
        ("outer",),
        ("inner",),
    ]
    metrics = coordinator.metrics_snapshot()
    assert metrics["acquisitions"] == 2
    assert metrics["waited_acquisitions"] == 0
    assert metrics["max_active_writers"] == 1
    assert metrics["active_writers"] == 0
    assert metrics["hold_seconds_max"] >= 0
    conn.close()


def test_reentrant_transaction_on_another_connection_fails_fast(tmp_path: Path):
    db_path = tmp_path / "cross-connection.db"
    first = sqlite3.connect(db_path, timeout=0.05)
    second = sqlite3.connect(db_path, timeout=0.05)
    first.execute("CREATE TABLE entries(value TEXT)")
    first.commit()
    coordinator = get_writer_coordinator(db_path)

    started = time.perf_counter()
    with coordinator.transaction(first, begin_immediate=True):
        first.execute("INSERT INTO entries(value) VALUES ('outer')")
        with pytest.raises(RuntimeError, match="different SQLite connection"):
            with coordinator.transaction(second, begin_immediate=True):
                pass
    elapsed = time.perf_counter() - started

    assert elapsed < 0.5
    assert first.execute("SELECT value FROM entries").fetchall() == [("outer",)]
    first.close()
    second.close()


def test_commit_failure_rolls_back_before_next_writer_is_admitted(tmp_path: Path):
    db_path = tmp_path / "deferred-foreign-key.db"
    first = sqlite3.connect(db_path)
    second = sqlite3.connect(db_path)
    for conn in (first, second):
        conn.execute("PRAGMA foreign_keys=ON")
    first.executescript(
        """
        CREATE TABLE parents(id INTEGER PRIMARY KEY);
        CREATE TABLE children(
            parent_id INTEGER,
            FOREIGN KEY(parent_id) REFERENCES parents(id)
                DEFERRABLE INITIALLY DEFERRED
        );
        """
    )
    first.commit()
    coordinator = get_writer_coordinator(db_path)

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        with coordinator.transaction(first, begin_immediate=True):
            first.execute("INSERT INTO children(parent_id) VALUES (99)")

    assert first.in_transaction is False
    with coordinator.transaction(second, begin_immediate=True):
        second.execute("INSERT INTO parents(id) VALUES (99)")
    assert second.execute("SELECT id FROM parents").fetchall() == [(99,)]
    first.close()
    second.close()


def test_bootstrap_fallback_rolls_back_commit_failure():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(
        """
        CREATE TABLE parents(id INTEGER PRIMARY KEY);
        CREATE TABLE children(
            parent_id INTEGER,
            FOREIGN KEY(parent_id) REFERENCES parents(id)
                DEFERRABLE INITIALLY DEFERRED
        );
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        with _write_transaction(conn, begin_immediate=True):
            conn.execute("INSERT INTO children(parent_id) VALUES (7)")

    assert conn.in_transaction is False
    conn.execute("INSERT INTO parents(id) VALUES (7)")
    conn.commit()
    assert conn.execute("SELECT id FROM parents").fetchall() == [(7,)]
    conn.close()


def test_interrupted_ticket_is_skipped_and_later_writer_progresses(
    tmp_path: Path,
    monkeypatch,
):
    coordinator = get_writer_coordinator(tmp_path / "interrupted.db")
    original_wait = coordinator._condition.wait
    interrupted_errors: list[BaseException] = []
    successor_entered = threading.Event()

    def interrupt_one_waiter(timeout=None):
        if threading.current_thread().name == "interrupted-writer":
            raise InterruptedError("synthetic waiter interruption")
        return original_wait(timeout)

    def interrupted_writer() -> None:
        try:
            with coordinator.permit():
                pytest.fail("interrupted waiter must not enter")
        except BaseException as exc:
            interrupted_errors.append(exc)

    def successor_writer() -> None:
        with coordinator.permit():
            successor_entered.set()

    with coordinator.permit():
        monkeypatch.setattr(coordinator._condition, "wait", interrupt_one_waiter)
        interrupted = threading.Thread(
            target=interrupted_writer,
            name="interrupted-writer",
            daemon=True,
        )
        interrupted.start()
        interrupted.join(timeout=2)
        assert not interrupted.is_alive()
        assert len(interrupted_errors) == 1
        assert isinstance(interrupted_errors[0], InterruptedError)

        monkeypatch.setattr(coordinator._condition, "wait", original_wait)
        successor = threading.Thread(target=successor_writer, daemon=True)
        successor.start()
        assert not successor_entered.wait(timeout=0.05)

    successor.join(timeout=2)
    assert not successor.is_alive()
    assert successor_entered.is_set()


def test_different_databases_admit_writers_independently(tmp_path: Path):
    paths = [tmp_path / "a.db", tmp_path / "b.db"]
    conns = [sqlite3.connect(path, check_same_thread=False) for path in paths]
    for conn in conns:
        conn.execute("CREATE TABLE entries(value TEXT)")
        conn.commit()
    coordinators = [get_writer_coordinator(path) for path in paths]
    entered = threading.Event()
    release = threading.Event()

    def hold_first_database() -> None:
        with coordinators[0].transaction(conns[0], begin_immediate=True):
            conns[0].execute("INSERT INTO entries(value) VALUES ('a')")
            entered.set()
            assert release.wait(timeout=5)

    thread = threading.Thread(target=hold_first_database, daemon=True)
    thread.start()
    assert entered.wait(timeout=5)

    started = time.perf_counter()
    with coordinators[1].transaction(conns[1], begin_immediate=True):
        conns[1].execute("INSERT INTO entries(value) VALUES ('b')")
    elapsed = time.perf_counter() - started

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert elapsed < 1.0
    assert conns[0].execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 1
    assert conns[1].execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 1
    for conn in conns:
        conn.close()


def test_ingest_preparation_and_reads_do_not_hold_writer_permit(
    tmp_path: Path,
    monkeypatch,
):
    db_path = tmp_path / "permit-scope.db"
    slow = MessageStore(db_path)
    fast = MessageStore(db_path)
    coordinator = slow.writer_coordinator
    entered = threading.Event()
    release = threading.Event()

    from hermes_lcm import store as store_module

    original_protect = store_module.protect_message_for_ingest

    def blocking_protect(msg, **kwargs):
        if msg.get("content") == "slow-preparation":
            entered.set()
            assert release.wait(timeout=5)
        return original_protect(msg, **kwargs)

    monkeypatch.setattr(store_module, "protect_message_for_ingest", blocking_protect)
    preparation = threading.Thread(
        target=lambda: slow.append(
            "slow", {"role": "user", "content": "slow-preparation"}
        ),
        daemon=True,
    )
    preparation.start()
    assert entered.wait(timeout=5)
    assert coordinator.metrics_snapshot()["active_writers"] == 0
    fast.append("fast", {"role": "user", "content": "fast-write"})
    release.set()
    preparation.join(timeout=5)
    assert not preparation.is_alive()

    store_id = fast.append("read", {"role": "user", "content": "slow-read"})
    read_entered = threading.Event()
    read_release = threading.Event()
    original_row_to_dict = slow._row_to_dict

    def blocking_row_to_dict(row):
        read_entered.set()
        assert read_release.wait(timeout=5)
        return original_row_to_dict(row)

    monkeypatch.setattr(slow, "_row_to_dict", blocking_row_to_dict)
    reader = threading.Thread(target=lambda: slow.get(store_id), daemon=True)
    reader.start()
    assert read_entered.wait(timeout=5)
    assert coordinator.metrics_snapshot()["active_writers"] == 0
    fast.append("fast", {"role": "user", "content": "write-during-read"})
    read_release.set()
    reader.join(timeout=5)
    assert not reader.is_alive()

    slow.close()
    fast.close()


def test_clear_debt_serializes_its_read_check_write_with_new_debt(tmp_path: Path):
    db_path = tmp_path / "clear-debt-race.db"
    clearer = LifecycleStateStore(db_path)
    recorder = LifecycleStateStore(db_path)
    clearer.bind_session("session", conversation_id="conversation")
    clearer.record_debt("conversation", kind="old", size_estimate=1)
    read_complete = threading.Event()
    release_clear = threading.Event()
    record_complete = threading.Event()
    original_get = clearer.get_by_conversation

    def blocking_get(conversation_id):
        state = original_get(conversation_id)
        if not read_complete.is_set():
            read_complete.set()
            assert release_clear.wait(timeout=5)
        return state

    clearer.get_by_conversation = blocking_get
    clear_thread = threading.Thread(
        target=lambda: clearer.clear_debt("conversation"),
        daemon=True,
    )
    record_thread = threading.Thread(
        target=lambda: (
            recorder.record_debt("conversation", kind="new", size_estimate=9),
            record_complete.set(),
        ),
        daemon=True,
    )
    try:
        clear_thread.start()
        assert read_complete.wait(timeout=5)
        record_thread.start()
        assert not record_complete.wait(timeout=0.15)
    finally:
        release_clear.set()
        clear_thread.join(timeout=5)
        record_thread.join(timeout=5)
        del clearer.get_by_conversation

    assert not clear_thread.is_alive()
    assert not record_thread.is_alive()
    state = clearer.get_by_conversation("conversation")
    assert state.debt_kind == "new"
    assert state.debt_size_estimate == 9
    clearer.close()
    recorder.close()


def test_eleven_engine_clones_share_one_writer_and_preserve_exact_rows(tmp_path: Path):
    db_path = tmp_path / "eleven-clones.db"
    engine_count = 11
    rounds = 8
    construction_gate = threading.Event()

    def construct_engine() -> LCMEngine:
        assert construction_gate.wait(timeout=5)
        return LCMEngine(config=LCMConfig(database_path=str(db_path)))

    with ThreadPoolExecutor(max_workers=engine_count) as pool:
        futures = [pool.submit(construct_engine) for _ in range(engine_count)]
        construction_gate.set()
        engines = [future.result(timeout=30) for future in futures]
    coordinator = engines[0]._store.writer_coordinator

    try:
        assert len(engines) == engine_count
        assert all(engine._store.writer_coordinator is coordinator for engine in engines)
        assert all(engine._dag.writer_coordinator is coordinator for engine in engines)
        assert all(engine._lifecycle.writer_coordinator is coordinator for engine in engines)
        assert coordinator.metrics_snapshot()["owner_count"] == 33

        barrier = threading.Barrier(len(engines))

        def write_clone(index: int) -> None:
            engine = engines[index]
            session_id = f"session-{index}"
            conversation_id = f"conversation-{index}"
            barrier.wait(timeout=15)
            engine._lifecycle.bind_session(session_id, conversation_id=conversation_id)
            for round_index in range(rounds):
                store_id = engine._store.append(
                    session_id,
                    {
                        "role": "user",
                        "content": f"message-{index}-{round_index}",
                    },
                    token_estimate=round_index + 1,
                    conversation_id=conversation_id,
                )
                engine._dag.add_node(
                    SummaryNode(
                        session_id=session_id,
                        depth=0,
                        summary=f"summary-{index}-{round_index}",
                        source_ids=[store_id],
                        source_type="messages",
                        token_count=1,
                        source_token_count=round_index + 1,
                    )
                )
                engine._lifecycle.advance_frontier(
                    conversation_id,
                    session_id,
                    store_id,
                )

        with ThreadPoolExecutor(max_workers=len(engines)) as pool:
            futures = [pool.submit(write_clone, index) for index in range(len(engines))]
            for future in futures:
                future.result(timeout=30)

        conn = sqlite3.connect(db_path)
        try:
            expected_rows = engine_count * rounds
            assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == expected_rows
            assert conn.execute("SELECT COUNT(*) FROM summary_nodes").fetchone()[0] == expected_rows
            assert conn.execute("SELECT COUNT(*) FROM lcm_lifecycle_state").fetchone()[0] == engine_count
            assert conn.execute(
                "SELECT COUNT(*) FROM lcm_lifecycle_state WHERE current_frontier_store_id > 0"
            ).fetchone()[0] == engine_count
            assert conn.execute(
                """
                SELECT COUNT(*)
                FROM summary_nodes n
                JOIN messages m
                  ON m.store_id = CAST(json_extract(n.source_ids, '$[0]') AS INTEGER)
                 AND m.session_id = n.session_id
                WHERE n.source_type = 'messages'
                  AND n.summary = REPLACE(m.content, 'message-', 'summary-')
                """
            ).fetchone()[0] == expected_rows
            assert conn.execute(
                """
                SELECT COUNT(*)
                FROM lcm_lifecycle_state l
                JOIN (
                    SELECT conversation_id, session_id, MAX(store_id) AS max_store_id
                    FROM messages
                    GROUP BY conversation_id, session_id
                ) m
                  ON m.conversation_id = l.conversation_id
                 AND m.session_id = l.current_session_id
                 AND m.max_store_id = l.current_frontier_store_id
                """
            ).fetchone()[0] == engine_count
            assert conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0] == expected_rows
            assert conn.execute("SELECT COUNT(*) FROM nodes_fts_docsize").fetchone()[0] == expected_rows
            assert check_external_content_fts_integrity(
                conn, build_message_fts_spec()
            )["status"] == "pass"
            assert check_external_content_fts_integrity(
                conn, build_nodes_fts_spec()
            )["status"] == "pass"
            assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        finally:
            conn.close()

        metrics = coordinator.metrics_snapshot()
        assert metrics["max_active_writers"] == 1
        assert metrics["active_writers"] == 0
        assert metrics["wait_seconds_p95"] >= 0
        assert metrics["wait_seconds_p99"] >= metrics["wait_seconds_p95"]
        assert metrics["hold_seconds_p99"] >= metrics["hold_seconds_p95"]
    finally:
        for engine in engines:
            engine.shutdown()

    closed_metrics = coordinator.metrics_snapshot()
    assert closed_metrics["owner_count"] == 0
    assert closed_metrics["checkpoint_attempt_count"] == 1
    assert closed_metrics["checkpoint_success_count"] == 1


def test_unused_coordinator_registry_entry_can_be_collected_and_recreated(tmp_path: Path):
    db_path = tmp_path / "collectable.db"
    coordinator = get_writer_coordinator(db_path)
    original_ref = weakref.ref(coordinator)
    del coordinator
    gc.collect()

    assert original_ref() is None
    recreated = get_writer_coordinator(db_path)
    assert recreated.database_path == str(db_path.resolve())


class _CheckpointCursor:
    def __init__(self, result):
        self._result = result

    def fetchone(self):
        return self._result


class _CheckpointConnection:
    def __init__(self, *, result=None, error: sqlite3.Error | None = None):
        self.result = result
        self.error = error
        self.closed = False

    def execute(self, sql):
        assert sql == "PRAGMA wal_checkpoint(PASSIVE)"
        if self.error is not None:
            raise self.error
        return _CheckpointCursor(self.result)

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("suffix", "connection", "metric", "last_result"),
    [
        ("busy", _CheckpointConnection(result=(1, 10, 2)), "checkpoint_busy_count", "busy"),
        (
            "failure",
            _CheckpointConnection(error=sqlite3.OperationalError("checkpoint failed")),
            "checkpoint_failure_count",
            "error: checkpoint failed",
        ),
    ],
)
def test_checkpoint_metrics_distinguish_attempts_from_busy_and_failure(
    tmp_path: Path,
    suffix,
    connection,
    metric,
    last_result,
):
    coordinator = get_writer_coordinator(tmp_path / f"checkpoint-{suffix}.db")
    owner_token = coordinator.bind_owner()

    coordinator.close_owner(owner_token, connection)

    metrics = coordinator.metrics_snapshot()
    assert connection.closed is True
    assert metrics["checkpoint_attempt_count"] == 1
    assert metrics["checkpoint_success_count"] == 0
    assert metrics[metric] == 1
    assert metrics["checkpoint_last_result"] == last_result
