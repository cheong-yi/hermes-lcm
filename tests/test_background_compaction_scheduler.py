"""Deterministic concurrency contracts for the opt-in background scheduler."""

from __future__ import annotations

import threading
import time
import sqlite3
import os
from pathlib import Path

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


def _messages(prefix: str, count: int = 8):
    messages = [{"role": "system", "content": "system"}]
    for index in range(count):
        messages.append(
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"{prefix} {index} " + "payload " * 10,
            }
        )
    return messages


def _engine(db_path, index: int):
    config = LCMConfig(
        database_path=str(db_path),
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
    )
    engine = LCMEngine(config=config)
    engine.on_session_start(
        f"scheduler-session-{index}",
        conversation_id=f"scheduler-conversation-{index}",
        platform="test",
        context_length=1_000,
    )
    return engine


def test_scheduler_caps_eleven_conversations_at_two_workers(tmp_path, monkeypatch):
    lock = threading.Lock()
    active = 0
    maximum = 0

    def slow_summary(**_kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.08)
        with lock:
            active -= 1
        return "prepared summary", 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", slow_summary)
    engines = [_engine(tmp_path / "shared.db", index) for index in range(11)]
    try:
        for index, engine in enumerate(engines):
            messages = _messages(f"conversation-{index}")
            engine.ingest(messages)
            assert engine.schedule_background_compaction(messages) is True
        assert engines[0].wait_for_background_compaction(timeout=5.0) is True

        assert maximum == 2
        assert all(engine.get_async_compaction_status()["prepared_batches"] == 1 for engine in engines)
    finally:
        for engine in engines:
            engine.shutdown()


def test_two_slow_summaries_overlap_within_one_and_a_half_calls(tmp_path, monkeypatch):
    lock = threading.Lock()
    first_started = None
    last_finished = None

    def slow_summary(**_kwargs):
        nonlocal first_started, last_finished
        with lock:
            now = time.perf_counter()
            first_started = now if first_started is None else min(first_started, now)
        time.sleep(0.20)
        with lock:
            last_finished = time.perf_counter()
        return "prepared summary", 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", slow_summary)
    engines = [_engine(tmp_path / "overlap.db", index) for index in range(2)]
    try:
        for index, engine in enumerate(engines):
            messages = _messages(f"overlap-{index}")
            engine.ingest(messages)
            engine.schedule_background_compaction(messages)
        assert engines[0].wait_for_background_compaction(timeout=3.0) is True
        elapsed = last_finished - first_started

        assert elapsed <= 0.30
    finally:
        for engine in engines:
            engine.shutdown()


def test_scheduler_coalesces_one_conversation_and_releases_writer_during_llm(tmp_path, monkeypatch):
    calls = 0
    maximum_for_conversation = 0
    active_for_conversation = 0
    lock = threading.Lock()
    engine = _engine(tmp_path / "coalesce.db", 0)

    def inspect_summary(**_kwargs):
        nonlocal calls, maximum_for_conversation, active_for_conversation
        assert engine._store.writer_coordinator.metrics_snapshot()["active_writers"] == 0
        with lock:
            calls += 1
            active_for_conversation += 1
            maximum_for_conversation = max(maximum_for_conversation, active_for_conversation)
        time.sleep(0.08)
        with lock:
            active_for_conversation -= 1
        return "prepared summary", 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", inspect_summary)
    messages = _messages("coalesced")
    try:
        engine.ingest(messages)
        for _ in range(8):
            engine.schedule_background_compaction(messages)
        assert engine.wait_for_background_compaction(timeout=3.0) is True

        assert maximum_for_conversation == 1
        assert calls == 1
        assert engine.get_async_compaction_status()["prepared_batches"] == 1
    finally:
        engine.shutdown()


def test_foreground_writer_does_not_wait_for_background_summary(tmp_path, monkeypatch):
    background_started = threading.Event()
    release_background = threading.Event()

    def selective_summary(**_kwargs):
        if threading.current_thread().name.startswith("lcm-background"):
            background_started.set()
            assert release_background.wait(2.0)
            return "background summary", 1
        return "foreground summary", 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", selective_summary)
    background = _engine(tmp_path / "foreground.db", 0)
    foreground = _engine(tmp_path / "foreground.db", 1)
    try:
        background_messages = _messages("background")
        background.ingest(background_messages)
        background.schedule_background_compaction(background_messages)
        assert background_started.wait(1.0)

        foreground_messages = _messages("foreground")
        started = time.perf_counter()
        compacted = foreground.compress(
            foreground_messages,
            current_tokens=foreground.threshold_tokens + 1,
        )
        elapsed = time.perf_counter() - started

        assert compacted != foreground_messages
        assert elapsed <= 0.50
    finally:
        release_background.set()
        background.wait_for_background_compaction(timeout=3.0)
        background.shutdown()
        foreground.shutdown()


def test_preparing_claim_has_durable_bounded_lease_and_closed_state(tmp_path):
    engine = _engine(tmp_path / "lease.db", 0)
    engine._config.async_background_compaction_worker_enabled = False
    messages = _messages("lease")
    try:
        engine.ingest(messages)
        before = time.time()
        batch = engine.prepare_background_compaction_once(messages, leave_state="preparing")
        row = engine._store.connection.execute(
            """
            SELECT owner_id, attempt_token, lease_expires_at, attempt_count
            FROM lcm_prepared_compactions WHERE batch_id = ?
            """,
            (batch.batch_id,),
        ).fetchone()

        assert row[0]
        assert row[1]
        assert row[2] >= before + (2 * engine._config.summary_timeout_ms / 1000) + 30
        assert row[3] == 1
        with pytest.raises(sqlite3.IntegrityError):
            engine._store.connection.execute(
                "UPDATE lcm_prepared_compactions SET state = 'unknown' WHERE batch_id = ?",
                (batch.batch_id,),
            )
    finally:
        engine._store.connection.rollback()
        engine.shutdown()


def test_shutdown_cancels_inflight_preparation_before_storage_close(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_summary(**_kwargs):
        started.set()
        assert release.wait(3.0)
        return "late summary", 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", slow_summary)
    db_path = tmp_path / "shutdown.db"
    engine = _engine(db_path, 0)
    messages = _messages("shutdown")
    engine.ingest(messages)
    assert started.wait(1.0)
    scheduler = engine._background_compaction_scheduler

    started_shutdown = time.perf_counter()
    engine.shutdown()
    shutdown_elapsed = time.perf_counter() - started_shutdown
    release.set()

    assert shutdown_elapsed <= 0.50
    assert scheduler.wait_idle(timeout=3.0) is True
    assert scheduler.metrics_snapshot()["failed"] == 0
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT state FROM lcm_prepared_compactions").fetchone()[0] == "pending"
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def _open_fds_for(path: Path) -> list[str]:
    fd_root = Path("/proc/self/fd")
    if not fd_root.exists():
        return []
    matches: list[str] = []
    for entry in fd_root.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if str(path) in target:
            matches.append(target)
    return matches


def test_profile_rebind_cancels_inflight_preparation_and_releases_old_fds(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_summary(**_kwargs):
        started.set()
        assert release.wait(3.0)
        return "late profile summary", 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", slow_summary)
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    config = LCMConfig(
        database_path="",
        fresh_tail_count=2,
        leaf_chunk_tokens=20,
        context_threshold=0.10,
        async_background_compaction_enabled=True,
        async_background_compaction_worker_enabled=True,
    )
    engine = LCMEngine(config=config, hermes_home=str(home_a))
    engine.on_session_start(
        "profile-a-session",
        conversation_id="profile-a-conversation",
        platform="test",
        context_length=1_000,
    )
    messages = _messages("profile-a")
    engine.ingest(messages)
    assert started.wait(1.0)
    old_scheduler = engine._background_compaction_scheduler
    old_db = home_a / "lcm.db"

    started_rebind = time.perf_counter()
    engine.on_session_start(
        "profile-b-session",
        conversation_id="profile-b-conversation",
        platform="test",
        context_length=1_000,
        hermes_home=str(home_b),
    )
    rebind_elapsed = time.perf_counter() - started_rebind

    assert rebind_elapsed <= 0.50
    assert Path(engine._store.db_path) == home_b / "lcm.db"
    assert _open_fds_for(old_db) == []

    # A cancelled background callback must not leak across threads and poison
    # new foreground summary work on the rebound profile.
    monkeypatch.setattr(
        "hermes_lcm.engine.summarize_with_escalation",
        lambda **_kwargs: ("new profile foreground summary", 1),
    )
    _chunk, _tokens, foreground_summary, _level, _attempts = (
        engine._summarize_leaf_chunk_with_rescue(
            [{"role": "user", "content": "new profile foreground work"}]
        )
    )
    assert foreground_summary == "new profile foreground summary"

    release.set()
    assert old_scheduler.wait_idle(timeout=3.0) is True
    assert old_scheduler.metrics_snapshot()["failed"] == 0
    with sqlite3.connect(old_db) as conn:
        assert conn.execute("SELECT state FROM lcm_prepared_compactions").fetchone()[0] == "pending"
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    engine.shutdown()


def test_configured_db_rebind_releases_cancelled_claim_for_retry(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def slow_summary(**_kwargs):
        started.set()
        assert release.wait(3.0)
        return "late configured-db summary", 1

    monkeypatch.setattr("hermes_lcm.engine.summarize_with_escalation", slow_summary)
    db_path = tmp_path / "configured.db"
    engine = _engine(db_path, 0)
    messages = _messages("configured")
    engine.ingest(messages)
    assert started.wait(1.0)
    scheduler = engine._background_compaction_scheduler

    engine.on_session_start(
        "new-profile-session",
        conversation_id="new-profile-conversation",
        platform="test",
        context_length=1_000,
        hermes_home=str(tmp_path / "new-profile"),
    )
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT state FROM lcm_prepared_compactions").fetchone()[0] == "pending"

    release.set()
    assert scheduler.wait_idle(timeout=3.0) is True
    assert scheduler.metrics_snapshot()["failed"] == 0

    engine._config.async_background_compaction_worker_enabled = False
    engine.on_session_start(
        "scheduler-session-0",
        conversation_id="scheduler-conversation-0",
        platform="test",
        context_length=1_000,
    )
    retried = engine.prepare_background_compaction_once(messages)
    assert retried.state == "ready"
    assert retried.attempt_count == 2
    engine.shutdown()
