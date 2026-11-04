"""Focused tests for host-owned event identity and projection boundaries."""

from __future__ import annotations

import json
from threading import Event

import pytest

from agent.event_identity import ensure_host_event_identities
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import HostEventIdentityIntegrityError, LCMEngine


EVENT_ONE = "event:00000000-0000-4000-8000-000000000001"
EVENT_TWO = "event:00000000-0000-4000-8000-000000000002"


def _engine(path, session_id: str) -> LCMEngine:
    instance = LCMEngine(config=LCMConfig(database_path=str(path)))
    instance._session_id = session_id
    instance._conversation_id = f"conversation:{session_id}"
    return instance


def _event(content: str, event_id: str | None = None) -> dict:
    message = {"role": "user", "content": content}
    if event_id is not None:
        message["display_metadata"] = {"host_event_id": event_id}
    return message


def test_explicit_event_id_is_idempotent_across_restart(tmp_path):
    database = tmp_path / "identity.db"
    first = _engine(database, "session-a")
    try:
        first._ingest_messages([_event("same", EVENT_ONE)])
        assert first._store.get_session_count("session-a") == 1
    finally:
        first.shutdown()

    second = _engine(database, "session-a")
    try:
        second._ingest_messages([_event("same", EVENT_ONE)])
        assert second._store.get_session_count("session-a") == 1
    finally:
        second.shutdown()


def test_same_id_in_another_session_is_independent(tmp_path):
    database = tmp_path / "session-scoped.db"
    first = _engine(database, "session-a")
    second = _engine(database, "session-b")
    try:
        first._ingest_messages([_event("same", EVENT_ONE)])
        second._ingest_messages([_event("different session", EVENT_ONE)])
        assert first._store.get_session_count("session-a") == 1
        assert second._store.get_session_count("session-b") == 1
    finally:
        first.shutdown()
        second.shutdown()


def test_incompatible_reuse_surfaces_without_mutating_store(tmp_path):
    engine = _engine(tmp_path / "conflict.db", "session-a")
    try:
        engine._ingest_messages([_event("original", EVENT_ONE)])
        with pytest.raises(HostEventIdentityIntegrityError):
            engine._ingest_messages([_event("changed", EVENT_ONE)])
        assert engine._store.get_session_count("session-a") == 1
        assert engine._store.get_session_messages("session-a")[0]["content"] == "original"
    finally:
        engine.shutdown()


def test_unbound_equal_rows_remain_distinct(tmp_path):
    engine = _engine(tmp_path / "unbound.db", "session-a")
    try:
        engine._ingest_messages([_event("same")])
        engine._ingest_messages([_event("same")])
        assert engine._store.get_session_count("session-a") == 2
    finally:
        engine.shutdown()


def test_projection_and_generated_lookalikes_do_not_replace_sources(tmp_path):
    engine = _engine(tmp_path / "projection.db", "session-a")
    try:
        source_a = _event("same", EVENT_ONE)
        source_b = _event("same", EVENT_TWO)
        projection = {
            "role": "user",
            "content": "same",
            "display_metadata": {
                "source_event_ids": [EVENT_TWO, EVENT_ONE, EVENT_ONE],
            },
        }
        generated = {
            "role": "assistant",
            "content": "unproven generated input",
            "display_metadata": {"_hermes_generated_replay_v1": True},
        }
        generated_summary = {
            "role": "assistant",
            "content": "proven generated summary",
            "display_metadata": {
                "_hermes_generated_replay_v1": True,
                "source_event_ids": [],
            },
        }
        engine._ingest_messages([source_a, source_b, projection, generated, generated_summary])
        rows = engine._store.get_session_messages("session-a")
        assert [row["content"] for row in rows] == [
            "same",
            "same",
            "unproven generated input",
        ]
        assert projection["display_metadata"]["source_event_ids"] == [
            EVENT_TWO,
            EVENT_ONE,
            EVENT_ONE,
        ]
    finally:
        engine.shutdown()


def test_bound_authority_survives_nested_ingest_scope(tmp_path):
    engine = _engine(tmp_path / "nested.db", "session-a")
    try:
        messages = [_event("nested")]
        ensure_host_event_identities(messages)
        bound_id = messages[0]["display_metadata"]["host_event_id"]
        engine.bind_replay_rows(messages)
        original_ingest = engine._ingest_messages_with_host_authority
        nested = {"entered": False}

        def ingest_with_nested_scope(bound_messages):
            if not nested["entered"]:
                nested["entered"] = True
                engine._ingest_messages(bound_messages)
            return original_ingest(bound_messages)

        engine._ingest_messages_with_host_authority = ingest_with_nested_scope
        engine._ingest_messages(messages)
        assert engine._store.get_session_count("session-a") == 1
        assert engine._pending_host_replay_authority == (bound_id,)
        engine.clear_replay_authority()
        assert engine._pending_host_replay_authority is None
    finally:
        engine.shutdown()


def test_background_snapshot_keeps_identity_metadata_and_session_gate(
    tmp_path,
    monkeypatch,
):
    engine = _engine(tmp_path / "background.db", "session-a")
    try:
        messages = [_event("background")]
        ensure_host_event_identities(messages)
        bound_id = messages[0]["display_metadata"]["host_event_id"]
        scheduler = engine._background_compaction_scheduler
        submitted = Event()

        def hold_scheduler_drain():
            submitted.set()

        monkeypatch.setattr(scheduler, "_drain_locked", hold_scheduler_drain)
        assert scheduler.submit(engine, messages) is True
        assert submitted.wait(timeout=1) is True
        job = scheduler._queued.get(engine._conversation_id)
        assert job is not None
        assert job.session_id == "session-a"
        assert job.messages[0]["display_metadata"]["host_event_id"] == bound_id
        messages[0]["display_metadata"]["host_event_id"] = EVENT_ONE
        assert job.messages[0]["display_metadata"]["host_event_id"] == bound_id
    finally:
        engine.shutdown()


def test_r1_mixed_known_new_physical_event_identity(tmp_path):
    evidence = {}
    oracles = {}

    def snapshot(engine, session_id):
        return [
            {
                "store_id": row["store_id"],
                "role": row["role"],
                "content": row["content"],
            }
            for row in engine._store.get_session_messages(session_id)
        ]

    def mapping(engine, event_id):
        return engine._store.read_metadata_json(
            engine._host_event_mapping_key(event_id)
        )

    def mapping_store_id(value):
        return value.get("store_id") if isinstance(value, dict) else None

    session_id = "r1-mixed-known-new"
    mixed = _engine(tmp_path / "r1-mixed.db", session_id)
    try:
        event_a = _event("physical event A", EVENT_ONE)
        event_b = _event("physical event B", EVENT_TWO)

        mixed._ingest_messages([event_a])
        first_rows = snapshot(mixed, session_id)
        first_a_mapping = mapping(mixed, EVENT_ONE)
        evidence["1_first_explicit_event"] = {
            "rows": first_rows,
            "event_one_mapping": first_a_mapping,
        }
        oracles["1_first_explicit_event"] = (
            len(first_rows) == 1
            and mapping_store_id(first_a_mapping) == first_rows[0]["store_id"]
        )

        mixed._ingest_messages([event_a, event_b])
        mixed_rows = snapshot(mixed, session_id)
        mixed_a_mapping = mapping(mixed, EVENT_ONE)
        mixed_b_mapping = mapping(mixed, EVENT_TWO)
        inserted_after_first = sorted(
            {row["store_id"] for row in mixed_rows}
            - {row["store_id"] for row in first_rows}
        )
        evidence["2_mixed_known_new"] = {
            "rows": mixed_rows,
            "event_one_mapping": mixed_a_mapping,
            "event_two_mapping": mixed_b_mapping,
            "inserted_store_ids": inserted_after_first,
        }
        oracles["2_mixed_known_new"] = (
            len(mixed_rows) == 2
            and mixed_a_mapping == first_a_mapping
            and inserted_after_first == [mapping_store_id(mixed_b_mapping)]
            and mapping_store_id(mixed_b_mapping)
            != mapping_store_id(mixed_a_mapping)
        )

        mixed._ingest_messages([event_a, event_b])
        replay_rows = snapshot(mixed, session_id)
        replay_a_mapping = mapping(mixed, EVENT_ONE)
        replay_b_mapping = mapping(mixed, EVENT_TWO)
        evidence["3_exact_replay"] = {
            "rows": replay_rows,
            "event_one_mapping": replay_a_mapping,
            "event_two_mapping": replay_b_mapping,
            "added_rows": len(replay_rows) - len(mixed_rows),
        }
        oracles["3_exact_replay"] = (
            replay_rows == mixed_rows
            and replay_a_mapping == mixed_a_mapping
            and replay_b_mapping == mixed_b_mapping
        )

        mixed._ingest_messages([event_a, event_a, event_b])
        repeated_rows = snapshot(mixed, session_id)
        repeated_a_mapping = mapping(mixed, EVENT_ONE)
        repeated_b_mapping = mapping(mixed, EVENT_TWO)
        evidence["4_same_batch_repeated_ids"] = {
            "rows": repeated_rows,
            "event_one_mapping": repeated_a_mapping,
            "event_two_mapping": repeated_b_mapping,
            "added_rows": len(repeated_rows) - len(replay_rows),
        }
        oracles["4_same_batch_repeated_ids"] = (
            repeated_rows == replay_rows
            and repeated_a_mapping == replay_a_mapping
            and repeated_b_mapping == replay_b_mapping
        )
    finally:
        mixed.shutdown()

    equal_explicit_session = "r1-equal-explicit"
    equal_explicit = _engine(tmp_path / "r1-equal-explicit.db", equal_explicit_session)
    try:
        equal_explicit._ingest_messages([
            _event("equal content", EVENT_ONE),
            _event("equal content", EVENT_TWO),
        ])
        equal_explicit_rows = snapshot(equal_explicit, equal_explicit_session)
        equal_explicit_a_mapping = mapping(equal_explicit, EVENT_ONE)
        equal_explicit_b_mapping = mapping(equal_explicit, EVENT_TWO)
        evidence["5_equal_content_distinct_ids"] = {
            "rows": equal_explicit_rows,
            "event_one_mapping": equal_explicit_a_mapping,
            "event_two_mapping": equal_explicit_b_mapping,
        }
        oracles["5_equal_content_distinct_ids"] = (
            len(equal_explicit_rows) == 2
            and mapping_store_id(equal_explicit_a_mapping)
            != mapping_store_id(equal_explicit_b_mapping)
            and {
                mapping_store_id(equal_explicit_a_mapping),
                mapping_store_id(equal_explicit_b_mapping),
            }
            == {row["store_id"] for row in equal_explicit_rows}
        )
    finally:
        equal_explicit.shutdown()

    unbound_session = "r1-equal-unbound"
    unbound = _engine(tmp_path / "r1-equal-unbound.db", unbound_session)
    try:
        unbound._ingest_messages([_event("equal unbound content")])
        first_unbound_rows = snapshot(unbound, unbound_session)
        unbound._ingest_messages([_event("equal unbound content")])
        second_unbound_rows = snapshot(unbound, unbound_session)
        evidence["6_separate_equal_unbound_events"] = {
            "after_first": first_unbound_rows,
            "after_second": second_unbound_rows,
            "inserted_store_ids": sorted(
                {row["store_id"] for row in second_unbound_rows}
                - {row["store_id"] for row in first_unbound_rows}
            ),
        }
        oracles["6_separate_equal_unbound_events"] = (
            len(first_unbound_rows) == 1
            and len(second_unbound_rows) == 2
            and len({
                row["store_id"] for row in second_unbound_rows
            }) == 2
        )
    finally:
        unbound.shutdown()

    print(
        "R1_EVIDENCE="
        + json.dumps(
            {"evidence": evidence, "oracles": oracles},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    assert all(oracles.values()), oracles
