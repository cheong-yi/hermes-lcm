from __future__ import annotations

from types import SimpleNamespace

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.extraction as lcm_extraction
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import HostEventIdentityIntegrityError, LCMEngine
from hermes_lcm.tokens import count_messages_tokens


def _engine(config: LCMConfig, session_id: str) -> LCMEngine:
    engine = LCMEngine(config=config)
    engine.on_session_start(
        session_id,
        conversation_id=f"conversation-{session_id}",
        platform="discord",
        context_length=100_000,
    )
    return engine


def test_cancelled_summary_and_extraction_publish_nothing(tmp_path, monkeypatch):
    engine = _engine(
        LCMConfig(
            database_path=str(tmp_path / "cancel.db"),
            fresh_tail_count=2,
            leaf_chunk_tokens=1,
        ),
        "cancel-session",
    )
    cancelled = {"value": False}

    def summary_then_cancel(**_kwargs):
        cancelled["value"] = True
        return "late summary", 1

    engine._compression_cancelled_check = lambda: cancelled["value"]
    monkeypatch.setattr(lcm_engine, "summarize_with_escalation", summary_then_cancel)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old source " + "x" * 200},
        {"role": "user", "content": "fresh request"},
        {"role": "assistant", "content": "fresh answer"},
    ]
    try:
        result = engine.compress(
            messages,
            current_tokens=count_messages_tokens(messages),
            force=True,
        )
        assert result == messages
        assert engine._dag.get_session_nodes(engine._session_id) == []
        assert engine.last_compression_status == "noop"
        assert engine.last_compression_noop_reason == "compression cancelled by host"
    finally:
        engine.shutdown()

    cancelled["value"] = False

    def extraction_then_cancel(*_args, **_kwargs):
        cancelled["value"] = True
        return "late extracted decision"

    monkeypatch.setattr(lcm_extraction, "_call_extraction_llm", extraction_then_cancel)
    output = tmp_path / "extractions"
    assert lcm_extraction.extract_before_compaction(
        "serialized input",
        str(output),
        session_id="cancel-session",
        cancelled_check=lambda: cancelled["value"],
    ) is False
    assert not output.exists()


def test_cancellation_inside_publication_transaction_rolls_back(tmp_path, monkeypatch):
    engine = _engine(
        LCMConfig(
            database_path=str(tmp_path / "cancel-transaction.db"),
            fresh_tail_count=1,
            leaf_chunk_tokens=1,
        ),
        "cancel-transaction-session",
    )
    cancelled = {"value": False}
    engine._compression_cancelled_check = lambda: cancelled["value"]
    monkeypatch.setattr(
        lcm_engine,
        "summarize_with_escalation",
        lambda **_kwargs: ("transactional summary", 1),
    )
    publish_leaf = engine._publication.publish_leaf

    def cancel_after_insert(intent, **kwargs):
        after_publish = kwargs["after_publish"]

        def wrapped_after_publish(conn, published_intent, node_id):
            cancelled["value"] = True
            return after_publish(conn, published_intent, node_id)

        kwargs["after_publish"] = wrapped_after_publish
        return publish_leaf(intent, **kwargs)

    monkeypatch.setattr(engine._publication, "publish_leaf", cancel_after_insert)
    messages = [
        {"role": "user", "content": "old source " + "x" * 200},
        {"role": "assistant", "content": "fresh tail"},
    ]
    try:
        result = engine.compress(messages, current_tokens=10_000, force=True)
        assert result == messages
        assert engine._dag.get_session_nodes(engine._session_id) == []
        assert engine._last_compacted_store_id == 0
        assert engine.get_status()["lifecycle"]["current_frontier_store_id"] == 0
        assert engine.last_compression_status == "noop"
        assert engine.last_compression_noop_reason == "compression cancelled by host"
    finally:
        engine.shutdown()


def test_durable_host_id_mapping_is_idempotent_and_changed_rows_conflict(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "physical.db"))
    session_id = "physical-session"
    original = [
        {
            "role": "user",
            "content": "duplicate literal",
            "_row_id": 1,
        },
        {
            "role": "assistant",
            "content": "duplicate literal",
            "_row_id": 2,
        },
    ]

    first = _engine(config, session_id)
    try:
        first._ingest_messages(original)
        assert first._store.get_session_count(session_id) == 2
        assert not hasattr(first, "_resolve_host_replay_store_id")
    finally:
        first.shutdown()

    restarted = _engine(config, session_id)
    try:
        restarted._ingest_messages(original)
        assert restarted._store.get_session_count(session_id) == 2

        changed = [
            {
                "role": "user",
                "content": "changed physical row",
                "_row_id": 1,
            },
            original[1],
        ]
        with pytest.raises(HostEventIdentityIntegrityError):
            restarted._ingest_messages(changed)
        assert restarted._store.get_session_count(session_id) == 2
    finally:
        restarted.shutdown()


def test_generated_replay_provenance_skips_only_marked_rows(tmp_path):
    engine = _engine(
        LCMConfig(database_path=str(tmp_path / "synthetic.db")),
        "synthetic-session",
    )
    content = "literal content identical to generated context"
    generated = {
        "role": "user",
        "content": content,
        "display_metadata": {"_hermes_generated_replay_v1": True},
    }
    literal = {"role": "user", "content": content, "_row_id": 10}
    try:
        engine.bind_replay_rows([generated])
        engine._ingest_messages([generated])
        assert engine._store.get_session_count(engine._session_id) == 1

        engine.bind_replay_rows([literal])
        engine._ingest_messages([literal])
        rows = engine._store.get_session_messages(engine._session_id)
        assert [row["content"] for row in rows] == [content, content]
    finally:
        engine.shutdown()


def test_assertion_worker_keeps_attempt_cancellation_check_after_host_clear(
    tmp_path,
    monkeypatch,
):
    engine = _engine(
        LCMConfig(
            database_path=str(tmp_path / "assertion-cancel.db"),
            assertions_enabled=True,
            assertion_extraction_enabled=True,
        ),
        "assertion-cancel-session",
    )
    messages = [{"role": "user", "content": "durable assertion source"}]
    captured = {}

    class CapturingThread:
        def __init__(self, *, target, args, name, daemon):
            captured.update(target=target, args=args, name=name, daemon=daemon)

        def start(self):
            return None

    slot = SimpleNamespace(acquire=lambda blocking=False: True, release=lambda: None)

    def cancelled_check():
        return True

    try:
        engine._ingest_messages(messages)
        engine._compression_cancelled_check = cancelled_check
        monkeypatch.setattr(lcm_engine, "_ASSERTION_EXTRACTION_PROCESS_SLOT", slot)
        monkeypatch.setattr(lcm_engine.threading, "Thread", CapturingThread)
        assert engine._schedule_pre_compaction_assertions(messages) is True
        engine._compression_cancelled_check = None
        assert captured["args"][-1] is cancelled_check
        assert captured["args"][-1]() is True
    finally:
        engine.shutdown()


def _a0_event(number, content):
    return {
        "role": "user",
        "content": content,
        "display_metadata": {
            "host_event_id": f"event:00000000-0000-4000-8000-{number:012d}",
        },
    }


def _a0_durable_snapshot(database):
    import sqlite3

    conn = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        schema = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        virtual = [
            name for name, sql in schema
            if sql and "CREATE VIRTUAL TABLE" in sql.upper()
        ]
        tables = {}
        for name, _sql in schema:
            if name in virtual or any(name.startswith(v + "_") for v in virtual):
                continue
            quoted = '"' + name.replace('"', '""') + '"'
            cursor = conn.execute(f"SELECT * FROM {quoted}")
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            if name == "messages":
                rows.sort(key=lambda row: row[columns.index("store_id")])
            else:
                rows.sort(key=repr)
            tables[name] = {"columns": columns, "rows": rows}
        assert {"messages", "metadata", "summary_nodes", "lcm_lifecycle_state"} <= tables.keys()
        assert "messages_fts" in virtual
        fts = conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ? ORDER BY rowid",
            ("a0probe",),
        ).fetchall()
        return {"tables": tables, "fts_probe_rowids": fts}
    finally:
        conn.close()


def _a0_memory_snapshot(engine, batch):
    from copy import deepcopy

    names = (
        "_ingest_cursor", "_ingest_cursor_needs_reconcile",
        "_pending_host_replay_authority", "_bound_host_replay_session_id",
        "_bound_host_replay_input_id", "_last_active_replay_source_identities",
        "_last_active_replay_messages", "_last_active_replay_store_ids",
        "_last_fully_ingested_snapshot_session_id",
        "_last_fully_ingested_snapshot_conversation_id",
        "_last_fully_ingested_snapshot_signature",
        "_compression_boundary_ingest_pending",
        "_compression_boundary_active_placeholder_digest_budget",
        "_compression_boundary_active_placeholder_digest_ordinals",
        "_compression_boundary_stored_placeholder_digest_counts",
        "_foreground_rebind_session_id", "_foreground_rebind_previous_session_id",
        "_foreground_rebind_previous_platform",
        "_foreground_rebind_previous_conversation_id",
        "_foreground_rebind_parent_session_id",
        "_generated_ignored_active_replay_placeholder_message_ids",
        "_ignored_message_count", "_ignore_pattern_dropped_count",
        "_last_ingest_reconciliation", "_current_compress_store_ids_by_message_id",
        "_current_compress_placeholder_identity_counts",
        "_session_id", "_conversation_id", "_last_compacted_store_id",
        "_ingest_scope_depth", "_active_host_replay_authority",
        "_active_host_event_store_ids", "_active_host_event_fingerprints",
    )
    snapshot = {name: deepcopy(getattr(engine, name)) for name in names}
    snapshot["associations"] = {
        key: {
            "message_object_id": id(message),
            "message": deepcopy(message),
            "payload_identity": deepcopy(identity),
            "store_id": store_id,
        }
        for key, (message, identity, store_id)
        in engine._current_active_replay_store_associations_by_message_id.items()
    }
    snapshot["caller_messages"] = deepcopy(batch)
    snapshot["caller_list_id"] = id(batch)
    return snapshot


def _a0_emit(evidence):
    import json

    def encode(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        raise TypeError(f"Unsupported A0 evidence type: {type(value).__name__}")

    print("A0_EVIDENCE=" + json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), default=encode,
    ), flush=True)


@pytest.mark.parametrize("fault", [
    "before_final_mapping",
    "after_mapping_phase_before_owner_commit",
])
def test_host_event_mapping_failure_rolls_back_ingest_batch(
    tmp_path, monkeypatch, caplog, fault,
):
    import logging

    database = tmp_path / "a0-mapping.db"
    config = LCMConfig(database_path=str(database))
    engine = _engine(config, "a0-mapping-session")
    batch = [
        _a0_event(1, "a0probe original"),
        _a0_event(2, "a0probe equal new event"),
        _a0_event(3, "a0probe equal new event"),
    ]
    event_ids = [m["display_metadata"]["host_event_id"] for m in batch]
    keys = [engine._host_event_mapping_key(event_id) for event_id in event_ids]
    sentinel = RuntimeError("A0 injected persistence failure")
    trace = []
    inserted_ids = []
    completed_keys = []
    phase_maps = None
    caught = None
    try:
        engine._ingest_messages([batch[0]])
        initial_a = engine._store.read_metadata_json(keys[0])
        assert initial_a is not None
        assert engine._store.get_session_count(engine._session_id) == 1
        assert all(engine._store.read_metadata_json(k) is None for k in keys[1:])
        engine.bind_replay_rows(batch)
        assert engine._ingest_cursor_needs_reconcile is True
        assert engine._ingest_scope_depth == 0
        assert not engine._session_ignored and not engine._session_stateless
        assert not engine._compiled_ignore_message_patterns
        assert engine._store._conn is not None
        assert not engine._store._conn.in_transaction
        assert engine._pending_host_replay_authority == tuple(event_ids)
        assert engine._bound_host_replay_input_id == id(batch)
        assert engine._bound_host_replay_session_id == engine._session_id
        before_durable = _a0_durable_snapshot(database)
        before_memory = _a0_memory_snapshot(engine, batch)
        transient_refs = {
            name: getattr(engine, name) for name in (
                "_active_host_replay_authority", "_active_host_event_store_ids",
                "_active_host_event_fingerprints",
            )
        }
        association_refs = {
            key: value[0] for key, value
            in engine._current_active_replay_store_associations_by_message_id.items()
        }
        append = engine._store._append_protected_batch
        write_mapping = engine._store.write_metadata_json
        mapping_phase = engine._persist_host_event_mappings

        def record(kind, **values):
            trace.append({
                "kind": kind,
                "in_transaction": engine._store._conn.in_transaction,
                "needs_reconcile": engine._ingest_cursor_needs_reconcile,
                **values,
            })

        def observed_append(*args, **kwargs):
            record("append_enter")
            result = append(*args, **kwargs)
            inserted_ids.extend(result)
            record("append_return", store_ids=list(result))
            return result

        def observed_metadata(target_keys, serialized, **kwargs):
            record("metadata_enter", keys=list(target_keys))
            if fault == "before_final_mapping" and keys[2] in target_keys:
                record("raise", at="before_final_mapping")
                raise sentinel
            result = write_mapping(target_keys, serialized, **kwargs)
            completed_keys.extend(target_keys)
            record("metadata_return", keys=list(target_keys))
            return result

        def observed_phase(messages, store_ids):
            nonlocal phase_maps
            record(
                "mapping_phase_enter", same_batch=messages is batch,
                store_ids=list(store_ids),
            )
            result = mapping_phase(messages, store_ids)
            phase_maps = [engine._store.read_metadata_json(k) for k in keys]
            record("mapping_phase_returned", mappings=phase_maps)
            if fault == "after_mapping_phase_before_owner_commit":
                record("raise", at="after_mapping_phase_before_owner_commit")
                raise sentinel
            return result

        caplog.set_level(logging.DEBUG, logger="hermes_lcm.engine")
        caplog.clear()
        with monkeypatch.context() as patch:
            patch.setattr(engine._store, "_append_protected_batch", observed_append)
            patch.setattr(engine._store, "write_metadata_json", observed_metadata)
            patch.setattr(engine, "_persist_host_event_mappings", observed_phase)
            try:
                engine._ingest_messages(batch)
            except Exception as exc:
                caught = exc
            after_durable = _a0_durable_snapshot(database)
            after_memory = _a0_memory_snapshot(engine, batch)
            after_maps = [engine._store.read_metadata_json(k) for k in keys]
            kinds = [item["kind"] for item in trace]
            raises = [item for item in trace if item["kind"] == "raise"]
            phase_enters = [item for item in trace if item["kind"] == "mapping_phase_enter"]
            metadata_enters = [
                item["keys"] for item in trace if item["kind"] == "metadata_enter"
            ]
            expected_completed = keys[:2] if fault == "before_final_mapping" else keys
            admission = {
                "exact_sentinel": caught is sentinel,
                "one_append": kinds.count("append_enter") == kinds.count("append_return") == 1,
                "two_distinct_new_ids": (
                    len(inserted_ids) == len(set(inserted_ids)) == 2
                    and initial_a["store_id"] not in inserted_ids
                ),
                "one_mapping_phase": (
                    len(phase_enters) == 1 and phase_enters[0]["same_batch"]
                    and phase_enters[0]["store_ids"] == [initial_a["store_id"], *inserted_ids]
                ),
                "exact_mapping_entry_order": metadata_enters == [[k] for k in keys],
                "earlier_delegates_completed": completed_keys == expected_completed,
                "intended_fault_once": len(raises) == 1 and raises[0]["at"] == fault,
                "mapping_phase_completion": (
                    kinds.count("mapping_phase_returned")
                    == (0 if fault == "before_final_mapping" else 1)
                ),
                "original_a_preserved": after_maps[0] == initial_a,
                "pre_attempt_reconcile_true": before_memory["_ingest_cursor_needs_reconcile"] is True,
            }
            if fault == "after_mapping_phase_before_owner_commit":
                admission["all_phase_mappings_present"] = (
                    phase_maps is not None
                    and phase_maps[0] == initial_a
                    and all(isinstance(m, dict) for m in phase_maps)
                    and [m["store_id"] for m in phase_maps]
                    == [initial_a["store_id"], *inserted_ids]
                )
            table_names = sorted(
                set(before_durable["tables"]) | set(after_durable["tables"])
            )
            contract = {
                **{
                    f"durable:{name}": (
                        before_durable["tables"].get(name)
                        == after_durable["tables"].get(name)
                    )
                    for name in table_names
                },
                "durable:fts_visibility": (
                    before_durable["fts_probe_rowids"] == after_durable["fts_probe_rowids"]
                ),
                **{
                    f"memory:{name}": before_memory[name] == after_memory[name]
                    for name in before_memory
                },
                **{
                    f"transient_reference:{name}": getattr(engine, name) is reference
                    for name, reference in transient_refs.items()
                },
                "association_references": (
                    association_refs.keys()
                    == engine._current_active_replay_store_associations_by_message_id.keys()
                    and all(
                        engine._current_active_replay_store_associations_by_message_id[k][0] is ref
                        for k, ref in association_refs.items()
                    )
                ),
                "transaction_closed": not engine._store._conn.in_transaction,
                "writer_released": (
                    engine._store.writer_coordinator.metrics_snapshot()["active_writers"] == 0
                ),
                "no_success_log": not any(
                    r.name == "hermes_lcm.engine" and r.getMessage().startswith("Ingested ")
                    for r in caplog.records
                ),
            }
            _a0_emit({
                "case": fault, "admission": admission, "contract": contract,
                "trace": trace, "inserted_ids": inserted_ids,
                "before_durable": before_durable, "after_durable": after_durable,
                "before_memory": before_memory, "after_memory": after_memory,
                "after_mappings": after_maps,
                "fingerprints": [engine._host_event_payload_fingerprint(m) for m in batch],
                "exception": None if caught is None else {
                    "type": type(caught).__name__, "message": str(caught),
                    "is_sentinel": caught is sentinel,
                },
            })
            assert all(admission.values()), {"A0_ADMISSION_FAILED": admission}
            assert all(contract.values()), {
                "A0_UNCHANGED_STATE_CONTRACT": [k for k, ok in contract.items() if not ok],
            }

        # Recovery is reached only after rollback equality succeeds.
        engine._ingest_messages(batch)
        assert engine._store.get_session_count(engine._session_id) == 3
        recovered = [engine._store.read_metadata_json(k) for k in keys]
        assert recovered[0] == initial_a
        assert len({m["store_id"] for m in recovered}) == 3
        committed = _a0_durable_snapshot(database)
        engine._ingest_messages(batch)
        assert _a0_durable_snapshot(database) == committed
    finally:
        engine.shutdown()
    restarted = _engine(config, "a0-mapping-session")
    try:
        restarted._ingest_messages(batch)
        assert _a0_durable_snapshot(database) == committed
        assert [restarted._store.read_metadata_json(k) for k in keys] == recovered
    finally:
        restarted.shutdown()


def test_host_event_mapping_ingest_joins_caller_transaction(tmp_path):
    database = tmp_path / "a0-owner.db"
    engine = _engine(LCMConfig(database_path=str(database)), "a0-owner-session")
    batch = [_a0_event(4, "a0probe caller owned event")]
    key = engine._host_event_mapping_key(batch[0]["display_metadata"]["host_event_id"])
    sentinel = RuntimeError("A0 caller owned rollback")
    trace = []
    caught = None
    try:
        before = _a0_durable_snapshot(database)
        assert not engine._store._conn.in_transaction
        try:
            with engine._store.write_transaction():
                sentinel_id = engine._store.append(
                    engine._session_id,
                    {"role": "user", "content": "a0probe unrelated caller sentinel"},
                    conversation_id=engine._conversation_id,
                )
                engine.bind_replay_rows(batch)
                engine._ingest_messages(batch)
                mapping = engine._store.read_metadata_json(key)
                trace.append({
                    "kind": "nested_ingest_returned",
                    "in_transaction": engine._store._conn.in_transaction,
                    "sentinel_id": sentinel_id, "mapping": mapping,
                    "writer_rows": engine._store.get_session_count(engine._session_id),
                })
                raise sentinel
        except Exception as exc:
            caught = exc
        after = _a0_durable_snapshot(database)
        admission = {
            "exact_sentinel": caught is sentinel,
            "successful_nested_ingest": len(trace) == 1,
            "outer_still_owns_transaction": len(trace) == 1 and trace[0]["in_transaction"],
            "two_uncommitted_rows": len(trace) == 1 and trace[0]["writer_rows"] == 2,
            "mapped_distinct_event": (
                len(trace) == 1 and isinstance(trace[0]["mapping"], dict)
                and trace[0]["mapping"]["store_id"] != trace[0]["sentinel_id"]
            ),
        }
        contract = {
            "durable_unchanged": before == after,
            "mapping_absent": engine._store.read_metadata_json(key) is None,
            "transaction_closed": not engine._store._conn.in_transaction,
            "writer_released": (
                engine._store.writer_coordinator.metrics_snapshot()["active_writers"] == 0
            ),
        }
        _a0_emit({
            "case": "caller_owned_transaction", "admission": admission,
            "contract": contract, "trace": trace,
            "before_durable": before, "after_durable": after,
            "exception": None if caught is None else {
                "type": type(caught).__name__, "message": str(caught),
                "is_sentinel": caught is sentinel,
            },
        })
        assert all(admission.values()), {"A0_OWNER_ADMISSION_FAILED": admission}
        assert all(contract.values()), {"A0_OWNER_ROLLBACK_CONTRACT": contract}
    finally:
        engine.shutdown()
