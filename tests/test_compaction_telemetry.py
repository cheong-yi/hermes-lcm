"""Tests for B2 compaction telemetry (per-conversation snapshot in metadata)."""

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_test.db")
    e = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    e._session_id = "test-session"
    e._conversation_id = "conv-1"
    e.update_model("gpt-test", 200000, provider="openai-codex", api_mode="responses")
    return e


def _hot_usage(prompt=1000, read=400, write=50):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": 100,
        "total_tokens": prompt + 100,
        "input_tokens": prompt - read - write,
        "cache_read_tokens": read,
        "cache_write_tokens": write,
    }


def _cold_usage(prompt=1000):
    # Cache keys present (so cache_metrics_available) but both zero -> cold.
    return {"prompt_tokens": prompt, "cache_read_tokens": 0, "cache_write_tokens": 0}


def _telemetry(engine):
    return engine.get_status().get("compaction_telemetry")


def test_records_per_turn_snapshot(engine):
    engine.update_from_response(_hot_usage(prompt=1050, read=400, write=50))
    t = _telemetry(engine)
    assert t is not None
    assert t["cache_state"] == "hot"
    assert t["consecutive_cold_observations"] == 0
    assert t["turns_since_leaf_compaction"] == 1
    assert t["last_observed_prompt_tokens"] == 1050
    assert t["last_observed_cache_read"] == 400
    assert t["last_observed_cache_write"] == 50
    assert t["peak_prompt_tokens_since_leaf_compaction"] == 1050
    assert t["provider"] == "openai-codex"
    assert t["model"] == "gpt-test"
    assert t["activity_band"] == "low"


def test_turns_since_increments_and_peak_is_max(engine):
    engine.update_from_response(_hot_usage(prompt=1000))
    engine.update_from_response(_hot_usage(prompt=600))
    t = _telemetry(engine)
    assert t["turns_since_leaf_compaction"] == 2
    assert t["peak_prompt_tokens_since_leaf_compaction"] == 1000  # max across turns


def test_cold_streak_counts_and_hot_resets(engine):
    engine.update_from_response(_cold_usage())
    engine.update_from_response(_cold_usage())
    assert _telemetry(engine)["cache_state"] == "cold"
    assert _telemetry(engine)["consecutive_cold_observations"] == 2
    engine.update_from_response(_hot_usage())
    assert _telemetry(engine)["cache_state"] == "hot"
    assert _telemetry(engine)["consecutive_cold_observations"] == 0


def test_idle_turn_is_skipped(engine):
    engine.update_from_response(_hot_usage())
    before = _telemetry(engine)["turns_since_leaf_compaction"]
    # No prompt tokens and no cache keys at all -> no signal -> no write.
    engine.update_from_response({"completion_tokens": 5})
    assert _telemetry(engine)["turns_since_leaf_compaction"] == before


def test_resets_on_compaction(engine):
    engine.update_from_response(_hot_usage(prompt=1000))
    engine.update_from_response(_hot_usage(prompt=1000))
    assert _telemetry(engine)["turns_since_leaf_compaction"] == 2

    # Simulate a leaf compaction happening between turns.
    engine.compression_count += 1
    engine._last_compaction_duration_ms = 12.5
    engine.update_from_response(_hot_usage(prompt=400))

    t = _telemetry(engine)
    assert t["turns_since_leaf_compaction"] == 0
    assert t["total_compactions"] == 1
    assert t["last_leaf_compaction_at"] is not None
    assert t["last_compaction_duration_ms"] == 12.5
    assert t["peak_prompt_tokens_since_leaf_compaction"] == 400  # reset to current


def test_no_conversation_records_nothing(engine):
    engine._conversation_id = ""
    engine.update_from_response(_hot_usage())
    assert _telemetry(engine) is None


def test_unknown_cache_state_when_no_cache_signal(engine):
    # Prompt tokens but no cache keys -> still recorded, state unknown.
    engine.update_from_response({"prompt_tokens": 800})
    t = _telemetry(engine)
    assert t is not None
    assert t["cache_state"] == "unknown"


def test_store_roundtrip_and_skip_unchanged(tmp_path):
    store = MessageStore(tmp_path / "t.db")
    assert store.read_compaction_telemetry("c1") is None
    record = {"conversation_id": "c1", "cache_state": "hot", "turns_since_leaf_compaction": 3}
    store.write_compaction_telemetry("c1", record)
    assert store.read_compaction_telemetry("c1") == record

    # Unchanged payload must not rewrite the row.
    key = store._compaction_telemetry_key("c1")
    store.write_compaction_telemetry("c1", dict(record))
    row = store._conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    assert row is not None

    # Empty conversation id is a no-op.
    store.write_compaction_telemetry("", {"x": 1})
    assert store.read_compaction_telemetry("") is None
    store.close()
