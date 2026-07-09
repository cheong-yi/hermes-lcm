import json
import importlib.util
import sys
from pathlib import Path

from hermes_lcm import tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine


class _FakeAgent:
    def __init__(self, engine: LCMEngine, session_id: str, parent_session_id: str = ""):
        self.engine = engine
        self.session_id = session_id
        self._parent_session_id = parent_session_id
        self._memory_write_origin = "assistant_tool"
        self._memory_write_context = "foreground"
        self.log_prefix = ""
        self.enabled_toolsets = {"terminal", "file", "web"}

    def start_session(self) -> None:
        self.engine.on_session_start(
            self.session_id,
            platform="cli",
            conversation_id=f"conversation:{self.session_id}",
        )

    def ingest_history_after_background_marker(self, history):
        self._memory_write_origin = "background_review"
        self._memory_write_context = "background_review"
        self.engine.ingest(history)

    def ingest_history_as_foreground(self, history):
        self.engine.ingest(history)


class _ForegroundAgent(_FakeAgent):
    pass


def _engine(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"))
    return LCMEngine(config=config, hermes_home=str(tmp_path / "home"))


def _load_plugin_entrypoint_module(module_name: str):
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(repo_root / "__init__.py"),
        submodule_search_locations=[str(repo_root)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_first_ingest_rechecks_background_review_marker_after_missed_bind_time(tmp_path):
    engine = _engine(tmp_path)
    agent = _FakeAgent(engine, "review-session", parent_session_id="foreground-session")
    history = [
        {"role": "user", "content": "foreground question"},
        {"role": "assistant", "content": "foreground answer"},
    ]

    try:
        # Host bug shape: on_session_start fires while the agent still exposes
        # the foreground default, so bind-time auxiliary detection misses.
        agent.start_session()
        assert engine._store.get_session_count("review-session") == 0

        # By first post-LLM ingest the host has set the background-review marker.
        # LCM must re-check before writing the replay snapshot as a new session.
        agent.ingest_history_after_background_marker(history)

        assert engine._store.get_session_count("review-session") == 0
        assert engine._thread_context_has_auxiliary_session("review-session")
    finally:
        engine.shutdown()


def test_late_first_ingest_auxiliary_reclassification_restores_foreground_view(tmp_path):
    engine = _engine(tmp_path)
    foreground = _ForegroundAgent(engine, "foreground-session")
    review = _FakeAgent(engine, "review-session", parent_session_id="foreground-session")

    try:
        foreground.start_session()
        foreground.ingest_history_as_foreground(
            [{"role": "user", "content": "operator durable anchor"}]
        )
        assert engine.current_session_id == "foreground-session"
        assert engine.current_conversation_id == "conversation:foreground-session"

        # Host bug shape: bind-time detection sees the foreground defaults, so
        # on_session_start briefly rebinds the child as a normal foreground.
        review.start_session()
        assert engine.current_session_id == "review-session"

        review.ingest_history_after_background_marker(
            [{"role": "user", "content": "background review replay must not persist"}]
        )

        assert engine._store.get_session_count("review-session") == 0
        assert engine._thread_context_has_auxiliary_session("review-session")
        assert engine.current_session_id == "foreground-session"
        assert engine.current_conversation_id == "conversation:foreground-session"
        assert engine.side_channel_active is True

        status = json.loads(lcm_tools.lcm_status({}, engine=engine))
        assert status["session_id"] == "foreground-session"
        assert status["session_filters"]["side_channel_active"] is True
        assert status["session_filters"]["side_channel_session_id"] == "review-session"

        grep = json.loads(lcm_tools.lcm_grep({"query": "operator"}, engine=engine))
        assert grep["session_scope"] == "current"
        assert [hit["session_id"] for hit in grep["results"]] == ["foreground-session"]
    finally:
        engine.shutdown()


def test_foreground_post_turn_rebinds_after_late_auxiliary_reclassification(tmp_path):
    lcm_plugin = _load_plugin_entrypoint_module("hermes_lcm_post_hook_rebind_regression")
    engine = _engine(tmp_path)
    foreground = _ForegroundAgent(engine, "foreground-session")
    review = _FakeAgent(engine, "review-session", parent_session_id="foreground-session")

    try:
        foreground.start_session()
        foreground.ingest_history_as_foreground(
            [{"role": "user", "content": "operator durable anchor"}]
        )

        review.start_session()
        review.ingest_history_after_background_marker(
            [{"role": "user", "content": "background review replay must not persist"}]
        )
        assert engine.current_session_id == "foreground-session"
        assert engine.bound_session_id == "review-session"
        assert engine.side_channel_active is True

        # The auxiliary thread has ended; the next post_llm_call is a foreground
        # turn. The hook must compare against the bound ingest session, not the
        # operator-facing current_session_id, before deciding whether to rebind.
        engine._clear_thread_context_stateless()
        lcm_plugin._ensure_engine_bound_to_session(
            engine,
            foreground.session_id,
            platform="cli",
            conversation_id="conversation:foreground-session",
        )
        engine.ingest(
            [
                {"role": "user", "content": "operator durable anchor"},
                {"role": "assistant", "content": "foreground follow-up"},
            ]
        )

        assert engine.bound_session_id == "foreground-session"
        assert engine.current_session_id == "foreground-session"
        assert engine._store.get_session_count("review-session") == 0
        foreground_rows = engine._store.get_range("foreground-session", limit=10)
        review_rows = engine._store.get_range("review-session", limit=10)
        assert any(row["content"] == "foreground follow-up" for row in foreground_rows)
        assert all(row["content"] != "foreground follow-up" for row in review_rows)
    finally:
        engine.shutdown()


def test_stacked_late_auxiliary_binds_restore_original_foreground(tmp_path):
    engine = _engine(tmp_path)
    foreground = _ForegroundAgent(engine, "foreground-session")
    first_review = _FakeAgent(engine, "review-session-a", parent_session_id="foreground-session")
    second_review = _FakeAgent(engine, "review-session-b", parent_session_id="foreground-session")

    try:
        foreground.start_session()
        foreground.ingest_history_as_foreground(
            [{"role": "user", "content": "operator durable anchor"}]
        )

        first_review.start_session()
        assert engine.current_session_id == "review-session-a"

        second_review.start_session()
        assert engine.current_session_id == "review-session-b"

        second_review.ingest_history_after_background_marker(
            [{"role": "user", "content": "second background review replay"}]
        )

        assert engine._store.get_session_count("review-session-a") == 0
        assert engine._store.get_session_count("review-session-b") == 0
        assert engine.current_session_id == "foreground-session"
        assert engine.current_conversation_id == "conversation:foreground-session"
        assert engine.bound_session_id == "review-session-b"
    finally:
        engine.shutdown()


def test_confirmed_foreground_ingest_clears_late_auxiliary_restore_candidate(tmp_path):
    engine = _engine(tmp_path)
    first_foreground = _ForegroundAgent(engine, "foreground-a")
    second_foreground = _ForegroundAgent(engine, "foreground-b")
    review = _FakeAgent(engine, "review-session", parent_session_id="foreground-b")

    try:
        first_foreground.start_session()
        first_foreground.ingest_history_as_foreground(
            [{"role": "user", "content": "first foreground turn"}]
        )

        second_foreground.start_session()
        second_foreground.ingest_history_as_foreground(
            [{"role": "user", "content": "second foreground turn"}]
        )

        review.start_session()
        review.ingest_history_after_background_marker(
            [{"role": "user", "content": "late background review replay"}]
        )

        assert engine._store.get_session_count("foreground-a") == 1
        assert engine._store.get_session_count("foreground-b") == 1
        assert engine._store.get_session_count("review-session") == 0
        assert engine.current_session_id == "foreground-b"
        assert engine.current_conversation_id == "conversation:foreground-b"
        assert engine.bound_session_id == "review-session"
    finally:
        engine.shutdown()


def test_first_ingest_recheck_does_not_flag_foreground_session(tmp_path):
    engine = _engine(tmp_path)
    agent = _ForegroundAgent(engine, "foreground-session")
    history = [{"role": "user", "content": "real foreground turn"}]

    try:
        agent.start_session()
        agent.ingest_history_as_foreground(history)

        assert engine._store.get_session_count("foreground-session") == 1
        assert not engine._thread_context_has_auxiliary_session("foreground-session")
    finally:
        engine.shutdown()


def test_first_ingest_recheck_does_not_poison_session_reset_foreground(tmp_path):
    engine = _engine(tmp_path)
    first = _ForegroundAgent(engine, "foreground-a")
    second = _ForegroundAgent(engine, "foreground-b")

    try:
        first.start_session()
        first.ingest_history_as_foreground([{"role": "user", "content": "before reset"}])
        engine.on_session_reset()

        second.start_session()
        second.ingest_history_as_foreground([{"role": "user", "content": "after reset"}])

        assert engine._store.get_session_count("foreground-a") == 1
        assert engine._store.get_session_count("foreground-b") == 1
        assert not engine._thread_context_has_auxiliary_session("foreground-b")
    finally:
        engine.shutdown()
