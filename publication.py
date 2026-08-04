"""Atomic, idempotent publication of foreground leaf summaries.

Leaf preparation is deliberately split from publication.  Capturing source
identity is read-only, summarization can take arbitrarily long without owning a
SQLite writer slot, and only the final validation/insert/frontier CAS runs in a
single ``BEGIN IMMEDIATE`` transaction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from .db_bootstrap import (
    configure_connection,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .sqlite_writer import WriterCoordinator, get_writer_coordinator


_SOURCE_COLUMNS = (
    "store_id, session_id, source, conversation_id, role, content, "
    "tool_call_id, tool_calls, tool_name, timestamp, token_estimate, pinned"
)
_MAX_SOURCE_IDENTITIES = 4096


class PublicationCaptureError(RuntimeError):
    """Base class for a capture that is already stale or invalid."""

    def __init__(self, message: str, *, frontier_store_id: int = 0) -> None:
        super().__init__(message)
        self.frontier_store_id = int(frontier_store_id)


class StalePublicationCapture(PublicationCaptureError):
    """The lifecycle binding/frontier changed before summarization started."""


class PublicationSourceMismatch(PublicationCaptureError):
    """The requested source rows are absent or outside the captured binding."""


@dataclass(frozen=True)
class SourceIdentity:
    """Immutable content identity for one bounded message source row."""

    store_id: int
    fingerprint: str


@dataclass(frozen=True)
class PublicationIntent:
    """Immutable captured publication inputs plus an optional prepared summary."""

    conversation_id: str
    session_id: str
    expected_frontier_store_id: int
    new_frontier_store_id: int
    source_store_ids: tuple[int, ...]
    validation_store_ids: tuple[int, ...]
    source_identities: tuple[SourceIdentity, ...]
    coverage_key: str
    summary: str = ""
    token_count: int = 0
    source_token_count: int = 0
    created_at: float = 0.0
    earliest_at: float | None = None
    latest_at: float | None = None
    expand_hint: str = ""

    def with_summary(
        self,
        *,
        summary: str,
        token_count: int,
        source_token_count: int,
        created_at: float = 0.0,
        earliest_at: float | None = None,
        latest_at: float | None = None,
        expand_hint: str = "",
    ) -> "PublicationIntent":
        """Return a prepared copy without mutating the pre-LLM capture."""

        if not summary:
            raise ValueError("publication summary must not be empty")
        return replace(
            self,
            summary=summary,
            token_count=max(0, int(token_count)),
            source_token_count=max(0, int(source_token_count)),
            created_at=float(created_at or time.time()),
            earliest_at=earliest_at,
            latest_at=latest_at,
            expand_hint=expand_hint or "",
        )


@dataclass(frozen=True)
class PublicationResult:
    """Outcome returned only after the publication transaction has completed."""

    status: str
    node_id: int | None
    frontier_store_id: int
    reason: str = ""

    @property
    def canonical(self) -> bool:
        return self.status in {"published", "already_published"}


AfterInsertHook = Callable[[sqlite3.Connection, PublicationIntent, int], None]
BeforeInsertHook = Callable[[sqlite3.Connection, PublicationIntent], None]


def _row_fingerprint(row: Iterable[object]) -> str:
    encoded = json.dumps(
        list(row),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coverage_key(
    *,
    conversation_id: str,
    session_id: str,
    expected_frontier_store_id: int,
    new_frontier_store_id: int,
    source_store_ids: tuple[int, ...],
    source_identities: tuple[SourceIdentity, ...],
) -> str:
    payload = [
        conversation_id,
        session_id,
        int(expected_frontier_store_id),
        int(new_frontier_store_id),
        list(source_store_ids),
        [[identity.store_id, identity.fingerprint] for identity in source_identities],
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class AtomicPublicationStore:
    """Dedicated lazy connection for one atomic leaf-publication boundary."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        writer_coordinator: WriterCoordinator | None = None,
        after_insert: AfterInsertHook | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._writer_coordinator = writer_coordinator or get_writer_coordinator(
            self.db_path
        )
        self._writer_owner_token = self._writer_coordinator.bind_owner()
        self._init_lock = threading.Lock()
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._after_insert = after_insert
        self._published_count = 0
        self._already_published_count = 0
        self._stale_count = 0
        self._source_mismatch_count = 0

    @property
    def writer_coordinator(self) -> WriterCoordinator:
        return self._writer_coordinator

    def _ensure_connection(self) -> sqlite3.Connection:
        # Connection initialization has its own mutex. Never hold the local DB
        # lock while acquiring the shared writer permit: every configured
        # write path uses the opposite (permit -> local lock) ordering.
        with self._init_lock:
            if self._conn is not None:
                return self._conn
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,
            )
            try:
                refuse_schema_version_too_new(conn)
                configure_connection(
                    conn,
                    coordinator=self._writer_coordinator,
                    local_lock=self._lock,
                )
                run_versioned_migrations(
                    conn,
                    coordinator=self._writer_coordinator,
                    local_lock=self._lock,
                )
                conn.row_factory = sqlite3.Row
            except BaseException:
                conn.close()
                raise
            self._conn = conn
            return conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the dedicated connection for read-only diagnostics/tracing."""

        return self._ensure_connection()

    @staticmethod
    def _validate_source_ids(
        source_store_ids: Iterable[int],
        *,
        allow_empty: bool = False,
    ) -> tuple[int, ...]:
        source_ids = tuple(int(store_id) for store_id in source_store_ids)
        if not source_ids and not allow_empty:
            raise ValueError("publication needs at least one source row")
        if len(source_ids) > _MAX_SOURCE_IDENTITIES:
            raise ValueError(
                f"publication source identity limit is {_MAX_SOURCE_IDENTITIES}"
            )
        if any(store_id <= 0 for store_id in source_ids):
            raise ValueError("publication source ids must be positive")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("publication source ids must be unique")
        return source_ids

    def _read_source_rows(
        self,
        conn: sqlite3.Connection,
        source_ids: tuple[int, ...],
    ) -> list[sqlite3.Row]:
        placeholders = ",".join("?" for _ in source_ids)
        rows = conn.execute(
            f"SELECT {_SOURCE_COLUMNS} FROM messages "
            f"WHERE store_id IN ({placeholders}) ORDER BY store_id",
            source_ids,
        ).fetchall()
        by_id = {int(row["store_id"]): row for row in rows}
        return [by_id[store_id] for store_id in source_ids if store_id in by_id]

    def capture_leaf_intent(
        self,
        *,
        conversation_id: str,
        session_id: str,
        expected_frontier_store_id: int,
        new_frontier_store_id: int,
        source_store_ids: Iterable[int],
        validation_store_ids: Iterable[int] | None = None,
    ) -> PublicationIntent:
        """Capture exact lifecycle and source identity before any LLM work."""

        conversation_id = str(conversation_id or "")
        session_id = str(session_id or "")
        expected = int(expected_frontier_store_id or 0)
        new_frontier = int(new_frontier_store_id or 0)
        source_ids = self._validate_source_ids(source_store_ids, allow_empty=True)
        validation_ids = self._validate_source_ids(
            source_ids if validation_store_ids is None else validation_store_ids
        )
        if not set(source_ids).issubset(validation_ids):
            raise ValueError("publication source ids must be validated")
        if not conversation_id or not session_id:
            raise ValueError("publication requires conversation and session ids")
        if new_frontier <= expected:
            raise ValueError("publication frontier must advance monotonically")

        conn = self._ensure_connection()
        with self._lock:
            state = conn.execute(
                "SELECT current_session_id, current_frontier_store_id "
                "FROM lcm_lifecycle_state WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            current_frontier = int(state["current_frontier_store_id"] or 0) if state else 0
            if (
                state is None
                or str(state["current_session_id"] or "") != session_id
                or current_frontier != expected
            ):
                raise StalePublicationCapture(
                    "lifecycle binding/frontier changed before capture",
                    frontier_store_id=current_frontier,
                )
            rows = self._read_source_rows(conn, validation_ids)

        if len(rows) != len(validation_ids):
            raise PublicationSourceMismatch(
                "one or more publication source rows are missing",
                frontier_store_id=current_frontier,
            )
        for row in rows:
            if (
                # Pre-v5/explicit helper writes can legitimately have a blank
                # conversation id. Session identity plus the exact captured
                # row fingerprint keeps those legacy sources bounded without
                # claiming they belong to another named conversation.
                str(row["conversation_id"] or "")
                not in {"", conversation_id}
            ):
                raise PublicationSourceMismatch(
                    "publication source row is outside the captured binding",
                    frontier_store_id=current_frontier,
                )

        identities = tuple(
            SourceIdentity(
                store_id=int(row["store_id"]),
                fingerprint=_row_fingerprint(tuple(row)),
            )
            for row in rows
        )
        return PublicationIntent(
            conversation_id=conversation_id,
            session_id=session_id,
            expected_frontier_store_id=expected,
            new_frontier_store_id=new_frontier,
            source_store_ids=source_ids,
            validation_store_ids=validation_ids,
            source_identities=identities,
            coverage_key=_coverage_key(
                conversation_id=conversation_id,
                session_id=session_id,
                expected_frontier_store_id=expected,
                new_frontier_store_id=new_frontier,
                source_store_ids=source_ids,
                source_identities=identities,
            ),
        )

    @staticmethod
    def _state_result(
        status: str,
        *,
        frontier_store_id: int,
        reason: str,
    ) -> PublicationResult:
        return PublicationResult(
            status=status,
            node_id=None,
            frontier_store_id=int(frontier_store_id),
            reason=reason,
        )

    def publish_leaf(
        self,
        intent: PublicationIntent,
        *,
        before_insert: BeforeInsertHook | None = None,
        after_publish: AfterInsertHook | None = None,
        failure_hook: str = "",
    ) -> PublicationResult:
        """Validate and publish one canonical D0 node in one transaction."""

        if not intent.summary:
            raise ValueError("publication intent has no prepared summary")
        if intent.new_frontier_store_id <= intent.expected_frontier_store_id:
            raise ValueError("publication frontier must advance monotonically")

        conn = self._ensure_connection()
        result: PublicationResult
        with self._writer_coordinator.transaction(
            conn,
            local_lock=self._lock,
            begin_immediate=True,
        ):
            existing = conn.execute(
                "SELECT node_id FROM summary_nodes WHERE coverage_key = ?",
                (intent.coverage_key,),
            ).fetchone()
            state = conn.execute(
                "SELECT current_session_id, current_frontier_store_id "
                "FROM lcm_lifecycle_state WHERE conversation_id = ?",
                (intent.conversation_id,),
            ).fetchone()
            current_frontier = int(state["current_frontier_store_id"] or 0) if state else 0

            # Retried/lost responses must remain recognizable after the first
            # committed publish advanced the frontier (and after best-effort GC
            # may have rewritten source payloads).
            if existing is not None:
                result = PublicationResult(
                    status="already_published",
                    node_id=int(existing["node_id"]),
                    frontier_store_id=current_frontier,
                    reason="coverage key already committed",
                )
            elif (
                state is None
                or str(state["current_session_id"] or "") != intent.session_id
                or current_frontier != intent.expected_frontier_store_id
            ):
                result = self._state_result(
                    "stale",
                    frontier_store_id=current_frontier,
                    reason="lifecycle binding/frontier no longer matches capture",
                )
            else:
                rows = self._read_source_rows(conn, intent.validation_store_ids)
                identities = tuple(
                    SourceIdentity(
                        store_id=int(row["store_id"]),
                        fingerprint=_row_fingerprint(tuple(row)),
                    )
                    for row in rows
                )
                if identities != intent.source_identities:
                    result = self._state_result(
                        "source_mismatch",
                        frontier_store_id=current_frontier,
                        reason="source rows changed after capture",
                    )
                else:
                    if before_insert is not None:
                        before_insert(conn, intent)
                    cursor = conn.execute(
                        """
                        INSERT INTO summary_nodes(
                            session_id, depth, summary, token_count,
                            source_token_count, source_ids, source_type,
                            created_at, earliest_at, latest_at, expand_hint,
                            coverage_key
                        ) VALUES (?, 0, ?, ?, ?, ?, 'messages', ?, ?, ?, ?, ?)
                        """,
                        (
                            intent.session_id,
                            intent.summary,
                            intent.token_count,
                            intent.source_token_count,
                            json.dumps(intent.source_store_ids),
                            intent.created_at or time.time(),
                            intent.earliest_at,
                            intent.latest_at,
                            intent.expand_hint,
                            intent.coverage_key,
                        ),
                    )
                    node_id = int(cursor.lastrowid)
                    if failure_hook == "after_canonical_insert":
                        raise RuntimeError("injected async promotion failure")
                    if self._after_insert is not None:
                        self._after_insert(conn, intent, node_id)
                    updated = conn.execute(
                        """
                        UPDATE lcm_lifecycle_state
                        SET current_frontier_store_id = ?, updated_at = ?
                        WHERE conversation_id = ?
                          AND current_session_id = ?
                          AND current_frontier_store_id = ?
                          AND ? > current_frontier_store_id
                        """,
                        (
                            intent.new_frontier_store_id,
                            time.time(),
                            intent.conversation_id,
                            intent.session_id,
                            intent.expected_frontier_store_id,
                            intent.new_frontier_store_id,
                        ),
                    )
                    if updated.rowcount != 1:
                        # BEGIN IMMEDIATE makes this unreachable for cooperating
                        # writers, but never commit a node if the CAS contract is
                        # broken by an external writer/trigger.
                        raise RuntimeError("publication frontier CAS lost after insert")
                    if after_publish is not None:
                        after_publish(conn, intent, node_id)
                    result = PublicationResult(
                        status="published",
                        node_id=node_id,
                        frontier_store_id=intent.new_frontier_store_id,
                    )

        if result.status == "published":
            self._published_count += 1
        elif result.status == "already_published":
            self._already_published_count += 1
        elif result.status == "stale":
            self._stale_count += 1
        elif result.status == "source_mismatch":
            self._source_mismatch_count += 1
        return result

    def metrics_snapshot(self) -> dict[str, int]:
        return {
            "published": self._published_count,
            "already_published": self._already_published_count,
            "stale": self._stale_count,
            "source_mismatch": self._source_mismatch_count,
        }

    def close(self) -> None:
        owner_token = self._writer_owner_token
        if owner_token is None:
            return
        self._writer_coordinator.close_owner(
            owner_token,
            self._conn,
            local_lock=self._lock,
        )
        self._writer_owner_token = None
        self._conn = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass
