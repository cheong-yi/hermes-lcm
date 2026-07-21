"""Durable non-canonical leaf preparation and bounded scheduling.

Nothing in this module writes ``summary_nodes``.  Preparation persists exact
publication inputs in v7 tables; canonical visibility remains owned by
``AtomicPublicationStore`` and its one ``BEGIN IMMEDIATE`` transaction.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .db_bootstrap import (
    configure_connection,
    refuse_schema_version_too_new,
    run_versioned_migrations,
)
from .publication import PublicationIntent, SourceIdentity
from .sqlite_writer import WriterCoordinator, canonical_db_path, get_writer_coordinator


PREPARED_STATES = frozenset(
    {"pending", "preparing", "ready", "promoted", "rejected", "failed", "superseded"}
)
TERMINAL_STATES = frozenset({"promoted", "rejected", "failed", "superseded"})
ACTIVE_STATES = frozenset({"pending", "preparing", "ready"})

_PROCESS_OWNER_ID = f"{os.getpid()}:{uuid.uuid4().hex}"
_RECOVERY_LOCK = threading.Lock()
_RECOVERED_PATHS: set[str] = set()


class PreparedPromotionRejected(RuntimeError):
    """Abort a promotion transaction with a stable rejection reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _BackgroundPreparationCancelled(RuntimeError):
    """Internal cooperative cancellation that never becomes a failed batch."""


@dataclass(frozen=True)
class PreparedBatch:
    batch_id: str
    conversation_id: str
    session_id: str
    state: str
    frontier_start_store_id: int
    frontier_end_store_id: int
    source_ids: tuple[int, ...]
    validation_source_ids: tuple[int, ...]
    source_identity_hashes: tuple[str, ...]
    policy_fingerprint: str
    summary_route_fingerprint: str
    coverage_key: str
    expected_leaf_count: int
    prepared_leaf_count: int
    attempt_count: int
    next_retry_at: float | None
    last_error: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class PreparedPromotionResult:
    promoted: bool
    reason: str
    node_id: int | None = None
    frontier_store_id: int = 0


def _json_tuple(value: str | None, cast: Callable[[Any], Any]) -> tuple[Any, ...]:
    try:
        decoded = json.loads(value or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = []
    if not isinstance(decoded, list):
        return ()
    return tuple(cast(item) for item in decoded)


def _batch_from_row(row: sqlite3.Row) -> PreparedBatch:
    return PreparedBatch(
        batch_id=str(row["batch_id"]),
        conversation_id=str(row["conversation_id"]),
        session_id=str(row["session_id"]),
        state=str(row["state"]),
        frontier_start_store_id=int(row["frontier_start_store_id"] or 0),
        frontier_end_store_id=int(row["frontier_end_store_id"] or 0),
        source_ids=_json_tuple(row["source_ids"], int),
        validation_source_ids=_json_tuple(row["validation_source_ids"], int),
        source_identity_hashes=_json_tuple(row["source_identity_hashes"], str),
        policy_fingerprint=str(row["policy_fingerprint"] or ""),
        summary_route_fingerprint=str(row["summary_route_fingerprint"] or ""),
        coverage_key=str(row["coverage_key"] or ""),
        expected_leaf_count=int(row["expected_leaf_count"] or 0),
        prepared_leaf_count=int(row["prepared_leaf_count"] or 0),
        attempt_count=int(row["attempt_count"] or 0),
        next_retry_at=(
            float(row["next_retry_at"])
            if row["next_retry_at"] is not None
            else None
        ),
        last_error=str(row["last_error"] or ""),
        created_at=float(row["created_at"] or 0),
        updated_at=float(row["updated_at"] or 0),
    )


class PreparedCompactionStore:
    """One engine-bound helper for durable prepared batches."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        writer_coordinator: WriterCoordinator | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.canonical_path = canonical_db_path(self.db_path)
        self._writer_coordinator = writer_coordinator or get_writer_coordinator(self.db_path)
        self._writer_owner_token = self._writer_coordinator.bind_owner()
        self._owner_id = f"{_PROCESS_OWNER_ID}:{uuid.uuid4().hex}"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            refuse_schema_version_too_new(self._conn)
            configure_connection(
                self._conn,
                coordinator=self._writer_coordinator,
                local_lock=self._lock,
            )
            run_versioned_migrations(
                self._conn,
                coordinator=self._writer_coordinator,
                local_lock=self._lock,
            )
            self._conn.row_factory = sqlite3.Row
            self.recover_orphans_once()
        except BaseException:
            self._conn.close()
            self._writer_coordinator.close_owner(self._writer_owner_token, None)
            self._writer_owner_token = None
            raise

    @property
    def writer_coordinator(self) -> WriterCoordinator:
        return self._writer_coordinator

    @property
    def owner_id(self) -> str:
        return self._owner_id

    def recover_orphans_once(self) -> int:
        """Reclaim crashed-process leases once for this canonical DB path."""

        with _RECOVERY_LOCK:
            if self.canonical_path in _RECOVERED_PATHS:
                return 0
            _RECOVERED_PATHS.add(self.canonical_path)
        now = time.time()
        try:
            with self._writer_coordinator.transaction(
                self._conn,
                local_lock=self._lock,
                begin_immediate=True,
            ):
                cursor = self._conn.execute(
                    """
                    UPDATE lcm_prepared_compactions
                    SET state = 'pending', owner_id = NULL, attempt_token = NULL,
                        lease_expires_at = NULL, heartbeat_at = NULL,
                        next_retry_at = ?, last_error = ?, updated_at = ?
                    WHERE state = 'preparing'
                      AND (
                        owner_id IS NULL
                        OR owner_id NOT LIKE ?
                        OR lease_expires_at IS NULL
                        OR lease_expires_at <= ?
                      )
                    """,
                    (
                        now,
                        "orphaned preparation recovered without publication",
                        now,
                        f"{_PROCESS_OWNER_ID}:%",
                        now,
                    ),
                )
                return int(cursor.rowcount or 0)
        except BaseException:
            with _RECOVERY_LOCK:
                _RECOVERED_PATHS.discard(self.canonical_path)
            raise

    def get_batch(self, batch_id: str) -> PreparedBatch | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                (str(batch_id),),
            ).fetchone()
        return _batch_from_row(row) if row is not None else None

    def ready_batch(self, conversation_id: str, session_id: str) -> PreparedBatch | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM lcm_prepared_compactions
                WHERE conversation_id = ? AND session_id = ? AND state = 'ready'
                ORDER BY created_at DESC LIMIT 1
                """,
                (conversation_id, session_id),
            ).fetchone()
        return _batch_from_row(row) if row is not None else None

    def claim(
        self,
        intent: PublicationIntent,
        *,
        policy_fingerprint: str,
        summary_route_fingerprint: str,
        fresh_tail_count: int,
        leaf_chunk_tokens: int,
        max_batches: int,
        lease_seconds: float,
    ) -> tuple[PreparedBatch | None, str | None]:
        """Create/claim one generation before any LLM or tokenization work."""

        now = time.time()
        attempt_token = uuid.uuid4().hex
        batch_id = uuid.uuid4().hex
        attempt_number = 1
        max_batches = max(1, int(max_batches or 1))
        lease_seconds = max(30.0, float(lease_seconds))
        with self._writer_coordinator.transaction(
            self._conn,
            local_lock=self._lock,
            begin_immediate=True,
        ):
            existing = self._conn.execute(
                """
                SELECT * FROM lcm_prepared_compactions
                WHERE conversation_id = ? AND session_id = ? AND coverage_key = ?
                  AND state IN ('pending', 'preparing', 'ready', 'failed')
                ORDER BY created_at DESC LIMIT 1
                """,
                (intent.conversation_id, intent.session_id, intent.coverage_key),
            ).fetchone()
            if existing is not None:
                batch = _batch_from_row(existing)
                if batch.state == "ready":
                    return batch, None
                if batch.state == "preparing":
                    lease_expires_at = existing["lease_expires_at"]
                    if (
                        lease_expires_at is not None
                        and float(lease_expires_at) > now
                    ):
                        return batch, None
                    attempt_number = max(1, batch.attempt_count + 1)
                    cursor = self._conn.execute(
                        """
                        UPDATE lcm_prepared_compactions
                        SET attempt_count = ?, owner_id = ?, attempt_token = ?,
                            lease_expires_at = ?, heartbeat_at = ?,
                            next_retry_at = NULL, updated_at = ?
                        WHERE batch_id = ? AND state = 'preparing'
                          AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
                        """,
                        (
                            attempt_number,
                            self._owner_id,
                            attempt_token,
                            now + lease_seconds,
                            now,
                            now,
                            batch.batch_id,
                            now,
                        ),
                    )
                    if cursor.rowcount != 1:
                        return self.get_batch(batch.batch_id), None
                    claimed = self._conn.execute(
                        "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                        (batch.batch_id,),
                    ).fetchone()
                    return _batch_from_row(claimed), attempt_token
                if batch.state == "pending":
                    attempt_number = max(1, batch.attempt_count + 1)
                    cursor = self._conn.execute(
                        """
                        UPDATE lcm_prepared_compactions
                        SET state = 'preparing', attempt_count = ?, owner_id = ?,
                            attempt_token = ?, lease_expires_at = ?, heartbeat_at = ?,
                            next_retry_at = NULL, updated_at = ?
                        WHERE batch_id = ? AND state = 'pending'
                        """,
                        (
                            attempt_number,
                            self._owner_id,
                            attempt_token,
                            now + lease_seconds,
                            now,
                            now,
                            batch.batch_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        return self.get_batch(batch.batch_id), None
                    claimed = self._conn.execute(
                        "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                        (batch.batch_id,),
                    ).fetchone()
                    return _batch_from_row(claimed), attempt_token
                if (batch.next_retry_at or 0) > now:
                    return batch, None
                # Failed generations are terminal.  Once backoff expires a new
                # generation carries the incremented attempt ordinal.
                attempt_number = batch.attempt_count + 1

            active_count = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_prepared_compactions
                    WHERE conversation_id = ?
                      AND state IN ('pending', 'preparing', 'ready')
                    """,
                    (intent.conversation_id,),
                ).fetchone()[0]
                or 0
            )
            if active_count >= max_batches:
                return None, None

            source_ids_json = json.dumps(intent.source_store_ids, separators=(",", ":"))
            validation_ids_json = json.dumps(intent.validation_store_ids, separators=(",", ":"))
            identity_hashes = json.dumps(
                [identity.fingerprint for identity in intent.source_identities],
                separators=(",", ":"),
            )
            ordered_lineage = json.dumps(
                [[identity.store_id, identity.fingerprint] for identity in intent.source_identities],
                separators=(",", ":"),
            )
            self._conn.execute(
                """
                INSERT INTO lcm_prepared_compactions(
                    batch_id, conversation_id, session_id, state,
                    frontier_start_store_id, frontier_end_store_id,
                    fresh_tail_count, leaf_chunk_tokens,
                    policy_fingerprint, summary_route_fingerprint, coverage_key,
                    source_ids, validation_source_ids, source_identity_hashes,
                    ordered_lineage, expected_leaf_count, prepared_leaf_count,
                    attempt_count, owner_id, attempt_token, lease_expires_at,
                    heartbeat_at, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          1, 0, 0, NULL, NULL, NULL, NULL, ?, ?)
                """,
                (
                    batch_id,
                    intent.conversation_id,
                    intent.session_id,
                    intent.expected_frontier_store_id,
                    intent.new_frontier_store_id,
                    int(fresh_tail_count),
                    int(leaf_chunk_tokens),
                    policy_fingerprint,
                    summary_route_fingerprint,
                    intent.coverage_key,
                    source_ids_json,
                    validation_ids_json,
                    identity_hashes,
                    ordered_lineage,
                    now,
                    now,
                ),
            )
            cursor = self._conn.execute(
                """
                UPDATE lcm_prepared_compactions
                SET state = 'preparing', attempt_count = ?, owner_id = ?,
                    attempt_token = ?, lease_expires_at = ?, heartbeat_at = ?,
                    updated_at = ?
                WHERE batch_id = ? AND state = 'pending'
                """,
                (
                    attempt_number,
                    self._owner_id,
                    attempt_token,
                    now + lease_seconds,
                    now,
                    now,
                    batch_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("prepared batch claim transition failed")
            row = self._conn.execute(
                "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            return _batch_from_row(row), attempt_token

    def heartbeat(self, batch_id: str, attempt_token: str, lease_seconds: float) -> bool:
        now = time.time()
        with self._writer_coordinator.transaction(
            self._conn,
            local_lock=self._lock,
            begin_immediate=True,
        ):
            cursor = self._conn.execute(
                """
                UPDATE lcm_prepared_compactions
                SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE batch_id = ? AND state = 'preparing'
                  AND owner_id = ? AND attempt_token = ?
                """,
                (
                    now,
                    now + max(30.0, float(lease_seconds)),
                    now,
                    batch_id,
                    self._owner_id,
                    attempt_token,
                ),
            )
            return cursor.rowcount == 1

    def mark_ready(
        self,
        batch_id: str,
        attempt_token: str,
        intent: PublicationIntent,
    ) -> PreparedBatch:
        now = time.time()
        if not intent.summary:
            raise ValueError("prepared summary payload must not be empty")
        with self._writer_coordinator.transaction(
            self._conn,
            local_lock=self._lock,
            begin_immediate=True,
        ):
            row = self._conn.execute(
                "SELECT state, owner_id, attempt_token FROM lcm_prepared_compactions WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("prepared batch disappeared before ready transition")
            if (
                str(row["state"]) != "preparing"
                or str(row["owner_id"] or "") != self._owner_id
                or str(row["attempt_token"] or "") != attempt_token
            ):
                raise RuntimeError("prepared batch lease lost before ready transition")
            self._conn.execute(
                """
                INSERT INTO lcm_prepared_summary_nodes(
                    pending_id, batch_id, conversation_id, session_id, depth,
                    summary, token_count, source_token_count, source_ids,
                    previous_pending_ids, created_at, earliest_at, latest_at,
                    expand_hint
                ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, '[]', ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    summary = excluded.summary,
                    token_count = excluded.token_count,
                    source_token_count = excluded.source_token_count,
                    source_ids = excluded.source_ids,
                    created_at = excluded.created_at,
                    earliest_at = excluded.earliest_at,
                    latest_at = excluded.latest_at,
                    expand_hint = excluded.expand_hint
                """,
                (
                    uuid.uuid4().hex,
                    batch_id,
                    intent.conversation_id,
                    intent.session_id,
                    intent.summary,
                    intent.token_count,
                    intent.source_token_count,
                    json.dumps(intent.source_store_ids, separators=(",", ":")),
                    intent.created_at or now,
                    intent.earliest_at,
                    intent.latest_at,
                    intent.expand_hint,
                ),
            )
            cursor = self._conn.execute(
                """
                UPDATE lcm_prepared_compactions
                SET state = 'ready', prepared_leaf_count = expected_leaf_count,
                    frontier_start_store_id = ?, frontier_end_store_id = ?,
                    coverage_key = ?, source_ids = ?, validation_source_ids = ?,
                    source_identity_hashes = ?, ordered_lineage = ?,
                    owner_id = NULL, attempt_token = NULL,
                    lease_expires_at = NULL, heartbeat_at = NULL,
                    last_error = NULL, next_retry_at = NULL, updated_at = ?
                WHERE batch_id = ? AND state = 'preparing'
                  AND owner_id = ? AND attempt_token = ?
                """,
                (
                    intent.expected_frontier_store_id,
                    intent.new_frontier_store_id,
                    intent.coverage_key,
                    json.dumps(intent.source_store_ids, separators=(",", ":")),
                    json.dumps(intent.validation_store_ids, separators=(",", ":")),
                    json.dumps(
                        [identity.fingerprint for identity in intent.source_identities],
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        [[identity.store_id, identity.fingerprint] for identity in intent.source_identities],
                        separators=(",", ":"),
                    ),
                    now,
                    batch_id,
                    self._owner_id,
                    attempt_token,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("prepared batch ready transition lost its lease")
            ready = self._conn.execute(
                "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            return _batch_from_row(ready)

    def mark_failed(
        self,
        batch_id: str,
        attempt_token: str,
        error: BaseException,
        *,
        base_backoff_seconds: float,
    ) -> PreparedBatch:
        now = time.time()
        message = str(error).strip()[:1000] or type(error).__name__
        with self._writer_coordinator.transaction(
            self._conn,
            local_lock=self._lock,
            begin_immediate=True,
        ):
            row = self._conn.execute(
                "SELECT attempt_count FROM lcm_prepared_compactions WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            attempts = int(row[0] or 1) if row else 1
            base = max(0.05, float(base_backoff_seconds or 0.05))
            backoff = min(3600.0, base * (2 ** min(5, max(0, attempts - 1))))
            cursor = self._conn.execute(
                """
                UPDATE lcm_prepared_compactions
                SET state = 'failed', last_error = ?, next_retry_at = ?,
                    owner_id = NULL, attempt_token = NULL,
                    lease_expires_at = NULL, heartbeat_at = NULL, updated_at = ?
                WHERE batch_id = ? AND state = 'preparing'
                  AND owner_id = ? AND attempt_token = ?
                """,
                (
                    message,
                    now + backoff,
                    now,
                    batch_id,
                    self._owner_id,
                    attempt_token,
                ),
            )
            if cursor.rowcount != 1:
                current = self._conn.execute(
                    "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()
                if current is None:
                    raise RuntimeError("prepared failure transition lost batch")
                return _batch_from_row(current)
            failed = self._conn.execute(
                "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            return _batch_from_row(failed)

    def reject(self, batch_id: str, reason: str, *, superseded: bool = False) -> PreparedBatch:
        state = "superseded" if superseded else "rejected"
        now = time.time()
        with self._writer_coordinator.transaction(
            self._conn,
            local_lock=self._lock,
            begin_immediate=True,
        ):
            row = self._conn.execute(
                "SELECT state FROM lcm_prepared_compactions WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if row is None:
                raise KeyError(batch_id)
            current = str(row["state"])
            if current == "promoted":
                raise ValueError("promoted batches are immutable")
            if current in TERMINAL_STATES:
                result = self._conn.execute(
                    "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()
                return _batch_from_row(result)
            if current not in {"rejected", "superseded"}:
                self._conn.execute(
                    """
                    UPDATE lcm_prepared_compactions
                    SET state = ?, rejected_reason = ?, owner_id = NULL,
                        attempt_token = NULL, lease_expires_at = NULL,
                        heartbeat_at = NULL, updated_at = ?
                    WHERE batch_id = ?
                    """,
                    (state, str(reason)[:1000], now, batch_id),
                )
            result = self._conn.execute(
                "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            return _batch_from_row(result)

    def load_intent(self, batch_id: str) -> PublicationIntent:
        with self._lock:
            batch = self._conn.execute(
                "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            node = self._conn.execute(
                "SELECT * FROM lcm_prepared_summary_nodes WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
        if batch is None or node is None:
            raise PreparedPromotionRejected("prepared_payload_missing")
        source_ids = _json_tuple(batch["source_ids"], int)
        validation_ids = _json_tuple(batch["validation_source_ids"], int)
        identity_hashes = _json_tuple(batch["source_identity_hashes"], str)
        if len(validation_ids) != len(identity_hashes):
            raise PreparedPromotionRejected("source_identity_mismatch")
        identities = tuple(
            SourceIdentity(store_id=store_id, fingerprint=fingerprint)
            for store_id, fingerprint in zip(validation_ids, identity_hashes)
        )
        return PublicationIntent(
            conversation_id=str(batch["conversation_id"]),
            session_id=str(batch["session_id"]),
            expected_frontier_store_id=int(batch["frontier_start_store_id"] or 0),
            new_frontier_store_id=int(batch["frontier_end_store_id"] or 0),
            source_store_ids=source_ids,
            validation_store_ids=validation_ids,
            source_identities=identities,
            coverage_key=str(batch["coverage_key"]),
            summary=str(node["summary"]),
            token_count=int(node["token_count"] or 0),
            source_token_count=int(node["source_token_count"] or 0),
            created_at=float(node["created_at"] or 0),
            earliest_at=(float(node["earliest_at"]) if node["earliest_at"] is not None else None),
            latest_at=(float(node["latest_at"]) if node["latest_at"] is not None else None),
            expand_hint=str(node["expand_hint"] or ""),
        )

    def promotion_callbacks(
        self,
        batch_id: str,
        *,
        live_policy_fingerprint: str,
        live_summary_route_fingerprint: str,
    ) -> tuple[Callable[[sqlite3.Connection, PublicationIntent], None], Callable[[sqlite3.Connection, PublicationIntent, int], None]]:
        def validate(conn: sqlite3.Connection, intent: PublicationIntent) -> None:
            row = conn.execute(
                "SELECT * FROM lcm_prepared_compactions WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if row is None or str(row["state"]) != "ready":
                raise PreparedPromotionRejected("batch_not_ready")
            if str(row["conversation_id"]) != intent.conversation_id or str(row["session_id"]) != intent.session_id:
                raise PreparedPromotionRejected("binding_mismatch")
            if str(row["policy_fingerprint"]) != live_policy_fingerprint:
                raise PreparedPromotionRejected("policy_fingerprint_mismatch")
            if str(row["summary_route_fingerprint"]) != live_summary_route_fingerprint:
                raise PreparedPromotionRejected("summary_route_fingerprint_mismatch")
            if int(row["prepared_leaf_count"] or 0) != int(row["expected_leaf_count"] or 0):
                raise PreparedPromotionRejected("prepared_payload_incomplete")

        def promoted(conn: sqlite3.Connection, intent: PublicationIntent, _node_id: int) -> None:
            now = time.time()
            cursor = conn.execute(
                """
                UPDATE lcm_prepared_compactions
                SET state = 'promoted', promoted_at = ?, updated_at = ?
                WHERE batch_id = ? AND state = 'ready'
                """,
                (now, now, batch_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("prepared batch promotion state CAS lost")
            conn.execute(
                """
                UPDATE lcm_prepared_compactions
                SET state = 'superseded', rejected_reason = 'newer generation promoted',
                    owner_id = NULL, attempt_token = NULL,
                    lease_expires_at = NULL, heartbeat_at = NULL, updated_at = ?
                WHERE conversation_id = ? AND batch_id <> ?
                  AND state IN ('pending', 'preparing', 'ready')
                  AND created_at <= (
                    SELECT created_at FROM lcm_prepared_compactions WHERE batch_id = ?
                  )
                """,
                (now, intent.conversation_id, batch_id, batch_id),
            )

        return validate, promoted

    @staticmethod
    def supersede_for_foreground_in_transaction(
        conn: sqlite3.Connection,
        intent: PublicationIntent,
        _node_id: int,
    ) -> None:
        now = time.time()
        conn.execute(
            """
            UPDATE lcm_prepared_compactions
            SET state = 'superseded', rejected_reason = 'foreground publication won',
                owner_id = NULL, attempt_token = NULL,
                lease_expires_at = NULL, heartbeat_at = NULL, updated_at = ?
            WHERE conversation_id = ?
              AND state IN ('pending', 'preparing', 'ready')
            """,
            (now, intent.conversation_id),
        )

    def status(
        self,
        conversation_id: str = "",
        *,
        enabled: bool = False,
        worker_enabled: bool = False,
        live_policy_fingerprint: str = "",
        live_summary_route_fingerprint: str = "",
    ) -> dict[str, Any]:
        where = " WHERE conversation_id = ?" if conversation_id else ""
        params = (conversation_id,) if conversation_id else ()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT state, COUNT(*) AS count FROM lcm_prepared_compactions{where} GROUP BY state",
                params,
            ).fetchall()
            pending_nodes = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_prepared_summary_nodes n
                    JOIN lcm_prepared_compactions b ON b.batch_id = n.batch_id
                    WHERE b.state IN ('pending', 'preparing', 'ready')
                    """ + (" AND b.conversation_id = ?" if conversation_id else ""),
                    params,
                ).fetchone()[0]
                or 0
            )
            oldest = self._conn.execute(
                """
                SELECT MIN(created_at) FROM lcm_prepared_compactions
                WHERE state IN ('pending', 'preparing', 'ready')
                """ + (" AND conversation_id = ?" if conversation_id else ""),
                params,
            ).fetchone()[0]
            last_rejection = self._conn.execute(
                """
                SELECT rejected_reason FROM lcm_prepared_compactions
                WHERE rejected_reason IS NOT NULL AND rejected_reason <> ''
                """ + (" AND conversation_id = ?" if conversation_id else "") +
                " ORDER BY updated_at DESC LIMIT 1",
                params,
            ).fetchone()
            last_error = self._conn.execute(
                """
                SELECT last_error FROM lcm_prepared_compactions
                WHERE last_error IS NOT NULL AND last_error <> ''
                """ + (" AND conversation_id = ?" if conversation_id else "") +
                " ORDER BY updated_at DESC LIMIT 1",
                params,
            ).fetchone()
            stale_policy = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_prepared_compactions
                    WHERE state = 'ready' AND policy_fingerprint <> ?
                    """ + (" AND conversation_id = ?" if conversation_id else ""),
                    (live_policy_fingerprint, *params),
                ).fetchone()[0]
                or 0
            )
            stale_route = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_prepared_compactions
                    WHERE state = 'ready' AND summary_route_fingerprint <> ?
                    """ + (" AND conversation_id = ?" if conversation_id else ""),
                    (live_summary_route_fingerprint, *params),
                ).fetchone()[0]
                or 0
            )
            expired_retry = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_prepared_compactions
                    WHERE state = 'failed' AND next_retry_at IS NOT NULL
                      AND next_retry_at <= ?
                    """ + (" AND conversation_id = ?" if conversation_id else ""),
                    (time.time(), *params),
                ).fetchone()[0]
                or 0
            )
            orphan_summaries = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM lcm_prepared_summary_nodes n
                    LEFT JOIN lcm_prepared_compactions b ON b.batch_id = n.batch_id
                    WHERE b.batch_id IS NULL
                    """ + (" AND n.conversation_id = ?" if conversation_id else ""),
                    params,
                ).fetchone()[0]
                or 0
            )
        counts = {str(row["state"]): int(row["count"] or 0) for row in rows}
        now = time.time()
        return {
            "enabled": bool(enabled),
            "worker_enabled": bool(worker_enabled),
            "pending_batches": counts.get("pending", 0),
            "preparing_batches": counts.get("preparing", 0),
            "prepared_batches": counts.get("ready", 0),
            "promoted_batches": counts.get("promoted", 0),
            "rejected_batches": counts.get("rejected", 0),
            "failed_batches": counts.get("failed", 0),
            "superseded_batches": counts.get("superseded", 0),
            "pending_summaries": pending_nodes,
            "oldest_pending_age_seconds": (max(0.0, now - float(oldest)) if oldest is not None else None),
            "last_rejected_reason": (
                str(last_rejection["rejected_reason"] or "") if last_rejection else ""
            ) or None,
            "last_error": (str(last_error["last_error"] or "") if last_error else "") or None,
            "stale_ready_policy_batches": stale_policy,
            "stale_ready_route_batches": stale_route,
            "expired_retry_batches": expired_retry,
            "orphan_pending_summaries": orphan_summaries,
        }

    def release_owned_preparations(self) -> int:
        now = time.time()
        with self._writer_coordinator.transaction(
            self._conn,
            local_lock=self._lock,
            begin_immediate=True,
        ):
            cursor = self._conn.execute(
                """
                UPDATE lcm_prepared_compactions
                SET state = 'pending', owner_id = NULL, attempt_token = NULL,
                    lease_expires_at = NULL, heartbeat_at = NULL,
                    next_retry_at = ?, last_error = ?, updated_at = ?
                WHERE state = 'preparing' AND owner_id = ?
                """,
                (now, "owner shut down before preparation completed", now, self._owner_id),
            )
            return int(cursor.rowcount or 0)

    def close(self) -> None:
        token = self._writer_owner_token
        if token is None:
            return
        self.release_owned_preparations()
        self._writer_coordinator.close_owner(token, self._conn, local_lock=self._lock)
        self._writer_owner_token = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        try:
            self.close()
        except Exception:
            pass


class LeaseHeartbeat:
    """Bounded lease refresh that never spans summary work with a DB permit."""

    def __init__(
        self,
        store: PreparedCompactionStore,
        batch_id: str,
        attempt_token: str,
        lease_seconds: float,
    ) -> None:
        self._store_ref = weakref.ref(store)
        self._batch_id = batch_id
        self._attempt_token = attempt_token
        self._lease_seconds = lease_seconds
        self._interval = max(1.0, min(10.0, lease_seconds / 4.0))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="lcm-background-lease-heartbeat",
            daemon=True,
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            store = self._store_ref()
            if store is None:
                return
            try:
                if not store.heartbeat(
                    self._batch_id,
                    self._attempt_token,
                    self._lease_seconds,
                ):
                    return
            except Exception:
                return

    def __enter__(self) -> "LeaseHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self._stop.set()
        self._thread.join(timeout=min(1.0, self._interval))


@dataclass
class _ScheduledJob:
    engine_ref: weakref.ReferenceType[Any]
    messages: list[dict[str, Any]]
    conversation_id: str
    engine_id: int
    cancel_event: threading.Event


class BackgroundCompactionScheduler:
    """One canonical-path scheduler with two summarizers total."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        self._condition = threading.Condition(threading.RLock())
        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="lcm-background",
        )
        self._active_conversations: set[str] = set()
        self._active_jobs: dict[str, _ScheduledJob] = {}
        self._queued: OrderedDict[str, _ScheduledJob] = OrderedDict()
        self._futures: set[Future[Any]] = set()
        self._maximum_active_workers = 0
        self._submitted = 0
        self._coalesced = 0
        self._completed = 0
        self._failed = 0
        self._cancelled = 0

    def submit(self, engine: Any, messages: list[dict[str, Any]]) -> bool:
        conversation_id = str(getattr(engine, "_conversation_id", "") or "")
        if not conversation_id or not messages:
            return False
        job = _ScheduledJob(
            engine_ref=weakref.ref(engine),
            messages=[dict(message) for message in messages],
            conversation_id=conversation_id,
            engine_id=id(engine),
            cancel_event=threading.Event(),
        )
        with self._condition:
            self._submitted += 1
            if conversation_id in self._active_conversations or conversation_id in self._queued:
                self._queued[conversation_id] = job
                self._queued.move_to_end(conversation_id)
                self._coalesced += 1
            else:
                self._queued[conversation_id] = job
            self._drain_locked()
            return True

    def _drain_locked(self) -> None:
        while len(self._active_conversations) < 2 and self._queued:
            conversation_id = next(
                (
                    queued_id
                    for queued_id in self._queued
                    if queued_id not in self._active_conversations
                ),
                None,
            )
            if conversation_id is None:
                break
            job = self._queued.pop(conversation_id)
            engine = job.engine_ref()
            if engine is None:
                continue
            self._active_conversations.add(conversation_id)
            self._active_jobs[conversation_id] = job
            self._maximum_active_workers = max(
                self._maximum_active_workers,
                len(self._active_conversations),
            )
            future = self._executor.submit(self._run_job, job)
            self._futures.add(future)
            future.add_done_callback(
                lambda completed, cid=conversation_id: self._job_done(cid, completed)
            )

    @staticmethod
    def _run_job(job: _ScheduledJob) -> None:
        engine = job.engine_ref()
        if engine is None:
            return
        engine.prepare_background_compaction_once(
            job.messages,
            _scheduled=True,
            _cancel_event=job.cancel_event,
        )

    def _job_done(self, conversation_id: str, future: Future[Any]) -> None:
        with self._condition:
            self._futures.discard(future)
            self._active_conversations.discard(conversation_id)
            self._active_jobs.pop(conversation_id, None)
            self._completed += 1
            try:
                future.result()
            except BaseException:
                self._failed += 1
            self._drain_locked()
            self._condition.notify_all()

    def cancel_engine(self, engine: Any) -> int:
        """Cancel queued/in-flight jobs without waiting for an LLM call."""

        engine_id = id(engine)
        cancelled = 0
        with self._condition:
            for conversation_id, job in list(self._queued.items()):
                if job.engine_id != engine_id:
                    continue
                job.cancel_event.set()
                self._queued.pop(conversation_id, None)
                cancelled += 1
            for job in self._active_jobs.values():
                if job.engine_id != engine_id or job.cancel_event.is_set():
                    continue
                job.cancel_event.set()
                cancelled += 1
            self._cancelled += cancelled
            self._condition.notify_all()
        return cancelled

    def wait_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while self._active_conversations or self._queued or self._futures:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def metrics_snapshot(self) -> dict[str, int | str]:
        with self._condition:
            return {
                "database_path": self.database_path,
                "active_workers": len(self._active_conversations),
                "maximum_active_workers": self._maximum_active_workers,
                "queued_conversations": len(self._queued),
                "submitted": self._submitted,
                "coalesced": self._coalesced,
                "completed": self._completed,
                "failed": self._failed,
                "cancelled": self._cancelled,
            }


_SCHEDULER_LOCK = threading.Lock()
_SCHEDULERS: weakref.WeakValueDictionary[str, BackgroundCompactionScheduler] = (
    weakref.WeakValueDictionary()
)


def get_background_compaction_scheduler(path: str | Path) -> BackgroundCompactionScheduler:
    key = canonical_db_path(path)
    with _SCHEDULER_LOCK:
        scheduler = _SCHEDULERS.get(key)
        if scheduler is None:
            scheduler = BackgroundCompactionScheduler(key)
            _SCHEDULERS[key] = scheduler
        return scheduler


class BackgroundCompactionMixin:
    """Engine-facing prepare/promote/status API for leaf-only background work."""

    def _async_policy_fingerprint(self) -> str:
        config = self._config
        payload = {
            "protocol": "async_compaction_v1",
            "fresh_tail_count": int(config.fresh_tail_count),
            "leaf_chunk_tokens": int(config.leaf_chunk_tokens),
            "configured_context_threshold": float(config.context_threshold),
            "effective_context_threshold": float(getattr(self, "context_threshold", config.context_threshold)),
            "context_threshold_source": str(getattr(self, "_context_threshold_source", "") or ""),
            "dynamic_leaf_chunk_enabled": bool(config.dynamic_leaf_chunk_enabled),
            "dynamic_leaf_chunk_max": int(config.dynamic_leaf_chunk_max),
            "ignore_message_patterns": list(config.ignore_message_patterns or []),
            "ignore_message_patterns_source": str(config.ignore_message_patterns_source or ""),
            "sensitive_patterns_enabled": bool(config.sensitive_patterns_enabled),
            "sensitive_patterns": list(config.sensitive_patterns or []),
            "sensitive_patterns_source": str(config.sensitive_patterns_source or ""),
            "large_output_externalization_enabled": bool(config.large_output_externalization_enabled),
            "large_output_externalization_threshold_chars": int(config.large_output_externalization_threshold_chars),
            "large_output_transcript_gc_enabled": bool(config.large_output_transcript_gc_enabled),
            "custom_instructions": str(config.custom_instructions or ""),
            "l2_budget_ratio": float(config.l2_budget_ratio),
            "l3_truncate_tokens": int(config.l3_truncate_tokens),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _async_summary_route_fingerprint(self) -> str:
        config = self._config
        payload = {
            "protocol": "async_summary_route_v1",
            "summary_model": str(config.summary_model or ""),
            "summary_fallback_models": list(config.summary_fallback_models or []),
            "provider": str(getattr(self, "provider", "") or ""),
            "model": str(getattr(self, "model", "") or ""),
            "api_mode": str(getattr(self, "api_mode", "") or ""),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _background_lease_seconds(self) -> float:
        timeout_seconds = max(0.0, float(self._config.summary_timeout_ms) / 1000.0)
        return max(30.0, timeout_seconds * 2.0 + 30.0)

    @staticmethod
    def _tool_call_ids(message: dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or call.get("tool_call_id") or "").strip()
            if call_id:
                ids.add(call_id)
        return ids

    def _background_leaf_candidate(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], PublicationIntent] | None:
        working_messages = self._ingest_messages(messages)
        leading = self._leading_anchor_count(working_messages)
        fresh_tail_start = max(leading, len(working_messages) - int(self._config.fresh_tail_count))
        while leading < fresh_tail_start and self._is_replayed_context_scaffold_message(working_messages[leading]):
            leading += 1
        candidate = list(working_messages[leading:fresh_tail_start])
        if not candidate:
            return None

        previous_map = self._current_compress_store_ids_by_message_id
        self._current_compress_store_ids_by_message_id = self._get_publication_store_id_map(candidate)
        try:
            filtered: list[dict[str, Any]] = []
            drop_dependent = False
            for message in candidate:
                role = str(message.get("role") or "")
                ignored = (
                    self._matches_ignore_message_patterns(message)
                    or self._mapped_stored_row_matches_ignore_message_patterns(message)
                )
                if ignored:
                    drop_dependent = True
                    continue
                if drop_dependent and role in {"assistant", "tool"}:
                    continue
                if role in {"user", "system"}:
                    drop_dependent = False
                filtered.append(message)
            candidate = filtered
        finally:
            self._current_compress_store_ids_by_message_id = previous_map
        if not candidate:
            return None

        # Never prepare half of an assistant tool-call/result group across the
        # fresh-tail boundary.  The whole suffix remains foreground-fresh.
        tail = working_messages[fresh_tail_start:]
        first_tail_tool_id = ""
        if tail and str(tail[0].get("role") or "") == "tool":
            first_tail_tool_id = str(tail[0].get("tool_call_id") or "").strip()
        if first_tail_tool_id:
            for index in range(len(candidate) - 1, -1, -1):
                if first_tail_tool_id in self._tool_call_ids(candidate[index]):
                    candidate = candidate[:index]
                    break
        if not candidate:
            return None

        raw_tokens = self._count_messages_tokens_for_background(candidate)
        working_threshold = (
            self._working_leaf_chunk_tokens(raw_tokens)
            if self._config.dynamic_leaf_chunk_enabled
            else int(self._config.leaf_chunk_tokens)
        )
        if raw_tokens < working_threshold:
            return None
        if self._config.dynamic_leaf_chunk_enabled:
            candidate = self._select_oldest_leaf_chunk(candidate, working_threshold)
        if not candidate:
            return None

        source_map = self._get_publication_store_id_map(candidate)
        if len(source_map) != len(candidate) or len(set(source_map.values())) != len(candidate):
            return None
        source_ids = tuple(source_map[id(message)] for message in candidate)
        lifecycle = self._lifecycle.get_by_conversation(self._conversation_id)
        expected_frontier = int(self._last_compacted_store_id or 0)
        if lifecycle is not None and lifecycle.current_session_id == self._session_id:
            expected_frontier = max(expected_frontier, int(lifecycle.current_frontier_store_id or 0))
        new_frontier = max(source_ids)
        if new_frontier <= expected_frontier:
            return None
        intent = self._publication.capture_leaf_intent(
            conversation_id=self._conversation_id,
            session_id=self._session_id,
            expected_frontier_store_id=expected_frontier,
            new_frontier_store_id=new_frontier,
            source_store_ids=source_ids,
            validation_store_ids=source_ids,
        )
        return candidate, intent

    def _count_messages_tokens_for_background(self, messages: list[dict[str, Any]]) -> int:
        # Kept as a narrow method so tests can assert tokenization occurs after
        # the durable claim without reaching into the token module.
        from .tokens import count_messages_tokens

        return count_messages_tokens(messages)

    def prepare_background_compaction_once(
        self,
        messages: list[dict[str, Any]],
        *,
        leave_state: str = "",
        _scheduled: bool = False,
        _cancel_event: threading.Event | None = None,
    ) -> PreparedBatch | None:
        if not bool(self._config.async_background_compaction_enabled):
            return None
        if not self._session_id or not self._conversation_id or self._bypasses_lcm_context_management():
            return None
        lifecycle_lock = self._background_compaction_storage_lock
        with lifecycle_lock:
            if _cancel_event is not None and _cancel_event.is_set():
                return None
            candidate = self._background_leaf_candidate(messages)
            if candidate is None:
                return None
            selected_messages, capture = candidate
            prepared_store = self._prepared_compactions
            message_store = self._store
            batch, attempt_token = prepared_store.claim(
                capture,
                policy_fingerprint=self._async_policy_fingerprint(),
                summary_route_fingerprint=self._async_summary_route_fingerprint(),
                fresh_tail_count=self._config.fresh_tail_count,
                leaf_chunk_tokens=self._config.leaf_chunk_tokens,
                max_batches=self._config.async_background_compaction_max_batches,
                lease_seconds=self._background_lease_seconds(),
            )
        if batch is None or attempt_token is None:
            return batch
        if leave_state == "preparing":
            return batch

        final_capture = {"intent": capture}

        def recapture(attempt_messages: list[dict[str, Any]]) -> None:
            with lifecycle_lock:
                if _cancel_event is not None and _cancel_event.is_set():
                    raise _BackgroundPreparationCancelled()
                source_map = self._get_publication_store_id_map(attempt_messages)
                if len(source_map) != len(attempt_messages):
                    raise RuntimeError("background source mapping changed during preparation")
                source_ids = tuple(source_map[id(message)] for message in attempt_messages)
                final_capture["intent"] = self._publication.capture_leaf_intent(
                    conversation_id=self._conversation_id,
                    session_id=self._session_id,
                    expected_frontier_store_id=capture.expected_frontier_store_id,
                    new_frontier_store_id=max(source_ids),
                    source_store_ids=source_ids,
                    validation_store_ids=source_ids,
                )

        heartbeat = LeaseHeartbeat(
            prepared_store,
            batch.batch_id,
            attempt_token,
            self._background_lease_seconds(),
        )
        try:
            with heartbeat:
                if _cancel_event is not None and _cancel_event.is_set():
                    raise _BackgroundPreparationCancelled()
                if self._config.extraction_enabled:
                    self._run_pre_compaction_extraction(selected_messages)
                if _cancel_event is not None and _cancel_event.is_set():
                    raise _BackgroundPreparationCancelled()
                self._thread_context.leaf_publication_capture_callback = recapture
                try:
                    compacted_chunk, _source_tokens, summary_text, _level, _attempts = (
                        self._summarize_leaf_chunk_with_rescue(
                            selected_messages,
                            focus_topic=self._derive_auto_focus_topic(messages),
                        )
                    )
                finally:
                    try:
                        del self._thread_context.leaf_publication_capture_callback
                    except AttributeError:
                        pass
                if _cancel_event is not None and _cancel_event.is_set():
                    raise _BackgroundPreparationCancelled()
                intent = final_capture["intent"]
                source_tokens = self._count_messages_tokens_for_background(compacted_chunk)
                from .tokens import count_tokens

                summary_tokens = count_tokens(summary_text)
                expand_hint = self._extract_expand_hint(summary_text)
            with lifecycle_lock:
                if _cancel_event is not None and _cancel_event.is_set():
                    return batch
                earliest_at, latest_at = message_store.get_time_bounds(intent.source_store_ids)
                prepared_intent = intent.with_summary(
                    summary=summary_text,
                    token_count=summary_tokens,
                    source_token_count=source_tokens,
                    created_at=time.time(),
                    earliest_at=earliest_at,
                    latest_at=latest_at,
                    expand_hint=expand_hint,
                )
                return prepared_store.mark_ready(
                    batch.batch_id,
                    attempt_token,
                    prepared_intent,
                )
        except _BackgroundPreparationCancelled:
            return batch
        except Exception as exc:
            with lifecycle_lock:
                if _cancel_event is not None and _cancel_event.is_set():
                    return batch
                return prepared_store.mark_failed(
                    batch.batch_id,
                    attempt_token,
                    exc,
                    base_backoff_seconds=self._config.async_background_compaction_retry_backoff_seconds,
                )

    def _fresh_tail_allows_prepared(self, batch: PreparedBatch) -> bool:
        rows = self._store.get_range(
            batch.session_id,
            start_id=1,
            conversation_id=batch.conversation_id,
        )
        non_system_ids = [
            int(row.get("store_id") or 0)
            for row in rows
            if str(row.get("role") or "") != "system"
        ]
        tail_count = max(0, int(self._config.fresh_tail_count))
        if len(non_system_ids) <= tail_count:
            return False
        safe_ids = non_system_ids[:-tail_count] if tail_count else non_system_ids
        return bool(safe_ids and batch.frontier_end_store_id <= max(safe_ids))

    def promote_prepared_compaction(
        self,
        batch_id: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> PreparedPromotionResult:
        if not bool(self._config.async_background_compaction_enabled):
            return PreparedPromotionResult(False, "disabled")
        batch = self._prepared_compactions.get_batch(batch_id)
        if batch is None:
            return PreparedPromotionResult(False, "batch_not_found")
        if batch.state != "ready":
            if batch.state == "superseded":
                return PreparedPromotionResult(False, "frontier_mismatch")
            return PreparedPromotionResult(False, "batch_not_ready")
        if messages:
            self._ingest_messages(messages)
        live_policy = self._async_policy_fingerprint()
        live_route = self._async_summary_route_fingerprint()
        if batch.policy_fingerprint != live_policy:
            self._prepared_compactions.reject(batch_id, "policy_fingerprint_mismatch")
            return PreparedPromotionResult(False, "policy_fingerprint_mismatch")
        if batch.summary_route_fingerprint != live_route:
            self._prepared_compactions.reject(batch_id, "summary_route_fingerprint_mismatch")
            return PreparedPromotionResult(False, "summary_route_fingerprint_mismatch")
        if not self._fresh_tail_allows_prepared(batch):
            self._prepared_compactions.reject(batch_id, "fresh_tail_boundary_mismatch")
            return PreparedPromotionResult(False, "fresh_tail_boundary_mismatch")
        try:
            intent = self._prepared_compactions.load_intent(batch_id)
            before_insert, after_publish = self._prepared_compactions.promotion_callbacks(
                batch_id,
                live_policy_fingerprint=live_policy,
                live_summary_route_fingerprint=live_route,
            )
            result = self._publication.publish_leaf(
                intent,
                before_insert=before_insert,
                after_publish=after_publish,
                failure_hook=str(getattr(self, "_async_compaction_publish_failure_hook", "") or ""),
            )
        except PreparedPromotionRejected as exc:
            self._prepared_compactions.reject(batch_id, exc.reason)
            return PreparedPromotionResult(False, exc.reason)

        if result.status == "published":
            self._last_compacted_store_id = max(
                self._last_compacted_store_id,
                result.frontier_store_id,
            )
            if result.node_id is not None:
                node = self._dag.get_node(result.node_id)
                if node is not None:
                    self._invalidate_rollups_for_published_node(node)
            return PreparedPromotionResult(
                True,
                "promoted",
                node_id=result.node_id,
                frontier_store_id=result.frontier_store_id,
            )
        if result.status == "already_published":
            self._prepared_compactions.reject(
                batch_id,
                "canonical_source_overlap",
                superseded=True,
            )
            return PreparedPromotionResult(False, "canonical_source_overlap", frontier_store_id=result.frontier_store_id)
        if result.status == "source_mismatch":
            self._prepared_compactions.reject(batch_id, "source_identity_mismatch")
            return PreparedPromotionResult(False, "source_identity_mismatch", frontier_store_id=result.frontier_store_id)
        if result.status == "stale":
            self._prepared_compactions.reject(batch_id, "frontier_mismatch", superseded=True)
            return PreparedPromotionResult(False, "frontier_mismatch", frontier_store_id=result.frontier_store_id)
        self._prepared_compactions.reject(batch_id, result.reason or result.status)
        return PreparedPromotionResult(False, result.reason or result.status, frontier_store_id=result.frontier_store_id)

    def reject_prepared_compaction(self, batch_id: str, *, reason: str) -> PreparedBatch:
        return self._prepared_compactions.reject(batch_id, reason)

    def get_async_compaction_status(self) -> dict[str, Any]:
        status = self._prepared_compactions.status(
            self.current_conversation_id,
            enabled=self._config.async_background_compaction_enabled,
            worker_enabled=self._config.async_background_compaction_worker_enabled,
            live_policy_fingerprint=self._async_policy_fingerprint(),
            live_summary_route_fingerprint=self._async_summary_route_fingerprint(),
        )
        scheduler = getattr(self, "_background_compaction_scheduler", None)
        status["scheduler"] = (
            scheduler.metrics_snapshot()
            if scheduler is not None
            else {
                "active_workers": 0,
                "maximum_active_workers": 0,
                "queued_conversations": 0,
            }
        )
        return status

    def schedule_background_compaction(self, messages: list[dict[str, Any]]) -> bool:
        if not (
            self._config.async_background_compaction_enabled
            and self._config.async_background_compaction_worker_enabled
        ):
            return False
        return self._background_compaction_scheduler.submit(self, messages)

    def wait_for_background_compaction(self, timeout: float | None = None) -> bool:
        return self._background_compaction_scheduler.wait_idle(timeout)

    def _cancel_background_compaction(self) -> int:
        scheduler = getattr(self, "_background_compaction_scheduler", None)
        if scheduler is None:
            return 0
        return scheduler.cancel_engine(self)

    def _try_promote_ready_background(
        self,
        working_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        if not (
            self._config.async_background_compaction_enabled
            and self._config.async_background_compaction_worker_enabled
        ):
            return None
        batch = self._prepared_compactions.ready_batch(self._conversation_id, self._session_id)
        if batch is None:
            return None
        result = self.promote_prepared_compaction(batch.batch_id, working_messages)
        if not result.promoted:
            return None
        leading = self._leading_anchor_count(working_messages)
        durable_tail = [
            self._store.to_openai_msg(row)
            for row in self._store.get_range(
                self._session_id,
                start_id=result.frontier_store_id + 1,
                conversation_id=self._conversation_id,
            )
            if str(row.get("role") or "") != "system"
        ]
        self._refresh_raw_backlog_debt(working_messages)
        compressed = self._assemble_context(
            working_messages[0] if leading else None,
            durable_tail,
        )
        self.compression_count += 1
        self._last_compression_status = "compacted"
        self._last_compression_noop_reason = ""
        self._ingest_cursor = len(compressed)
        self._ingest_cursor_needs_reconcile = False
        compressed = self._sanitize_active_context_messages(compressed)
        self._write_generated_ignored_placeholder_hash_counts(
            self._generated_placeholder_digest_budget_for_active_replay(compressed)
        )
        self._write_generated_ignored_placeholder_hash_ordinals(
            self._generated_placeholder_digest_ordinals_for_active_replay(compressed)
        )
        return compressed
