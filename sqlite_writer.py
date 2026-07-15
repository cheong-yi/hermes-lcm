"""Process-wide, canonical-path SQLite writer admission.

SQLite has one writer slot per database.  LCM engines are cloned per agent and
therefore own separate connections, so connection-local locks cannot prevent
same-process write transactions from colliding.  This module adds one fair,
reentrant permit per canonical database path while leaving reads independent.
"""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
import weakref
from collections import deque
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import ContextManager, Iterator


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: weakref.WeakValueDictionary[str, "WriterCoordinator"] = (
    weakref.WeakValueDictionary()
)
_SAMPLE_LIMIT = 2048


def canonical_db_path(path: str | Path) -> str:
    """Return the stable registry key for a database path.

    ``Path.resolve(strict=False)`` normalizes relative paths and resolves every
    existing symlink component while still supporting a database file that has
    not been created yet.  ``normcase`` preserves the same guarantee on
    case-folding filesystems.
    """

    resolved = Path(path).expanduser().resolve(strict=False)
    return os.path.normcase(os.path.realpath(os.fspath(resolved)))


def get_writer_coordinator(path: str | Path) -> "WriterCoordinator":
    """Return the process-wide coordinator for ``path``."""

    key = canonical_db_path(path)
    with _REGISTRY_LOCK:
        coordinator = _REGISTRY.get(key)
        if coordinator is None:
            coordinator = WriterCoordinator(key)
            _REGISTRY[key] = coordinator
        return coordinator


def _percentile(samples: deque[int], percentile: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = max(0, math.ceil((percentile / 100.0) * len(ordered)) - 1)
    return ordered[index] / 1_000_000_000.0


class WriterCoordinator:
    """Fair, reentrant writer admission and bounded timing telemetry."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        self._condition = threading.Condition(threading.RLock())
        self._next_ticket = 0
        self._serving_ticket = 0
        self._abandoned_tickets: set[int] = set()
        self._owner_thread_id: int | None = None
        self._owner_depth = 0
        self._owner_started_ns = 0
        self._transaction_connection: sqlite3.Connection | None = None
        self._transaction_depth = 0
        self._acquisitions = 0
        self._waited_acquisitions = 0
        self._active_writers = 0
        self._max_active_writers = 0
        self._wait_ns_total = 0
        self._hold_ns_total = 0
        self._wait_samples_ns: deque[int] = deque(maxlen=_SAMPLE_LIMIT)
        self._hold_samples_ns: deque[int] = deque(maxlen=_SAMPLE_LIMIT)
        self._owner_tokens: set[int] = set()
        self._next_owner_token = 1
        self._checkpoint_attempt_count = 0
        self._checkpoint_success_count = 0
        self._checkpoint_busy_count = 0
        self._checkpoint_failure_count = 0
        self._checkpoint_last_result = "not_attempted"

    def bind_owner(self) -> int:
        """Register one storage helper and return its idempotent close token."""

        with self._condition:
            token = self._next_owner_token
            self._next_owner_token += 1
            self._owner_tokens.add(token)
            return token

    def _acquire(self) -> bool:
        thread_id = threading.get_ident()
        with self._condition:
            if self._owner_thread_id == thread_id:
                self._owner_depth += 1
                return False

            ticket = self._next_ticket
            self._next_ticket += 1
            wait_started_ns = time.perf_counter_ns()
            queued = ticket != self._serving_ticket or self._owner_thread_id is not None
            try:
                while ticket != self._serving_ticket or self._owner_thread_id is not None:
                    self._condition.wait()
            except BaseException:
                self._abandoned_tickets.add(ticket)
                self._skip_abandoned_tickets_locked()
                self._condition.notify_all()
                raise

            wait_ns = time.perf_counter_ns() - wait_started_ns
            self._owner_thread_id = thread_id
            self._owner_depth = 1
            self._owner_started_ns = time.perf_counter_ns()
            self._acquisitions += 1
            if queued:
                self._waited_acquisitions += 1
            self._wait_ns_total += wait_ns
            self._wait_samples_ns.append(wait_ns)
            self._active_writers += 1
            self._max_active_writers = max(
                self._max_active_writers,
                self._active_writers,
            )
            return True

    def _skip_abandoned_tickets_locked(self) -> None:
        while (
            self._owner_thread_id is None
            and self._serving_ticket in self._abandoned_tickets
        ):
            self._abandoned_tickets.remove(self._serving_ticket)
            self._serving_ticket += 1

    def _release(self, outermost: bool) -> None:
        with self._condition:
            if not outermost:
                self._owner_depth -= 1
                return
            if self._owner_thread_id != threading.get_ident():
                raise RuntimeError("writer permit released by a non-owner thread")
            hold_ns = time.perf_counter_ns() - self._owner_started_ns
            self._hold_ns_total += hold_ns
            self._hold_samples_ns.append(hold_ns)
            self._active_writers -= 1
            self._owner_thread_id = None
            self._owner_depth = 0
            self._owner_started_ns = 0
            self._serving_ticket += 1
            self._skip_abandoned_tickets_locked()
            self._condition.notify_all()

    @contextmanager
    def permit(self) -> Iterator["WriterCoordinator"]:
        """Acquire the path-scoped permit without managing a transaction."""

        outermost = self._acquire()
        try:
            yield self
        finally:
            self._release(outermost)

    @contextmanager
    def write_region(
        self,
        local_lock: ContextManager[object] | None = None,
    ) -> Iterator[None]:
        """Acquire in the invariant order: path permit, then helper lock."""

        lock_context = local_lock if local_lock is not None else nullcontext()
        with self.permit():
            with lock_context:
                yield

    @contextmanager
    def transaction(
        self,
        connection: sqlite3.Connection,
        *,
        local_lock: ContextManager[object] | None = None,
        begin_immediate: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        """Run one transaction under path-scoped and helper-local admission.

        Nested use on the same connection joins the outer transaction. A
        same-thread nested write on another connection fails immediately:
        reentrant admission cannot make SQLite grant a second writer slot, so
        waiting there would only self-timeout behind the outer connection.
        """

        with self.write_region(local_lock):
            with self._condition:
                if (
                    self._transaction_connection is not None
                    and self._transaction_connection is not connection
                ):
                    raise RuntimeError(
                        "cannot nest a writer transaction on a different SQLite connection"
                    )
                self._transaction_connection = connection
                self._transaction_depth += 1
            try:
                started_transaction = not connection.in_transaction
                if started_transaction:
                    connection.execute(
                        "BEGIN IMMEDIATE" if begin_immediate else "BEGIN"
                    )
                try:
                    yield connection
                    if started_transaction and connection.in_transaction:
                        connection.commit()
                except BaseException:
                    if started_transaction and connection.in_transaction:
                        connection.rollback()
                    raise
            finally:
                with self._condition:
                    self._transaction_depth -= 1
                    if self._transaction_depth == 0:
                        self._transaction_connection = None

    def close_owner(
        self,
        owner_token: int,
        connection: sqlite3.Connection | None,
        *,
        local_lock: ContextManager[object] | None = None,
    ) -> None:
        """Close a helper and checkpoint only when its owner is the final one."""

        lock_context = local_lock if local_lock is not None else nullcontext()
        with self.permit():
            with lock_context:
                with self._condition:
                    if owner_token not in self._owner_tokens:
                        return
                    self._owner_tokens.remove(owner_token)
                    final_owner = not self._owner_tokens
                    if final_owner and connection is not None:
                        self._checkpoint_attempt_count += 1
                        try:
                            row = connection.execute(
                                "PRAGMA wal_checkpoint(PASSIVE)"
                            ).fetchone()
                            busy = bool(row and int(row[0] or 0))
                            if busy:
                                self._checkpoint_busy_count += 1
                                self._checkpoint_last_result = "busy"
                            else:
                                self._checkpoint_success_count += 1
                                self._checkpoint_last_result = "ok"
                        except sqlite3.Error as exc:
                            self._checkpoint_failure_count += 1
                            self._checkpoint_last_result = f"error: {exc}"
                if connection is not None:
                    connection.close()

    def metrics_snapshot(self) -> dict[str, int | float | str]:
        """Return bounded wait/hold telemetry for diagnostics and stress runs."""

        with self._condition:
            return {
                "database_path": self.database_path,
                "acquisitions": self._acquisitions,
                "waited_acquisitions": self._waited_acquisitions,
                "active_writers": self._active_writers,
                "max_active_writers": self._max_active_writers,
                "owner_count": len(self._owner_tokens),
                "checkpoint_attempt_count": self._checkpoint_attempt_count,
                "checkpoint_success_count": self._checkpoint_success_count,
                "checkpoint_busy_count": self._checkpoint_busy_count,
                "checkpoint_failure_count": self._checkpoint_failure_count,
                "checkpoint_last_result": self._checkpoint_last_result,
                "wait_seconds_total": self._wait_ns_total / 1_000_000_000.0,
                "wait_seconds_p50": _percentile(self._wait_samples_ns, 50),
                "wait_seconds_p95": _percentile(self._wait_samples_ns, 95),
                "wait_seconds_p99": _percentile(self._wait_samples_ns, 99),
                "wait_seconds_max": (
                    max(self._wait_samples_ns, default=0) / 1_000_000_000.0
                ),
                "hold_seconds_total": self._hold_ns_total / 1_000_000_000.0,
                "hold_seconds_p50": _percentile(self._hold_samples_ns, 50),
                "hold_seconds_p95": _percentile(self._hold_samples_ns, 95),
                "hold_seconds_p99": _percentile(self._hold_samples_ns, 99),
                "hold_seconds_max": (
                    max(self._hold_samples_ns, default=0) / 1_000_000_000.0
                ),
            }
