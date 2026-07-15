# LCM Concurrency and Durable Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every behavior change. Complete one MUW at a time and stop for the configured Luna/Ponytail review before starting the next.

**Goal:** Remove same-process SQLite writer collisions and make foreground and background summary publication atomic under eleven concurrent Hermes conversations.

**Architecture:** Share a canonical-path writer coordinator across clone-owned storage helpers, then route canonical summary publication through one compare-and-set transaction. Persist optional background preparation separately and promote it through the same atomic boundary.

**Tech Stack:** Python 3.11+, stdlib `sqlite3`, `threading`, `concurrent.futures`, pytest.

---

### Task 1: MUW 2 writer coordinator contract

**Files:**
- Create: `sqlite_writer.py`
- Modify: `store.py`, `dag.py`, `lifecycle_state.py`, `engine.py`, `db_bootstrap.py`
- Test: `tests/test_sqlite_writer_coordination.py`

- [ ] Write failing tests proving canonical aliases share one coordinator,
  distinct databases do not, eleven helpers never overlap write regions, and
  summarization/read work is not covered by the permit.
- [ ] Run
  `python -m pytest -q tests/test_sqlite_writer_coordination.py` and confirm the
  import/API failures are caused by the missing coordinator.
- [ ] Implement `get_writer_coordinator(path)` and a ticketed reentrant
  `coordinator.transaction(connection, begin_immediate=False)` context manager
  that rolls back on error, records wait/hold metrics, and reference-counts
  helper owners so only the final close checkpoints.
- [ ] Inject the coordinator into every engine-bound store/DAG/lifecycle helper
  and wrap all mutation+commit/rollback regions, including startup migrations.
- [ ] Run the new tests, the existing WAL/migration/concurrency tests, and the
  eleven-clone stress command; require zero lock errors and `quick_check=ok`.
- [ ] Commit the slice using the repository Lore commit format.
- [ ] Dispatch a Luna-default Ponytail-style reviewer. Address all correctness,
  context-integrity, and performance blockers before Task 2.

### Task 2: MUW 3A atomic foreground publication

**Files:**
- Create: `publication.py`
- Modify: `db_bootstrap.py`, `dag.py`, `lifecycle_state.py`, `engine.py`, `compaction.py`
- Test: `tests/test_atomic_publication.py`

- [ ] Write failing tests for a two-clone same-range race, committed-response-lost
  retry, source identity mismatch, frontier mismatch, and injected failure after
  node insert.
- [ ] Run `python -m pytest -q tests/test_atomic_publication.py`; confirm every
  test fails because the publication API/schema does not exist.
- [ ] Add schema v6 `summary_nodes.coverage_key` and its unique partial index.
- [ ] Implement immutable `PublicationIntent`/`PublicationResult` values and
  `AtomicPublicationStore.publish_leaf()` using exactly one `BEGIN IMMEDIATE`
  transaction and CAS frontier update.
- [ ] Replace foreground `dag.add_node()` + separate frontier commit with the
  publisher. Keep summary generation outside the transaction and update GC and
  in-memory markers only after success.
- [ ] Run targeted publication, core compaction, migration, and eleven-clone
  stress tests. Require one canonical node, monotonic frontier, rollback on
  injection, and no more than one publication commit.
- [ ] Commit the slice with Lore trailers.
- [ ] Dispatch a fresh Luna-default Ponytail reviewer and clear every blocker
  before Task 3.

### Task 3: MUW 3B durable background preparation/recovery

**Files:**
- Create: `prepared_compaction.py`
- Modify: `db_bootstrap.py`, `config.py`, `engine.py`, `compaction.py`, `diagnostics.py`
- Modify/activate: `tests/test_async_background_compaction_design.py`
- Test: `tests/test_background_compaction_scheduler.py`

- [ ] Remove the file-wide strict xfail and run the existing design tests to
  capture the RED state for the absent API.
- [ ] Add schema v7 `lcm_prepared_compactions` with the closed state values and
  indexes for conversation/state recovery scans.
- [ ] Implement durable prepare, reject, recover, status, and atomic promotion;
  promotion must use the 3A transaction and batch-state commit atomically.
- [ ] Implement one opt-in scheduler per canonical DB path/process with two total
  workers, per-conversation coalescing, durable owner/attempt leases, bounded
  heartbeat/backoff, and foreground supersession.
- [ ] Add deterministic tests proving at most two simultaneous summarizers for
  eleven conversations, no duplicate conversation job, no permit held during
  summarization, safe restart recovery, and foreground progress after failure.
- [ ] Run all background, publication, writer, diagnostics, and eleven-thread
  stress tests; require zero invisible-to-visible leakage before promotion.
- [ ] Commit the slice with Lore trailers.
- [ ] Dispatch a fresh Luna-default Ponytail reviewer and clear every blocker.

### Task 4: Full verification and activation packet

**Files:**
- Modify if needed: `benchmarking/stress.py`, `scripts/lcm_stress_check.py`
- Create: `/tmp/hermes-lcm-concurrency-2-3b-validation/` runtime artifacts only

- [ ] Run the complete pytest suite in a temporary `HERMES_HOME`.
- [ ] Run ruff and the release stress smoke with eleven clones.
- [ ] Compare uncontended and contended benchmark artifacts against the baseline;
  reject >20% uncontended regression unless deterministic evidence explains it.
- [ ] Verify branch diff, schema migrations from v4/v5/v6, downgrade refusal,
  `PRAGMA quick_check`, and absence of cache/residue.
- [ ] Produce a cutover/rollback receipt. Do not update the live plugin or restart
  the gateway without a separate final activation decision.
