# LCM Concurrency and Durable Publication Design

## Goal

Keep LCM's lossless context behavior while making one SQLite database reliable
and responsive under eleven concurrent Hermes conversations.

## Constraints

- Preserve the current prompt/context semantics and source lineage.
- SQLite remains the durable store and therefore still has one writer slot.
- Never hold a writer permit during LLM calls, tokenization, extraction, reads,
  or active-context assembly.
- Default behavior remains synchronous until background compaction is explicitly
  enabled.
- No new dependencies and no changes to Hermes core.

## Considered approaches

1. **Increase SQLite timeouts and add retries.** Small, but it only turns lock
   failures into longer stalls and can repeat non-idempotent work. Rejected as
   the primary fix.
2. **One process-wide writer queue.** It can batch writes, but every call becomes
   asynchronous and failure propagation becomes harder. It is a possible later
   throughput MUW, not the narrow reliability repair.
3. **Path-scoped coordination plus atomic publication.** Recommended. It matches
   SQLite's existing one-writer rule, removes competing in-process transactions,
   and makes each summary publication idempotent without serializing LLM work.

## MUW 2: DB-path writer coordination

Add a small `sqlite_writer.py` module with a process-wide registry keyed by the
canonical database path. `WriterCoordinator` owns a ticketed, reentrant permit
and compact wait/hold counters. Every storage helper created by an engine clone
receives the same coordinator for the same database; different databases remain
independent.

Existing per-connection locks remain because Python SQLite connections must not
be used concurrently. All mutation paths use one fixed order: DB-path permit,
then helper-local lock, then SQLite transaction. The permit covers only schema
mutation or `execute`/`commit`/`rollback` write regions. LLM, selection,
serialization, hashing, extraction, tokenization, and read-only work remain
outside it. The coordinator reference-counts bound helpers so only the last
owner performs a graceful shutdown checkpoint.

Eleven-clone acceptance:

- all clones for one canonical path share one coordinator, including symlink and
  relative aliases;
- at most one in-process write transaction is active for that path;
- concurrent store, DAG, and lifecycle writes complete without `SQLITE_BUSY`;
- a blocked summarizer does not block reads or unrelated appends;
- uncontended microbenchmark throughput does not regress by more than 20% and
  contended p95/p99 writer wait improves versus the current timeout behavior.

## MUW 3A: atomic idempotent foreground publication

Add a focused `AtomicPublicationStore` using one lazy SQLite connection and the
same writer coordinator. The compactor captures an expected lifecycle frontier,
source row identities, a deterministic coverage key, and the intended new
frontier before summarization. Summarization remains outside any transaction.

Publication uses one `BEGIN IMMEDIATE` transaction to:

1. validate the current conversation/session frontier;
2. validate the source rows and their content identity;
3. return `already_published` when the coverage key already exists;
4. insert the leaf summary node;
5. compare-and-set the lifecycle frontier from the expected value to the new
   monotonic value; and
6. commit both changes together.

A CAS loser returns `stale` and reloads canonical context. Best-effort transcript
GC and in-memory bookkeeping happen only after commit. Schema v6 adds a nullable
`coverage_key` to `summary_nodes` with a unique partial index, so legacy nodes
remain valid.

Acceptance includes barrier-controlled two-clone races, response-lost retry,
rollback after injected partial failure, monotonic frontier checks, and a trace
assertion proving one publication transaction/commit.

## MUW 3B: durable bounded background preparation and recovery

Schema v7 adds `lcm_prepared_compactions`. Rows move through the closed state
machine `preparing -> ready -> promoted` or a terminal
`failed/rejected/superseded` state. They persist:

- conversation/session and expected frontier;
- ordered source IDs and content fingerprints;
- policy and summary-route fingerprints;
- deterministic coverage key and proposed frontier;
- summary payload, token/lineage metadata, attempts, error, and timestamps.

Prepared summaries are never queried by DAG readers. Promotion reuses the 3A
transaction and updates the batch state in that same commit. Startup converts
orphaned `preparing` rows to retryable failure/pending state; it never treats
them as canonical.

The optional process-wide scheduler is keyed by database path, coalesces work by
conversation, and caps summarization at two workers total for that path/process
(not two workers per clone) for the eleven-thread target. Durable claim leases
carry an owner/attempt token and expire after at least twice the configured
summary timeout plus 30 seconds; a bounded heartbeat prevents live work from
being reclaimed.
It never blocks foreground compaction: foreground publication supersedes stale
background work, while background failures use bounded backoff and remain
visible through status/doctor diagnostics. Background mode stays disabled by
default.

## Performance and observability

The deterministic stress lane uses eleven cloned engines and reports throughput,
p50/p95/p99 permit wait, maximum concurrent writers, publication transaction
count, background queue depth, and worker concurrency. CI asserts structural
invariants and generous non-flaky bounds; the release benchmark records timing
without treating noisy wall-clock values as correctness. The calibrated target
for the current workstation is foreground ingest p95 <= 250 ms, p99 <= 500 ms,
normal writer hold p95 <= 50 ms, and atomic publication p95 <= 100 ms for a
bounded four-leaf batch. CI treats correctness and relative regression as hard
gates and records absolute timing as evidence unless its storage class is
calibrated.

The expected result is fewer writer timeouts and lower contended tail latency,
not parallel SQLite writes. If the single-writer queue remains the dominant cost
after these MUWs, batching or database partitioning is a separate performance
project.

## Rollout

Each MUW lands as a local checkpoint with targeted and full tests, followed by a
Luna-default Ponytail-style review. The live plugin and gateway are not changed
during development. Final activation requires a fresh backup, gateway quiescence,
schema verification, restart, and an eleven-thread bounded smoke.
