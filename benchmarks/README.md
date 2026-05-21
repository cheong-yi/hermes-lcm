# hermes-lcm deterministic benchmarks

This directory contains deterministic replay fixtures and policy files for benchmark-driven LCM preset work.

The benchmark harness is offline by default:

- no live provider calls
- deterministic summarization stub
- no live Hermes config mutation
- writes isolated to the requested output directory

## Run the default replay suite

```bash
python scripts/lcm_benchmark.py \
  --fixture benchmarks/fixtures/long_history_canaries.json \
  --fixture benchmarks/fixtures/repeated_compaction_chatter.json \
  --output benchmarks/runs/local-smoke \
  --json
```

Use `--allow-external-output` when writing outside the repository:

```bash
python scripts/lcm_benchmark.py \
  --fixture benchmarks/fixtures/repeated_compaction_chatter.json \
  --output /tmp/hermes-lcm-benchmark \
  --allow-external-output \
  --json
```

When no `--policy` is supplied, the harness loads built-in policies:

- `baseline_272k`, current long-context baseline
- `codex_gpt_long_context_candidate`, initial Codex/GPT long-context candidate
- `pressure_smoke`, a deliberately small benchmark-only policy that proves pressure/chatter metrics trigger compaction

`pressure_smoke` is not a runtime preset recommendation. It is a control policy for validating benchmark signals.

## Output files

The harness writes:

- `metrics.jsonl`, one serialized replay result per fixture/policy pair
- `summary.json`, aggregate provenance, metric summary, and ranked policy comparison
- per-run `metrics.json` files under fixture/policy output directories

Summary metadata includes:

- `benchmark_version`
- `generated_at_utc`
- `fixture_suite`
- `policy_versions`
- `metric_summary`
- `policy_comparison`

The comparison score is intentionally conservative. It rewards canary recall and stability, then penalizes failures, repeated-compaction risk, and excessive fresh-tail pressure. Treat it as a harness signal, not as proof that a policy is ready to become `preset: auto`.

## Metrics added for preset research

Each replay records:

- `post_compaction_headroom_tokens`
- `post_compaction_headroom_ratio`
- `fresh_tail_tokens`
- `fresh_tail_pressure_ratio`
- `estimated_next_turn_tokens`
- `repeated_compaction_risk`
- `active_canary_recall`
- `retrieval_canary_recall`

These are the first benchmark-quality signals for issue #189. Runtime preset selection, `/lcm preset suggest`, `/lcm preset apply`, live-provider tuning, and automatic config edits remain out of scope.
