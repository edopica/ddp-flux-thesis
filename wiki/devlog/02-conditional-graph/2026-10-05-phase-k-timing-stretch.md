# Phase K timing stretch: 1×A100 versus 4×A100 NCCL on Leonardo

Date: 2026-10-05
Host: `leonardo` (CINECA), 4× NVIDIA A100-SXM-64GB, driver 535.274.02
Nodes/jobs: GPU check `lrdn3078` (salloc `59434335`), correctness `lrdn0790`
(`59434349`), baseline `lrdn1990` (`59434611`), DDP `lrdn1847` (`59435161`)
Flux worktree: `~/projects/flux-integration` at `ddp/integration-m1-spike`
HEAD `55171e65` dirty (git fields read `unavailable` on the synced tree); no
Flux source changed this session
Julia: 1.12.6, project `~/projects/flux-integration/test`
Predecessor: [Leonardo Phase K K1/K2 4-rank NCCL](2026-10-05-leonardo-k1-k2-4rank-nccl.md)
Campaign record: `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/CAMPAIGN.md`
Supervisor report: `reports/conditional-graph-phase-k-timing-stretch.md`

## Outcome

The optional post-Phase K timing stretch is complete on Leonardo. The
correctness preflight passed (`single` trajectory `maxabs=2.980232e-08`, `ddp`
`maxabs=0.0`, both `TIMING_CHECK_RESULT source=trajectory PASS` at the declared
Float32 `atol=2.0e-6`, `rtol=1.0e-5`), baseline completed 5/5 timed repeats,
and DDP completed 5/5 timed repeats. Measured diagnostic speedup:
`median(t_total_s baseline) / median(slowest_rank_s ddp) = 0.465324 /
0.314365 = 1.480203`. All jobs exited 0; no Flux source changed; 0
`Precompiling packages` lines in all four console logs; no commits.

## Implemented

- `scripts/examples/conditional_graph_timing.jl` - standalone timing CLI with
  `single` (one process consumes all scheduled shards through
  `CGReference.global_gradient`, one native Optimisers update per step) and
  `ddp` (one local shard per rank, `DistributedOptimizerState` with the real
  backend). Untimed warm-up, rebuild from the prepared artifact before every
  repeat, validation only at record boundaries with the time accumulated
  separately, `CUDA.synchronize()` at timing boundaries, and
  `slowest_rank_s = maximum(Allgather(local_elapsed))` for DDP. Rank 0 writes
  `records.tsv` (append+flush every `--record-every` steps), `steps.tsv`,
  `repeats.tsv`, and `metadata.txt`.
- `scripts/checks/conditional_graph_timing_check.jl` - bounded driver with
  `--phase correctness|baseline|ddp|all`, captured child exit codes and
  elapsed times, `done` markers, and `summary.txt`.
- `scripts/plots/plot_phase_k_timing.py` - strict-schema static plotter (8
  PNGs plus `figures-metadata.txt` with input SHA-256 hashes) and `--follow`
  live viewer (status, last step, record age, refresh of `timing-live.png`).
- Additive `CGReference.reference_step!` in
  `scripts/examples/conditional_graph/reference.jl`: a native scheduled step
  that asserts a globally absent head stays out of the AD result and unchanged
  by the update. The existing demo/checkpoint paths are unchanged; the timing
  CLI uses this entry point.

## Local validation (CPU, 4 ranks)

Driver `--phase all` on the prepared K2 artifact (steps 4, warm-up 1,
record-every 2): `single` and `ddp` children exit 0; the driver compared the
`single` and `ddp` final dumps and printed
`TIMING_CHECK_RESULT source=single-vs-ddp PASS maxabs=2.980232e-08 nbad=0` and
`TIMING_RANK_DUMP rank=0..3 equal=true`. The local driver run had no trajectory
file; the Leonardo preflight below provides the trajectory gate. The local
scratch evidence
(`/tmp/opencode/phase-k-timing-local/`) was copied verbatim to
`artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/local-smoke/`
(`driver.log`, `summary.txt`).

## Leonardo campaign

Sequence and exact commands are in the campaign record; all four jobs ran in
separate `srun` allocations on `boost_usr_prod`:

1. Targeted `rsync -avzR` of the new/changed thesis files (no `--delete`, no
   full tree, `flux-integration` untouched), then remote md5/SHA verification
   before any job.
2. GPU check `59434335` (`lrdn3078`, 4× A100 listed, 4 s).
3. Correctness preflight `59434349` (`lrdn0790`, 4m32s): `single` 166.1 s,
   `ddp` 58.2 s, both PASS; driver `TIMING_STRETCH_SUMMARY backend=nccl
   ranks=4 speedup=na status=PASS`.
4. Baseline `59434611` (`lrdn1990`, 3m40s, 1 process on device 0).
5. DDP `59435161` (`lrdn1847`, 3m41s, 4 ranks on devices `[1,2,3,0]`) with the
   local live viewer running against the streamed rank-0 `records.tsv`.
6. Pull-back via `rsync -avz` (exit 0, no forward sync afterward); static
   figures generated locally with `uv run --offline --with matplotlib --with
   numpy`.

Prepared artifact reused (SHA verified before the run): schedule digest
`ce6b41aa...`, dataset digest `658e3dd3...`, prepared SHA
`7f0d38d9...`. Fixed work: 120 global updates × global batch 128 = 15360
samples per repeat, exposures a=6400 / b=8960.

## Measured results

- Baseline `t_total_s` per repeat 0.626333, 0.439836, 0.832694, 0.465324,
  0.444683; median 0.465324 (median 257.884817 updates/s).
- DDP `slowest_rank_s` per repeat 0.714849, 0.344812, 0.282049, 0.314365,
  0.266138; median 0.314365 (median 381.721898 updates/s).
- Speedup `0.465324 / 0.314365 = 1.480203` (measured diagnostic; not an
  acceptance condition and not a production-speedup claim). Baseline repeat 3
  is the slowest baseline repeat (0.832694 s, driven by t_val_s 0.535641 s);
  DDP repeat 1 is the slowest DDP repeat (0.714849 s, residual warm-up
  variance).
- Final validation at step 120 matched the K2 Leonardo metrics (task A
  accuracy 0.847, task B 0.739; task B loss absolute difference 7.6e-09,
  about 9.8e-09 relative, Float32 accumulation).
- Evidence checklist in `CAMPAIGN.md`: 5 repeats each, 120 records each, 600
  step rows each, 4 DDP `TIMING_RANK_DONE` markers, both `done` markers PASS.
- Known deviation: the stored `summary.txt` reflects the last
  (`--phase ddp`) invocation because each phase overwrote the run-root file;
  per-phase PASS evidence is in the child `done`/`console.log` files.

## Live viewer

The viewer ran from 2026-10-05T19:08:18Z (WAITING) to 19:14:33Z (DONE); it
wrote `viewer/timing-live.png` and `viewer-live.log` (126 WAITING, 218
RUNNING, 2 DONE lines; 22 RUNNING lines carry data). Documented caveat: the
ssh pipe delivered `records.tsv` in a late buffered burst, so mid-run
`last_step` stayed 0 and all 120 rows arrived about 16 s before job end. The
viewer was live, not a replay, and the pulled `ddp/records.tsv` is the
authoritative record.

## Documentation updated

- New supervisor report `reports/conditional-graph-phase-k-timing-stretch.md`.
- `wiki/context/current.md` (new top section), `wiki/context/leonardo.md`
  (timing-stretch status), `README.md` (summary, checkpoint lines, CW-K row),
  `wiki/quickstart.md` (next action).
- `temp/conditional_graph_phase_k_prompt.md` (header status, section 1 evidence
  row, section 11 status and next action),
  `temp/conditional_graph_wrapper_implementation_plan.md` (exact next action,
  Phase K status, post-K stretch status),
  `reports/conditional-graph-phase-k.md` (status line and unrun section
  pointer).
- `artifacts/logs/conditional-wrapper/timing/stretch-2026-10-05-leonardo/local-smoke/`
  (`driver.log`, `summary.txt`) preserves the local same-host check.

## Failures

No command failed: no `status=FAIL`, no `ERROR: LoadError`, all driver and
child exit codes 0. No `wiki/failures/` entry is needed for this session.
The only recorded deviations are the buffered live stream and the
single-`summary.txt` overwrite, both in `CAMPAIGN.md`.

## Remaining

The deferred-behavior report (`temp/conditional_graph_deferred_behaviors_prompt.md`)
is the only remaining Phase K item.

**Exact next action:** write the deferred-behavior report.
