# 2026-09-30 — Remote tooling review, `hpc` → `bocconi` rename, per-server docs

## Goal

Prepare the remote workflow for the Phase F two- and four-rank runs and for the upcoming Leonardo (CINECA GPU) server:

1. Rename the `hpc` server alias to `bocconi` in SSH config and everywhere in the docs.
2. Review and clean up `scripts/remote/`.
3. Split per-server instructions out of the global `remote` skill into wiki pages, so adding Leonardo does not bloat the skill.

No remote machine was contacted for this work; all checks are local except `ssh -G bocconi` (config parsing only).

## Rename

- `~/.ssh/config`: `Host hpc` → `Host bocconi`. Only the alias changed; `HostName slogin.hpc.unibocconi.it`, `User 3320522`, and `KexAlgorithms` are unchanged; no key change.
- `scripts/remote/hosts/hpc.conf` → `hosts/bocconi.conf`. The `WRAPPER` line is byte-identical (`git show HEAD:...` comparison); the composed precompile command was verified through a fake `ssh` and matches the pre-rename output exactly.
- References updated: `AGENTS.md`, `Makefile` (comments and login-node error strings), `README.md`, `wiki/context/{current,remote-nodes,deathstar}.md`, `ADR-0005`, historical devlogs/reports, `reports/phase-c7-report.typ`.
- Deliberately unchanged: real hostnames `slogin.hpc.unibocconi.it` and `deathstar-hpc.sm.unibocconi.it`, historical filenames (`20260716-c6b-hpc-tests.md`, `20260730-hpc-unreachable.md`), and generic "HPC clusters" wording (e.g. ADR-0003).
- `reports/phase-c7-report.pdf` is a compiled artifact and still contains the old "HPC GPU Deployment" text; only the `.typ` source was updated (not recompiled).

## Tool review and cleanups

- Deleted stale one-off scripts `run_tests.sh` and `setup_mpi.jl` (superseded by `make` + `run.sh`; `run_tests.sh` referenced a deleted test file).
- Moved `update_wiki.sh` to `scripts/setup/` (not remote tooling).
- `run.sh`: rejects invalid host names and host names without `hosts/<host>.conf` before SSH, so a typo cannot run unwrapped on a cluster login node. Added `hosts/deathstar.conf` with an empty `WRAPPER` so the non-Slurm host passes validation.
- `sync_code.sh`: both rsyncs now use `--delete`. The Flux sync excludes `.git`, `.julia` and protects `test/Manifest.toml` (remote-resolved); the root `Manifest.toml` is still synced on purpose (`wiki/devlog/20260716-c6b-hpc-tests.md`). The thesis sync protects everything in `.rsyncignore`. Dry-run semantics verified locally: stale files would be deleted; protected paths (`.git`, `.julia`, `test/Manifest.toml`) are not.
- `precompile.sh` logic untouched; only the usage example changed. The wrapper composition is unchanged.
- Known caveat (documented, not changed): `run.sh` wraps commands in `bash -l`, which resets CUDA env; GPU/NCCL work may need a direct `ssh` + `salloc` + `mpiexecjl` command (see `wiki/context/bocconi.md`).

## Docs and skill structure

- `remote` skill (`~/.config/opencode/skills/remote/SKILL.md`) is now an index: shared sequence (sync → setup → precompile → run), tool table, and a server table that links to per-server pages and states each server's critical constraint.
- New `wiki/context/bocconi.md`: environment, CPU-target/cache rule, make-arg vs env-prefix rule, GPU access pattern, usage sequence.
- `wiki/context/remote-nodes.md` rewritten as the server index and shared workflow.
- `wiki/context/deathstar.md`: Bocconi rename plus GPU `nvidia-smi`/NCCL safety rules moved in from the skill.
- New `wiki/context/leonardo.md` stub: planned CINECA GPU cluster, explicitly marked not operational; `run.sh` rejects it until `hosts/leonardo.conf` exists.
- `wiki/quickstart.md`: context links updated (bocconi/leonardo added) and historical devlog descriptions renamed.

## Verification

- `bash -n` on all changed scripts: OK (shellcheck not installed).
- Fake-`ssh` test: `NTASKS=1 CPUS_PER_TASK=8 scripts/remote/run.sh bocconi "make precompile-all FLUX_REPO_PATH=../ddp_flux"` prints the exact pre-rename salloc command.
- `run.sh definitely-not-a-host true` and `run.sh ../evil true` exit 1 before any SSH.
- rsync dry-run: `--delete` removes only stale files; `test/Manifest.toml`, `.git`, `.julia` are protected.
- `ssh -G bocconi` resolves hostname/user correctly.
- `rg -n '\bhpc\b'` outside `artifacts/` leaves only allowed hostnames and historical filenames; `git diff --check` clean.

## Next action

Phase F: run the conditional-graph CPU/MPI gates at two and four ranks per `temp/conditional_graph_wrapper_implementation_plan.md` (the four-rank case must prove weighting). First sync and precompile on `bocconi` (the remote cache is cold after the source sync), then run the gates.
