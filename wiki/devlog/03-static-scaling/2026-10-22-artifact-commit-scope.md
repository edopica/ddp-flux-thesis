# Benchmark artifact commit scope — 2026-10-22

## Request

Commit the benchmark files while leaving generated logs and build output
untracked. A full scan of the worktree preceded the commit.

## What the scan found

The worktree looked large, but the 12 GB under `artifacts/` is already ignored:
3,690 files, dominated by the `artifacts/**/*.npz` arrays, `__pycache__/`, and
`*.jls`. Of the 628 untracked files, 24.0 MB would have been added.

The untracked set was a mix of three things:

- benchmark code and documents: `scripts/` (38 files), `experiments/`, `wiki/`,
  `reports/`;
- evidence bundles: `artifacts/logs/` (499 files, 13.5 MB) and
  `artifacts/baselines/` (9 files, 0.5 MB);
- presentation build residue: 27 `dist/slides/*.png` renders, 13
  `components/test_*` scratch renders, a byte-identical duplicate deck, and a
  stray one-page `out.pdf` at the repository root (5.8 MB).

The evidence bundles are not disposable. Their `SHA256SUMS` lists the logs, and
`README.md`, the wiki, and each bundle's `RESULTS.md`/`PROVENANCE.md` link to
them; dropping the raw logs alone would have broken 16 document links. The
`artifacts/logs/**/source/` snapshots are 76% byte-identical to `scripts/`, but
the remaining files are the historical versions that produced the runs, so they
were kept.

## Decision

Commit the code, the documents, and the evidence bundles. Ignore only
regenerable presentation output and the stray root build:

```gitignore
presentations/**/dist/
presentations/**/components/test_*
/out.pdf
.ipynb_checkpoints/
```

Result: 582 files, 15.4 MB committed; 46 files, 5.8 MB ignored; no document link
lost. The arrays remain ignored by the existing `artifacts/**/*.npz` rule.

`presentations/static-scaling-benchmark/Makefile` set
`OUTPUT = benchmark_presentation.pdf`, while this devlog series and the
supervisor next action name `presentation.pdf`. The two files were
byte-identical (sha256 `ad24d743…`), so `OUTPUT` was corrected to
`presentation.pdf` and the stale duplicate was deleted. `make -n pdf` now
targets `presentation.pdf`.

## Checks

```bash
git ls-files --others --ignored --exclude-standard | wc -l   # 3690
git ls-files --others --ignored --exclude-standard -z | xargs -0 du -ch | tail -1  # 12G
git diff --check
```

No training, remote job, or measurement ran. No artifact content changed.

## Next action

Push the commit. Raw logs remain tracked by design; a future "no logs in git"
policy needs a separate `git rm --cached` decision because it would break the
existing evidence links.
