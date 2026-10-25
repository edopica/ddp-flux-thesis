# Cache diagnostic preparation failures — 2026-10-25

- Initial sandboxed SSH availability check returned 255: DNS resolution for `login.leonardo.cineca.it` failed. Retried the same command with the required network escalation; SSH succeeded (`login02`) and Slurm showed no active jobs. No remote workload had been submitted.
- Local source discovery tried absent `scripts/remote/leonardo/static-full-campaign.sbatch`, `scripts/static_scaling/full_dataset_timing.py`, and old CUDA `src/memory.jl`/`src/pool.jl` glob locations. Correct paths are `run-static-full-campaign.sh`, `timing_torch.py`, and version-specific CUDA memory sources discovered with `rg --files`. Corrected reads succeeded. Harness agent also encountered old CUDA source/env path lookups; these did not run training.
- Skill discovery found the required remote skill at `/home/kurapica/.config/opencode/skills/remote/SKILL.md`; some searched paths were absent. The skill was read before remote execution, together with the Leonardo server page.

No campaign result is inferred from these preparation checks.

- Harness development caught a Julia range-expression syntax error during actual helper inclusion. Fixed by putting the range expression on one line; actual inclusion and recursive parse checks now pass. Bare `Meta.parseall` alone can return an error expression and is insufficient validation.
- Launcher source discovery tried nonexistent `scripts/static_scaling/runtime_capture*`; runtime capture is embedded in the existing launcher. Corrected inspection succeeded.
- The first sandboxed rsync dry-run also returned DNS error 255. Required escalated retry completed successfully. No files were changed by the dry-run.
