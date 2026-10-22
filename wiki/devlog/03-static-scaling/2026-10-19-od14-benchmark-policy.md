# OD-14 benchmark policy — 2026-10-19

## Request and decision

The user requested formalization of OD-14 after the timing review response and the 300-update diagnostic.
The [decision](../../../experiments/static-scaling/OD-14-benchmark-acceptance.md) records qualified descriptive throughput and scaling claims.
It separates short-horizon correctness gates from long-run diagnostics and adopts no long-run numerical-equivalence threshold.
Stronger numerical and training-quality claims require prospective criteria and distinct acceptance evidence.
The final Phase 6 cross-framework report requires a deterministic-mode sensitivity in both frameworks.

The specification, decision table, and TOML policy now agree.
Historical artifacts retain their original claims and limitations.
The printed diagnostic includes parameter `rel_l2=0.00225564012`, slightly above the summary's `0.00224`.
The decision records the printed value without converting it into an acceptance bound.

## Checks

Local checks parsed the TOML, compared its original tolerance tables, and checked the decision's documentation links.
Output: `OD14_POLICY_CHECK PASS: TOML parsed; original tolerance tables unchanged; decision links resolve`.
`git diff --check` also passed.

Exact commands from the repository root:

```bash
python3 -c 'import pathlib,tomllib,subprocess,re; p=pathlib.Path("experiments/static-scaling/experiment.toml"); new=tomllib.loads(p.read_text()); old=tomllib.loads(subprocess.check_output(["git","show","HEAD:"+str(p)],text=True)); t=dict(new["tolerances"]); t.pop("scope_note"); assert t==old["tolerances"]; assert new["benchmark_policy"]["numerical_equivalence_claim"]=="none"; d=pathlib.Path("experiments/static-scaling/OD-14-benchmark-acceptance.md"); links=re.findall(r"\]\(([^)]+)\)",d.read_text()); assert all((d.parent/x).exists() for x in links); print("OD14_POLICY_CHECK PASS: TOML parsed; original tolerance tables unchanged; decision links resolve")'
git diff --check
```

The initial patch context mismatch and recovery are in the [failure record](../../failures/2026-10-19-od14-patch-context.md).
No training or remote run occurred.

## Next action

Run the Phase 5 larger-batch memory pilot under OD-14.
Record the OD-7 batch and OD-8 memory-margin decisions before the full-dataset campaign.
