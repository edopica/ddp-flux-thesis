"""
Phase 2 negative-acceptance harness for the static-scaling checkers.

Purpose
-------
Reproduce the three acceptance gaps from
`reports/static-scaling-phase2-review.md` (R1, R2, R3) as mutations against
scratch copies, and prove that the fixed checkers REJECT them while the
unmutated inputs still PASS:

  * mutation_prep_epoch49 -- R1: hardlink-copy (`cp -al`) the Python-side PREP
    directory, change `augmentation[49, 0, 0]` to the next valid crop value
    (0..8), keep the copied `schedule_manifest.json` stale. Expected rejection:
    `prep.schedule_files_sha256_bytes FAIL` plus `prep.semantic_checks SKIP`;
    the original file hash must stay unchanged.
  * mutation_fixture_label -- R2: copy the Julia-side fixture directory, set
    `data_fixture.npz::train_labels[0] = 1`, keep the copied
    `fixture_manifest.json` stale. Expected rejection:
    `fixtures.file_sha256_bytes FAIL`; the original fixture hashes must stay
    unchanged.
  * mutation_gn_julia / mutation_gn_python -- R3: set
    `stage1.block0.gn1.weight[0] = 2.0` in copied bundles and update that
    array's sha256 plus `bundle_hash` so the mutation is self-consistent
    (hashes cannot catch it). Expected rejection:
    `bundle.gn_initial_values FAIL` while `bundle.manifest_sha256` and
    `bundle.bundle_hash` still PASS.

Checkers run: `check_dataset.py` (Python) and `check_all.jl` (Julia data check
plus the Flux operator check when Flux resolves on the stacked load path).

CLI
---
  python3 scripts/static_scaling/negative_tests_local.py \\
    [--prep-dir P] [--fixture-dir F] [--bundle B] [--bundle-manifest M] [--workdir W]

Defaults:
  fixture-dir      artifacts/baselines/static-scaling/2026-10-16-phase2 (real)
  bundle           <fixture-dir>/initial_bundle.npz
  bundle-manifest  artifacts/logs/static-scaling/2026-10-16-phase2/prepared-manifests/initial_bundle.manifest.json
  prep-dir         unset -> build
                   `python3 scripts/static_scaling/make_fixtures.py --synthetic
                   --synthetic-prep --out-dir <workdir>/synthetic` and use that
                   same directory as PREP + fixture-dir + bundle location for
                   the Python-side tests
  workdir          tempfile.mkdtemp(prefix="phase2-negtest-")

When --prep-dir is given (e.g. Leonardo `$PREP` with the real fixtures, or a
consistent local triple), the Python-side tests use it together with the given
fixture-dir/bundle/bundle-manifest; the caller is responsible for consistency.

Local default run:
  TMPDIR=/tmp/opencode python3 scripts/static_scaling/negative_tests_local.py \\
    --workdir /tmp/opencode/phase2-negtest

Leonardo real-artifact run (compute node; `$PREP` paths as in
artifacts/logs/static-scaling/2026-10-16-phase2/COMMANDS.md):
  PREP=/leonardo_work/IscrC_UNMASKED/epicazio/static-scaling-prepared
  /leonardo_work/IscrC_UNMASKED/epicazio/venvs/pytorch-scaling/bin/python \\
    scripts/static_scaling/negative_tests_local.py \\
    --prep-dir "$PREP" \\
    --fixture-dir artifacts/baselines/static-scaling/2026-10-16-phase2 \\
    --bundle "$PREP/initial_bundle.npz" \\
    --bundle-manifest "$PREP/initial_bundle.manifest.json" \\
    --workdir "$TMPDIR/phase2-negtest"
  (invoke the harness under the pinned venv python: it runs every checker with
  sys.executable, so all checkers see the same interpreter)

Output: one `NEGTEST <test> ok <detail>` line per test, the full checker output
at <workdir>/logs/<test>.log, then `NEGTEST PASS` (exit 0) or
`NEGTEST FAIL <names>` (exit 1). All scratch data stays under the workdir.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np

import dataset_adapter as da

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECK_JULIA = os.path.join("scripts", "static_scaling", "check_all.jl")
CHECK_PYTHON = os.path.join("scripts", "static_scaling", "check_dataset.py")
MAKE_FIXTURES = os.path.join("scripts", "static_scaling", "make_fixtures.py")
JULIA_ENV_DIR = os.path.join(REPO_ROOT, "scripts", "static_scaling", "julia-env")
DEFAULT_FIXTURE_DIR = os.path.join(REPO_ROOT, "artifacts", "baselines",
                                   "static-scaling", "2026-10-16-phase2")
DEFAULT_BUNDLE_MANIFEST = os.path.join(
    REPO_ROOT, "artifacts", "logs", "static-scaling", "2026-10-16-phase2",
    "prepared-manifests", "initial_bundle.manifest.json")
DEFAULT_FLUX_REPO = "/home/kurapica/Projects/ddp_flux/flux-integration"

GN_TARGET = "stage1.block0.gn1.weight"
GN_VALUE = np.float32(2.0)

CHECK_LINE = re.compile(r"^CHECK (\S+) (PASS|FAIL|SKIP)(?: (.*))?$", re.MULTILINE)


def julia_load_path(flux_repo: str) -> str:
    return ":".join([JULIA_ENV_DIR, flux_repo, "@stdlib"])


def parse_checks(text: str) -> dict:
    checks = {}
    for m in CHECK_LINE.finditer(text):
        checks[m.group(1)] = m.group(2)
    return checks


def final_status(text: str, tag: str):
    """(status, fail_names) of the last `tag PASS|FAIL ...` line, or (None, [])."""
    status, rest = None, ""
    for m in re.finditer(r"^%s (PASS|FAIL)(?: (.*))?$" % re.escape(tag), text, re.MULTILINE):
        status, rest = m.group(1), m.group(2) or ""
    names = [n for n in rest.split(",") if n] if status == "FAIL" else []
    return status, names


def last_nonempty_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def sha256_dir(directory: str) -> dict:
    out = {}
    for root, _dirs, files in os.walk(directory):
        for name in sorted(files):
            path = os.path.join(root, name)
            out[os.path.relpath(path, directory)] = da.sha256_file(path)
    return out


class Runner:
    def __init__(self, workdir: str):
        self.workdir = workdir
        self.logs_dir = os.path.join(workdir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

    def run(self, name: str, cmd, env, printable: str = None):
        log_path = os.path.join(self.logs_dir, name + ".log")
        shown = printable if printable is not None else \
            " ".join(shlex.quote(str(c)) for c in cmd)
        print("NEGTEST exec: (cd %s) %s" % (REPO_ROOT, shown), flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = proc.stdout.decode("utf-8", errors="replace")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("NEGTEST log: %s (exit %d)" % (log_path, proc.returncode), flush=True)
        return proc.returncode, text


class Config:
    def __init__(self, args):
        self.workdir = os.path.abspath(args.workdir) if args.workdir else \
            tempfile.mkdtemp(prefix="phase2-negtest-")
        os.makedirs(self.workdir, exist_ok=True)
        self.fixture_dir = os.path.abspath(args.fixture_dir)
        self.bundle = os.path.abspath(
            args.bundle or os.path.join(self.fixture_dir, "initial_bundle.npz"))
        self.bundle_manifest = os.path.abspath(args.bundle_manifest or DEFAULT_BUNDLE_MANIFEST)
        self.prep_dir = os.path.abspath(args.prep_dir) if args.prep_dir else None
        self.flux_repo = os.path.abspath(os.environ.get("FLUX_REPO_PATH") or DEFAULT_FLUX_REPO)
        self.flux_expected = os.path.isfile(os.path.join(self.flux_repo, "Project.toml"))
        self.runner = Runner(self.workdir)
        self.py_prep = None
        self.py_fixture_dir = None
        self.py_bundle = None
        self.py_bundle_manifest = None


def prepare(args):
    cfg = Config(args)
    problems = []
    if not os.path.isdir(cfg.fixture_dir):
        problems.append("fixture-dir not found: %s" % cfg.fixture_dir)
    if not os.path.isfile(cfg.bundle):
        problems.append("bundle not found: %s" % cfg.bundle)
    if not os.path.isfile(cfg.bundle_manifest):
        problems.append("bundle-manifest not found: %s" % cfg.bundle_manifest)
    if cfg.prep_dir is not None and not os.path.isdir(cfg.prep_dir):
        problems.append("prep-dir not found: %s" % cfg.prep_dir)
    if problems:
        return cfg, problems

    if cfg.prep_dir is None:
        synth = os.path.join(cfg.workdir, "synthetic")
        bundle = os.path.join(synth, "initial_bundle.npz")
        manifest = os.path.join(synth, "initial_bundle.manifest.json")
        fixture_manifest = os.path.join(synth, "fixture_manifest.json")
        if not (os.path.isfile(fixture_manifest) and os.path.isfile(bundle)
                and os.path.isfile(manifest)):
            cmd = [sys.executable, MAKE_FIXTURES, "--synthetic", "--synthetic-prep",
                   "--out-dir", synth]
            rc, _ = cfg.runner.run("build_synthetic", cmd, os.environ.copy())
            if rc != 0:
                return cfg, ["synthetic PREP build failed (exit %d); see %s"
                             % (rc, os.path.join(cfg.runner.logs_dir, "build_synthetic.log"))]
        else:
            print("NEGTEST build reuse: %s (synthetic fixtures + bundle present)" % synth,
                  flush=True)
        cfg.py_prep = synth
        cfg.py_fixture_dir = synth
        cfg.py_bundle = bundle
        cfg.py_bundle_manifest = manifest
    else:
        cfg.py_prep = cfg.prep_dir
        cfg.py_fixture_dir = cfg.fixture_dir
        cfg.py_bundle = cfg.bundle
        cfg.py_bundle_manifest = cfg.bundle_manifest

    if not cfg.flux_expected:
        print("NEGTEST warn: Flux repo not found at %s; operator checks will SKIP"
              % cfg.flux_repo, flush=True)
    return cfg, []


def python_cmd(cfg, prep, fixture, bundle, manifest):
    return [sys.executable, CHECK_PYTHON, "--prep-dir", prep, "--fixture-dir", fixture,
            "--bundle", bundle, "--bundle-manifest", manifest]


def julia_cmd(fixture, bundle, manifest):
    return ["julia", "--startup-file=no", CHECK_JULIA, "--fixture-dir", fixture,
            "--bundle", bundle, "--bundle-manifest", manifest]


def run_julia(cfg, name, fixture, bundle, manifest):
    cmd = julia_cmd(fixture, bundle, manifest)
    printable = "JULIA_LOAD_PATH=%s %s" % (
        shlex.quote(julia_load_path(cfg.flux_repo)),
        " ".join(shlex.quote(str(c)) for c in cmd))
    return cfg.runner.run(name, cmd, julia_env(cfg.flux_repo), printable)


def julia_env(flux_repo: str) -> dict:
    env = os.environ.copy()
    env["JULIA_LOAD_PATH"] = julia_load_path(flux_repo)
    return env


def rewrite_bundle_gn(bundle_path: str, manifest_path: str):
    """Self-consistent R3 mutation: gn1.weight[0]=2.0 with updated hashes."""
    with np.load(bundle_path) as z:
        payload = {k: np.array(z[k]) for k in z.files}
    if GN_TARGET not in payload:
        raise AssertionError("bundle lacks %s" % GN_TARGET)
    old = float(payload[GN_TARGET][0])
    payload[GN_TARGET][0] = GN_VALUE
    np.savez(bundle_path, **payload)
    manifest = da.read_json(manifest_path)
    entries = manifest.get("arrays", [])
    found = False
    for e in entries:
        if e.get("name") == GN_TARGET:
            e["sha256"] = da.sha256_c_order(payload[GN_TARGET])
            found = True
    if not found:
        raise AssertionError("manifest lacks %s" % GN_TARGET)
    manifest["bundle_hash"] = da.bundle_hash_of_arrays(
        [payload[e["name"]] for e in entries])
    da.write_json(manifest_path, manifest)
    return old


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_positive_python(cfg: Config) -> str:
    cmd = python_cmd(cfg, cfg.py_prep, cfg.py_fixture_dir, cfg.py_bundle,
                     cfg.py_bundle_manifest)
    rc, text = cfg.runner.run("positive_python", cmd, os.environ.copy())
    checks = parse_checks(text)
    final = last_nonempty_line(text)
    assert rc == 0, "exit=%d (want 0); final=%r" % (rc, final)
    assert final == "PHASE2_DATASET_CHECK PASS", "final line is %r" % final
    failed = sorted(n for n, s in checks.items() if s == "FAIL")
    assert not failed, "FAIL check lines: %s" % ",".join(failed)
    return "rc=0 final=PHASE2_DATASET_CHECK PASS checks=%d failed=0" % len(checks)


def test_positive_julia(cfg: Config) -> str:
    rc, text = run_julia(cfg, "positive_julia", cfg.fixture_dir, cfg.bundle,
                         cfg.bundle_manifest)
    checks = parse_checks(text)
    data_status, data_fail = final_status(text, "PHASE2_JULIA_DATA_CHECK")
    assert rc == 0, "exit=%d (want 0)" % rc
    assert data_status == "PASS", "PHASE2_JULIA_DATA_CHECK=%s fail=%s" % (data_status, data_fail)
    detail = "rc=0 data=PASS checks=%d" % len(checks)
    op_status, op_fail = final_status(text, "PHASE2_JULIA_OPERATOR_CHECK")
    if cfg.flux_expected:
        assert op_status == "PASS", \
            "Flux expected at %s but operator=%s fail=%s" % (cfg.flux_repo, op_status, op_fail)
        detail += " operator=PASS"
    else:
        assert op_status is None or op_status == "PASS", \
            "operator=%s fail=%s without a resolvable Flux" % (op_status, op_fail)
        detail += " operator=%s" % (op_status or "SKIP")
    return detail


def test_mutation_prep_epoch49(cfg: Config) -> str:
    scratch_root = os.path.join(cfg.workdir, "mutation_prep_epoch49")
    scratch_prep = os.path.join(scratch_root, "prep")
    if os.path.exists(scratch_prep):
        shutil.rmtree(scratch_prep)
    os.makedirs(scratch_root, exist_ok=True)
    subprocess.run(["cp", "-al", cfg.py_prep, scratch_prep], cwd=REPO_ROOT, check=True)

    orig_aug = os.path.join(cfg.py_prep, "augmentation.npy")
    sha_before = da.sha256_file(orig_aug)
    aug = np.array(np.load(orig_aug, mmap_mode="r"))
    old = int(aug[49, 0, 0])
    new = (old + 1) % 9
    aug[49, 0, 0] = new
    scratch_aug = os.path.join(scratch_prep, "augmentation.npy")
    os.remove(scratch_aug)
    np.save(scratch_aug, aug)

    cmd = python_cmd(cfg, scratch_prep, cfg.py_fixture_dir, cfg.py_bundle,
                     cfg.py_bundle_manifest)
    rc, text = cfg.runner.run("mutation_prep_epoch49", cmd, os.environ.copy())
    checks = parse_checks(text)
    _status, fail_names = final_status(text, "PHASE2_DATASET_CHECK")
    assert rc != 0, "exit=%d (want nonzero)" % rc
    assert checks.get("prep.schedule_files_sha256_bytes") == "FAIL", \
        "prep.schedule_files_sha256_bytes=%s (want FAIL)" % checks.get("prep.schedule_files_sha256_bytes")
    assert checks.get("prep.semantic_checks") == "SKIP", \
        "prep.semantic_checks=%s (want SKIP)" % checks.get("prep.semantic_checks")
    assert "prep.schedule_files_sha256_bytes" in fail_names, \
        "final FAIL list %s lacks prep.schedule_files_sha256_bytes" % fail_names
    sha_after = da.sha256_file(orig_aug)
    assert sha_after == sha_before, "original augmentation.npy changed during the test"
    return ("rc=%d reject=prep.schedule_files_sha256_bytes skip=prep.semantic_checks "
            "aug49_0_0=%d->%d original_sha_unchanged=true" % (rc, old, new))


def test_mutation_fixture_label(cfg: Config) -> str:
    scratch_root = os.path.join(cfg.workdir, "mutation_fixture_label")
    scratch_fixtures = os.path.join(scratch_root, "fixtures")
    if os.path.exists(scratch_fixtures):
        shutil.rmtree(scratch_fixtures)
    os.makedirs(scratch_root, exist_ok=True)
    before = sha256_dir(cfg.fixture_dir)
    shutil.copytree(cfg.fixture_dir, scratch_fixtures)

    path = os.path.join(scratch_fixtures, "data_fixture.npz")
    with np.load(path) as z:
        arrays = {k: np.array(z[k]) for k in z.files}
    old = int(arrays["train_labels"][0])
    arrays["train_labels"][0] = 1
    np.savez(path, **arrays)

    rc, text = run_julia(cfg, "mutation_fixture_label", scratch_fixtures, cfg.bundle,
                         cfg.bundle_manifest)
    checks = parse_checks(text)
    _status, fail_names = final_status(text, "PHASE2_JULIA_DATA_CHECK")
    assert rc != 0, "exit=%d (want nonzero)" % rc
    assert checks.get("fixtures.file_sha256_bytes") == "FAIL", \
        "fixtures.file_sha256_bytes=%s (want FAIL)" % checks.get("fixtures.file_sha256_bytes")
    assert "fixtures.file_sha256_bytes" in fail_names, \
        "final FAIL list %s lacks fixtures.file_sha256_bytes" % fail_names
    after = sha256_dir(cfg.fixture_dir)
    assert after == before, "original fixture directory changed during the test"
    return ("rc=%d reject=fixtures.file_sha256_bytes label0=%d->1 "
            "original_dir_unchanged=true" % (rc, old))


def test_mutation_gn_julia(cfg: Config) -> str:
    root = os.path.join(cfg.workdir, "mutation_gn_julia")
    os.makedirs(root, exist_ok=True)
    bundle = os.path.join(root, "initial_bundle.npz")
    manifest = os.path.join(root, "initial_bundle.manifest.json")
    shutil.copy2(cfg.bundle, bundle)
    shutil.copy2(cfg.bundle_manifest, manifest)
    old = rewrite_bundle_gn(bundle, manifest)

    rc, text = run_julia(cfg, "mutation_gn_julia", cfg.fixture_dir, bundle, manifest)
    checks = parse_checks(text)
    _status, fail_names = final_status(text, "PHASE2_JULIA_DATA_CHECK")
    assert rc != 0, "exit=%d (want nonzero)" % rc
    assert checks.get("bundle.gn_initial_values") == "FAIL", \
        "bundle.gn_initial_values=%s (want FAIL)" % checks.get("bundle.gn_initial_values")
    assert checks.get("bundle.manifest_sha256") == "PASS", \
        "bundle.manifest_sha256=%s (want PASS)" % checks.get("bundle.manifest_sha256")
    assert checks.get("bundle.bundle_hash") == "PASS", \
        "bundle.bundle_hash=%s (want PASS)" % checks.get("bundle.bundle_hash")
    assert "bundle.gn_initial_values" in fail_names, \
        "final FAIL list %s lacks bundle.gn_initial_values" % fail_names
    detail = ("rc=%d reject=bundle.gn_initial_values gn0=%g->%g manifest_sha256=PASS "
              "bundle_hash=PASS" % (rc, old, float(GN_VALUE)))
    if cfg.flux_expected:
        op_status, op_fail = final_status(text, "PHASE2_JULIA_OPERATOR_CHECK")
        assert op_status == "PASS", "operator=%s fail=%s" % (op_status, op_fail)
        detail += " operator=PASS"
    return detail


def test_mutation_gn_python(cfg: Config) -> str:
    root = os.path.join(cfg.workdir, "mutation_gn_python")
    os.makedirs(root, exist_ok=True)
    bundle = os.path.join(root, "initial_bundle.npz")
    manifest = os.path.join(root, "initial_bundle.manifest.json")
    shutil.copy2(cfg.py_bundle, bundle)
    shutil.copy2(cfg.py_bundle_manifest, manifest)
    old = rewrite_bundle_gn(bundle, manifest)

    cmd = python_cmd(cfg, cfg.py_prep, cfg.py_fixture_dir, bundle, manifest)
    rc, text = cfg.runner.run("mutation_gn_python", cmd, os.environ.copy())
    checks = parse_checks(text)
    _status, fail_names = final_status(text, "PHASE2_DATASET_CHECK")
    assert rc != 0, "exit=%d (want nonzero)" % rc
    assert checks.get("bundle.gn_initial_values") == "FAIL", \
        "bundle.gn_initial_values=%s (want FAIL)" % checks.get("bundle.gn_initial_values")
    assert checks.get("bundle.manifest_sha256") == "PASS", \
        "bundle.manifest_sha256=%s (want PASS)" % checks.get("bundle.manifest_sha256")
    assert checks.get("bundle.bundle_hash") == "PASS", \
        "bundle.bundle_hash=%s (want PASS)" % checks.get("bundle.bundle_hash")
    assert "bundle.gn_initial_values" in fail_names, \
        "final FAIL list %s lacks bundle.gn_initial_values" % fail_names
    return ("rc=%d reject=bundle.gn_initial_values gn0=%g->%g manifest_sha256=PASS "
            "bundle_hash=PASS" % (rc, old, float(GN_VALUE)))


TESTS = (
    ("positive_python", test_positive_python),
    ("positive_julia", test_positive_julia),
    ("mutation_prep_epoch49", test_mutation_prep_epoch49),
    ("mutation_fixture_label", test_mutation_fixture_label),
    ("mutation_gn_julia", test_mutation_gn_julia),
    ("mutation_gn_python", test_mutation_gn_python),
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 2 negative-acceptance harness (review R1/R2/R3 mutations)")
    p.add_argument("--prep-dir", default=None,
                   help="PREP for the Python-side tests; unset builds a synthetic all-in-one")
    p.add_argument("--fixture-dir", default=DEFAULT_FIXTURE_DIR,
                   help="Julia-side fixture dir (default: real 2026-10-16-phase2 artifacts)")
    p.add_argument("--bundle", default=None,
                   help="bundle npz (default: <fixture-dir>/initial_bundle.npz)")
    p.add_argument("--bundle-manifest", default=DEFAULT_BUNDLE_MANIFEST,
                   help="bundle manifest json (default: prepared-manifests/initial_bundle.manifest.json)")
    p.add_argument("--workdir", default=None,
                   help="scratch dir (default: tempfile.mkdtemp(prefix=phase2-negtest-))")
    args = p.parse_args(argv)

    cfg, problems = prepare(args)
    if problems:
        for problem in problems:
            print("NEGTEST setup error: %s" % problem, file=sys.stderr)
        print("NEGTEST FAIL setup")
        return 1

    failures = []
    for name, fn in TESTS:
        try:
            detail = fn(cfg)
            print("NEGTEST %s ok %s" % (name, detail), flush=True)
        except Exception as err:  # noqa: BLE001 - harness must report every test
            traceback.print_exc(file=sys.stderr)
            print("NEGTEST %s FAIL %s" % (name, err), flush=True)
            failures.append(name)

    print("NEGTEST workdir %s" % cfg.workdir, flush=True)
    if failures:
        print("NEGTEST FAIL %s" % ",".join(failures))
        return 1
    print("NEGTEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
