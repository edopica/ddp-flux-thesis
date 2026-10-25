#!/usr/bin/env python3
"""Artifact-only checker for the bounded fixture timing campaign.

The checker verifies timing arithmetic, workload identity, replica state
hashes, input hashes, source-snapshot integrity, driver summaries, and the
recorded execution policy. It does not establish numerical equivalence.

Reproduction::

  python3 scripts/static_scaling/check_timing.py \\
    artifacts/logs/static-scaling/2026-10-18-timing-pilot/campaign-61314790
  python3 scripts/static_scaling/check_timing.py --self-test
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import tempfile
from pathlib import Path


SCHEMA = "static-timing-pilot-v1"
PROVENANCE_SCHEMA = "static-timing-provenance-v1"
WORKLOAD = "alternating-two-fixture-batches"
TIMING = "training-only"
FLUX_PRECISION = "FP32 strict FMA, TF32 off"
FLUX_GUARD = "presence + nonfinite loss/gradient rejection inside timed window"
LEGACY_FLUX_GUARD = "inside timed window"
SINGLE_GPU_GUARD = "not applicable: single-GPU baseline"
DEFAULT_DRIVER_MARKER = "STATIC_TIMING_campaign PASS"
FLUX_HASH_KEYS = {"param", "m", "v", "beta_power"}
PYTORCH_HASH_KEYS = {"param", "m", "v", "step"}
PYTORCH_PRECISION = {
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
}
DEFAULT_FIXTURE_SHA256 = "6614a00ba7cac588b331b65cef4240f27624944c7a9df0383fdbeee3124066b3"
DEFAULT_BUNDLE_SHA256 = "a790b03c1d68361a09171ae53870bd49ffad73c80144b2db54263bed6f9d0317"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
FLUX_REQUIRED_SOURCE = {
    "timing_flux.jl",
    "check_timing.py",
    "precision.jl",
    "phase4_common.jl",
    "model_flux.jl",
    "dataset_adapter.py",
    "replay.jl",
}
PYTORCH_REQUIRED_SOURCE = {
    "timing_torch.py",
    "phase4_torch.py",
    "model_torch.py",
    "dataset_adapter.py",
}
REQUIRED_SOURCE = FLUX_REQUIRED_SOURCE | PYTORCH_REQUIRED_SOURCE
REQUIRED_DRIVER_LINES = (
    "FLUX_PRECOMPILE_OK",
    "STATIC_TIMING_campaign PASS",
    "TIMING_DRIVER_EXIT=0",
)
SUPPORTED_FRAMEWORKS = ("flux", "pytorch")
RECORD_NAME = re.compile(r"^(flux|pytorch)-w(\d+)-r(\d+)\.json$")


def _fail(path, message):
    raise AssertionError(f"{path}: {message}")


def _require(condition, path, message):
    if not condition:
        _fail(path, message)


def _is_hex64(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path):
    _require(path.is_file(), path, "missing file")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        _fail(path, f"invalid JSON: {exc}")


def _parse_frameworks(value):
    if isinstance(value, str):
        tokens = value.split(",")
    else:
        tokens = list(value)
    frameworks = []
    for token in tokens:
        token = str(token).strip().lower()
        if not token:
            continue
        if token == "torch":
            token = "pytorch"
        _require(token in SUPPORTED_FRAMEWORKS, value,
                 f"unknown framework {token!r}")
        if token not in frameworks:
            frameworks.append(token)
    _require(bool(frameworks), value, "no frameworks selected")
    return tuple(frameworks)


def _required_sources(frameworks):
    required = set()
    if "flux" in frameworks:
        required |= FLUX_REQUIRED_SOURCE
    if "pytorch" in frameworks:
        required |= PYTORCH_REQUIRED_SOURCE
    return required


def _flux_guard_matches(recorded, requested):
    if recorded == requested:
        return True
    return requested == FLUX_GUARD and recorded == LEGACY_FLUX_GUARD


def _check_record_set(root, frameworks, repeats):
    expected = {
        f"{framework}-w{world}-r{repeat}.json"
        for framework in frameworks
        for world in (1, 2, 4)
        for repeat in range(1, repeats + 1)
    }
    actual = {
        path.name for path in root.iterdir()
        if path.is_file() and RECORD_NAME.fullmatch(path.name)
    }
    extra = sorted(actual - expected)
    _require(not extra, root, f"unexpected framework/repeat records: {extra}")


def _check_source_snapshot(root, frameworks=SUPPORTED_FRAMEWORKS):
    frameworks = _parse_frameworks(frameworks)
    manifest = root / "source-sha256.txt"
    _require(manifest.is_file(), manifest, "missing source manifest")
    source_dir = root / "source"
    _require(source_dir.is_dir(), source_dir, "missing source snapshot")
    listed = {}
    for raw in manifest.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        match = SOURCE_LINE.fullmatch(raw)
        _require(match is not None, manifest, f"malformed line: {raw!r}")
        digest, remote = match.groups()
        marker = "/source/"
        _require(marker in remote, manifest, f"path is not inside source/: {remote}")
        relative = remote.split(marker, 1)[1]
        target = source_dir / relative
        _require(target.is_file(), target, "snapshot file is missing")
        _require(_sha256(target) == digest, target, "snapshot hash mismatch")
        listed[Path(relative).name] = digest
    missing = _required_sources(frameworks) - set(listed)
    _require(not missing, manifest, f"missing required sources: {sorted(missing)}")
    return listed


def _check_driver_log(root, driver_marker=DEFAULT_DRIVER_MARKER):
    driver = root / "driver.log"
    _require(driver.is_file(), driver, "missing driver log")
    text = driver.read_text(errors="replace")
    for line in ("FLUX_PRECOMPILE_OK", driver_marker, "TIMING_DRIVER_EXIT=0"):
        _require(line in text, driver, f"missing {line!r}")


def _check_provenance(root, source_hashes, frameworks=SUPPORTED_FRAMEWORKS,
                      flux_guard=FLUX_GUARD):
    frameworks = _parse_frameworks(frameworks)
    path = root / "provenance.json"
    provenance = _load_json(path)
    _require(provenance.get("schema") == PROVENANCE_SCHEMA, path, "schema mismatch")
    timing = provenance.get("timing_source_sha256")
    _require(isinstance(timing, dict), path, "timing_source_sha256 must be an object")
    for name, digest in timing.items():
        _require(name in source_hashes, path, f"source {name} is not in the snapshot")
        _require(_is_hex64(digest), path, f"invalid hash for {name}")
        _require(source_hashes[name] == digest, path, f"recorded hash differs for {name}")
    phase4 = provenance.get("phase4_source_sha256")
    _require(isinstance(phase4, dict), path, "phase4_source_sha256 must be an object")
    drift = provenance.get("shared_helper_drift")
    _require(isinstance(drift, list) and drift, path, "shared_helper_drift must be a non-empty list")
    for name in drift:
        _require(name in timing and name in phase4, path, f"drift {name} is not recorded")
        _require(timing[name] != phase4[name], path, f"drift {name} has equal hashes")
    policy = provenance.get("execution_policy")
    _require(isinstance(policy, dict), path, "execution_policy must be an object")
    for framework in frameworks:
        _require(isinstance(policy.get(framework), dict), path, f"missing {framework} policy")
    if "flux" in frameworks:
        _require(policy["flux"].get("cudnn_deterministic") is False, path, "Flux policy must record non-deterministic algorithm selection")
        _require(_flux_guard_matches(policy["flux"].get("nonfinite_guard"), flux_guard),
                 path, f"Flux guard placement mismatch: {policy['flux'].get('nonfinite_guard')!r} != {flux_guard!r}")
    if "pytorch" in frameworks:
        _require(policy["pytorch"].get("cudnn_deterministic") is True, path, "PyTorch policy must record deterministic cuDNN")
        _require(policy["pytorch"].get("cudnn_benchmark") is False, path, "PyTorch policy must disable the cuDNN benchmark")
        _require(policy["pytorch"].get("nonfinite_guard") == "outside timed loop", path, "PyTorch guard placement mismatch")
    qualifications = provenance.get("qualifications")
    _require(isinstance(qualifications, list) and qualifications, path, "qualifications must be a non-empty list")
    return provenance


def _check_flux_record(path, record, world, fixture_sha256, bundle_sha256,
                       flux_guard=FLUX_GUARD):
    _require(record["schema"] == SCHEMA, path, "schema mismatch")
    _require(record["framework"] == "flux", path, "framework mismatch")
    _require(record["world_size"] == world, path, "world_size mismatch")
    _require(record["global_batch"] == 12, path, "global_batch mismatch")
    _require(record["updates"] == 300, path, "updates mismatch")
    _require(record["warmup"] == 20, path, "warmup mismatch")
    _require(record["workload"] == WORKLOAD, path, "workload mismatch")
    _require(record["timing"] == TIMING, path, "timing view mismatch")
    _require(record["precision"] == FLUX_PRECISION, path, "precision policy mismatch")
    if "guard_policy" in record:
        allowed = {flux_guard}
        if world == 1:
            allowed.add(SINGLE_GPU_GUARD)
        _require(record["guard_policy"] in allowed or
                 _flux_guard_matches(record["guard_policy"], flux_guard),
                 path, f"guard policy mismatch: {record['guard_policy']!r}")
    _require(record["fixture_sha256"] == fixture_sha256, path, "fixture hash mismatch")
    _require(record["bundle_sha256"] == bundle_sha256, path, "bundle hash mismatch")
    _require(record["pass_checks"] is True, path, "pass_checks is not true")
    _require(bool(record.get("job")), path, "missing job")
    _require(bool(record.get("node")), path, "missing node")
    hashes = record.get("hashes")
    _require(isinstance(hashes, dict), path, "missing state hashes")
    _require(set(hashes) == FLUX_HASH_KEYS, path, f"state hash keys mismatch: {sorted(hashes)}")
    for key, digest in hashes.items():
        _require(_is_hex64(digest), path, f"invalid {key} hash")
    durations = record.get("rank_seconds")
    _require(isinstance(durations, list), path, "missing rank_seconds")
    _require(len(durations) == world, path, "rank_seconds length mismatch")
    return durations


def _check_pytorch_record(path, record, world, fixture_sha256, bundle_sha256):
    _require(record["schema"] == SCHEMA, path, "schema mismatch")
    _require(record["framework"] == "pytorch", path, "framework mismatch")
    _require(record["world_size"] == world, path, "world_size mismatch")
    _require(record["global_batch"] == 12, path, "global_batch mismatch")
    _require(record["updates"] == 300, path, "updates mismatch")
    _require(record["warmup"] == 20, path, "warmup mismatch")
    _require(record["workload"] == WORKLOAD, path, "workload mismatch")
    _require(record["timing"] == TIMING, path, "timing view mismatch")
    _require(record["precision"] == PYTORCH_PRECISION, path, "precision policy mismatch")
    _require(record["fixture_sha256"] == fixture_sha256, path, "fixture hash mismatch")
    _require(record["bundle_sha256"] == bundle_sha256, path, "bundle hash mismatch")
    _require(record["pass_checks"] is True, path, "pass_checks is not true")
    _require(bool(record.get("job")), path, "missing job")
    _require(bool(record.get("node")), path, "missing node")
    ranks = record.get("ranks")
    _require(isinstance(ranks, list), path, "missing ranks")
    _require(sorted(x.get("rank") for x in ranks) == list(range(world)), path, "rank set mismatch")
    _require(len({x.get("device") for x in ranks}) == world, path, "device set mismatch")
    first = ranks[0].get("hashes")
    _require(isinstance(first, dict), path, "missing state hashes")
    _require(set(first) == PYTORCH_HASH_KEYS, path, f"state hash keys mismatch: {sorted(first)}")
    for key in ("param", "m", "v"):
        _require(_is_hex64(first[key]), path, f"invalid {key} hash")
    _require(first["step"] == record["updates"], path, "optimizer step mismatch")
    _require(all(x.get("hashes") == first for x in ranks), path, "replica hashes differ")
    return [x["seconds"] for x in ranks]


def check_campaign(root, repeats, fixture_sha256, bundle_sha256, job_id=None,
                   verbose=True, frameworks=SUPPORTED_FRAMEWORKS,
                   flux_guard=FLUX_GUARD,
                   driver_marker=DEFAULT_DRIVER_MARKER):
    frameworks = _parse_frameworks(frameworks)
    root = Path(root)
    _require(root.is_dir(), root, "campaign root is missing")
    _check_record_set(root, frameworks, repeats)
    source_hashes = _check_source_snapshot(root, frameworks)
    _check_driver_log(root, driver_marker)
    provenance = _check_provenance(root, source_hashes, frameworks, flux_guard)
    inferred = re.fullmatch(r"campaign-(\d+)", root.name)
    if inferred is not None:
        if job_id is not None:
            _require(inferred.group(1) == job_id, root, "job-id argument does not match the directory")
        job_id = inferred.group(1)
    identity = None
    medians = {}
    rows = []
    seen_jobs = set()
    for framework in frameworks:
        for world in (1, 2, 4):
            values = []
            for repeat in range(1, repeats + 1):
                path = root / f"{framework}-w{world}-r{repeat}.json"
                record = _load_json(path)
                if framework == "flux":
                    durations = _check_flux_record(
                        path, record, world, fixture_sha256, bundle_sha256,
                        flux_guard=flux_guard)
                else:
                    durations = _check_pytorch_record(path, record, world, fixture_sha256, bundle_sha256)
                key = tuple(record[k] for k in (
                    "schema", "global_batch", "updates", "warmup", "fixture_sha256",
                    "bundle_sha256", "workload", "timing", "node", "job",
                ))
                if identity is None:
                    identity = key
                _require(key == identity, path, "workload, node, job, or policy mismatch")
                seen_jobs.add(record["job"])
                if job_id is not None:
                    _require(record["job"] == job_id, path, "job mismatch")
                _require(len(durations) == world, path, "duration count mismatch")
                _require(all(math.isfinite(t) and t > 0 for t in durations), path, "invalid duration")
                seconds = max(durations)
                _require(math.isclose(seconds, record["seconds"], rel_tol=1e-12), path, "seconds mismatch")
                _require(math.isclose(record["images_per_second"], record["global_batch"] * record["updates"] / seconds, rel_tol=1e-12), path, "throughput mismatch")
                _require(math.isclose(record["update_seconds"], seconds / record["updates"], rel_tol=1e-12), path, "update time mismatch")
                values.append(seconds)
            _require(len(seen_jobs) == 1, root, f"multiple job IDs: {sorted(seen_jobs)}")
            median = statistics.median(values)
            medians[framework, world] = median
            rows.append((framework, world, median, min(values), max(values)))
    if inferred is not None:
        _require(seen_jobs == {job_id}, root, "job ID does not match the directory")
    if verbose:
        print("Fixture training-only pilot: batch=12, updates=300, warmup=20")
        print(f"TIMING_PROVENANCE job={next(iter(seen_jobs))} node={identity[8]} source_files={len(source_hashes)} drift={','.join(provenance['shared_helper_drift'])}")
        print("framework W median_s min_s max_s images/s update_ms speedup efficiency")
        for framework, world, median, low, high in rows:
            speedup = medians[framework, 1] / median
            print(framework, world, f"{median:.6f} {low:.6f} {high:.6f} {3600/median:.3f} {median/300*1000:.3f} {speedup:.4f} {speedup/world:.4f}")
        print("TIMING_ARTIFACT_CHECK PASS")
    return rows


def _write_synthetic(root, frameworks=SUPPORTED_FRAMEWORKS, repeats=1,
                     driver_marker=DEFAULT_DRIVER_MARKER,
                     flux_guard=LEGACY_FLUX_GUARD,
                     job="12345", node="test-node", flux_extra=None):
    frameworks = _parse_frameworks(frameworks)
    source = root / "source"
    source.mkdir(parents=True)
    files = {
        "timing_flux.jl": "flux",
        "check_timing.py": "checker",
        "precision.jl": "precision",
        "phase4_common.jl": "common-v2",
        "model_flux.jl": "model-flux",
        "dataset_adapter.py": "dataset",
        "replay.jl": "replay",
    }
    if "pytorch" in frameworks:
        files.update({
            "timing_torch.py": "torch",
            "phase4_torch.py": "p4torch-v2",
            "model_torch.py": "model-torch",
        })
    for name, text in files.items():
        (source / name).write_text(text + "\n")
    hashes = {name: _sha256(source / name) for name in files}
    manifest = "\n".join(
        f"{digest}  /remote/campaign/source/{name}" for name, digest in sorted(hashes.items())
    )
    (root / "source-sha256.txt").write_text(manifest + "\n")
    (root / "driver.log").write_text(
        f"FLUX_PRECOMPILE_OK\n{driver_marker}\nTIMING_DRIVER_EXIT=0\n")
    phase4 = {
        name: hashes[name] for name in (
            "model_flux.jl", "precision.jl", "dataset_adapter.py", "replay.jl")
    }
    if "pytorch" in frameworks:
        phase4["model_torch.py"] = hashes["model_torch.py"]
    phase4["phase4_common.jl"] = hashlib.sha256(b"common-v1\n").hexdigest()
    drift = ["phase4_common.jl"]
    if "pytorch" in frameworks:
        phase4["phase4_torch.py"] = hashlib.sha256(b"p4torch-v1\n").hexdigest()
        drift.append("phase4_torch.py")
    execution_policy = {
        "flux": {"cudnn_deterministic": False,
                 "nonfinite_guard": flux_guard},
    }
    if "pytorch" in frameworks:
        execution_policy["pytorch"] = {
            "cudnn_deterministic": True, "cudnn_benchmark": False,
            "nonfinite_guard": "outside timed loop"}
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "timing_source_sha256": hashes,
        "phase4_source_sha256": phase4,
        "shared_helper_drift": drift,
        "execution_policy": execution_policy,
        "qualifications": ["timing only; no numerical-equivalence claim"],
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for framework in frameworks:
        for world in (1, 2, 4):
            for repeat in range(1, repeats + 1):
                path = root / f"{framework}-w{world}-r{repeat}.json"
                seconds = 5.0 + world
                hashes = {key: hashlib.sha256(f"{framework}{key}".encode()).hexdigest() for key in ("param", "m", "v", "beta_power")}
                record = {
                    "schema": SCHEMA,
                    "framework": framework,
                    "world_size": world,
                    "global_batch": 12,
                    "updates": 300,
                    "warmup": 20,
                    "seconds": seconds,
                    "images_per_second": 12 * 300 / seconds,
                    "update_seconds": seconds / 300,
                    "node": node,
                    "job": job,
                    "fixture_sha256": DEFAULT_FIXTURE_SHA256,
                    "bundle_sha256": DEFAULT_BUNDLE_SHA256,
                    "workload": WORKLOAD,
                    "timing": TIMING,
                    "pass_checks": True,
                }
                if framework == "flux":
                    record["precision"] = FLUX_PRECISION
                    record["rank_seconds"] = [seconds] * world
                    record["hashes"] = hashes
                    if flux_extra is not None:
                        extra = (flux_extra(world, repeat)
                                 if callable(flux_extra) else flux_extra)
                        record.update(extra)
                else:
                    record["precision"] = dict(PYTORCH_PRECISION)
                    record["ranks"] = [
                        {
                            "rank": rank,
                            "seconds": seconds,
                            "device": rank,
                            "hashes": {key: hashlib.sha256(f"{framework}{key}".encode()).hexdigest() for key in ("param", "m", "v")} | {"step": 300},
                        }
                        for rank in range(world)
                    ]
                path.write_text(json.dumps(record, indent=2) + "\n")


def run_self_tests():
    checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        def mutate_json(root, name, fn):
            path = root / name
            record = json.loads(path.read_text())
            fn(record)
            path.write_text(json.dumps(record, indent=2) + "\n")

        def reject(name, base, mutate, job_id="12345", **kwargs):
            nonlocal checks
            copy = Path(tmp) / f"mut-{name}"
            shutil.copytree(base, copy)
            mutate(copy)
            try:
                check_campaign(
                    copy, 1, DEFAULT_FIXTURE_SHA256, DEFAULT_BUNDLE_SHA256,
                    job_id, verbose=False, **kwargs)
            except Exception:
                checks += 1
            else:
                raise AssertionError(f"self-test mutation was accepted: {name}")

        base = Path(tmp) / "campaign-12345"
        _write_synthetic(base)
        check_campaign(base, 1, DEFAULT_FIXTURE_SHA256,
                       DEFAULT_BUNDLE_SHA256, "12345", verbose=False)
        checks += 1

        reject("remove-flux-hashes", base,
               lambda root: mutate_json(root, "flux-w1-r1.json",
                                        lambda r: r.pop("hashes")))
        reject("schema", base,
               lambda root: mutate_json(root, "pytorch-w1-r1.json",
                                        lambda r: r.__setitem__("schema", "wrong")))
        reject("flux-precision", base,
               lambda root: mutate_json(root, "flux-w2-r1.json",
                                        lambda r: r.__setitem__("precision", "FP32")))
        reject("pytorch-precision", base,
               lambda root: mutate_json(root, "pytorch-w4-r1.json",
                                        lambda r: r["precision"].__setitem__("cudnn_deterministic", False)))
        reject("job", base,
               lambda root: mutate_json(root, "flux-w4-r1.json",
                                        lambda r: r.__setitem__("job", "99999")))
        reject("pass-checks", base,
               lambda root: mutate_json(root, "pytorch-w2-r1.json",
                                        lambda r: r.pop("pass_checks")))
        reject("fixture-hash", base,
               lambda root: mutate_json(root, "flux-w1-r1.json",
                                        lambda r: r.__setitem__("fixture_sha256", "0" * 64)))
        reject("duration", base,
               lambda root: mutate_json(root, "pytorch-w1-r1.json",
                                        lambda r: r.__setitem__("seconds", r["seconds"] + 1.0)))
        reject("rank-missing", base,
               lambda root: mutate_json(root, "pytorch-w2-r1.json",
                                        lambda r: r["ranks"].pop()))
        reject("source-hash", base,
               lambda root: (root / "source" / "timing_flux.jl").write_text("tampered\n"))
        reject("provenance", base,
               lambda root: (root / "provenance.json").unlink())
        reject("driver", base,
               lambda root: (root / "driver.log").write_text("TIMING_DRIVER_EXIT=0\n"))
        reject("missing-framework", base,
               lambda root: (root / "pytorch-w1-r1.json").unlink())
        reject("extra-framework-record", base,
               lambda root: shutil.copy(root / "flux-w1-r1.json",
                                        root / "pytorch-w1-r2.json"))
        reject("flux-guard", base,
               lambda root: mutate_json(
                   root, "provenance.json",
                   lambda r: r["execution_policy"]["flux"].__setitem__(
                       "nonfinite_guard", "no guard")))

        flux_base = Path(tmp) / "campaign-23456"
        _write_synthetic(flux_base, frameworks=("flux",),
                         flux_guard=FLUX_GUARD, job="23456")
        check_campaign(flux_base, 1, DEFAULT_FIXTURE_SHA256,
                       DEFAULT_BUNDLE_SHA256, "23456", verbose=False,
                       frameworks=("flux",))
        checks += 1
        reject("flux-only-missing-run", flux_base,
               lambda root: (root / "flux-w4-r1.json").unlink(),
               job_id="23456", frameworks=("flux",))
        reject("flux-only-extra-framework", flux_base,
               lambda root: shutil.copy(root / "flux-w1-r1.json",
                                        root / "pytorch-w1-r1.json"),
               job_id="23456", frameworks=("flux",))
        reject("flux-only-missing-source", flux_base,
               lambda root: (root / "source" / "timing_flux.jl").unlink(),
               job_id="23456", frameworks=("flux",))
    print(f"TIMING_ARTIFACT_SELFTEST PASS checks={checks}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fixture-sha256", default=DEFAULT_FIXTURE_SHA256)
    parser.add_argument("--bundle-sha256", default=DEFAULT_BUNDLE_SHA256)
    parser.add_argument("--job-id")
    parser.add_argument("--frameworks", default="flux,pytorch",
                        help="comma-separated subset of flux,pytorch")
    parser.add_argument("--flux-guard", default=FLUX_GUARD,
                        help="expected Flux non-finite guard policy string")
    parser.add_argument("--driver-marker", default=DEFAULT_DRIVER_MARKER,
                        help="expected campaign driver PASS marker")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    if args.root is None:
        parser.error("root is required unless --self-test is used")
    try:
        frameworks = _parse_frameworks(args.frameworks)
    except AssertionError as exc:
        parser.error(str(exc))
    check_campaign(args.root, args.repeats, args.fixture_sha256,
                   args.bundle_sha256, args.job_id,
                   frameworks=frameworks, flux_guard=args.flux_guard,
                   driver_marker=args.driver_marker)


if __name__ == "__main__":
    main()
