#!/usr/bin/env python3
"""Independent recheck for two Flux-only timing variants from one revision job.

The expected layout is::

  <root>/current/      STATIC_TIMING_REVISION-CURRENT PASS
  <root>/historical/   STATIC_TIMING_REVISION-HISTORICAL PASS

Each variant must be a complete Flux-only timing campaign produced by
``timing_flux.jl``: five fresh-process repeats at worlds 1, 2, and 4, plus
``source/``, ``source-sha256.txt``, ``provenance.json``, and ``driver.log``.
The per-variant checks reuse ``check_timing.check_campaign`` with
``frameworks=("flux",)`` and the variant-specific driver marker and Flux guard.

The revision-specific checks compare the two variants:

  * all 15 runs exist for both variants with identical workload policy;
  * fixture and bundle hashes agree everywhere and match the CLI when given;
  * job and node are identical and match ``--job-id`` when given;
  * world 1 records ``single_gpu`` / ``single_gpu``;
  * worlds 2 and 4 record the guarded or historical-static protocol;
  * guard-policy strings are exactly the documented strings;
  * Flux commits differ and match the CLI when given;
  * the two source worktrees (``source.flux_repo_root``) differ;
  * source hashes are present and recorded.

This checker does not establish numerical equivalence and does not create or
modify timing records. It only reads them and writes
``<root>/revision-timing-report.json``.

Reproduction::

  python3 scripts/static_scaling/check_revision_timing.py \\
    --root artifacts/logs/static-scaling/<revision-job> \\
    --job-id <JOB> --current-flux-commit <SHA> \\
    --historical-flux-commit <SHA>
  python3 scripts/static_scaling/check_revision_timing.py --self-test
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import statistics
import sys
import tempfile
from pathlib import Path

from check_timing import FLUX_PRECISION, TIMING, WORKLOAD

CURRENT_MARKER = "STATIC_TIMING_REVISION-CURRENT PASS"
HISTORICAL_MARKER = "STATIC_TIMING_REVISION-HISTORICAL PASS"
SINGLE_GUARD = "not applicable: single-GPU baseline"
GUARDED_GUARD = "presence + nonfinite loss/gradient rejection inside timed window"
STATIC_GUARD = "no presence or nonfinite distributed validation"
REPORT_SCHEMA = "static-revision-timing-v1"
WORLDS = (1, 2, 4)
REPEATS = 5
GLOBAL_BATCH = 12
UPDATES = 300
WARMUP = 20


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _is_hex64(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _load_record_set(path, repeats):
    expected = {
        f"flux-w{world}-r{repeat}.json"
        for world in WORLDS
        for repeat in range(1, repeats + 1)
    }
    actual = {
        item.name for item in path.iterdir()
        if item.is_file() and item.name.startswith("flux-w")
        and item.name.endswith(".json")
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _require(not missing, f"{path}: missing runs: {missing}")
    _require(not extra, f"{path}: extra runs: {extra}")
    records = {}
    for world in WORLDS:
        records[world] = []
        for repeat in range(1, repeats + 1):
            record_path = path / f"flux-w{world}-r{repeat}.json"
            _require(record_path.is_file(), f"{record_path}: missing file")
            try:
                record = json.loads(record_path.read_text())
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{record_path}: invalid JSON: {exc}")
            records[world].append(record)
    return records


def _all_records(records):
    for world in WORLDS:
        for record in records[world]:
            yield world, record


def _unique(values, label):
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    _require(len(unique) == 1,
             f"{label} is not identical everywhere: {unique!r}")
    return unique[0]


def _value(record, key):
    if key in record:
        return record[key]
    for container_name in ("config", "ddp_config", "ddp_configuration",
                           "metadata", "meta", "protocol"):
        container = record.get(container_name)
        if isinstance(container, dict) and key in container:
            return container[key]
    return None


def _guard_policy(record):
    value = _value(record, "guard_policy")
    if value is not None:
        return value
    policy = record.get("execution_policy")
    if isinstance(policy, dict):
        flux = policy.get("flux")
        if isinstance(flux, dict):
            return flux.get("guard_policy", flux.get("nonfinite_guard"))
    return None


def _flux_commit(record):
    values = []
    if "flux_commit" in record:
        values.append(record["flux_commit"])
    source = record.get("source")
    if isinstance(source, dict) and "flux_commit" in source:
        values.append(source["flux_commit"])
    if not values:
        return None
    _require(all(value == values[0] for value in values),
             "flux_commit and source.flux_commit differ")
    return values[0]


def _repo_root(record):
    source = record.get("source")
    if not isinstance(source, dict):
        return None
    for key in ("flux_repo_root", "repo_root", "flux_root"):
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _source_hashes(record):
    source = record.get("source")
    containers = []
    if isinstance(source, dict):
        containers.append(source)
    containers.append(record)
    for container in containers:
        for key in ("source_sha256", "source_hashes", "timing_source_sha256",
                    "file_sha256", "sha256"):
            value = container.get(key)
            if not isinstance(value, dict) or not value:
                continue
            _require(all(_is_hex64(digest) for digest in value.values()),
                     f"invalid source hash dictionary: {sorted(value)}")
            return value
    return None


def _expected_protocol(variant, world):
    if world == 1:
        return ("single_gpu", "single_gpu", SINGLE_GUARD)
    if variant == "current":
        return ("guarded", "guarded_ddp", GUARDED_GUARD)
    return ("static", "historical_static_ddp", STATIC_GUARD)


def _summarise(records, repeats):
    summary = {}
    medians = {}
    for world in WORLDS:
        durations = [float(record["seconds"]) for record in records[world]]
        _require(len(durations) == repeats, "duration count mismatch")
        median = statistics.median(durations)
        medians[world] = median
        summary[world] = {
            "repeats": repeats,
            "durations": durations,
            "median_seconds": median,
            "min_seconds": min(durations),
            "max_seconds": max(durations),
            "images_per_second": GLOBAL_BATCH * UPDATES / median,
            "update_seconds": median / UPDATES,
            "update_milliseconds": median / UPDATES * 1000.0,
        }
    for world in WORLDS:
        speedup = medians[1] / medians[world]
        summary[world]["speedup_vs_w1"] = speedup
        summary[world]["parallel_efficiency"] = speedup / world
    return summary


def _check(name, condition, detail="", checks=None):
    item = {"name": name, "ok": bool(condition), "detail": detail}
    if checks is not None:
        checks.append(item)
    return item["ok"]


def _write_report(root, report):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "revision-timing-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def check_revision(root, current_dir="current", historical_dir="historical",
                   repeats=REPEATS, fixture_sha256=None, bundle_sha256=None,
                   job_id=None, current_flux_commit=None,
                   historical_flux_commit=None, synthetic=False,
                   verbose=True):
    import check_timing

    root = Path(root)
    current_path = root / current_dir
    historical_path = root / historical_dir
    checks = []
    variants = {}
    report = {
        "schema": REPORT_SCHEMA,
        "created": _utc_now(),
        "synthetic": bool(synthetic),
        "root": str(root.resolve()),
        "inputs": {
            "current_dir": current_dir,
            "historical_dir": historical_dir,
            "repeats": repeats,
            "fixture_sha256": fixture_sha256,
            "bundle_sha256": bundle_sha256,
            "job_id": job_id,
            "current_flux_commit": current_flux_commit,
            "historical_flux_commit": historical_flux_commit,
        },
        "variants": variants,
        "checks": checks,
        "failed": [],
        "pass": False,
        "argv": sys.argv,
        "command": " ".join(shlex.quote(item) for item in sys.argv),
    }

    try:
        _require(root.is_dir(), f"{root}: revision root is missing")
        for variant, path, marker, guard in (
                ("current", current_path, CURRENT_MARKER, GUARDED_GUARD),
                ("historical", historical_path, HISTORICAL_MARKER,
                 STATIC_GUARD)):
            _require(path.is_dir(), f"{path}: variant directory is missing")
            records = _load_record_set(path, repeats)
            fixture = fixture_sha256
            if fixture is None:
                fixture = _unique(
                    [record.get("fixture_sha256")
                     for _, record in _all_records(records)],
                    f"{variant} fixture_sha256")
            bundle = bundle_sha256
            if bundle is None:
                bundle = _unique(
                    [record.get("bundle_sha256")
                     for _, record in _all_records(records)],
                    f"{variant} bundle_sha256")
            check_timing.check_campaign(
                path, repeats, fixture, bundle, job_id, verbose=False,
                frameworks=("flux",), flux_guard=guard,
                driver_marker=marker)
            variants[variant] = {
                "dir": str(path.resolve()),
                "driver_marker": marker,
                "flux_guard": guard,
                "fixture_sha256": fixture,
                "bundle_sha256": bundle,
                "records": records,
            }
    except AssertionError as exc:
        _check("variant-load", False, str(exc), checks)
        return _finish(root, report, verbose)

    try:
        current = variants["current"]["records"]
        historical = variants["historical"]["records"]
        _check("runs", True, "15 runs per variant", checks)

        all_records = list(_all_records(current)) + list(
            _all_records(historical))
        current_records = [record for _, record in _all_records(current)]
        historical_records = [
            record for _, record in _all_records(historical)]

        fixture_values = [record.get("fixture_sha256")
                          for _, record in all_records]
        bundle_values = [record.get("bundle_sha256")
                         for _, record in all_records]
        fixture = _unique(fixture_values, "fixture_sha256")
        bundle = _unique(bundle_values, "bundle_sha256")
        _check("fixture_sha256",
               fixture_sha256 is None or fixture == fixture_sha256,
               fixture, checks)
        _check("bundle_sha256",
               bundle_sha256 is None or bundle == bundle_sha256,
               bundle, checks)

        jobs = [record.get("job") for _, record in all_records]
        nodes = [record.get("node") for _, record in all_records]
        job = _unique(jobs, "job")
        node = _unique(nodes, "node")
        _check("job", job_id is None or job == job_id, str(job), checks)
        _check("node", True, str(node), checks)

        policy_ok = all(
            record.get("global_batch") == GLOBAL_BATCH
            and record.get("updates") == UPDATES
            and record.get("warmup") == WARMUP
            and record.get("workload") == WORKLOAD
            and record.get("timing") == TIMING
            and record.get("precision") == FLUX_PRECISION
            for _, record in all_records)
        _check("workload", policy_ok,
               "batch=12 updates=300 warmup=20 alternating training-only",
               checks)

        for variant, records in (("current", current),
                                 ("historical", historical)):
            for world, record in _all_records(records):
                expected_api, expected_kind, expected_guard = (
                    _expected_protocol(variant, world))
                _check(f"{variant}.w{world}.distributed_api",
                       _value(record, "distributed_api") == expected_api,
                       f"{_value(record, 'distributed_api')!r} != "
                       f"{expected_api!r}", checks)
                _check(f"{variant}.w{world}.config_kind",
                       _value(record, "config_kind") == expected_kind,
                       f"{_value(record, 'config_kind')!r} != "
                       f"{expected_kind!r}", checks)
                _check(f"{variant}.w{world}.guard_policy",
                       _guard_policy(record) == expected_guard,
                       f"{_guard_policy(record)!r} != {expected_guard!r}",
                       checks)

        current_commits = [_flux_commit(record)
                           for record in current_records]
        historical_commits = [_flux_commit(record)
                              for record in historical_records]
        current_commit = _unique(current_commits, "current flux_commit")
        historical_commit = _unique(
            historical_commits, "historical flux_commit")
        _require(isinstance(current_commit, str) and current_commit,
                 "current flux_commit is missing")
        _require(isinstance(historical_commit, str) and historical_commit,
                 "historical flux_commit is missing")
        commit_ok = current_commit != historical_commit
        if current_flux_commit is not None:
            commit_ok = commit_ok and current_commit == current_flux_commit
        if historical_flux_commit is not None:
            commit_ok = commit_ok and (
                historical_commit == historical_flux_commit)
        _check("flux_commit", commit_ok,
               f"current={current_commit} historical={historical_commit}",
               checks)

        current_roots = [_repo_root(record) for record in current_records]
        historical_roots = [_repo_root(record)
                            for record in historical_records]
        current_root = _unique(current_roots, "current flux_repo_root")
        historical_root = _unique(historical_roots,
                                  "historical flux_repo_root")
        _require(isinstance(current_root, str) and current_root,
                 "current source.flux_repo_root is missing")
        _require(isinstance(historical_root, str) and historical_root,
                 "historical source.flux_repo_root is missing")
        _check("flux_repo_root", current_root != historical_root,
               f"current={current_root} historical={historical_root}",
               checks)

        source_hashes = {}
        for variant, records in (("current", current_records),
                                 ("historical", historical_records)):
            hashes = [_source_hashes(record) for record in records]
            _require(all(item is not None for item in hashes),
                     f"{variant} source hashes are missing")
            source_hashes[variant] = hashes[0]
        _check("source_hashes", True,
               "current=%d historical=%d files" %
               (len(source_hashes["current"]),
                len(source_hashes["historical"])), checks)

        summaries = {
            "current": _summarise(current, repeats),
            "historical": _summarise(historical, repeats),
        }
        ratios = {
            str(world): (summaries["historical"][world]["median_seconds"] /
                         summaries["current"][world]["median_seconds"])
            for world in WORLDS
        }
        for variant in ("current", "historical"):
            variants[variant]["flux_commit"] = (
                current_commit if variant == "current" else historical_commit)
            variants[variant]["flux_repo_root"] = (
                current_root if variant == "current" else historical_root)
            variants[variant]["source_hashes"] = source_hashes[variant]
            variants[variant]["summary"] = summaries[variant]
        report["fixture_sha256"] = fixture
        report["bundle_sha256"] = bundle
        report["job"] = job
        report["node"] = node
        report["summaries"] = summaries
        report["historical_over_current_median_ratio"] = ratios
    except AssertionError as exc:
        _check("cross-check", False, str(exc), checks)

    return _finish(root, report, verbose)


def _finish(root, report, verbose):
    passed = bool(report["checks"]) and all(
        item["ok"] for item in report["checks"])
    report["pass"] = passed
    report["failed"] = [item["name"] for item in report["checks"]
                        if not item["ok"]]
    report["finished"] = _utc_now()
    path = _write_report(root, report)
    if verbose:
        print(f"REVISION_TIMING report={path} synthetic={report['synthetic']}")
        if "summaries" in report:
            print("variant W median_s min_s max_s images/s update_ms "
                  "speedup efficiency")
            for variant in ("current", "historical"):
                summary = report["summaries"][variant]
                for world in WORLDS:
                    item = summary[world]
                    print(variant, world,
                          f"{item['median_seconds']:.6f} "
                          f"{item['min_seconds']:.6f} "
                          f"{item['max_seconds']:.6f} "
                          f"{item['images_per_second']:.3f} "
                          f"{item['update_milliseconds']:.3f} "
                          f"{item['speedup_vs_w1']:.4f} "
                          f"{item['parallel_efficiency']:.4f}")
            print("historical/current median duration ratio " + " ".join(
                f"W{world}={report['historical_over_current_median_ratio'][str(world)]:.4f}"
                for world in WORLDS))
        for item in report["checks"]:
            if not item["ok"]:
                print(f"REVISION_TIMING_CHECK FAIL {item['name']} "
                      f"{item['detail']}")
        print("REVISION_TIMING PASS" if passed else
              "REVISION_TIMING FAIL " + " ".join(report["failed"]))
    return report


def _synthetic_extra(variant, commit, repo_root):
    def extra(world, repeat):
        api, kind, guard = _expected_protocol(variant, world)
        digest = hashlib.sha256(
            f"{variant}-{world}-{repeat}".encode()).hexdigest()
        return {
            "distributed_api": api,
            "config_kind": kind,
            "guard_policy": guard,
            "flux_commit": commit,
            "source": {
                "flux_commit": commit,
                "flux_repo_root": repo_root,
                "source_sha256": {
                    "timing_flux.jl": digest,
                    "phase4_common.jl": hashlib.sha256(
                        b"phase4_common").hexdigest(),
                },
            },
        }
    return extra


def _write_synthetic_variant(root, variant, commit, repo_root):
    import check_timing

    marker = (CURRENT_MARKER if variant == "current"
              else HISTORICAL_MARKER)
    guard = (GUARDED_GUARD if variant == "current" else STATIC_GUARD)
    check_timing._write_synthetic(
        root, frameworks=("flux",), repeats=REPEATS,
        driver_marker=marker, flux_guard=guard,
        job="77777", node="synthetic-node",
        flux_extra=_synthetic_extra(variant, commit, repo_root))


def _mutate_json(path, mutate):
    record = json.loads(path.read_text())
    mutate(record)
    path.write_text(json.dumps(record, indent=2) + "\n")


def _mutate_variant_records(root, variant, mutate):
    for path in sorted((root / variant).glob("flux-w*-r*.json")):
        _mutate_json(path, mutate)


def run_self_tests():
    current_commit = "a" * 40
    historical_commit = "b" * 40
    checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "revision"
        current_root = str(base / "worktrees" / "current")
        historical_root = str(base / "worktrees" / "historical")
        _write_synthetic_variant(base / "current", "current",
                                 current_commit, current_root)
        _write_synthetic_variant(base / "historical", "historical",
                                 historical_commit, historical_root)
        valid = check_revision(
            base, repeats=REPEATS, job_id="77777",
            current_flux_commit=current_commit,
            historical_flux_commit=historical_commit,
            synthetic=True, verbose=False)
        if not valid["pass"]:
            raise AssertionError(
                "self-test synthetic revision was rejected: "
                + ",".join(valid["failed"]))
        if valid.get("synthetic") is not True:
            raise AssertionError("self-test report is not labelled synthetic")
        checks += 1

        def reject(name, mutate):
            nonlocal checks
            import shutil

            copy = Path(tmp) / f"mut-{name}"
            shutil.copytree(base, copy)
            mutate(copy)
            report = check_revision(
                copy, repeats=REPEATS, synthetic=True, verbose=False)
            if report["pass"]:
                raise AssertionError(
                    f"self-test mutation was accepted: {name}")
            checks += 1

        reject("missing-run", lambda root: (
            root / "current" / "flux-w2-r3.json").unlink())
        reject("fixture-hash", lambda root: _mutate_json(
            root / "historical" / "flux-w1-r2.json",
            lambda record: record.__setitem__("fixture_sha256", "0" * 64)))
        reject("distributed-api", lambda root: _mutate_variant_records(
            root, "current",
            lambda record: record.__setitem__("distributed_api", "static")))
        reject("guard-policy", lambda root: _mutate_variant_records(
            root, "historical",
            lambda record: record.__setitem__("guard_policy", "no guard")))
        reject("job", lambda root: _mutate_json(
            root / "current" / "flux-w4-r1.json",
            lambda record: record.__setitem__("job", "99999")))
        reject("node", lambda root: _mutate_json(
            root / "historical" / "flux-w2-r2.json",
            lambda record: record.__setitem__("node", "other-node")))
        reject("same-commit", lambda root: _mutate_variant_records(
            root, "historical",
            lambda record: (
                record.__setitem__("flux_commit", current_commit),
                record["source"].__setitem__("flux_commit", current_commit))))
        reject("same-repo-root", lambda root: _mutate_variant_records(
            root, "historical",
            lambda record: record["source"].__setitem__(
                "flux_repo_root", current_root)))
        reject("historical-guarded", lambda root: _mutate_variant_records(
            root, "historical",
            lambda record: (
                record.__setitem__("distributed_api", "guarded"),
                record.__setitem__("config_kind", "guarded_ddp"),
                record.__setitem__("guard_policy", GUARDED_GUARD))))
    print(f"REVISION_TIMING_SELFTEST PASS checks={checks}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Recheck two Flux-only timing variants from one revision "
                    "job")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--current-dir", default="current")
    parser.add_argument("--historical-dir", default="historical")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--fixture-sha256")
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--job-id")
    parser.add_argument("--current-flux-commit")
    parser.add_argument("--historical-flux-commit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_tests()
        return 0
    if args.root is None:
        parser.error("--root is required unless --self-test is used")
    report = check_revision(
        args.root, current_dir=args.current_dir,
        historical_dir=args.historical_dir, repeats=args.repeats,
        fixture_sha256=args.fixture_sha256,
        bundle_sha256=args.bundle_sha256, job_id=args.job_id,
        current_flux_commit=args.current_flux_commit,
        historical_flux_commit=args.historical_flux_commit)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
