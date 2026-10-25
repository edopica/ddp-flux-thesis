#!/usr/bin/env python3
"""CPU-only adversarial tests for cache diagnostic artifact acceptance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import check_cache_diagnostic as cache
import check_full_campaign as full


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def fixture(root):
    full._write_synthetic(root, repeats=2, batch=256)
    labels = ["%s-w1-r%d" % (framework, repeat) for framework in cache.FRAMEWORKS for repeat in (1, 2)]
    for path in (root / "records").glob("*.json"):
        if path.stem not in labels:
            path.unlink()
    identity = full._parse_pairs((root / "source-identity.txt").read_text())
    for name in ("cache_diagnostic.jl", "cache_diagnostic.py", "check_cache_diagnostic.py"):
        (root / "source" / name).write_text("synthetic " + name)
    manifest = dict(schema="static-cache-campaign-v1", repeats=2,
                    frameworks=list(cache.FRAMEWORKS), world_size=1,
                    global_batch=256, warmup=20, updates=20, cpus_per_task=4, epoch=0,
                    source_identity=identity, job="12345", node="synthetic-node")
    manifest["source_files"] = {str(path.relative_to(root)): full._sha256(path)
                                for path in (root / "source").rglob("*") if path.is_file()}
    write_json(root / "manifest.json", manifest)
    write_json(root / "resources.json", dict(schema="static-cache-resources-v1", cpus_per_task=4,
                                             ntasks=1, job="12345", node="synthetic-node"))
    write_json(root / "runtime-python.json", dict(python="3.11.17 synthetic", executable="/python",
            torch="2.9.1+cu126", cuda="12.6", cudnn=90000, torch_path="/torch/__init__.py"))
    for index, label in enumerate(labels):
        path = root / "records" / (label + ".json")
        record = full._load_json(path)
        record.update(updates=20, warmup=20, node="synthetic-node",
                      images_per_second=512.0, update_seconds=0.5)
        if record["framework"] == "pytorch":
            record["ranks"][0]["hashes"]["step"] = 20
            hashes = record["ranks"][0]["hashes"]
        else:
            hashes = record["hashes"]
        snapshots = []
        for number, phase in enumerate(cache.PHASES):
            epoch = (990, 995, 999, 1011, 1012, 1013)[number]
            counters = dict(allocator_used_bytes=100, allocator_reserved_bytes=200,
                            process_bytes=(500 - 20 * number) * 1024**2,
                            device_used_bytes=800 * 1024**2, total_bytes=64 * 1024**3)
            snapshots.append(dict(phase=phase, epoch=epoch, end_epoch=epoch + 0.1,
                                  counter_status={key: "measured" for key in counters}, **counters))
        record["cache_diagnostic"] = dict(schema="static-cache-diagnostic-v1", pid=100 + index,
            device_uuid="GPU-0", retention_policy="same model optimizer dataset post_training post_gc post_reclaim",
            peak_policy="point samples, not peaks", counter_semantics={key: "explicit semantics" for key in cache.METRICS},
            runtime_api_evidence={key: dict(api="loaded API", doc="loaded documentation") for key in cache.METRICS[:2]},
            state_invariant=dict(before=hashes, after=hashes, passed=True), snapshots=snapshots)
        write_json(path, record)
        full._write_gpu_trace(root / "samples" / (label + ".csv"), full._synthetic_trace(1), 1, pid_base=100 + index)


class CacheDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture(self.root)
        self.record_path = self.root / "records/flux-current-w1-r1.json"

    def mutate_record(self, function):
        record = full._load_json(self.record_path)
        function(record)
        write_json(self.record_path, record)

    def reject(self, function, message):
        self.mutate_record(function)
        with self.assertRaisesRegex(AssertionError, message):
            cache.check_campaign(self.root)

    def test_complete_artifact_passes(self):
        report = cache.check_campaign(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["runs"]), 4)
        self.assertEqual(report["runs"][0]["process_cache_release_drop_bytes"], 20 * 1024**2)

    def test_missing_phase(self):
        self.reject(lambda r: r["cache_diagnostic"]["snapshots"].pop(), "phases")

    def test_snapshot_during_training(self):
        self.reject(lambda r: r["cache_diagnostic"]["snapshots"][3].update(epoch=1009, end_epoch=1009.1), "precedes")

    def test_invalid_counter(self):
        for value in (True, -1, 1.2, float("nan"), None):
            with self.subTest(value=value):
                original = self.record_path.read_text()
                self.reject(lambda r: r["cache_diagnostic"]["snapshots"][0].update(process_bytes=value), "invalid measured")
                self.record_path.write_text(original)

    def test_missing_counter_reason(self):
        self.reject(lambda r: r["cache_diagnostic"]["snapshots"][0]["counter_status"].update(allocator_used_bytes="unavailable"), "unavailability")

    def test_unavailable_counter_is_explicit(self):
        def disable(record):
            for row in record["cache_diagnostic"]["snapshots"]:
                row["allocator_used_bytes"] = None
                row["counter_status"]["allocator_used_bytes"] = "unavailable: backend does not expose this counter"
        self.mutate_record(disable)
        self.assertEqual(cache.check_campaign(self.root)["status"], "PASS")

    def test_pid_mismatch(self):
        self.reject(lambda r: r["cache_diagnostic"].update(pid=9999), "PID differs")

    def test_gpu_mismatch(self):
        self.reject(lambda r: r["cache_diagnostic"].update(device_uuid="GPU-wrong"), "GPU differs")

    def test_retained_state_mutation(self):
        self.reject(lambda r: r["cache_diagnostic"]["state_invariant"].update(after={}), "state invariant")

    def test_allocator_accounting(self):
        self.reject(lambda r: r["cache_diagnostic"]["snapshots"][0].update(allocator_used_bytes=300), "used exceeds reserved")

    def test_missing_run(self):
        self.record_path.unlink()
        with self.assertRaisesRegex(AssertionError, "record set"):
            cache.check_campaign(self.root)

    def test_source_tampering(self):
        (self.root / "source/cache_diagnostic.jl").write_text("tampered")
        with self.assertRaisesRegex(AssertionError, "snapshot hash mismatch"):
            cache.check_campaign(self.root)

    def test_actual_resource_policy(self):
        path = self.root / "resources.json"
        resources = full._load_json(path)
        resources["cpus_per_task"] = 32
        write_json(path, resources)
        with self.assertRaisesRegex(AssertionError, "actual resource"):
            cache.check_campaign(self.root)

    def test_input_identity_mismatch(self):
        self.reject(lambda r: r.update(sample_order_sha256="0" * 64), "sample order hash differs")

    def test_insufficient_trace_density(self):
        path = self.root / "samples/flux-current-w1-r1.csv"
        full._write_gpu_trace(path, [(999, [500]), (1000, [500]), (1010, [500]), (1011, [500])], 1)
        with self.assertRaisesRegex(AssertionError, "training-window samples"):
            cache.check_campaign(self.root)


if __name__ == "__main__":
    unittest.main()
