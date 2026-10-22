"""Artifact-only reader for the bounded fixture timing campaign."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    identity = None
    medians = {}
    rows = []
    for framework in ("flux", "pytorch"):
        for world in (1, 2, 4):
            values = []
            for repeat in range(1, a.repeats + 1):
                path = a.root / f"{framework}-w{world}-r{repeat}.json"
                r = json.loads(path.read_text())
                assert r["pass_checks"] and r["framework"] == framework and r["world_size"] == world, path
                key = tuple(r[k] for k in ("global_batch", "updates", "warmup", "fixture_sha256", "bundle_sha256", "workload", "timing", "node"))
                if identity is None:
                    identity = key
                assert key == identity, (path, "workload/node mismatch")
                assert r["global_batch"] == 12 and r["updates"] == 300 and r["warmup"] == 20, path
                durations = r.get("rank_seconds")
                if framework == "pytorch":
                    records = r["ranks"]
                    assert sorted(x["rank"] for x in records) == list(range(world)), path
                    assert len({x["device"] for x in records}) == world, path
                    assert all(x["hashes"] == records[0]["hashes"] for x in records), path
                    durations = [x["seconds"] for x in records]
                assert len(durations) == world and all(math.isfinite(t) and t > 0 for t in durations), path
                seconds = max(durations)
                assert math.isclose(seconds, r["seconds"], rel_tol=1e-12), path
                assert math.isclose(r["images_per_second"], 12*300/seconds, rel_tol=1e-12), path
                assert math.isclose(r["update_seconds"], seconds/300, rel_tol=1e-12), path
                values.append(seconds)
            median = statistics.median(values)
            medians[framework, world] = median
            rows.append((framework, world, median, min(values), max(values)))
    print("Fixture training-only pilot: batch=12, updates=300, warmup=20")
    print("framework W median_s min_s max_s images/s update_ms speedup efficiency")
    for framework, world, median, low, high in rows:
        speedup = medians[framework, 1] / median
        print(framework, world, f"{median:.6f} {low:.6f} {high:.6f} {3600/median:.3f} {median/300*1000:.3f} {speedup:.4f} {speedup/world:.4f}")
    print("TIMING_ARTIFACT_CHECK PASS")


if __name__ == "__main__":
    main()
