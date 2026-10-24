"""CPU-only regressions for launcher control flow and preflight acceptance."""
import argparse
import ast
import contextlib
import io
import json
import hashlib
import math
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
import numpy as np

import check_full_campaign as c

LAUNCHER = Path(__file__).parents[1] / "remote/leonardo/run-static-full-campaign.sh"


def shell_function(name):
    text = LAUNCHER.read_text()
    return re.search(r"^" + name + r"\(\) \{\n.*?^\}", text, re.M | re.S).group()


class LauncherTests(unittest.TestCase):
    def test_batch256_schedule_cross_language(self):
        directory = Path(__file__).parent
        compile((directory / "timing_torch.py").read_text(), "timing_torch.py", "exec")
        tree = ast.parse((directory / "model_torch.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "schedule_eta")
        scope = dict(ETA_MAX=1e-3, WARMUP_UPDATES=975, TOTAL_UPDATES=9750,
                     math=math, _require=c._require)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "schedule", "exec"), scope)
        eta = scope["schedule_eta"]
        values = np.asarray([eta(t, warmup=1950, total=19500)
                             for t in range(1, 391)], dtype="<f4")
        self.assertAlmostEqual(eta(1950, warmup=1950, total=19500), 1e-3)
        self.assertEqual(eta(19500, warmup=1950, total=19500), 0)
        julia_function = re.search(r"^function schedule_eta\(.*?^end",
                                   (directory / "model_flux.jl").read_text(), re.M | re.S).group()
        result = subprocess.run(["julia", "--startup-file=no", "-e",
            "using SHA; const ETA_MAX=1e-3; const WARMUP_UPDATES=975; const TOTAL_UPDATES=9750;\n"
            + julia_function + '\nprintln(bytes2hex(sha256(reinterpret(UInt8, '
            'Float32[schedule_eta(t; W=1950, T=19500) for t in 1:390]))))'],
            capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), hashlib.sha256(values.tobytes()).hexdigest())

    def test_torch_gpu_uuid_normalisation(self):
        directory = Path(__file__).parent
        tree = ast.parse((directory / "timing_torch.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "gpu_uuid_string")
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]),
                     "gpu_uuid", "exec"), scope)
        normalise = scope["gpu_uuid_string"]

        class Properties:
            pass

        properties = Properties()
        properties.uuid = "abc"
        self.assertEqual(normalise(properties), "GPU-abc")
        properties.uuid = "GPU-abc"
        self.assertEqual(normalise(properties), "GPU-abc")

    def test_runtime_capture_julia_parses(self):
        text = LAUNCHER.read_text()
        match = re.search(r'"\$JULIA_BIN" --startup-file=no -e \'\n(.*?)\n\s*\'',
                          text, re.S)
        self.assertIsNotNone(match)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime-capture.jl"
            path.write_text(match.group(1))
            result = subprocess.run(
                ["julia", "--startup-file=no", "-e",
                 'Meta.parseall(read(ARGS[1], String)); '
                 'println("JULIA_RUNTIME_PARSE_OK")', str(path)],
                capture_output=True, text=True, check=True)
        self.assertIn("JULIA_RUNTIME_PARSE_OK", result.stdout)

    def run_shell(self, text):
        return subprocess.run(["bash", "-c", "set -euo pipefail\n" + text],
                              capture_output=True, text=True)

    def test_precompile_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            stub = root / "precompile.sh"
            stub.write_text("printf 'FLUX_PRECOMPILE_OK\\n'\n")
            result = self.run_shell(
                f'OUT={tmp}; PRECOMPILE={stub}; WORK=x; JULIA_BIN=x; JULIA_DEPOT_PATH=x\n'
                + shell_function("precompile_variant") + '\nprecompile_variant current /unused')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "logs/precompile-current.log").exists())

    def test_batch_256_only_and_fail_closed(self):
        for process_rc, checker_rc in ((0, 0), (1, 0), (0, 1)):
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "preflight").mkdir()
                text = f'''OUT={tmp}; PYTHON_BIN=checker; CHECKER=unused
PREFLIGHT_UPDATES=20; PREFLIGHT_WARMUP=20; EPOCH=0
run_preflight_candidate() {{ [ "$1" = 256 ]; return {process_rc}; }}
checker() {{ return {checker_rc}; }}
'''
                result = self.run_shell(text + shell_function("select_preflight_batch")
                                        + '\nselect_preflight_batch')
                selected = Path(tmp) / "preflight/selected-batch.txt"
                self.assertEqual(result.returncode == 0, process_rc == checker_rc == 0)
                self.assertEqual(selected.exists(), process_rc == checker_rc == 0)
                if selected.exists():
                    self.assertEqual(selected.read_text().strip(), "256")

    def test_ready_gate_precedes_timing(self):
        text = LAUNCHER.read_text()
        branches = text.split('if [ "$MODE" = "all" ] || [ "$MODE" = "preflight" ]; then')[1]
        before_timing = branches.split('PREFLIGHT_BATCH=$SELECTED')[0]
        self.assertEqual(before_timing.count('"$CHECKER" ready --root "$OUT"'), 2)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base_record = c._write_synthetic(self.root, batch=256)
        candidate = self.root / "preflight/b256"
        (candidate / "records").mkdir(parents=True)
        (candidate / "samples").mkdir()
        for framework in c.FRAMEWORKS:
            for world in c.WORLDS:
                label = "%s-w%d" % (framework, world)
                record = self.base_record(framework, world, False, 2, 256, 2)
                (candidate / "records" / (label + ".json")).write_text(json.dumps(record))
                ranks = candidate / "samples" / (label + ".ranks")
                ranks.mkdir()
                for rank in range(world):
                    (ranks / ("rank-%d.json" % rank)).write_text(json.dumps(
                        dict(rank=rank, pid=100 + rank, gpu_uuid="GPU-%d" % rank)))
                (candidate / "samples" / (label + ".csv")).write_text("".join(
                    "TIME,%.1f\n" % stamp + "".join(
                        "%d, GPU-%d, 1000\n" % (100 + rank, rank)
                        for rank in range(world)) for stamp in (1002.0, 1006.0)))
        self.args = argparse.Namespace(root=str(self.root), candidate=[(256, str(candidate))],
                                       updates=2, warmup=2, epoch=0, select=True,
                                       out=str(self.root / "preflight/decision.json"))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(c._preflight(self.args), 0)
        self.refresh_provenance()

    def refresh_provenance(self):
        path = self.root / "provenance.json"
        doc = json.loads(path.read_text())
        doc["preflight_decision_sha256"] = c._sha256(self.root / "preflight/decision.json")
        path.write_text(json.dumps(doc))

    def test_ready_and_stale_decisions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(c._check_ready(self.args), 0)
        path = self.root / "preflight/decision.json"
        original = path.read_text()
        mutations = [
            lambda d: d["candidates"]["256"].update(status="FAIL"),
            lambda d: d["source_identity"].update(THESIS_STATE_SHA256="0" * 64),
            lambda d: d["runtime"]["python"].update(torch="changed"),
            lambda d: d["resources"].update(cpus_per_task=16),
            lambda d: d.update(epoch=1),
            lambda d: d["candidates"]["256"].update(rows=[]),
        ]
        for mutate in mutations:
            doc = json.loads(original)
            mutate(doc)
            path.write_text(json.dumps(doc))
            self.refresh_provenance()
            with self.assertRaises(AssertionError):
                c._check_ready(self.args)

    def test_missing_wrong_gpu_and_single_sample_rejected(self):
        sample = self.root / "preflight/b256/samples/flux-current-w4.csv"
        for text in ("TIME,1002.0\n100, GPU-0, 1000\nTIME,1006.0\n100, GPU-0, 1000\n",
                     "TIME,1002.0\n100, GPU-wrong, 1000\n",
                     "TIME,1002.0\n" + "".join("%d, GPU-%d, 1000\n" % (100+r, r)
                                                for r in range(4))):
            sample.write_text(text)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(c._preflight(self.args), 1)


if __name__ == "__main__":
    unittest.main()
