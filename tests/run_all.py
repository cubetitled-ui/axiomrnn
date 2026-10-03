#!/usr/bin/env python3
"""
ALL FRAMEWORK TESTS, ONE COMMAND.

    python3 axiomrnn/tests/run_all.py

HARD RULE: no result is published until this is green. That rule exists
because three experiments in a row looked convincing while being wrong,
and only this check caught them.

Suites that NEED TensorFlow are marked and skipped with an explicit
notice -- skipping silently is not allowed.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PY = sys.executable
# separate venv holding TF: TF has no build for 3.14, lives in .venvs/ax
TF_PY = "/home/cune/.venvs/ax/bin/python"

SUITES = [
    ("GATE (numpy core, gradient proven)", ["python3", "-W", "ignore",
                                              "axon/gate.py"], False),
    ("PLANNER (DP optimal vs brute force)", ["python3", "-W", "ignore",
                                                   "tests/test_planner.py"], False),
    ("MEMORY MODEL (31 tests, Korthikanti)", ["python3", "-W", "ignore",
                                                       "../verify/test_cost.py"], False),
    ("CREDIT RULES (e-prop vs BPTT, quality price)",
     ["python3", "-W", "ignore", "tests/test_credit.py"], False),
    ("TF CORE (gradient, normalisation, custom cell)",
     [TF_PY, "-W", "ignore", "tests/test_tf_cells.py"], True),
    ("KERAS ASSEMBLY + training via keras.fit",
     [TF_PY, "-W", "ignore", "tests/test_keras_build.py"], True),
    # API audit. It exists because add_spiking(cell=...) was broken and
    # nobody noticed: nobody ever called the function.
    ("API AUDIT (every advertised function called)",
     [TF_PY, "-W", "ignore", "tests/test_api_surface.py"], True),
    ("C++ KERNELS (bit-packing, layout)",
     ["/bin/sh", "-c",
      "cd axcore && g++ -O3 -march=native -std=c++20 -include cstring "
      "kernels.cpp -o /tmp/axkernels 2>/dev/null && /tmp/axkernels"], False),
    # Examples inside the tests are a necessity, not belt-and-braces:
    # 01_minimal had already drifted from reality (it failed on import), and
    # the tropa files drifted so far from the core that they became
    # untrustworthy. An example nobody runs lies.
    ("EXAMPLE 02 (budget, NO TensorFlow)",
     ["python3", "-W", "ignore", "examples/02_budget.py"], False),
    ("EXAMPLE 01 (train a network)",
     [TF_PY, "-W", "ignore", "examples/01_minimal.py"], True),
    ("EXAMPLE 03 (custom neuron, no fork)",
     [TF_PY, "-W", "ignore", "examples/03_custom_neuron.py"], True),
    # T9 trains a real language model on 115k words. It is the slowest
    # suite here, so it runs last; if it times out the rest already ran.
    ("EXAMPLE 05 (T9 next-word prediction)",
     [TF_PY, "-W", "ignore", "examples/05_t9.py"], True),
    # Advanced set. The planning-only ones run on python3 with no TF,
    # which also exercises the axplan/axtf boundary from the other side.
    ("ADV 01 (embedding + stacked layers)",
     [TF_PY, "-W", "ignore", "examples/advanced/01_embedding_and_layers.py"],
     True),
    ("ADV 02 (horizon sweep)", ["python3", "-W", "ignore",
                                 "examples/advanced/02_horizon_sweep.py"], False),
    ("ADV 03 (credit rule, the real lever)",
     ["python3", "-W", "ignore",
      "examples/advanced/03_credit_rule.py"], False),
    ("ADV 04 (budget, codec, precision)",
     ["python3", "-W", "ignore",
      "examples/advanced/04_budget_and_precision.py"], False),
    ("ADV 05 (segmentation planner)",
     ["python3", "-W", "ignore",
      "examples/advanced/05_segmentation_planner.py"], False),
    ("ADV 06 (devices: CPU and GPU)",
     [TF_PY, "-W", "ignore", "examples/advanced/06_devices.py"], True),
    ("CAPACITY SOLVER (axplan.solve)",
     ["python3", "-W", "ignore", "tests/test_solve.py"], False),
    ("ADV 15 (what to build: the inverse question)",
     ["python3", "-W", "ignore",
      "examples/advanced/15_what_to_build.py"], False),
]


def run(name, argv, need_tf):
    print("\n" + "=" * 74)
    print(name)
    print("=" * 74)
    exe = argv[0]
    if need_tf and not os.path.exists(exe):
        print(f"  SKIPPED: {exe} not found")
        print("  Only this suite needs TensorFlow. axplan works without it.")
        return "SKIP"
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                       timeout=2400)
    dt = time.time() - t0
    out = (r.stdout or "") + (r.stderr or "")
    # print only what matters
    for ln in out.splitlines():
        skip = ("oneDNN", "cpu_feature", "absl::", "TF-TRT",
                "To enable the following", "port.cc", "external/local")
        # Filter keywords cover BOTH scripts' output: some suites still
        # print Russian summary lines, and dropping them would hide a
        # failure. Both wordings are matched deliberately.
        if any(k in ln for k in ("PASS", "FAIL", "PASSED", "FAILED",
                                 "PASSED", "FAILED",
                                 "ok]", "[ok]", "ALL TESTS", "Error",
                                 "error", "Traceback", "SKIPPED")) \
                and not any(k in ln for k in skip):
            print("  " + ln.strip())
    print(f"  -> code {r.returncode}, {dt:.1f}s")
    return "OK" if r.returncode == 0 else "FAIL"


def main():
    print(f"python: {PY}")
    print(f"TF venv: {TF_PY} "
          f"({'found' if os.path.exists(TF_PY) else 'MISSING'})")
    results = []
    for name, argv, need_tf in SUITES:
        results.append((name, run(name, argv, need_tf)))

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for name, st in results:
        mark = {"OK": "  OK  ", "FAIL": " FAIL", "SKIP": "SKIP"}[st]
        print(f"  [{mark}] {name}")
    bad = [n for n, s in results if s == "FAIL"]
    print("=" * 74)
    if bad:
        print(f"  FAILURES: {len(bad)}. Nothing may be published.")
        return 1
    print("  ALL GREEN. Results may be published.")
    return 0


if __name__ == "__main__":
    sys.exit(main())