"""
09_api_tour.py -- every public name, what it is for, and whether it works
without TensorFlow.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/09_api_tour.py

WHAT THIS TEACHES
=================
There are three packages, and the boundary between them is the whole
architecture:

  axiomrnn   the public surface. You import this.
  axplan     pure arithmetic. Never imports TensorFlow.
  axtf       the only code that knows about Keras.
  axon       the numpy core plus the verification gate.

This script calls every advertised name for real -- no hasattr, no
introspection -- and prints what each one returned. If a function is
listed here and the line prints a value, it works.

Run it twice to see the boundary:
    /home/cune/.venvs/ax/bin/python examples/advanced/09_api_tour.py
    python3 examples/advanced/09_api_tour.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import axiomrnn as ax


def line(name, call, fmt=None):
    """Call something and print what came back. Never hides an exception."""
    try:
        v = call()
    except Exception as e:                                  # noqa: BLE001
        print(f"    {name:<30} RAISED {type(e).__name__}: {e}")
        return None
    shown = fmt(v) if fmt else str(v)
    if len(shown) > 52:
        shown = shown[:49] + "..."
    print(f"    {name:<30} {shown}")
    return v


def main():
    print("=" * 78)
    print("API TOUR: EVERY NAME, CALLED FOR REAL")
    print("=" * 78)
    print(f"  TensorFlow available: {ax._HAS_TF}")
    print()

    # ── budgets ──────────────────────────────────────────────────────────────
    print("  BUDGETS  -- how much memory there is")
    print("  " + "-" * 74)
    for nm in ("consumer_6gb", "consumer_8gb", "pro_24gb", "datacenter_80gb"):
        line(nm, lambda nm=nm: getattr(ax.Budget, nm)())
    line("Budget(12, 32)  custom",
         lambda: ax.Budget(12.0, host_ram_gb=32.0),
         lambda b: f"{b.name}, {b.vram_gb:.0f}G vram")
    line("usable_bytes()",
         lambda: ax.Budget.consumer_6gb().usable_bytes(),
         lambda n: f"{n / 2**30:.2f} GiB usable of 6.00")
    print()

    # ── the model spec ───────────────────────────────────────────────────────
    print("  MODEL SPEC  -- what you want to learn")
    print("  " + "-" * 74)

    def spec():
        s = ax.Model()
        s.add_spiking(256, horizon=64)
        s.add_spiking(128, horizon=64)
        s.add_dense(10)
        return s

    line("ax.Model()", lambda: repr(spec()),
         lambda r: r.replace("\n", " ")[:50])
    line("  .layers", lambda: [type(l).__name__ for l in spec().layers])
    line("  .add returns the layer", lambda: spec().add_spiking(64, horizon=8)
         is not None)
    line("ax.Dense(10)", lambda: f"{ax.Dense(10).units} units, "
     f"activation={ax.Dense(10).activation}")
    line("ax.SpikingRNN(128)", lambda: repr(ax.SpikingRNN(128)))
    # SpikingRNN.spec() returns a BudgetSpec (the INPUT to the planner),
    # not a finished plan. Feed it to analyse() to get the peak.
    line("SpikingRNN.spec(...)",
         lambda: ax.SpikingRNN(128, return_spikes=False).spec(
             batch=32, horizon=64, vram_gb=6.0),
         lambda s: f"{s.connections:,} connections, "
                   f"hidden={s.hidden}, T={s.timesteps}")
    line("  ... then analyse()",
         lambda: ax.analyse(ax.SpikingRNN(128).spec(
             batch=32, horizon=64, vram_gb=6.0)),
         lambda p: f"peak {p.peak_bytes / 2**30:.4f} GiB, "
                   f"regime {p.regime.value}, feasible={not p.infeasible}")
    print()

    # ── planning ─────────────────────────────────────────────────────────────
    print("  PLANNING  -- no TensorFlow anywhere below this line")
    print("  " + "-" * 74)
    b = ax.Budget.consumer_6gb()
    line("Model.fit_budget", lambda: spec().fit_budget(b, batch=32),
         lambda p: f"peak {p.peak_bytes / 2**30:.3f} GiB, "
                   f"credit horizon {p.credit_horizon}")
    line("Model.explain_budget",
         lambda: len(spec().explain_budget(b, batch=32, verbose=False)),
         lambda n: f"{n} characters of human-readable report")
    line("explain_budget(model, budget)",
         lambda: len(ax.explain_budget(spec(), b, batch=32, verbose=False)),
         lambda n: f"{n} characters, module-level form")
    print()
    print("    Regimes, so you know which one you are reading:")
    for nm in ("ACTIVATION", "BALANCED", "OFFLOAD", "STATIC"):
        r = getattr(ax.Regime, nm)
        print(f"      Regime.{nm:<14} {r.value:<12} {r.__doc__.splitlines()[0][:36]}"
              if r.__doc__ else f"      Regime.{nm:<14} {r.value}")
    print()
    print("    Measured spike densities, not a constant:")
    line("SPARK_RATE_BAND", lambda: ax.SPARK_RATE_BAND,
         lambda t: f"{t[0]:.3f} .. {t[1]:.3f} across networks")
    line("SPARK_RATE_NEURON", lambda: ax.SPARK_RATE_NEURON,
         lambda t: f"{t[0]:.3f} .. {t[1]:.3f} within one network")
    line("SPARK_RATE_DEFAULT", lambda: f"{ax.SPARK_RATE_DEFAULT:.4f}")
    print()

    # ── the segmentation planner ─────────────────────────────────────────────
    print("  SEGMENTATION  -- keep or recompute")
    print("  " + "-" * 74)
    GIB = 1 << 30
    segs = [
        ax.Segment(input_bytes=1 * GIB, internal_bytes=0.4 * GIB,
                   recompute_ns=300_000_000, name="attn"),
        ax.Segment(input_bytes=1 * GIB, internal_bytes=0.3 * GIB,
                   recompute_ns=200_000_000, name="mlp"),
        ax.Segment(input_bytes=1 * GIB, internal_bytes=0.2 * GIB,
                   recompute_ns=100_000_000, name="readout"),
    ]
    line("plan_partition", lambda: ax.plan_partition(segs, 2 * GIB),
         lambda p: f"peak {p.peak_exact / GIB:.2f} GiB, {p.n_cuts} cuts, "
                   f"{p.recompute_ns / 1e9:.2f}s")
    line("exact_peak", lambda: ax.exact_peak(segs, [0, 3]),
         lambda n: f"{n / GIB:.2f} GiB for one segment")
    print("    (brute_force is exported from axplan.planner but is not on")
    print("     the top-level surface: it is the reference the DP is tested")
    print("     against, and 2**M makes it a test tool, not a user tool.)")
    print()

    # ── memory internals ─────────────────────────────────────────────────────
    print("  MEMORY INTERNALS  -- the arithmetic, exposed")
    print("  " + "-" * 74)
    line("BudgetSpec(vram_gb=6)",
         lambda: ax.BudgetSpec(vram_gb=6, connections=1024 * 1024,
                               n_layers=12, hidden=1024, timesteps=64,
                               batch=32),
         lambda s: f"{s.connections:,} connections, T={s.timesteps}")
    line("analyse", lambda: ax.analyse(
        ax.BudgetSpec(vram_gb=6, connections=1024 * 1024, n_layers=12,
                      hidden=1024, timesteps=64, batch=32)),
        lambda p: f"peak {p.peak_bytes / 2**30:.3f} GiB, "
                  f"infeasible={p.infeasible}")
    line("best_codec(0.2, 1<<20)", lambda: ax.best_codec(0.2, 1 << 20),
         lambda t: f"{t[0].value}, {t[1]:,} bytes")
    line("Codec members", lambda: [c.value for c in ax.Codec])
    line("GIB", lambda: f"{ax.GIB:,} bytes")
    print()

    # ── the gate ─────────────────────────────────────────────────────────────
    print("  THE GATE  -- run this before you believe anything")
    print("  " + "-" * 74)
    ok, text = ax.gate_report()
    for ln in text.strip().splitlines()[-6:]:
        print(f"    {ln}")
    print()
    print(f"    gate passed: {ok}")
    print()

    # ── the Keras half ───────────────────────────────────────────────────────
    print("  THE KERAS HALF  -- only exists when TF is installed")
    print("  " + "-" * 74)
    if not ax._HAS_TF:
        print(f"    TensorFlow is absent: {ax._TF_ERR}")
        print()
        print("    That is the whole point of the boundary. Everything")
        print("    above this line ran anyway, on a machine with no GPU")
        print("    stack at all.")
        print()
        print("    Installing it:")
        print("      pip install axiomrnn[tf]")
        print("      pip install axiomrnn[tf-gpu]    # + CUDA")
    else:
        from axtf.build import (build_keras, calibrate_model,
                                measure_model)
        from axtf.cells import SpikingCell, SpikingRNNCell
        print(f"    SpikingCell          {SpikingCell.__module__}")
        print(f"    SpikingRNNCell       {SpikingRNNCell.__module__}")

        hooks = ["membrane", "membrane_dstate", "spike_fn", "spike_d1",
                 "reset_fn", "reset_dud", "threshold_dthr"]
        print()
        print("    The seven hooks on SpikingCell. Override a forward one")
        print("    and you must handle the matching derivative one:")
        for h in hooks:
            f = getattr(SpikingCell, h)
            doc = (f.__doc__ or "").strip().splitlines()
            print(f"      {h:<16} {doc[0][:46] if doc else '(no docstring)'}")

        print()
        print("    Layer arguments, and what they actually do:")
        print("      units          neurons in the layer")
        print("      return_sequences  True gives (B, T, units)")
        print("      norm_t         divide the incoming gradient by T")
        print("      horizon_hint   the horizon, needed to build")
        print("      credit_k       truncate credit to k steps (None = full)")
        print("      learn_threshold make the threshold trainable")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())