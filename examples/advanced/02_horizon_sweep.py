"""
02_horizon_sweep.py -- how long a sequence can you afford, and what does it buy?

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/02_horizon_sweep.py

WHAT THIS TEACHES
=================
One command line, three questions answered honestly:

  1. What does each horizon cost in memory?    -> axplan
  2. Which credit rule fits your budget?      -> the planner
  3. Does a longer horizon actually help?     -> measured, not assumed

We answer (3) by TRAINING at each horizon and reporting held-out accuracy.
That is the only honest way, because "longer sequences must help" is a
hypothesis, not a theorem.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

import axiomrnn as ax


def synthetic(n=3000, T=32, classes=4, seed=0):
    """Label = which quarter of the sequence carried the signal.

    Longer T adds more signal-bearing steps AND more distractors, so this
    is a task where a longer horizon is not automatically better. That is
    the point: we want to see the trade-off, not a monotone win.
    """
    rng = np.random.default_rng(seed)
    blk = T // classes
    pat = rng.normal(0, 0.3, (classes, T, 8))
    for k in range(classes):
        pat[k, k * blk:(k + 1) * blk] += 3.0
    y = rng.integers(0, classes, n)
    x = pat[y] * (rng.random((n, T, 8)) < 0.2) + rng.normal(0, 1.0, (n, T, 8))
    return x.astype(np.float32), y.astype(np.int32)


def plan_only(width, T, layers, budget, batch=32):
    """Pure planning -- no TF required."""
    spec = ax.Model()
    for _ in range(layers):
        spec.add_spiking(width, horizon=T)
    spec.add_dense(8)
    return spec.fit_budget(budget, batch=batch), spec


def train_at(width, T, layers, classes, seed, epochs=6, batch=64):
    """Train and report held-out accuracy. Needs TF."""
    import tensorflow as tf
    from tensorflow import keras
    from axtf.build import calibrate_model
    from axtf.cells import SpikingCell, SpikingRNNCell

    x, y = synthetic(T=T, classes=classes, seed=seed)
    cut = int(0.8 * len(x))
    xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]

    inp = keras.Input(batch_shape=(None, T, 8))
    h = inp
    for i in range(layers):
        h = SpikingRNNCell(width, cell=SpikingCell(0.9, 2.0),
                           return_sequences=True, norm_t=True,
                           horizon_hint=T, name=f"sp{i}")(h)
    h = keras.layers.Flatten()(h)
    out = keras.layers.Dense(classes, name="logits")(h)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=["accuracy"], run_eagerly=True)

    calibrate_model(m, xtr[:512], target_rate=0.2, iters=6)
    m.fit(xtr, ytr, epochs=epochs, batch_size=batch, verbose=0)
    pred = m.predict(xte, verbose=0, batch_size=512).argmax(1)
    return float((pred == yte).mean())


def main():
    print("=" * 78)
    print("HORIZON: WHAT IT COSTS, WHAT IT BUYS")
    print("=" * 78)
    print("  Part 1 needs no TensorFlow at all.")
    print()

    WIDTH, LAYERS, CLASSES = 256, 2, 4
    BUDGET = ax.Budget.consumer_6gb()
    print(f"  width={WIDTH}  layers={LAYERS}  classes={CLASSES}")
    print(f"  budget: {BUDGET.name}")
    print()
    print(f"  {'T':>5}{'peak GiB':>11}{'credit k':>10}{'regime':>12}"
          f"{'batches':>10}{'fits':>7}")
    print("  " + "-" * 63)

    rows = []
    for T in (8, 16, 32, 64, 128):
        plan, _ = plan_only(WIDTH, T, LAYERS, BUDGET)
        from axplan.credit import crossover_batch
        xb = crossover_batch(T, hidden=WIDTH, n_layers=LAYERS)
        rows.append((T, plan))
        print(f"  {T:>5}{plan.peak_bytes / 2**30:>11.2f}"
              f"{plan.credit_horizon:>10}{plan.regime.value:>12}"
              f"{xb:>10.0f}{'yes' if not plan.infeasible else 'NO':>7}")
    print()
    print("  'batches' is the crossover batch: below it, BPTT is CHEAPER")
    print("  than the local rule, so the local rule is wasted complexity.")
    print()

    # ── Part 2: the planner's actual advice ─────────────────────────────
    print("  " + "-" * 63)
    print("  WHAT THE PLANNER RECOMMENDS")
    print("  " + "-" * 63)
    spec = ax.Model()
    for _ in range(LAYERS):
        spec.add_spiking(WIDTH, horizon=64)
    spec.add_dense(CLASSES)
    txt = spec.explain(BUDGET, batch=32, verbose=False)
    for line in txt.splitlines():
        if any(k in line for k in ("credit", "BPTT", "e-prop", "crossover",
                                   "costs", "local", "regime", "peak",
                                   "batch", "Cheaper", "DEARER", "cost")):
            print("  " + line.rstrip())
    print()

    # ── Part 3: measure the quality side ────────────────────────────────
    print("  " + "-" * 63)
    print("  PART 2 -- MEASURED QUALITY (needs TensorFlow)")
    print("  " + "-" * 63)
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("  TensorFlow absent; planning above is the whole story here.")
        return 0

    print(f"  {'T':>5}{'held-out accuracy':>22}{'vs T=8':>10}")
    print("  " + "-" * 39)
    base = None
    for T in (8, 16, 32):
        acc = train_at(WIDTH, T, LAYERS, CLASSES, seed=0)
        if base is None:
            base = acc
        print(f"  {T:>5}{acc:>22.4f}{acc - base:>+10.4f}")
    print()
    print("  Read this honestly: if accuracy falls as T grows, the extra")
    print("  steps added more distractors than signal. That is a real")
    print("  result about THIS task, not a bug in the framework.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())