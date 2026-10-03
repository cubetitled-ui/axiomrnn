"""
07_silence_diagnosis.py -- what to do when a spiking network learns nothing.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/07_silence_diagnosis.py

WHAT THIS TEACHES
=================
A spiking network that will not train is almost never a broken gradient.
It is a silent network: potentials stay below the threshold, nothing
spikes, and the gradient that reaches the weights is exactly zero.

This is the single most common way a beginner loses an afternoon. The
symptom is identical to a hundred other problems -- "it does not learn" --
and the cause is one number: the firing rate.

The script walks a network from dead to alive and prints the diagnostics
at each step, so you can recognise your own situation in the output.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

import axiomrnn as ax
from axtf.build import calibrate_model, measure_model


def dataset(n=3000, T=16, D=12, C=4, seed=0):
    rng = np.random.default_rng(seed)
    blk = T // C
    pat = rng.normal(0, 0.4, (C, T, D))
    for k in range(C):
        pat[k, k * blk:(k + 1) * blk] += 2.0
    y = rng.integers(0, C, n)
    x = (pat[y] * (rng.random((n, T, D)) < 0.3)
         + rng.normal(0, 0.6, (n, T, D)))
    return x.astype("float32"), y.astype("int32")


def build(T, width=128, threshold=None):
    """Assemble a network, optionally forcing an absurd threshold."""
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
    from axtf.cells import SpikingCell, SpikingRNNCell

    inp = keras.Input(batch_shape=(None, T, 12))
    x = inp
    for i in range(2):
        cell = SpikingRNNCell(width, cell=SpikingCell(0.9, 2.0),
                              return_sequences=True, norm_t=True,
                              horizon_hint=T, name=f"sp{i}")
        cell.build((None, T, 12 if i == 0 else width))
        if threshold is not None:
            cell.thr.assign(np.full(cell.thr.shape, float(threshold)))
        x = cell(x)
    x = keras.layers.Flatten()(x)
    out = keras.layers.Dense(4, name="logits")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"], run_eagerly=True)
    return m


def trial(label, threshold, calibrate_to=None, epochs=4, T=16, C=4):
    """Build, optionally calibrate, train, and report what happened."""
    X, Y = dataset(T=T, C=C)
    m = build(T, threshold=threshold)
    rows = measure_model(m, X[:512])
    before = [(r["firing"], r["dead"]) for r in rows]

    if calibrate_to is not None:
        calibrate_model(m, X[:512], target_rate=calibrate_to, iters=6)
    after = [(r["firing"], r["dead"]) for r in measure_model(m, X[:512])]

    h = m.fit(X, Y, epochs=epochs, batch_size=64, verbose=0)
    acc = float(h.history["accuracy"][-1])

    def fmt(pair):
        return "  ".join(f"{f:.4f}/{d:.2f}" for f, d in pair)

    print(f"  {label}")
    print(f"    firing/dead before  {fmt(before)}")
    print(f"    firing/dead after   {fmt(after)}")
    print(f"    loss {h.history['loss'][0]:.4f} -> "
          f"{h.history['loss'][-1]:.4f}   accuracy {acc:.4f} "
          f"(chance {1.0 / C:.4f})")
    print()
    return acc, max(f for f, _ in after)


def main():
    print("=" * 78)
    print("DIAGNOSING A SILENT SPIKING NETWORK")
    print("=" * 78)
    print("  Each row is firing rate / dead fraction, per spiking layer.")
    print("  Chance accuracy is 0.2500 for 4 classes.")
    print()

    print("  " + "-" * 74)
    print("  CASE 1. THRESHOLD FAR TOO HIGH -- the classic failure")
    print("  " + "-" * 74)
    acc1, rate1 = trial("threshold forced to 8.0 (nothing can reach it)",
                        threshold=8.0)
    silent = rate1 == 0.0
    if silent:
        print("    -> firing is exactly 0. The network cannot train and no")
        print("       amount of epochs will change that: there is no signal.")
        print("       The fix is calibration, not a different learning rate.")

    print("  " + "-" * 74)
    print("  CASE 2. THE SAME NETWORK, CALIBRATED")
    print("  " + "-" * 74)
    acc2, rate2 = trial("same weights, calibrate_model(target_rate=0.2)",
                        threshold=None, calibrate_to=0.2)
    print(f"    -> {acc2 - acc1:+.4f} accuracy from one calibration call.")
    print("       No architecture change, no hyperparameter change.")

    print("  " + "-" * 74)
    print("  CASE 3. THRESHOLD NEAR THE MIDDLE -- it just works")
    print("  " + "-" * 74)
    acc3, rate3 = trial("threshold 0.5, no calibration", threshold=0.5)

    print("  " + "-" * 74)
    print("  HOW TO READ YOUR OWN NUMBERS")
    print("  " + "-" * 74)
    print(f"""
    firing < 0.01   the layer is silent. Call calibrate_model(). If it
                    stays silent, the drive is too small: widen the layer
                    or raise the input scale, do not lower the threshold
                    forever -- there is a point where every neuron fires
                    and the temporal structure is gone.

    dead  > 0.5     more than half the neurons never fire. Same cause,
                    and the same fix. A dead neuron contributes nothing
                    and its incoming weights only collect gradient noise.

    0.01 - 0.10     sparse but alive. Workable, though a slow start: the
                    forward signal is small, so early training is noisy.

    0.10 - 0.40     the healthy band, and the default target is 0.25.

    > 0.50          everything fires. The code is now close to a dense
                    layer: the spike carries little information and the
                    temporal structure is washed out. Raise the threshold.

  MEASURED HERE:
    forced threshold 8.0   firing 0.0000   accuracy {acc1:.4f}
    after calibration      firing {rate2:.4f}   accuracy {acc2:.4f}
    threshold 0.5          firing {rate3:.4f}   accuracy {acc3:.4f}

  One number tells you which problem you have. Read it before you change
  anything else.
""")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())