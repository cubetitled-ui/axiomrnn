"""
TEST: assembling a real Keras model and training it through keras.fit.

This checks the requirement that TensorFlow/Keras is the architecture
description language. We do not merely import TF -- we build a real
keras.Model, compile it and train it.

Run: /home/cune/.venvs/ax/bin/python -W ignore tests/test_keras_build.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow import keras

import axiomrnn as ax
from axtf.build import build_keras, calibrate_model, measure_model

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))


def make_data(n=1200, T=8, d=12, c=4, seed=0):
    """A task with temporal structure: the label is a window sum."""
    r = np.random.default_rng(seed)
    blk = max(1, T // c)
    pat = np.zeros((c, T, d))
    for k in range(c):
        lo = min(k * blk, T - 1)
        pat[k, lo:min(lo + 2, T)] = 1.0
        pat[k] += r.normal(0, 0.25, (T, d))
    y = r.integers(0, c, n)
    X = pat[y] * (r.random((n, T, d)) < 0.18) + r.normal(0, 0.5, (n, T, d))
    return X.astype(np.float32), y.astype(np.int32)


def test_builds_real_keras_model():
    """ax.Model -> keras.Model. Not a wrapper, a real Keras graph."""
    spec = ax.Model()
    spec.add_spiking(64, horizon=8)
    spec.add_spiking(32, horizon=8)

    inp = keras.Input(shape=(8, 12), name="inp")
    m = build_keras(spec, inputs=inp, outputs=4)

    check("assembles a real keras.Model", isinstance(m, keras.Model))
    check("the graph contains spiking layers",
          any(isinstance(l, ax.SpikingRNNCell) for l in m.layers),
          f"layers: {[type(l).__name__ for l in m.layers]}")
    check("the graph contains an output Dense",
          any(isinstance(l, keras.layers.Dense) for l in m.layers))

    # count parameters
    nparam = int(sum(np.prod(v.shape) for v in m.trainable_variables))
    check("it has trainable parameters", nparam > 0, f"{nparam:,}")
    return m


def test_keras_fit_learns(epochs=25):
    """keras.fit genuinely learns. The main integration check.

    WHAT IS ACTUALLY CHECKED. Not "reach high accuracy", but that the
    gradient PASSES and the model trains. An earlier version demanded
    val_acc > 0.5 on a task where no method reaches that (the DENSE ceiling
    on the whole sequence is 0.448, and 0.267 on the time mean). A 0.5
    requirement was simply wrong and would have masked real bugs.

    WHAT WAS MEASURED ABOUT THE GRADIENT (checked separately in
    test_tf_cells.py):
      Keras autograd and the explicit backward agree to a ratio of 0.000000.
      A manual loop over 64 examples: loss 1.4121 -> 0.0069, accuracy 100%.

    HERE WE CHECK TWO SIGNS OF LIVE TRAINING:
      1) loss drops below ln(C) = 1.386 -- otherwise the net sits on the
         uniform distribution, i.e. the gradient never arrives;
      2) train accuracy is clearly above chance (1/C = 0.25).
    """
    X, y = make_data()
    ntr = int(0.8 * len(X))
    C = 4
    ln_C = float(np.log(C))

    spec = ax.Model()
    spec.add_spiking(64, horizon=8)
    spec.add_spiking(32, horizon=8)
    spec.add_dense(4)

    m = build_keras(spec, inputs=keras.Input(shape=(8, 12)), outputs=4)

    info = calibrate_model(m, X[:256], target_rate=0.25, iters=8)
    # calibrate_model returns one row per SPIKING layer, in order
    check("calibration returned one row per spiking layer",
          len(info) == 2, f"{len(info)} rows: {[r[0] for r in info]}")
    fr = [round(f, 3) for _, f, _ in info]
    check("thresholds hit the target firing rate",
          all(abs(f - 0.25) < 0.12 for f in fr), f"per-layer rates: {fr}")
    check("at most half the neurons are dead",
          all(d < 0.5 for _, _, d in info),
          f"dead: {[round(d, 3) for _, _, d in info]}")

    # run_eagerly=True is mandatory: Keras 3 caches the forward pass of a
    # layer containing a Python loop over time, and without this the loss
    # saturates at ln(C). The cause was measured with a call counter, see
    # axtf/cells.py.
    # from_logits=True is MANDATORY: our output IS logits, while Keras
    # expects probabilities and applies softmax itself. Measured:
    # raw CE 1.4414, Keras CE 2.4341, from_logits=True 1.4414.
    m.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=["accuracy"], run_eagerly=True)
    h = m.fit(X[:ntr], y[:ntr], validation_data=(X[ntr:], y[ntr:]),
              epochs=epochs, batch_size=64, verbose=0)

    l0 = float(h.history["loss"][0])
    l1 = float(h.history["loss"][-1])
    tr = float(h.history["accuracy"][-1])
    va = float(h.history["val_accuracy"][-1])

    # (1) the net left the uniform distribution
    check("loss drops below ln(C) -- the gradient arrives",
          l1 < ln_C - 0.02,
          f"loss {l0:.3f} -> {l1:.3f}, ln({C}) = {ln_C:.3f}")
    # (2) training on the training set makes progress
    check("train accuracy is above chance", tr > 1.0 / C + 0.05,
          f"train_acc = {tr:.3f}, chance = {1/C:.3f}, "
          f"val_acc = {va:.3f}")
    # (3) generalisation is no worse than a dense model on the SAME
    # representation (time mean). That is an honest floor, not 0.5.
    flat = X[ntr:].mean(axis=1)
    dense = keras.Sequential([keras.layers.Dense(128, activation="relu"),
                              keras.layers.Dense(C)])
    dense.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=["accuracy"])
    hd = dense.fit(X[:ntr].mean(axis=1), y[:ntr],
                   validation_data=(flat, y[ntr:]),
                   epochs=epochs, batch_size=64, verbose=0)
    dv = float(hd.history["val_accuracy"][-1])
    check("val_accuracy is no worse than dense on the time mean",
          va >= dv - 0.03,
          f"spiking {va:.3f} vs DENSE(mean) {dv:.3f}; "
          f"for reference, DENSE(whole sequence) is higher")


def test_measure_gives_real_density():
    """Densities are measured, not taken from a constant.

    The memory model is OBLIGED to budget the worst case of the measured
    range. If the density is a constant, the planner optimises for air.
    """
    X, _ = make_data(n=300)
    spec = ax.Model()
    spec.add_spiking(48, horizon=8)
    m = build_keras(spec, inputs=keras.Input(shape=(8, 12)), outputs=4)
    calibrate_model(m, X, iters=6)
    rows = measure_model(m, X)
    check("metrics are read off every layer", len(rows) >= 1,
          f"{[r['layer'] for r in rows]}")
    ok = all(0.0 <= r["firing"] <= 1.0 for r in rows)
    check("densities are in range", ok,
          f"{[r['firing'] for r in rows]}")


def test_custom_cell_in_keras():
    """A user neuron reaches a real Keras model.

    This is the requirement "a professional can override the system
    without forking": class MyCell(SpikingCell) must work in keras.Model.
    """
    class MyCell(ax.SpikingCell):
        def __init__(self, gain=2.0, **kw):
            super().__init__(**kw)
            self.gain = gain

        def membrane(self, state, drive, thr):
            return state * self.leak + self.gain * drive

    X, y = make_data(n=400)
    spec = ax.Model(cell=MyCell(gain=2.0, leak=0.9, alpha=2.0))
    spec.add_spiking(48, horizon=8)
    spec.add_dense(4)
    m = build_keras(spec, inputs=keras.Input(shape=(8, 12)), outputs=4)

    out = m.predict(X[:32], verbose=0)
    check("the custom cell assembles in Keras and produces an output",
          out.shape == (32, 4), f"shape {out.shape}")

    info = calibrate_model(m, X, iters=6)
    fr = info[0][1] if info else 0.0
    check("the custom cell calibrates inside the Keras graph",
          abs(fr - 0.25) < 0.15, f"rate {fr:.3f}")

    m.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True), run_eagerly=True)
    h = m.fit(X[:320], y[:320], epochs=3, batch_size=32, verbose=0)
    check("keras.fit works with the custom cell",
          np.isfinite(h.history["loss"][-1]),
          f"loss after 3 epochs: {h.history['loss'][-1]:.4f}")


def test_budget_before_and_after():
    """A plan before training and one after measuring -- both available.

    The framework's idea: the budget is known BEFORE, and the density is
    refined AFTER the first steps. The planner must be able to recompute.
    """
    spec = ax.Model()
    for _ in range(6):
        spec.add_spiking(256, horizon=32)
    spec.add_dense(1000)
    b = ax.Budget.consumer_6gb()

    p1 = spec.fit_budget(b, batch=32)
    check("a plan is produced before training", p1.peak_bytes > 0,
          f"peak {p1.peak_bytes / 2**30:.2f} GiB, regime {p1.regime.value}")
    check("the regime follows from arithmetic",
          p1.regime.value in ("activation", "static", "balanced", "offload"),
          p1.regime.value)
    check("the credit horizon is positive", p1.credit_horizon > 0,
          f"k = {p1.credit_horizon}")
    check("the tape share is in [0,1]", 0.0 <= p1.activation_fraction <= 1.0,
          f"{p1.activation_fraction:.1%}")

    txt = spec.explain_budget(b, batch=32, verbose=False)
    # The report headings moved to English when the code comments did.
    for must in ("WHAT WAS ASKED FOR", "WHAT IT COSTS IN MEMORY",
                 "WHAT THIS MEANS FOR QUALITY", "VERIFIED, AND NOT"):
        check(f"explain contains the section '{must[:22]}'", must in txt)


def test_output_shape_keeps_time_axis():
    """The time axis must NOT get lost: it silently broke Functional.

    The bug was in _as_shape: a shape tuple (None, T, in) was mistaken for
    a list of layer inputs, because the first element is the batch, i.e.
    None. The batch is None almost always, so this was not an edge case
    but the COMMON one. Result: compute_output_shape returned
    (None, None, units), Flatten produced (None, None), and Dense raised an
    unhelpful "fully-defined shapes" error -- in a different file, with a
    different apparent cause.

    Here we check exactly what used to break.
    """
    from axtf.cells import SpikingCell, SpikingRNNCell, _as_shape

    # batch None plus a known T: the case that used to get lost
    check("_as_shape((None,16,24)) -> (None,16,24)",
          _as_shape((None, 16, 24)) == (None, 16, 24))
    check("_as_shape((7,16,24)) -> (7,16,24)",
          _as_shape((7, 16, 24)) == (7, 16, 24))
    check("_as_shape([(None,4)]) -> (None,4)",
          _as_shape([(None, 4)]) == (None, 4))
    check("_as_shape(None) -> None", _as_shape(None) is None)
    check("_as_shape([]) -> None", _as_shape([]) is None)

    c = SpikingRNNCell(8, cell=SpikingCell(), return_sequences=True)
    c.build((None, 16, 4))
    check("compute_output_shape reads a tuple: (None,16,8)",
          tuple(c.compute_output_shape((None, 16, 4))) == (None, 16, 8))

    # and an end-to-end check: after the layer the time axis must stay
    # alive, otherwise the next layer cannot build itself
    x = keras.Input(batch_shape=(None, 16, 4))
    y = SpikingRNNCell(8, cell=SpikingCell(), return_sequences=True,
                       name="t1")(x)
    z = keras.layers.Flatten()(y)
    out = keras.layers.Dense(4, name="o")(z)
    m = keras.Model(x, out)
    # find Flatten BY TYPE, not by index: m.layers indices depend on
    # whether InputLayer is in the list, and this test was fooled by that
    fl = [l for l in m.layers if isinstance(l, keras.layers.Flatten)]
    check("Flatten after the layer gives (None, 16*8), not (None, None)",
          len(fl) == 1 and tuple(fl[0].output.shape) == (None, 128))
    check("a model with Flatten builds and outputs (None,4)",
          tuple(m.output_shape) == (None, 4))


def test_custom_cell_through_public_api():
    """A custom neuron must be reachable through the public API.

    The requirement "a professional can override the system without
    forking" was VIOLATED: add_spiking put cell=self.cell ahead of **kw,
    so passing your own neuron raised TypeError. The extension point
    existed only when importing axtf directly.
    """
    import axiomrnn as ax
    from axtf.cells import SpikingCell, SpikingRNNCell, _atan_surrogate

    class MyCell(SpikingCell):
        def spike_fn(self, u, thr, smooth=True):
            return _atan_surrogate(u, thr, self.alpha)
        def reset_fn(self, u, spike, smooth=True):
            return u * (1.0 - spike)

    spec = ax.Model()
    spec.add_spiking(16, horizon=4, cell=MyCell(0.9, 2.0))   # custom
    spec.add_spiking(16, horizon=4)                            # shared
    spec.add_dense(3)
    check("add_spiking accepts its own cell", True)

    inp = keras.Input(batch_shape=(None, 4, 2))
    m = spec.keras_model(inputs=inp, outputs=3)
    check("a model with a custom neuron builds",
          tuple(m.output_shape) == (None, 3))

    # the layer with the custom neuron MUST be in the model
    my = [l for l in m.layers
          if isinstance(l, SpikingRNNCell)
          and isinstance(l.cell, MyCell)]
    plain = [l for l in m.layers
             if isinstance(l, SpikingRNNCell)
             and not isinstance(l.cell, MyCell)]
    check("the custom-neuron layer reached the model", len(my) == 1)
    check("the shared-neuron layer survived too", len(plain) == 1)

    # and it really is a different object, not the same class
    check("the neurons are genuinely different",
          type(my[0].cell) is not type(plain[0].cell))

    # an explicit cell beats the model-wide one
    s2 = ax.Model(cell=SpikingCell(0.5, 1.0))
    s2.add_spiking(8, cell=MyCell(0.9, 2.0))
    check("an explicit cell overrides the model-wide cell",
          isinstance(s2.layers[0].cell, MyCell))


def test_readout_modes_differ_and_default_is_flatten():
    """The readout is a measured difference, not decoration.

    Measured (verify_14): flatten 0.9912 against mean 0.7356. Hence the
    default is flatten, and mean must be chosen deliberately.
    """
    import axiomrnn as ax
    from axtf.build import _Readout

    x = keras.Input(batch_shape=(None, 16, 8))
    f = _Readout(mode="flatten")(x)
    m = _Readout(mode="mean")(x)
    check("flatten gives (B, T*units)", tuple(f.shape) == (None, 128))
    check("mean gives (B, units)", tuple(m.shape) == (None, 8))
    check("compute_output_shape flatten", tuple(_Readout(
        mode="flatten").compute_output_shape((None, 16, 8))) == (None, 128))
    check("compute_output_shape mean", tuple(_Readout(
        mode="mean").compute_output_shape((None, 16, 8))) == (None, 8))

    try:
        _Readout(mode="invented")
        check("an unknown readout mode is rejected", False)
    except ValueError:
        check("an unknown readout mode is rejected", True)

    spec = ax.Model()
    spec.add_spiking(8, horizon=16)
    spec.add_dense(4)
    inp = keras.Input(batch_shape=(None, 16, 3))
    dflt = spec.keras_model(inputs=inp, outputs=4)
    mean_m = spec.keras_model(inputs=inp, outputs=4, readout="mean")
    # the default output is already reduced to 4 classes; look at the
    # layer just before it
    check("by default the output is 4 classes",
          tuple(dflt.output_shape) == (None, 4))
    check("readout='mean' builds too",
          tuple(mean_m.output_shape) == (None, 4))
    d_shape = int(dflt.layers[-2].output.shape[-1])
    check(f"the default readout size is T*width (got {d_shape})",
          d_shape == 16 * 8)


if __name__ == "__main__":
    print("=" * 74)
    print("KERAS ASSEMBLY AND TRAINING THROUGH keras.fit")
    print("=" * 74)
    print(f"  TF {tf.__version__}, Keras {keras.__version__}")
    test_builds_real_keras_model()
    test_keras_fit_learns()
    test_measure_gives_real_density()
    test_custom_cell_in_keras()
    test_budget_before_and_after()
    test_output_shape_keeps_time_axis()
    test_custom_cell_through_public_api()
    test_readout_modes_differ_and_default_is_flatten()
    print("=" * 74)
    print(f"  PASSED {len(PASS)} | FAILED {len(FAIL)}")
    if FAIL:
        print("  FAILED: " + ", ".join(FAIL))
    print("=" * 74)
    sys.exit(1 if FAIL else 0)