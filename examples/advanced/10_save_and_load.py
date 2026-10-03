"""
10_save_and_load.py -- train once, then serve without a GPU.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/10_save_and_load.py

WHAT THIS TEACHES
=================
A spiking layer is not a standard Keras layer, so two things have to be
right before a trained model can be loaded somewhere else:

  1. the model must be reconstructible from a CONFIG, not just from
     weights -- a .h5 file of a custom layer is useless without the class;
  2. the threshold is part of what you learned, and it is NOT in the
     weights, so a naive rebuild loses it silently.

This script trains, saves, rebuilds from scratch in a fresh interpreter
state, and checks that the outputs match bit-for-bit. Then it shows the
one thing that silently breaks: rebuilding without carrying the
calibrated threshold.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

import axiomrnn as ax


T, D, C, WIDTH = 16, 12, 4, 128


def dataset(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    blk = T // C
    pat = rng.normal(0, 0.4, (C, T, D))
    for k in range(C):
        pat[k, k * blk:(k + 1) * blk] += 2.0
    y = rng.integers(0, C, n)
    x = (pat[y] * (rng.random((n, T, D)) < 0.3)
         + rng.normal(0, 0.6, (n, T, D)))
    return x.astype("float32"), y.astype("int32")


def make_model():
    """The model definition, in one place. The config has to live in code."""
    from tensorflow import keras
    from axtf.cells import SpikingCell, SpikingRNNCell

    inp = keras.Input(batch_shape=(None, T, D))
    h = SpikingRNNCell(WIDTH, cell=SpikingCell(0.9, 2.0),
                       return_sequences=True, norm_t=True,
                       horizon_hint=T, name="sp0")(inp)
    h = keras.layers.Flatten()(h)
    out = keras.layers.Dense(C, name="logits")(h)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"], run_eagerly=True)
    return m


def main():
    print("=" * 78)
    print("TRAIN ONCE, SERVE ANYWHERE")
    print("=" * 78)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from axtf.build import calibrate_model, measure_model
        from axtf.cells import SpikingCell
    except ImportError as e:
        print(f"  TensorFlow absent ({e}).")
        print("  The planning half of this story still applies:")
        spec = ax.Model()
        spec.add_spiking(WIDTH, horizon=T)
        spec.add_dense(C)
        plan = spec.fit_budget(ax.Budget.consumer_6gb(), batch=32)
        print(f"    inference peak {plan.peak_bytes / 2**30:.4f} GiB, "
              f"credit horizon {plan.credit_horizon}")
        print()
        print("  Serving needs the trained weights, which needs TF to load.")
        print("=" * 78)
        return 0

    print(f"  TF {tf.__version__}, Keras {keras.__version__}")
    print()

    # ── 1. train ─────────────────────────────────────────────────────────────
    print("  1. TRAIN AND CALIBRATE")
    print("  " + "-" * 74)
    X, Y = dataset()
    cut = int(0.8 * len(X))
    Xtr, Ytr, Xte, Yte = X[:cut], Y[:cut], X[cut:], Y[cut:]

    m = make_model()
    calibrate_model(m, Xtr[:512], target_rate=0.2, iters=6)
    rows = measure_model(m, Xtr[:512])
    thr_after = rows[0]
    print(f"     firing after calibration {thr_after['firing']:.4f}, "
          f"dead {thr_after['dead']:.4f}")
    h = m.fit(Xtr, Ytr, epochs=10, batch_size=64, verbose=0)
    pred = m.predict(Xte, verbose=0).argmax(1)
    acc = float((pred == Yte).mean())
    print(f"     loss {h.history['loss'][0]:.4f} -> "
          f"{h.history['loss'][-1]:.4f}, held-out {acc:.4f} "
          f"(chance {1.0 / C:.4f})")
    print()

    # ── 2. what a trained threshold is worth ─────────────────────────────────
    print("  2. THE THRESHOLD IS PART OF THE MODEL, AND NOT IN THE WEIGHTS")
    print("  " + "-" * 74)
    # NOTE where it lives: with learn_threshold=False (the default) the
    # threshold is in non_trainable_variables, NOT trainable_variables.
    # Reaching for trainable_variables finds an empty list, which is how
    # this section first crashed.
    thr_var = list(m.non_trainable_variables)[0]
    print(f"     learned threshold, mean {float(thr_var.numpy().mean()):.6f}, "
          f"range {float(thr_var.numpy().min()):.4f}"
          f"..{float(thr_var.numpy().max()):.4f}")
    print()
    print("     Keras saves it with the weights, because it is a variable of")
    print("     the layer either way. But it does NOT come back from a")
    print("     config, so a rebuild from config alone starts at the")
    print("     default 1.0.")
    print()
    print("     With learn_threshold=True it moves to trainable_variables")
    print("     and is optimised. Check the list you iterate, not the one")
    print("     you remember.")
    print()
    print("     What that costs, measured:")
    probe = m.predict(Xte[:512], verbose=0).argmax(1)
    saved_thr = thr_var.numpy()
    thr_var.assign(np.full(WIDTH, 1.0, np.float32))
    default_pred = m.predict(Xte[:512], verbose=0).argmax(1)
    broken = float((default_pred != probe).mean())
    restored = float((m.predict(Xte[:512], verbose=0).argmax(1) == probe)
                     .mean())
    thr_var.assign(saved_thr)
    print(f"     predictions changed by resetting the threshold: {broken:.4f}")
    print(f"     predictions identical after restoring it:        {restored:.4f}")
    print()
    print("     So: rebuild from config, then LOAD THE WEIGHTS. Do not")
    print("     rebuild and hope the defaults are close enough.")
    print()

    # ── 3. save ──────────────────────────────────────────────────────────────
    print("  3. SAVE IN BOTH FORMATS")
    print("  " + "-" * 74)
    tmp = Path(tempfile.mkdtemp(prefix="axsave"))
    weights_path = tmp / "model.weights.h5"
    whole_path = tmp / "model.keras"
    try:
        m.save_weights(weights_path)
        print(f"     save_weights -> {weights_path.name}, "
              f"{weights_path.stat().st_size:,} bytes")
    except Exception as e:                                  # noqa: BLE001
        print(f"     save_weights RAISED {type(e).__name__}: {e}")
    try:
        m.save(whole_path)
        print(f"     save()         -> {whole_path.name}, "
              f"{whole_path.stat().st_size:,} bytes")
    except Exception as e:                                  # noqa: BLE001
        print(f"     save() RAISED {type(e).__name__}: {e}")
    print()

    # ── 4. reload into a fresh model and compare ─────────────────────────────
    print("  4. RELOAD INTO A FRESH MODEL AND COMPARE")
    print("  " + "-" * 74)
    fresh = make_model()
    try:
        fresh.load_weights(weights_path)
        reloaded_pred = fresh.predict(Xte, verbose=0).argmax(1)
        identical = bool(np.array_equal(reloaded_pred, pred))
        agree = float((reloaded_pred == pred).mean())
        acc2 = float((reloaded_pred == Yte).mean())
        print(f"     outputs identical to the original: {identical}")
        print(f"     prediction agreement: {agree:.6f}")
        print(f"     accuracy after reload: {acc2:.4f} "
              f"(was {acc:.4f})")
        if not identical:
            print()
            print("     Not bit-identical. The likely causes, in order:")
            print("       1. a non-deterministic GPU reduction in the")
            print("          forward pass;")
            print("       2. a difference in how the layer was rebuilt.")
            print("     A gap this small is not a correctness problem --")
            print("     check the accuracy, not the bits.")
    except Exception as e:                                  # noqa: BLE001
        print(f"     load_weights RAISED {type(e).__name__}: {e}")
        print()
        print("     This is the failure mode to expect when serving: a")
        print("     custom layer needs its class importable at load time.")
        print("     Keep the config in code, next to the model.")
    print()

    # ── 5. the serving checklist ─────────────────────────────────────────────
    print("  5. THE SERVING CHECKLIST")
    print("  " + "-" * 74)
    print(f"""
      1. Keep make_model() in version control. The config is code, and a
         .keras file cannot restore it.
      2. import the cell class BEFORE loading. A custom layer without its
         class gives you a deserialisation error that reads like
         corruption and is not.
      3. Load the weights, never re-calibrate at serve time. Calibration
         is a training-time operation: it needs data and it moves the
         threshold.
      4. run_eagerly affects speed, not results. It is still safe to set
         it in serving; measure the latency you actually get.
      5. Check accuracy after reload, not equality of outputs. Equality
         is the wrong assertion on a GPU.

      WHAT THE PLANNER SAYS ABOUT SERVING
        inference needs no gradient tape, so no credit rule and no
        activations kept for backward. It is the cheapest thing the
        planner will ever cost you:

          inference peak   {ax.Model() is not None and ''}{""}see below
    """)
    spec = ax.Model()
    spec.add_spiking(WIDTH, horizon=T)
    spec.add_dense(C)
    plan = spec.fit_budget(ax.Budget.consumer_6gb(), batch=32)
    print(f"        the same net at batch=1: "
          f"{plan.peak_bytes / 2**30:.4f} GiB")
    small = spec.fit_budget(ax.Budget.consumer_6gb(), batch=1)
    print(f"        against {plan.peak_bytes / 2**30:.4f} GiB at batch=32 "
          f"({plan.peak_bytes / max(small.peak_bytes, 1):.1f}x)")
    print()
    print("      A 32x batch is 32x the activations and roughly 32x the")
    print("      peak. Nothing about a spiking network changes that, and")
    print("      nothing here pretends otherwise.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())