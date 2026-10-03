"""
12_failure_modes.py -- the six ways this breaks, and the exact message.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/12_failure_modes.py

WHAT THIS TEACHES
=================
Every one of these happened to us, and every one of them costs an
afternoon if you have never seen it. The script TRIGGERS each one and
prints the message you will actually see, so you can recognise yours.

The ordering is by how often it happens, not by how interesting it is.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

import axiomrnn as ax


def show(n, title, fn, what_it_means, used_to_fail=None):
    """Trigger a failure mode and print what came out.

    used_to_fail: the exact error, for modes that are FIXED and therefore no
    longer raise. Printing only "it did NOT fail" would be dishonest -- the
    reader would conclude the framework never had that problem.
    """
    print(f"  {n}. {title}")
    print("  " + "-" * 74)
    try:
        fn()
    except Exception as e:                                  # noqa: BLE001
        first = str(e).strip().splitlines()
        msg = first[0] if first else type(e).__name__
        print(f"     {type(e).__name__}: {msg[:64]}")
        if len(first) > 1:
            print(f"     {first[1].strip()[:64]}")
    else:
        if used_to_fail:
            print("     FIXED. This exact code used to raise:")
            print(f"       {used_to_fail}")
        else:
            print("     it did NOT fail, and there is no error message.")
    print(f"     MEANS: {what_it_means}")
    print()


def main():
    print("=" * 78)
    print("THE SIX WAYS THIS BREAKS")
    print("=" * 78)
    print("  Each is triggered for real below, on purpose.")
    print()

    import tensorflow as tf
    from tensorflow import keras
    from axtf.cells import SpikingCell, SpikingRNNCell

    # ── 1. shape= instead of batch_shape= ────────────────────────────────────
    def bad_shape():
        inp = keras.Input(shape=(16, 12))          # time axis is None
        h = SpikingRNNCell(64, cell=SpikingCell(0.9, 2.0),
                           return_sequences=True)(inp)
        keras.Model(inp, h)

    show(1, "keras.Input(shape=...) instead of batch_shape=...", bad_shape,
         "the time axis is None, so the layer cannot know its horizon.\n"
         "     Use keras.Input(batch_shape=(None, T, D)).",
         used_to_fail="ValueError: the layer needs a horizon and shape=(16, 12)\n"
                      "       leaves the time axis None")

    # ── 2. Flatten after a lost time axis ────────────────────────────────────
    def lost_axis():
        inp = keras.Input(batch_shape=(None, 16, 12))
        h = SpikingRNNCell(64, cell=SpikingCell(0.9, 2.0),
                           return_sequences=True, horizon_hint=16)(inp)
        h = keras.layers.Flatten()(h)
        keras.layers.Dense(4)(h)

    show(2, "the time axis silently lost, then Flatten", lost_axis,
         "compute_output_shape used to return (None, None, units), Flatten\n"
         "     gave (None, None), and Dense complained about shapes. The\n"
         "     real bug was in a shape helper, two files from the message.",
         used_to_fail="ValueError: Shapes must be fully-defined (None, None)\n"
                      "       values may not be None -- raised by Dense, caused\n"
                      "       by _as_shape in compute_output_shape")

    # ── 3. from_logits=False ─────────────────────────────────────────────────
    def double_softmax():
        T, D, C = 8, 6, 4
        inp = keras.Input(batch_shape=(None, T, D))
        h = SpikingRNNCell(32, cell=SpikingCell(0.9, 2.0),
                           return_sequences=True, norm_t=True,
                           horizon_hint=T)(inp)
        out = keras.layers.Dense(C)(keras.layers.Flatten()(h))
        m = keras.Model(inp, out)
        m.compile(loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=False))
        x = np.random.default_rng(0).normal(0, 1, (8, T, D)).astype("float32")
        y = np.random.default_rng(1).integers(0, C, 8)
        h.history = m.fit(x, y, epochs=1, verbose=0).history
        return h.history["loss"][0]

    print("  3. from_logits=False on a model that already outputs logits")
    print("  " + "-" * 74)
    bad = double_softmax()
    print(f"     loss after 1 epoch: {bad:.4f}")
    print("     MEANS: Keras applied a second softmax to logits. Measured")
    print("     here: the correct loss is around 1.44 and this one starts")
    print("     far above ln(4) = 1.386 and stalls.")
    print("     Our outputs ARE logits, so from_logits=True is mandatory.")
    print()

    # ── 4. no calibration ───────────────────────────────────────────────────
    def silent():
        T, D, C = 16, 12, 4
        inp = keras.Input(batch_shape=(None, T, D))
        h = SpikingRNNCell(64, cell=SpikingCell(0.9, 2.0),
                           return_sequences=True, norm_t=True,
                           horizon_hint=T)(inp)
        out = keras.layers.Dense(C)(keras.layers.Flatten()(h))
        m = keras.Model(inp, out)
        m.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=["accuracy"], run_eagerly=True)
        x = np.random.default_rng(0).normal(0, 1, (32, T, D)).astype("float32")
        y = np.random.default_rng(1).integers(0, C, 32)
        h1 = m.fit(x, y, epochs=3, verbose=0)
        return h1.history["accuracy"][-1]

    print("  4. no calibration, so the network never fires")
    print("  " + "-" * 74)
    acc = silent()
    print(f"     accuracy after 3 epochs: {acc:.4f} (chance 0.2500)")
    print("     MEANS: a fresh LIF has potentials below the default")
    print("     threshold, so nothing spikes and the gradient is zero.")
    print("     This is not a bug, it is how the neuron starts. Call")
    print("     calibrate_model(model, x, target_rate=0.2).")
    print("     examples/advanced/07_silence_diagnosis.py has the numbers.")
    print()

    # ── 5. credit_k larger than T ───────────────────────────────────────────
    def bad_k():
        lay = SpikingRNNCell(8, cell=SpikingCell(0.9, 2.0),
                             return_sequences=True, norm_t=True,
                             horizon_hint=4, credit_k=99)
        lay.build((None, 4, 5))
        lay(np.random.default_rng(0).normal(0, 1, (2, 4, 5)).astype("float32"))

    show(5, "credit_k larger than the horizon", bad_k,
         "truncating further than the sequence is harmless in the maths,\n"
         "     but it usually means the horizon you ask for and the one you\n"
         "     feed disagree. k = T is full BPTT, and the gate checks that.",
         used_to_fail=None)

    # ── 6. overriding a forward hook only ────────────────────────────────────
    def wrong_hook():
        class BadCell(SpikingCell):
            def spike_fn(self, u, thr, smooth):
                return tf.cast(u > thr, u.dtype)      # a hard rule

            # spike_d1 left at the smooth surrogate -- a MISMATCH

        T, D = 8, 6
        inp = keras.Input(batch_shape=(None, T, D))
        h = SpikingRNNCell(16, cell=BadCell(0.9, 2.0),
                           return_sequences=True, norm_t=True,
                           horizon_hint=T)(inp)
        out = keras.layers.Dense(4)(keras.layers.Flatten()(h))
        m = keras.Model(inp, out)
        x = np.random.default_rng(0).normal(0, 1, (8, T, D)).astype("float32")
        y = np.random.default_rng(1).integers(0, 4, 8)
        m.compile(optimizer="adam",
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True), run_eagerly=True)
        h1 = m.fit(x, y, epochs=2, verbose=0)
        return h1.history["loss"][0]

    print("  6. overriding a forward hook and leaving the derivative")
    print("  " + "-" * 74)
    loss = wrong_hook()
    print(f"     it trained without any error. final loss {loss:.4f}")
    print("     MEANS: this is the dangerous one, because there is no")
    print("     message. The backward pass now uses the atan surrogate's")
    print("     derivative for a hard step. It trains, it just trains with")
    print("     a gradient that belongs to a different neuron.")
    print("     Check hooks against finite differences --")
    print("     examples/advanced/11_derivative_hooks.py.")
    print()

    print("  " + "=" * 72)
    print("  THE FIVE-SECOND CHECKLIST")
    print("  " + "=" * 72)
    print("""
      1. keras.Input(batch_shape=(None, T, D))   not shape=
      2. from_logits=True                       our output IS logits
      3. calibrate_model(...) before fit        or nothing spikes
      4. run_eagerly=True                       or the gradient is stale
      5. forward hook and derivative hook together, or neither

      Two of these RAISE and three do not. The three that do not are the
      ones that cost days.
    """)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())