"""
08_reproducibility.py -- what is deterministic here, and what is not.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/08_reproducibility.py

WHAT THIS TEACHES
=================
Before you report a comparison you have to know whether two runs differ
because of your change or because of the seed. This script finds out
empirically rather than promising anything.

Short version, measured below:

  * the numpy core is bit-exact under a fixed seed;
  * the PLANNER is bit-exact, it is pure arithmetic;
  * a Keras model is NOT reproducible across two builds unless you assign
    the weights, because the initialiser is unseeded.

That last one cost us a real mistake. Two gradient arms were compared with
two separately built layers, and the difference we attributed to a
parameter turned out to be two different initialisations. The fix was
`layer.kernel.assign(W)`, and the comparison became meaningful.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

import axiomrnn as ax


def main():
    print("=" * 78)
    print("REPRODUCIBILITY: MEASURED, NOT ASSUMED")
    print("=" * 78)

    # ── 1. the planner is pure arithmetic ────────────────────────────────────
    print()
    print("  1. THE PLANNER")
    print("  " + "-" * 74)
    spec = ax.Model()
    spec.add_spiking(1024, horizon=64)
    spec.add_dense(100)
    peaks = [spec.fit_budget(ax.Budget.consumer_6gb(), batch=32).peak_bytes
             for _ in range(3)]
    same_plan = len(set(peaks)) == 1
    print(f"     three identical calls, bytes: "
          f"{', '.join(str(p) for p in peaks)}")
    print(f"     bit-exact: {same_plan}")
    print()
    print("     No RNG is involved anywhere in axplan. The planner is")
    print("     arithmetic, so this is not a feature to be proud of -- it")
    print("     is what you need for a number you are going to act on.")
    print()

    # ── 2. the numpy core under a fixed seed ─────────────────────────────────
    print("  2. THE NUMPY CORE")
    print("  " + "-" * 74)
    from axon.lif import lif_forward

    def run(seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, (4, 12, 8)).astype(np.float32)
        w = rng.normal(0, 0.3, (8, 16)).astype(np.float32)
        return lif_forward(x, w, np.full(16, 0.6, np.float32), leak=0.9)[0]

    a, b = run(7), run(7)
    same_core = np.array_equal(a, b)
    print(f"     same seed twice, identical spikes: {same_core}")
    c = run(8)
    print(f"     different seed, different spikes: {not np.array_equal(a, c)}")
    print()
    print("     This is the layer worth trusting: fixed weights in, exact")
    print("     spikes out, no hidden state.")
    print()

    # ── 3. a Keras layer is NOT reproducible across builds ───────────────────
    print("  3. A KERAS LAYER, BUILT TWICE")
    print("  " + "-" * 74)
    try:
        import tensorflow as tf
        from tensorflow import keras
        from axtf.cells import SpikingCell, SpikingRNNCell
    except ImportError as e:
        print(f"     TensorFlow absent ({e}); sections 1 and 2 are the story.")
        print("=" * 78)
        return 0

    def build_default():
        lay = SpikingRNNCell(16, cell=SpikingCell(0.9, 2.0),
                             return_sequences=True, norm_t=True,
                             horizon_hint=8)
        lay.build((None, 8, 6))
        return lay

    k1 = build_default().kernel.numpy()
    k2 = build_default().kernel.numpy()
    identical = np.array_equal(k1, k2)
    print(f"     two builds, identical initial weights: {identical}")
    print(f"     weight difference: {float(np.abs(k1 - k2).max()):.6f}")
    print()
    if not identical:
        print("     So: glorot_uniform is UNSEEDED. Two builds of the same")
        print("     layer give different weights, which means two runs of")
        print("     'the same' model are two different models.")
        print()
        print("     The mistake this caused us: comparing two gradient arms")
        print("     from two separately built layers, then attributing the")
        print("     difference to the parameter under test.")
        print()
        print("     THE FIX, used everywhere in our tests:")
        print()
        print("       W = rng.normal(0, 1 / np.sqrt(DIN), (DIN, UNITS))")
        print("       layer.kernel.assign(W)")
        print("       layer.thr.assign(thr)")
        print()

        # prove the fix works
        rng = np.random.default_rng(3)
        DIN, UNITS = 6, 16
        W = rng.normal(0, 1 / np.sqrt(DIN), (DIN, UNITS)).astype(np.float32)
        thr = np.full(UNITS, 0.6, np.float32)
        xin = rng.normal(0, 1, (4, 8, DIN)).astype(np.float32)

        def grad_with_assigned():
            lay = build_default()
            lay.kernel.assign(W)
            lay.thr.assign(thr)
            xc = tf.constant(xin)
            with tf.GradientTape() as tape:
                out = lay(xc)
                loss = tf.reduce_mean(out)
            return float(tf.norm(tape.gradient(loss, lay.kernel)))

        g1, g2 = grad_with_assigned(), grad_with_assigned()
        print(f"     with assigned weights, two runs give "
              f"{g1:.8f} and {g2:.8f}")
        print(f"     identical: {g1 == g2}")
        print()
        print("     That is what makes a gradient comparison worth anything.")
        print()

    # ── 4. what you can and cannot control ──────────────────────────────────
    print("  4. WHAT TO SET WHEN YOU NEED REPRODUCIBILITY")
    print("  " + "-" * 74)
    print("""
      DO THIS
        python3 -c "import numpy as np, tensorflow as tf;
                   np.random.seed(0); tf.random.set_seed(0)"
        tf.keras.utils.set_random_seed(0)      # numpy + python + tf
        layer.kernel.assign(W)                 # for A/B comparisons
        cuDNN deterministic algorithms, if you are chasing exact equality
        record: git revision, TF version, device, dtype

      KNOW THAT THESE ARE NOT FULLY DETERMINISTIC EVEN WITH A SEED
        non-deterministic GPU reductions (atomics in reductions)
        thread scheduling on CPU
        cuBLAS algorithm selection, which varies with problem size
        our forward pass is a Python loop, so the order is fixed -- but
        the FLOATING POINT sum inside each step is the device's choice

      WHAT ACTUALLY MATTERS IN PRACTICE
        For a claim of the form "A scores X, B scores Y", run each
        arm 3 times and report the spread. A single run cannot tell you
        whether a gap is a result or a seed.

        For "does the fix change the gradient", assigned weights are not
        optional -- seeds do not help, because you are comparing two
        objects, not two runs of one object.
    """)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())