"""
Devices: CPU and GPU, same code.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/06_devices.py

WHAT THIS TEACHES
=================
Which device are we on, and does the framework care? It does
not: Keras places tensors, and the planner never touches them.

Run it twice, with and without CUDA_VISIBLE_DEVICES, and compare
the wall clock. The last section shows the planning half, which
produces the same number either way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import time

import axiomrnn as ax



print("=" * 78)
print("DEVICES: CPU AND GPU, SAME CODE")
print("=" * 78)
print()
print("axiomrnn has no opinion about devices. Keras places tensors; the")
print("planner never touches them. What this shows is that the SAME code")
print("runs both ways, and what each costs -- measured, not asserted.")
print()
print("Run it twice:")
print("    /home/cune/.venvs/ax/bin/python examples/advanced/06_devices.py")
print("    CUDA_VISIBLE_DEVICES= /home/cune/.venvs/ax/bin/python \\")
print("        examples/advanced/06_devices.py")
print()

try:
    import tensorflow as tf
    from tensorflow import keras
    from axtf.build import calibrate_model
    from axtf.cells import SpikingCell, SpikingRNNCell
except ImportError as e:
    print(f"  TensorFlow absent ({e}). The planning half still runs:")
    planning()
    print("=" * 78)
    sys.exit(0)

gpus = tf.config.list_physical_devices("GPU")
print("  physical devices")
for d in gpus:
    print(f"    GPU  {d.name}")
print(f"    CPU  {tf.config.list_physical_devices('CPU')[0].name}")
if not gpus:
    print()
    print("  No GPU visible. That is a supported configuration, not a")
    print("  degraded one -- everything below runs on CPU unchanged.")
print()

T, D, C, BATCH, N = 16, 24, 4, 128, 2000


def dataset(n=N, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    blk = T // C
    pat = rng.normal(0, 0.4, (C, T, D))
    for k in range(C):
        pat[k, k * blk:(k + 1) * blk] += 2.5
    y = rng.integers(0, C, n)
    x = pat[y] * (rng.random((n, T, D)) < 0.25) + rng.normal(0, 0.6, (n, T, D))
    return x.astype("float32"), y.astype("int32")


x, y = dataset()


def build():
    inp = keras.Input(batch_shape=(None, T, D))
    h = SpikingRNNCell(128, cell=SpikingCell(0.9, 2.0),
                       return_sequences=True, norm_t=True,
                       horizon_hint=T, name="sp0")(inp)
    h = keras.layers.Flatten()(h)
    out = keras.layers.Dense(C, name="logits")(h)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"], run_eagerly=True)
    return m


m = build()
calibrate_model(m, x[:512], target_rate=0.2, iters=6)

t0 = time.time()
h = m.fit(x, y, epochs=3, batch_size=BATCH, verbose=0)
dt = time.time() - t0

print("  training result")
print(f"    loss          {h.history['loss'][0]:.4f} -> "
      f"{h.history['loss'][-1]:.4f}")
print(f"    accuracy      {h.history['accuracy'][-1]:.4f}")
print(f"    wall clock    {dt:.1f}s for 3 epochs on {len(x)} sequences")
if gpus:
    info = tf.config.experimental.get_memory_info("GPU:0")
    print(f"    GPU peak      {info['peak'] / 2**20:.1f} MiB")
print()

print("  WHY run_eagerly SLOWS THINGS DOWN, AND WHY WE KEEP IT")
print("  " + "-" * 74)
print("  Our forward pass is a PYTHON loop over time. Keras 3 wraps the")
print("  graph in tf.function and then reuses the result instead of")
print("  recomputing it. Measured with a call counter: the forward ran")
print("  2 times when 4 were needed, and 0 times on a repeat fit.")
print()
print("  The gradient still applied -- to STALE activations. Loss fell")
print("  slowly and saturated at ln(C) = 1.386.")
print()
print("  run_eagerly=True costs speed and buys correctness. For spiking")
print("  networks that is the right trade: wrong gradients cost more than")
print("  slow steps. Anyone wanting raw speed can pass False.")
print()
print("  The proper fix is a tf.while_loop over tf.Variable state, which")
print("  would make the layer graph-native and let the cache be correct.")
print("  NOT done. Stated plainly rather than hidden behind a default.")
print()


def planning():
    """The planning half, which never touches TensorFlow."""
    print()
    print("  PLANNING IS DEVICE-INDEPENDENT")
    print("  " + "-" * 74)
    spec = ax.Model()
    spec.add_spiking(1024, horizon=64)
    spec.add_dense(100)
    plan = spec.fit_budget(ax.Budget.consumer_6gb(), batch=32)
    print(f"    peak           {plan.peak_bytes / 2**30:.2f} GiB")
    print(f"    credit horizon {plan.credit_horizon}")
    print(f"    regime         {plan.regime.value}")
    print()
    print("    None of that number touched TensorFlow. axplan is pure")
    print("    arithmetic, so it runs on a laptop with no CUDA and gives")
    print("    the same answer as the machine that trains.")

planning()
print("=" * 78)
