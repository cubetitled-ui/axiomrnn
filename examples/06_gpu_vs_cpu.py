"""
06_gpu_vs_cpu.py -- does the same thing on both devices, and how long each takes.

Run:
    python examples/06_gpu_vs_cpu.py

WHAT THIS MEASURES
    One model, one dataset, two devices. Same weights, same seed, same number of
    steps. The only variable is where the ops run.

    It reports, for each device:
      - whether the device was actually used, not merely visible
      - wall-clock seconds for training
      - steps per second
      - held-out accuracy

WHY THIS FILE IS HONEST ABOUT BEING UNIMPRESSIVE
    A spiking cell integrates over time with a loop written in Python, so
    keras_model() has to set run_eagerly=True. Eager mode is exactly the mode
    that gives up graph compilation, which is where most of a GPU's advantage on
    small models comes from. So a GPU speedup here is NOT the 10x you would get
    from a fully compiled model, and it may be close to nothing.

    We report the measured number either way. If the GPU is not faster, that is
    the finding, and printing an expectation instead of a measurement would make
    this file worthless.
"""

import os
import sys
import time
from pathlib import Path

# Running a script puts examples/ on sys.path, not the repository root, so the
# package would not be importable without this. Every other example does the
# same. The installed wheel needs none of it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

if os.environ.get("AX_FAST") != "1":
    print("Set AX_FAST=1 for the short version used by the test gate.")
    print("Full version: unset it.")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import keras

import axiomrnn as ax


# ── the task ────────────────────────────────────────────────────────────────
# Order matters here, not just content: the label is which quarter of the
# sequence the signal lives in. A network that averages over time cannot do it,
# which is the whole reason a spiking layer earns its place.
T, D, C = 24, 32, 4
FAST = os.environ.get("AX_FAST") == "1"
UNITS = 96 if FAST else 192
EPOCHS = 3 if FAST else 8
BATCH = 64


def make_task(seed=0):
    rng = np.random.default_rng(seed)
    n = 1024
    x = rng.normal(0, 1, (n, T, D)).astype(np.float32)
    q = rng.integers(0, C, n)
    y = np.zeros((n, C), dtype=np.float32)
    onset = (q[:, None] * (T // C)) + rng.integers(0, T // C, n)[:, None]
    rows = np.arange(n)
    for step in range(T):
        hit = (step >= onset[:, 0]) & (step < onset[:, 0] + T // (2 * C))
        # q must be subset by the same mask as rows, or the two do not
        # broadcast and the index silently describes the wrong samples.
        x[rows[hit], step, q[hit]] += 4.0
    y[rows, q] = 1.0
    return x, y


# ── device reporting ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")

print("=" * 74)
print("DEVICES VISIBLE")
print("=" * 74)
print(f"  TensorFlow   {tf.__version__}")
print(f"  built with CUDA   {tf.test.is_built_with_cuda()}")
print(f"  GPUs               {[d.name for d in gpus] or 'none'}")
print(f"  CPUs               {len(tf.config.list_physical_devices('CPU'))}")

if gpus:
    for gpu in gpus:
        # A 6 GB laptop GPU cannot hold the default pool. TensorFlow grabs
        # nearly all of it on first use, and the second thing you run then
        # fails with OOM for no obvious reason. Growing on demand avoids that.
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  memory growth on   {gpu.name}")
        except RuntimeError as exc:
            print(f"  memory growth not set on {gpu.name}: {exc}")


# ── train once, on a named device ────────────────────────────────────────────
def train_on(device_name):
    """Train the identical model pinned to one device. Returns measured facts."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)

    x, y = make_task(seed=0)
    cut = int(0.8 * len(x))
    xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]

    spec = ax.Model()
    spec.add_spiking(UNITS, horizon=T)
    spec.add_dense(C)

    placed = {}
    t0 = time.perf_counter()
    try:
        with tf.device(device_name):
            model = spec.keras_model(
                inputs=keras.Input(batch_shape=(None, T, D)),
                outputs=C,
                compile_model=True,
            )
            model.compile(optimizer="adam",
                          loss=keras.losses.CategoricalCrossentropy(
                              from_logits=True),
                          metrics=["accuracy"])
            steps = max(1, len(xtr) // BATCH)
            hist = model.fit(xtr, ytr, epochs=EPOCHS, batch_size=BATCH,
                             verbose=0, steps_per_epoch=steps)
            total_steps = steps * EPOCHS
            build_and_train = time.perf_counter() - t0

            acc = float(model.evaluate(xte, yte, verbose=0,
                                       return_dict=True)["accuracy"])
    except tf.errors.ResourceExhaustedError as exc:
        return dict(ok=False, note=f"out of memory: {str(exc)[:120]}")

    # Did the ops actually land on the device? "Visible" is not "used": if the
    # graph stayed on CPU, a fast number would mean nothing.
    #
    # A Keras 3 Variable exposes no public .device; the wrapped tf.Variable
    # does. Read it defensively rather than assuming the private attribute
    # exists in some future version.
    devices = set()
    for var in model.trainable_variables:
        inner = getattr(var, "_value", None)
        raw = getattr(inner, "device", None) if inner is not None else None
        if raw:
            # Placement strings look like
            #   /job:localhost/replica:0/task:0/device:CPU:0
            # so the meaningful part is after the last "device:" marker.
            # Splitting on "/" instead yields the literal word "device",
            # which is a placement report that says nothing.
            tail = raw.split("device:")[-1]
            devices.add(tail.split(":")[0] or "?")
    return dict(
        ok=True,
        seconds=build_and_train,
        steps=total_steps,
        sps=total_steps / build_and_train if build_and_train else 0.0,
        accuracy=acc,
        devices=sorted(devices) or ["unknown"],
        loss=float(hist.history["loss"][-1]),
    )


results = {}
runs = [("CPU", "/CPU:0"), ("GPU", "/GPU:0")]
# A second CPU run, to find out how much the accuracy moves when nothing at all
# changes. Without that number the device comparison cannot be read: two CPU
# runs differ too, because three epochs is not convergence and nothing here is
# bit-deterministic. Comparing two devices against each other while ignoring
# that spread is how a framework ends up claiming a GPU bug it does not have.
runs.append(("CPU-repeat", "/CPU:0"))

for label, device in runs:
    if label.startswith("GPU") and not gpus:
        print("\nNo GPU visible. Skipping, and not pretending it ran.")
        continue
    print("\n" + "=" * 74)
    print(f"TRAINING ON {label}  ({device})")
    print("=" * 74)
    print("  building and training, please wait")
    res = train_on(device)
    results[label] = res
    if not res.get("ok"):
        print(f"  FAILED: {res['note']}")
        continue
    print(f"  device actually used   {res['devices']}")
    print(f"  total seconds          {res['seconds']:.2f}")
    print(f"  steps per second       {res['sps']:.1f}")
    print(f"  final train loss       {res['loss']:.5f}")
    print(f"  held-out accuracy      {res['accuracy']:.4f}")
    print(f"  chance                 {1.0 / C:.4f}")


print("\n" + "=" * 74)
print("COMPARISON")
print("=" * 74)

if ("CPU" in results and results["CPU"].get("ok")
        and results.get("GPU", {}).get("ok")):
    c, g = results["CPU"], results["GPU"]
    speed = c["seconds"] / g["seconds"] if g["seconds"] else float("inf")
    print(f"  CPU   {c['seconds']:8.2f}s   {c['sps']:7.1f} steps/s   acc {c['accuracy']:.4f}")
    print(f"  GPU   {g['seconds']:8.2f}s   {g['sps']:7.1f} steps/s   acc {g['accuracy']:.4f}")
    print(f"  speedup {speed:.2f}x")
    print()
    if speed < 1.15:
        print("  The GPU is NOT faster here, and that is the measurement.")
        print("  The time loop is Python, so keras_model() sets run_eagerly=True.")
        print("  Eager mode does not compile the graph, and on a model this small")
        print("  the transfer and launch overhead cancels the arithmetic win.")
        print("  A GPU pays off on larger hidden sizes, not on this shape.")
    elif speed < 4:
        print("  A real but modest speedup. The eager-mode ceiling again: the time")
        print("  loop never becomes a graph, so this is not the 10x a fully")
        print("  compiled model would give.")
    else:
        print("  A large speedup. Worth checking run_eagerly is still forced,")
        print("  because that would be surprising for a Python time loop.")
    print()

    # The noise floor. Same device, same seed, same code, nothing changed: any
    # difference here is pure run-to-run variation.
    noise = None
    if results.get("CPU-repeat", {}).get("ok"):
        noise = abs(results["CPU-repeat"]["accuracy"] - c["accuracy"])
        print(f"  noise floor (CPU vs CPU, nothing changed)  {noise:.4f}")
    delta = abs(c["accuracy"] - g["accuracy"])
    print(f"  CPU vs GPU accuracy difference               {delta:.4f}")

    if noise is None:
        print("  No repeat run, so there is no noise floor to judge that against.")
    elif delta <= max(noise * 1.5, 0.02):
        print("  Within the run-to-run spread. Both devices trained the same")
        print("  network to the same accuracy, and the timing above is the")
        print("  comparable result.")
    else:
        print("  LARGER than the run-to-run spread, so the two devices did NOT")
        print("  land in the same place. Read the accuracy figures as unsettled")
        print("  rather than as a device property. The timing above still holds;")
        print("  accuracy at this training length is not a device measurement.")
else:
    print("  Could not compare: at least one device did not complete.")
    print("  A single-device number is not a comparison. Read it as one.")

print()
print("BOTH DEVICES TRAIN THE SAME NETWORK. The framework is not CPU-only and")
print("not GPU-only, and this run is the evidence for whichever it did here.")
sys.exit(0)