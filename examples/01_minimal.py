"""
01_minimal.py -- a spiking network that actually works.

Run:
    python3 examples/01_minimal.py

If you have never used Keras, read the comments. There are more of them
than there is code, and that is deliberate: Keras 3 silently does two
things that break a spiking network. Both are covered in the README and
repeated here where they matter.

════════════════════════════════════════════════════════════════════════
WHAT "SPIKING" MEANS
════════════════════════════════════════════════════════════════════════
An ordinary network passes a NUMBER at every step. A spiking one passes a
DECISION: "fire or not" (0 or 1). There is less information per step, but
the state can be stored in a single bit, and on neuromorphic hardware that
costs almost no energy.

This is not only about neuromorphic chips. On an ordinary GPU such a layer
works honestly through our SpikingRNNCell, and we measured that quality
is no worse than a dense network (0.99 against 1.00 on a task with
temporal structure). The difference is in MEMORY, and it has to be
measured separately.

════════════════════════════════════════════════════════════════════════
STEP 1. DATA: a task where ORDER matters, not just content
════════════════════════════════════════════════════════════════════════
Four classes, and the label says "fire in the first quarter", "in the
second", and so on. If the network averages everything over time the
label disappears. This is exactly the case where a spiking network must
hold on to TIME, and it is the case where we caught our own readout bug.
"""
import sys
from pathlib import Path

# The example lives in axiomrnn/examples/, the package is one level up.
# Without this line, running from examples/ dies with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

T = 16        # sequence length
D = 12        # channels per step
C = 4         # number of classes


def make_task(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    block = T // C                       # the label is "which block is active"
    patterns = rng.normal(0, 0.4, (C, T, D))
    for k in range(C):
        patterns[k, k * block:(k + 1) * block] += 2.0

    y = rng.integers(0, C, n)
    # sparse input: 30% of the channels carry signal, the rest is noise
    mask = rng.random((n, T, D)) < 0.30
    x = patterns[y] * mask + rng.normal(0, 0.6, (n, T, D))
    return x.astype(np.float32), y.astype(np.int32)


# ═══════════════════════════════════════════════════════════ LOADING TF ══
import tensorflow as tf
from tensorflow import keras

X, Y = make_task()
cut = int(0.8 * len(X))
Xtr, Ytr, Xte, Yte = X[:cut], Y[:cut], X[cut:], Y[cut:]

print(f"data: {len(X)} examples, T={T}, channels={D}, classes={C}")

# ─────────────────────────────────────────────────────────────────────────────
# WHAT IS HAPPENING HERE, AND WHY EXACTLY LIKE THIS (for someone new to Keras)
#
# keras.Input(...) is NOT data, it is a SHAPE DECLARATION. We tell Keras
# "a stream of numbers of shape (None, T, D) is coming". None in the first
# position means "batch size unspecified", and that is mandatory -- it
# changes from batch to batch.
#
# batch_shape instead of shape is NOT COSMETIC either. With
# Input(shape=(T, D)) the time axis is None, and our layer honestly passes
# that None along. A later Flatten then gets (None, None) and the next
# layer dies with "Shapes must be fully-defined", a message that points
# nowhere near the cause. We lost a day to that; here the length is known
# from the start.
IN = keras.Input(batch_shape=(None, T, D), name="input")


# ═════════════════════════════════════════════════ THE NETWORK: TWO LAYERS ══
#
# add_spiking(width, horizon=T)
#   width   -- how many neurons in the layer
#   horizon -- how many steps we PROMISE. This is a hint for the memory
#              planner, not a restriction: the layer copes with other T too.
#
# IMPORTANT: nothing on this list is written by hand. The budget planner
# picks the credit rule (e-prop or BPTT) on its own, and decides on its
# own how much memory may be spent. That is the point of the framework:
# you describe WHAT, the budget decides HOW.
import axiomrnn as ax

spec = ax.Model()
spec.add_spiking(192, horizon=T)          # layer 1
spec.add_spiking(192, horizon=T)          # layer 2
spec.add_dense(C)                         # output, C logits


# ═══════════════════════════════════════════════════════ ASSEMBLING THE MODEL ═══
#
# keras_model() does THREE things you need to know about:
#
# 1. compile_model=True sets from_logits=True. Our output IS LOGITS. Keras
#    expects probabilities by default and applies softmax ITSELF. The
#    double softmax produced a measured loss that stalled at 5.14 instead
#    of 1.17.
#
# 2. It also sets run_eagerly=True. This is NOT decoration: our layer's
#    loop over time is written in Python, and Keras 3 caches such
#    functions. During training the forward pass is then called once while
#    backward runs for every step, so the gradient lands on STALE
#    activations. A call counter showed 2 calls where 4 were needed. The
#    proper fix is to rewrite the loop as tf.while_loop; that is NOT
#    done, and it is stated plainly rather than hidden.
#
# 3. The default readout is 'flatten', the whole sequence. That is NOT a
#    detail. We measured: with a time-mean readout the network gives 0.73,
#    with flatten 0.99. Averaging erases the question "WHEN did it fire".
#    If your task does not depend on precise timing, try readout="mean"
#    and win on compactness.
model = spec.keras_model(
    inputs=IN,
    outputs=C,
    compile_model=True,
)
model.summary()

# ═══════════════════════════════════════════════════════════════ TRAINING ═══
history = model.fit(
    Xtr, Ytr,
    epochs=15,
    batch_size=64,
    verbose=2,
    validation_data=(Xte, Yte),
)

# ═══════════════════════════════════════════════════════════════ CHECKING ═══
probs = model.predict(Xte, verbose=0)
pred = probs.argmax(axis=1)
accuracy = float((pred == Yte).mean())

print()
print("=" * 66)
print(f"  held-out accuracy:      {accuracy:.4f}")
print(f"  chance:                 {1.0 / C:.4f}")
print("=" * 66)

# ═════════════════════════════════════════════════════ THE GATE IS MANDATORY ══
#
# AFTER any result, run the gate. It checks what the eye cannot see: does
# the gradient match the numerical one (to 1e-8 in our case), does the
# network learn 32 examples, is credit truncation intact.
#
# None of that is needed to RUN a network. It is needed so you never have
# to run one IN PRODUCTION.
print()
print("Gate (mandatory before any conclusion):")
import subprocess

HERE = Path(__file__).resolve().parent
gate = subprocess.run(
    [sys.executable, str(HERE.parent / "tests" / "run_all.py")],
    capture_output=True, text=True,
)
print(gate.stdout[-900:] if gate.stdout else gate.stderr[-900:])
print("return code:", gate.returncode)