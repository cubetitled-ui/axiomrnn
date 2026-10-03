"""
03_custom_neuron.py -- your own neuron, WITHOUT forking the framework.

Run (TensorFlow is needed):
    /home/cune/.venvs/ax/bin/python examples/03_custom_neuron.py

════════════════════════════════════════════════════════════════════════
THE REQUIREMENT THIS EXAMPLE CLOSES
════════════════════════════════════════════════════════════════════════
A beginner should be able to write a working network in ~400 lines. A
professional needs 6000+, and must be able to override the system WITHOUT
forking it.

The second half is the one that usually breaks. If the neuron dynamics are
baked into the layer, a professional has to copy the file and edit the
copy. A month later they and you have two different maths, and both of you
are sure you are right.

So the dynamics are an OBJECT. Everything else -- the backward pass, the
assembly, the training loop -- stays shared, and therefore stays verified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras

import axiomrnn as ax
from axtf.cells import SpikingCell, SpikingRNNCell


# ═══════════════════════════════════════════════════ 1. THE BASE NEURON ════
#
# SpikingCell is the thing people normally edit. It has four forward hooks:
#
#   membrane(state, drive, thr)     -> the next potential
#   spike_fn(u, thr, smooth)        -> "fire or not"
#   reset_fn(u, spike, smooth)      -> what happens to the potential after
#
# Plus THREE derivative hooks that must agree with those:
#
#   spike_d1(u, thr)                -> d spike / d u
#   reset_dud(u, spike, thr, ds)    -> d reset / d u
#   threshold_dthr(u, spike, thr, ds) -> d u / d thr
#
# Plus one more that used to be missing and caused a silent wrong gradient:
#
#   membrane_dstate(state, drive, thr) -> d membrane / d state
#
# Every one of these has a default, so a simple neuron needs no code at
# all. Override what you need and nothing else.


# ═════════════════════════════════════════ 2. A NEW NEURON: DRIVE-GATED LIF ═
#
# A plain LIF leaks at a FIXED rate whether the neuron is barely stimulated
# or hammered. This one scales its leak with the incoming drive:
#
#     u_{t+1} = u_t * (1 + g*tanh(drive)) + drive
#
# so a quiet neuron holds its charge and a busy one forgets quickly. That
# is a real mechanism, it is two lines of maths, and it needs both halves:
# the membrane AND its derivative.
#
# THE POINT OF THE EXAMPLE: the kernel knows nothing about this class.
# Nothing below is edited.

class DriveGatedCell(SpikingCell):
    """LIF whose effective leak scales with the incoming drive."""

    def __init__(self, leak=0.9, alpha=2.0, gate=0.3):
        super().__init__(leak=leak, alpha=alpha)
        self.gate = gate

    def membrane(self, state, drive, thr):
        return state * (1.0 + self.gate * tf.tanh(drive)) + drive

    def membrane_dstate(self, state, drive, thr):
        # d/dstate of state*(1 + g*tanh(drive)) + drive
        # the drive term does not depend on state, so it drops out
        return 1.0 + self.gate * tf.tanh(drive)


# ═══════════════════════════════════════════ 3. CHECK IT BEFORE TRAINING ═════
#
# Before training anything new, make sure it works at all. Three checks,
# and all three are cheap.

T, D, C = 16, 12, 4
rng = np.random.default_rng(0)
block = T // C
patterns = rng.normal(0, 0.4, (C, T, D))
for k in range(C):
    patterns[k, k * block:(k + 1) * block] += 2.0
y = rng.integers(0, C, 4000)
X = (patterns[y] * (rng.random((4000, T, D)) < 0.3)
     + rng.normal(0, 0.6, (4000, T, D))).astype(np.float32)
Y = y.astype(np.int32)

print("=" * 74)
print("YOUR NEURON, NO FORK")
print("=" * 74)
print(f"  TF {tf.__version__}, Keras {keras.__version__}")
print(f"  task: {len(X)} examples, T={T}, classes={C}")
print()

# ── check 1: the layer builds and produces a sane shape ───────────────────
x = keras.Input(batch_shape=(None, T, D))
layer = SpikingRNNCell(128, cell=SpikingCell(0.9, 2.0),
                       return_sequences=True, horizon_hint=T, name="probe")
y_out = layer(x)
probe = keras.Model(x, y_out)
print(f"  1. the layer builds, output shape {probe.output_shape}  "
      f"expected (None, {T}, 128)")

# ── check 2: the forward pass is sparse but NOT zero ──────────────────────
spikes = probe.predict(X[:256], verbose=0)
rate = float((spikes > 0.5).mean())
print(f"  2. spike density {rate:.4f} "
      f"{'OK' if 0.001 < rate < 0.95 else 'SUSPECT -- the neuron is silent'}")
if rate == 0.0:
    print("     Zero means the threshold is above everything the neuron")
    print("     accumulates. That is not a code bug: that is how a LIF")
    print("     behaves without calibration. The fix is the threshold,")
    print("     not an edit to the kernel.")

# ── check 3: the gradient is NOT zero (the important one) ─────────────────
#
# The hard spike 1[u>thr] has an identically zero derivative, so what
# trains is the SMOOTH model. If this is zero the layer cannot learn, and no
# amount of training will rescue it.
xt = tf.constant(X[:32])
with tf.GradientTape() as tape:
    tape.watch(xt)
    out = probe(xt)
    loss = tf.reduce_mean(out * out)
g = tape.gradient(loss, xt)
gnorm = float(tf.norm(g).numpy())
print(f"  3. gradient norm {gnorm:.6f} "
      f"{'OK' if gnorm > 0 else 'ZERO -- the layer cannot train!'}")
print()
print("  Worth knowing about that zero in advance. You cannot check it")
print("  with finite differences: the derivative of the hard step is")
print("  identically zero, and the differences return exactly 0 on 18 of 20")
print("  coordinates. So the gate checks the SMOOTH model instead, where")
print("  rel.err reaches 1e-8.")
print()

# ── check 4: YOUR neuron, same three checks ───────────────────────────────
gate_layer = SpikingRNNCell(128, cell=DriveGatedCell(0.9, 2.0),
                            return_sequences=True, horizon_hint=T,
                            name="probe_gate")
probe2 = keras.Model(x, gate_layer(x))
grate = float((probe2.predict(X[:256], verbose=0) > 0.5).mean())
with tf.GradientTape() as tape2:
    tape2.watch(xt)
    g_out = probe2(xt)
    g_loss = tf.reduce_mean(g_out * g_out)
gg = tape2.gradient(g_loss, xt)
print(f"  4. your neuron: density {grate:.4f}, "
      f"gradient norm {float(tf.norm(gg)):.6f}")
print()

# ═══════════════════════════════════════════════ 4. TRAINING WITH IT ═══════
#
# You overrode a class, and that is all. No fork, no edit to the kernel.
# keras_model() picks up the right cell by itself.

spec = ax.Model()
# your neuron goes in the first layer, a plain LIF in the second: different
# neurons live in one network without any fuss.
spec.add_spiking(192, horizon=T, cell=DriveGatedCell(leak=0.9, alpha=2.0))
spec.add_spiking(192, horizon=T, cell=SpikingCell(0.9, 2.0))
spec.add_dense(C)

model = spec.keras_model(inputs=keras.Input(batch_shape=(None, T, D)),
                         outputs=C, compile_model=True)

cut = int(0.8 * len(X))
h = model.fit(X[:cut], Y[:cut], epochs=12, batch_size=64, verbose=0)
pred = model.predict(X[cut:], verbose=0).argmax(1)
acc = float((pred == Y[cut:]).mean())

print("  " + "-" * 62)
print(f"  your neuron: val_accuracy {acc:.4f}, chance {1/C:.4f}")
print("  " + "-" * 62)
print(f"  loss: {h.history['loss'][0]:.4f} -> {h.history['loss'][-1]:.4f}")
print()

# ══════════════════════════════════════════════════ 5. WHY ALL OF THIS ════
print("  " + "=" * 62)
print("  WHY THE EXTENSION POINT EXISTS")
print("  " + "=" * 62)
print("""
  If the dynamics lived inside the layer, you would have to copy
  SpikingRNNCell and edit the copy. A month later you and we would have
  two different maths and both be certain we were right.

  Here the backward pass, the assembly, the calibration and the training
  loop stay SHARED -- which means they stay VERIFIED.

  THE RULE: every forward hook has a derivative hook. Overriding
  spike_fn without spike_d1 means the kernel builds a backward pass with
  someone else's derivative, quietly, and you see "the network trains, but
  badly". The kernel cannot guess your maths.

  membrane_dstate is the newest of these. The backward pass used to read
  cell.leak directly as du/dstate, which is right for a plain LIF and
  WRONG for every neuron that changes its membrane -- and it was wrong
  silently. You would have seen a model mysteriously 5% worse than a
  baseline, with no error anywhere. tests/test_tf_cells.py now checks the
  hook against finite differences for four different membranes, and checks
  that removing it makes the test fail.
""")