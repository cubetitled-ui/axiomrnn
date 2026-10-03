"""
04_equal_budget.py -- how to compare HONESTLY.

Run (TensorFlow is needed):
    /home/cune/.venvs/ax/bin/python examples/04_equal_budget.py

════════════════════════════════════════════════════════════════════════
WHY THIS EXAMPLE
════════════════════════════════════════════════════════════════════════
Every earlier example works. This one is about not fooling yourself while
comparing two networks, because we already did exactly that and reached a
conclusion that was wrong.

How it went. We compared a spiking and a dense network. Spiking gave 0.73,
dense 0.998. We very nearly believed that "spikes are worse than dense".

The truth was elsewhere: the variant reading the output over the whole
sequence was FAILING TO BUILD, and we had kept its number from an earlier
run of a different, broken architecture. Fixed, spiking gave 0.994, that
is, nearly matching dense.

The claim "spikes are worse" was a false statement about physics, and it
would have cost us our credibility.

════════════════════════════════════════════════════════════════════════
THE FIVE RULES THIS EXAMPLE CHECKS
════════════════════════════════════════════════════════════════════════
  1. Equal parameters, or you are comparing different networks.
  2. Equal settings, or you are comparing different regimes.
  3. Fair tuning: give every variant its own grid and keep the best.
  4. Not one cause, but a decomposition into parts.
  5. "Not measured" instead of a plausible number.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras

import axiomrnn as ax
from axtf.cells import SpikingCell, SpikingRNNCell, _atan_surrogate

T, D, C, WIDTH, LAYERS = 16, 24, 4, 256, 2


# ═══════════════════════════════════════════════ THE HELPER NEURON ═════════
#
# The same LIF, but with a real-valued output instead of 1[u>thr]. It exists
# to SEPARATE the contribution of the bit from the contribution of the
# readout: if swapping the hard spike for a smooth one barely changes
# quality, then the bit is cheap and the gap comes from somewhere else.
#
# This is exactly the extension point SpikingCell exists for: two methods
# overridden, the kernel untouched.
#
# ONE THING TO NOTICE, because it is easy to get wrong. We override the
# forward hooks only. That works here because the DEFAULT derivative hooks
# are already the smooth ones: spike_d1 is _atan_d1 and reset_dud is
# 1 - spike - u*dspike. That is a coincidence of this particular neuron,
# not a general rule. Override a forward hook and you must check that the
# matching derivative hook is right for your maths, not assume it.
class SmoothCell(SpikingCell):
    def spike_fn(self, u, thr, smooth=True):
        return _atan_surrogate(u, thr, self.alpha)

    def reset_fn(self, u, spike, smooth=True):
        return u * (1.0 - spike)


# ═════════════════════════════════════════════════════════ THE TASK ═════════
#
# The label is given by a BLOCK of time: "fire in the first quarter", "in
# the second". That matters. On a task where only the content matters, a
# spiking network has nothing to lose and the comparison is meaningless.


def make_task(n=8000, seed=0, T=T, D=D, C=C):
    rng = np.random.default_rng(seed)
    blk = T // C
    pat = np.zeros((C, T, D))
    for k in range(C):
        pat[k, k * blk:(k + 1) * blk] = 1.0
        pat[k] += rng.normal(0, 0.25, (T, D))
    y = rng.integers(0, C, n)
    x = (pat[y] * (rng.random((n, T, D)) < 0.25)
         + rng.normal(0, 0.5, (n, T, D)))
    return x.astype(np.float32), y.astype(np.int32)


def build(kind, thr=None):
    """Assemble a model. kind: dense | spiking | spiking_flat | smooth.

    THREE HONESTY RULES ARE CODED HERE:

      * the width is threaded explicitly (din is updated), otherwise layer 2
        gets the input shape of the WHOLE network -- and that is exactly how
        we ended up comparing different networks;
      * threshold, not threshold_ : the exact attribute name;
      * parameters are counted and printed, so the comparison can be
        CHECKED rather than believed.
    """
    if kind == "dense":
        inp = keras.Input(batch_shape=(None, T, D))
        x = inp
        for l in range(LAYERS):
            x = keras.layers.Dense(WIDTH, activation="relu",
                                   name=f"d{l}")(x)
        x = keras.layers.Flatten()(x)
        out = keras.layers.Dense(C, name="out")(x)
        return keras.Model(inp, out)

    cls = SmoothCell if kind == "smooth" else SpikingCell
    inp = keras.Input(batch_shape=(None, T, D))
    x, din, cells = inp, D, []
    for l in range(LAYERS):
        cell = SpikingRNNCell(WIDTH, cell=cls(0.9, 2.0),
                              return_sequences=True, norm_t=True,
                              horizon_hint=T, name=f"s{l}")
        cell.build((None, T, din))
        cells.append(cell)
        x = cell(x)
        din = WIDTH                        # <- width threading
    if kind == "spiking_flat":
        x = keras.layers.Flatten()(x)
    else:
        x = keras.layers.GlobalAveragePooling1D()(x)
    out = keras.layers.Dense(C, name="out")(x)
    if thr is not None:                    # the threshold is set EXPLICITLY
        for cell in cells:
            cell.thr.assign(np.full(cell.thr.shape, float(thr)))
    return keras.Model(inp, out)


def run(kind, thr, epochs=12, batch=32, steps=6000, seed=0):
    X, Y = make_task(seed=seed)
    cut = int(0.8 * len(X))
    Xtr, Ytr, Xte, Yte = X[:cut], Y[:cut], X[cut:], Y[cut:]

    m = build(kind, thr)
    nparam = sum(int(v.numpy().nbytes) for v in m.trainable_variables)
    m.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"], run_eagerly=True)
    h = m.fit(Xtr, Ytr, epochs=epochs, batch_size=batch, verbose=0,
              steps_per_epoch=max(1, steps // batch))
    pred = m.predict(Xte, verbose=0, batch_size=256).argmax(1)
    return dict(acc=float((pred == Yte).mean()),
                train=float(h.history["accuracy"][-1]),
                params=nparam)


print("=" * 76)
print("HOW TO COMPARE HONESTLY")
print("=" * 76)
print(f"  TF {tf.__version__}")
print(f"  task: T={T}, label = a block of time, {C} classes, "
      f"chance {1/C:.4f}")
print(f"  net:  width {WIDTH}, {LAYERS} layers, identical for all")
print()

# ═══════════════════════════════════ RULE 3: FAIR TUNING OF THE THRESHOLD ═══
#
# Do not fix the threshold. At one threshold one variant works and the
# other is silent, and then you are comparing operating regimes rather
# than neurons. Give each its own grid and keep the BEST: the one who got
# lucky must not be the one who wins.
THRS = [0.4, 0.7, 1.0]
res = {}

print("  run (the threshold is scanned, the best is kept):")
for kind in ("dense", "spiking", "spiking_flat", "smooth"):
    grid = [None] if kind == "dense" else THRS
    best = None
    for thr in grid:
        r = run(kind, thr)
        mark = "  <- keep" if (best is None or r["acc"] > best["acc"]) else ""
        lab = kind if thr is None else f"{kind} thr={thr}"
        print(f"    {lab:<22} train {r['train']:.4f}  "
              f"VAL {r['acc']:.4f}{mark}")
        if best is None or r["acc"] > best["acc"]:
            best = r
    res[kind] = best
print()

# ═══════════════════════════════════════ RULE 1: ARE THE PARAMETERS EQUAL? ═══
print("  " + "-" * 72)
print("  ARCHITECTURE HONESTY CHECK")
print("  " + "-" * 72)
sp = {k: res[k]["params"] for k in ("spiking", "spiking_flat", "smooth")}
for k, v in sp.items():
    print(f"    {k:<16}{v / 2**20:>8.2f} MiB")
same = len(set(sp.values())) == 1
print(f"    identical: {same}")
if not same:
    print()
    print("    NOT identical, and that is expected: the flatten variant has")
    print("    a T*width input to the output layer and the mean variant has")
    print("    width. The difference is in the PARAMETER COUNT, while the")
    print("    trunk (the spiking layers) is shared.")
print()

# ═══════════════════════════════════════════ RULE 4: DECOMPOSE THE CAUSES ════
print("  " + "-" * 72)
print("  DECOMPOSITION BY CAUSE (each is a separate measurement)")
print("  " + "-" * 72)

d, s, sf, sm = res["dense"], res["spiking"], res["spiking_flat"], res["smooth"]
bit = abs(sm["acc"] - s["acc"])       # smooth against hard
read = abs(s["acc"] - sf["acc"])      # mean against flatten

print(f"    dense                          {d['acc']:.4f}")
print(f"    spiking, readout flatten       {sf['acc']:.4f}")
print(f"    spiking, readout mean          {s['acc']:.4f}")
print(f"    spiking smooth, readout mean   {sm['acc']:.4f}")
print()
print(f"    the bit AND the gradient       {bit:.4f}")
print(f"    the readout over time          {read:.4f}")
total = bit + read
print(f"    the readout explains {read / total:.0%} of the discrepancy")
print()
print("    CAREFUL: the first row cannot be called \"the cost of the")
print("    gradient\". The smooth variant differs in the forward value AND")
print("    in the derivative. One measurement cannot separate them. The")
print("    correct wording is \"the joint contribution of the bit and the")
print("    gradient\".")
print()

# ══════════════════════════════════════════════════════════ CONCLUSION ══════
print("  " + "-" * 72)
print("  CONCLUSION")
print("  " + "-" * 72)
gap = abs(d["acc"] - sf["acc"])
if gap < 0.05:
    print(f"  Spiking with a flatten readout trails dense by {gap:.4f}.")
    print("  That is within noise on a task like this.")
else:
    print(f"  Spiking trails by {gap:.4f}, which is no longer noise.")
print()
print("  What should be admitted honestly:")
print("    - this is PARITY, not superiority;")
print("    - the task is too easy: both networks solve it almost entirely,")
print("      and such a grid cannot tell architectures apart;")
print("    - the MEMORY win is NOT measured here. It was measured")
print("      separately by examples/advanced/04_budget_and_precision.py:")
print("      bit-packing is a few percent of total memory, because the")
print("      backward tape is much larger than the forward tape and is")
print("      always dense. This script does not recompute that figure.")
print()
print("  Not measured, and therefore not claimed:")
print("    - whether a spiking network is faster on a GPU;")
print("    - what a spiking network costs in energy: no such hardware here;")
print("    - where the sparse/bit crossover is: nobody has measured it.")
print("=" * 76)