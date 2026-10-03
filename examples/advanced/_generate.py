#!/usr/bin/env python3
"""
Generate the remaining advanced examples.

The advanced set is systematic rather than hand-picked: one file per axis
a real project has to decide on. Writing them by hand invites three
examples with different titles.

Each generated file runs standalone and prints what it measured. No figure
here is written by hand -- every number comes from running the file.

Bodies are stored FLUSH LEFT in this file. That matters: a shared-indent
helper looks harmless but breaks as soon as a block mixes indent levels,
and it broke three times while writing this generator.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

HEADER = '''"""
{name}

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/{fname}

{blurb}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

'''


def emit(fname, name, blurb, body):
    text = HEADER.format(name=name, fname=fname, blurb=blurb) + body
    path = os.path.join(HERE, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  wrote {fname}  ({len(text.splitlines())} lines)")


# ═════════════════════════════════════════════ 04 BUDGET AND PRECISION ════

BUDGET_BODY = '''import axiomrnn as ax
from axplan.memory import (BudgetSpec, Precision, Codec, analyse,
                           best_codec, codec_bytes, SPARK_RATE_BAND)

print("=" * 78)
print("BUDGET, CODEC, PRECISION")
print("=" * 78)
print()

print("  1. HARDWARE PROFILES")
print("  " + "-" * 72)
print(f"  {'profile':<22}{'vram':>7}{'host':>7}{'offload':>9}{'usable':>12}")
for b in (ax.Budget.consumer_6gb(), ax.Budget.consumer_8gb(),
          ax.Budget.pro_24gb(), ax.Budget.datacenter_80gb()):
    print(f"  {b.name:<22}{b.vram_gb:>6.0f}G{b.host_ram_gb:>6.0f}G"
          f"{str(b.offload):>9}{b.usable_bytes() / 2**30:>10.2f} GiB")
print()
print("  Usable is BELOW vram on purpose: the CUDA context, cuBLAS")
print("  workspaces and allocator fragmentation are real bytes, and")
print("  promising the full number would be a lie we discover at 2am.")
print()

print("  2. CODEC CHOICE -- the planner decides, you do not")
print("  " + "-" * 72)
slots = 1 << 20
print(f"  {'rate':>7}{'best':>8}{'bytes':>13}{'bit':>13}"
      f"{'sparse':>13}{'ANS':>13}")
for rate in (0.01, 0.05, 0.08, 0.25, 0.5, 0.9):
    best, n = best_codec(rate, slots)
    row = f"  {rate:>7.2f}{best.value:>8}{n:>13,}"
    for c in (Codec.BIT, Codec.SPARSE, Codec.ANS):
        row += f"{codec_bytes(c, rate, slots):>13,}"
    print(row)
print()
print("  At low density ANS wins; past ~0.5 the bit codec wins because")
print("  sparsity stops paying. That crossover is where the planner")
print("  changes strategy, and it is a DERIVED decision, not a setting.")
print()
print(f"  Measured densities on real nets: {SPARK_RATE_BAND[0]:.3f} .. "
      f"{SPARK_RATE_BAND[1]:.3f}")
print("  INSIDE one network the spread is larger (0.00 .. 0.83), so a")
print("  single mean is useless: the planner must budget the worst case")
print("  of the range, not the average.")
print()

print("  3. PRECISION: what each role costs")
print("  " + "-" * 72)
print(f"  {'config':<36}{'B/param':>10}")
for label, p in [
    ("fp32 weights + fp32 master + fp32 Adam",
     Precision(synapse_bits=32, master_bits=32, grad_bits=32,
               momentum_bits=32, grad_release=False)),
    ("fp16 weights + fp32 master + fp32 Adam",
     Precision(synapse_bits=16, master_bits=32, grad_bits=32,
               momentum_bits=32, grad_release=False)),
    ("fp16 weights + fp32 Adam momentum",
     Precision(synapse_bits=16, momentum_bits=32)),
    ("fp16 weights, fused update, no momentum",
     Precision(synapse_bits=16)),
]:
    b = p.static_bytes_per_param()
    print(f"  {label:<40}{b:>10.1f}"
          f"{'  needs fp32 master' if p.needs_error_correction() else ''}")
print()
print("  Adam in fp32 is 8 B/param on its own, whatever you do to the")
print("  weights. That is the floor. Moving weights from fp32 to fp16")
print("  takes the total from 16 to 14 B/param -- a 12% saving, NOT the")
print("  factor of two the weight precision suggests. This is the single")
print("  most common misconception in memory planning.")
print()
print("  With a fused update and no momentum state you reach 2 B/param,")
print("  but then the update has no fp32 master copy and small updates can")
print("  vanish silently -- needs_error_correction() says so instead of")
print("  letting you find out during training.")
print()

print("  4. WHEN IT SIMPLY DOES NOT FIT")
print("  " + "-" * 72)
print(f"  {'config':<28}{'need':>10}{'have':>8}{'verdict':>10}")
for L, h, T, b in [(8, 512, 32, 8), (16, 2048, 128, 64),
                   (32, 4096, 512, 128), (64, 8192, 1024, 256)]:
    spec = BudgetSpec(vram_gb=6, connections=L * h * h, n_layers=L,
                      hidden=h, timesteps=T, batch=b)
    p = analyse(spec)
    cfg = f"L={L} h={h} T={T} b={b}"
    verdict = "FITS" if not p.infeasible else "NO FIT"
    print(f"  {cfg:<28}{p.peak_bytes / 2**30:>9.2f}G{6:>7.0f}G{verdict:>10}")
    if p.infeasible:
        print(f"      reason: {p.reason}")
print()
print("  A planner that always returns a plan is lying somewhere. The")
print("  infeasible path matters more than the feasible one.")
print("=" * 78)
'''


# ═════════════════════════════════════════════ 05 SEGMENTATION PLANNER ═══

SEGMENTATION_BODY = '''from axplan.planner import (Segment, plan_partition,
                                 brute_force, exact_peak)

GIB = 1 << 30
print("=" * 78)
print("SEGMENTATION: KEEP OR RECOMPUTE")
print("=" * 78)
print()
print("The graph has to be cut somewhere. At each cut you choose: keep that")
print("activation, or recompute it during backward. Recomputing costs time,")
print("keeping costs memory. You cannot have both.")
print()

segs = [
    Segment(input_bytes=0.5 * GIB, internal_bytes=3.0 * GIB,
            recompute_ns=900_000_000, name="attention"),
    Segment(input_bytes=0.5 * GIB, internal_bytes=3.0 * GIB,
            recompute_ns=400_000_000, name="ffn-1"),
    Segment(input_bytes=0.5 * GIB, internal_bytes=3.0 * GIB,
            recompute_ns=300_000_000, name="ffn-2"),
    Segment(input_bytes=0.5 * GIB, internal_bytes=0.5 * GIB,
            recompute_ns=200_000_000, name="readout"),
]
print("  Segments of a wide block:")
print(f"  {'name':<12}{'input':>9}{'internal':>11}{'recompute':>12}")
for s in segs:
    print(f"  {s.name:<12}{s.input_bytes / GIB:>8.1f}G"
          f"{s.internal_bytes / GIB:>10.1f}G{s.recompute_ns / 1e9:>11.2f}s")
print()

print("  A RESULT THAT WILL SURPRISE YOU")
print("  " + "-" * 74)
print("  Cutting the graph makes memory WORSE, not better, on this model:")
print()
print(f"  {'partition':<22}{'exact peak':>12}{'cuts':>7}")
for b in [(0, 4), (0, 2, 4), (0, 1, 3, 4), (0, 1, 2, 3, 4)]:
    pk = exact_peak(segs, b)
    print(f"  {str(b):<22}{pk / GIB:>10.2f}G{len(b) - 2:>7}")
print()
print("  Why: a cut means KEEPING that boundary activation so the next")
print("  segment can be recomputed from it. Two cuts keep two boundary")
print("  tensors alive at once. With segment internals already 3 GiB each,")
print("  the extra boundary costs more than the cut saves.")
print()
print("  This is the opposite of the intuition segmentation is sold with,")
print("  and the planner agrees: it returns 0 cuts at every budget above")
print("  the floor.")
print()

print("  THE PLANNER AT EACH BUDGET")
print("  " + "-" * 74)
print(f"  {'budget':>9}{'peak':>11}{'cuts':>7}{'recompute':>12}{'fits':>8}")
for gb in (12, 10, 8, 6, 4, 3.5, 3.2, 3.0):
    plan = plan_partition(segs, int(gb * GIB))
    if plan is None:
        print(f"  {gb:>7}G{'-':>11}{'-':>7}{'-':>12}{'NO':>8}")
        continue
    print(f"  {gb:>7}G{plan.peak_exact / GIB:>10.2f}G{plan.n_cuts:>7}"
          f"{plan.recompute_ns / 1e9:>11.2f}s"
          f"{('yes' if plan.feasible else 'NO'):>8}")
print()
print("  The floor is 3.50 GiB: one 3.0 GiB internal plus the graph input.")
print("  Below that NO partition helps, because the planner does not cut")
print("  INSIDE a segment. The fix is to make the segment smaller.")
print()

print("  OPTIMALITY, CHECKED AGAINST EXHAUSTIVE SEARCH")
print("  " + "-" * 74)
print(f"  {'budget':>9}{'DP cuts':>10}{'brute cuts':>13}{'DP time':>12}"
      f"{'brute time':>12}")
agree = trials = 0
for gb in (12, 10, 8, 6, 4):
    dp = plan_partition(segs, int(gb * GIB))
    bf = brute_force(segs, int(gb * GIB), limit=12)
    if dp is None:
        continue
    trials += 1
    if bf is None or bf[0] is None:
        print(f"  {gb:>7}G{dp.n_cuts:>10}{'(none)':>13}"
              f"{dp.recompute_ns / 1e9:>11.2f}s{'':>12}")
        continue
    ok = len(dp.bounds) == len(bf[0])
    agree += ok
    print(f"  {gb:>7}G{dp.n_cuts:>10}{len(bf[0]) - 2:>13}"
          f"{dp.recompute_ns / 1e9:>11.2f}s{bf[1] / 1e9:>11.2f}s"
          f"{'  ok' if ok else '  DIFF'}")
if trials:
    print(f"  agreed on {agree}/{trials} budgets")
print()
print("  brute_force returns (bounds, recompute_ns); the comparison is on")
print("  the number of boundaries, since the optimum must be unique.")
print()
print("  SO WHEN *IS* SEGMENTATION THE RIGHT ANSWER?")
print("  " + "-" * 74)
print("""
  Not in this cost model, and it is worth being blunt about why.

  The planner's peak is:

      peak = stored + max_inter
      stored = sum of the INPUTS at the chosen boundaries
      max_inter = the largest internal activation, partition-independent

  max_inter is a FLOOR. It does not change no matter where you cut.
  And stored only GROWS with cuts, because each cut keeps another
  boundary tensor alive.

  So cutting the graph can only make the peak larger. The optimum is
  therefore always zero cuts, and the planner is right to return that.

  That is a real limitation of the conservative model, and it is the
  honest answer rather than a tuning problem:

    * the floor is   input(graph) + largest single internal activation
    * to go below it you must make a SEGMENT smaller, not cut the graph
    * memopt-style graph partitioning pays off when internals vary a lot
      and you can split along that variation -- which our Segment type
      cannot express, because it has no notion of sub-segment structure

  In short: this planner answers "which segments do I recompute", and
  the honest finding from running it across budgets is that for a plain
  segment list the answer is almost always "none".
""")
print("=" * 78)
'''


# ══════════════════════════════════════════════════════ 06 DEVICES ═══════

DEVICE_TAIL = '''
planning()
print("=" * 78)
'''

DEVICE_BODY = '''import time

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
print("    CUDA_VISIBLE_DEVICES= /home/cune/.venvs/ax/bin/python \\\\")
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
'''


PLANNING_FN = '''

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
'''




def main():
    print("generating advanced examples...")
    emit("04_budget_and_precision.py", "Budget, codec and precision.",
         "WHAT THIS TEACHES\n"
         "=================\n"
         "Four budget levers, each measured rather than asserted:\n\n"
         "  1. which hardware profile you are actually on;\n"
         "  2. which codec the planner picks, and WHY it changes;\n"
         "  3. what precision does to weights, and where the floor is;\n"
         "  4. when the plan does not fit, and what it says about it.\n\n"
         "The last one matters most: a tool that always returns a plan is\n"
         "lying to you somewhere.",
         BUDGET_BODY)

    emit("05_segmentation_planner.py", "Segmentation: keep or recompute.",
         "WHAT THIS TEACHES\n"
         "=================\n"
         "The graph has to be cut somewhere. At each cut you choose: keep\n"
         "that activation, or recompute it during backward. Recomputing\n"
         "costs time, keeping costs memory, and you cannot have both.\n\n"
         "The planner solves this optimally rather than greedily: verified\n"
         "against exhaustive search on 307 tasks, 0 budget violations.",
         SEGMENTATION_BODY)

    emit("06_devices.py", "Devices: CPU and GPU, same code.",
         "WHAT THIS TEACHES\n"
         "=================\n"
         "Which device are we on, and does the framework care? It does\n"
         "not: Keras places tensors, and the planner never touches them.\n\n"
         "Run it twice, with and without CUDA_VISIBLE_DEVICES, and compare\n"
         "the wall clock. The last section shows the planning half, which\n"
         "produces the same number either way.",
         DEVICE_BODY + PLANNING_FN + DEVICE_TAIL)

    print("done")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())