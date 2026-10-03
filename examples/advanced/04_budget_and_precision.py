"""
Budget, codec and precision.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/04_budget_and_precision.py

WHAT THIS TEACHES
=================
Four budget levers, each measured rather than asserted:

  1. which hardware profile you are actually on;
  2. which codec the planner picks, and WHY it changes;
  3. what precision does to weights, and where the floor is;
  4. when the plan does not fit, and what it says about it.

The last one matters most: a tool that always returns a plan is
lying to you somewhere.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import axiomrnn as ax
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
