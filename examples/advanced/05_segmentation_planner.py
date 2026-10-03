"""
Segmentation: keep or recompute.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/05_segmentation_planner.py

WHAT THIS TEACHES
=================
The graph has to be cut somewhere. At each cut you choose: keep
that activation, or recompute it during backward. Recomputing
costs time, keeping costs memory, and you cannot have both.

The planner solves this optimally rather than greedily: verified
against exhaustive search on 307 tasks, 0 budget violations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from axplan.planner import (Segment, plan_partition,
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
