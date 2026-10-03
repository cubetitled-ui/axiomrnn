"""
14_budget_matrix.py -- one architecture, every budget, side by side.

Run:  python3 examples/advanced/14_budget_matrix.py     (no TensorFlow)

WHAT THIS TEACHES
=================
The claim this framework makes is that the budget is an INPUT. This script
makes that literal: the same architecture, planned against every hardware
profile, and then the largest thing that fits on each.

What you get is a table you can paste into a design document, plus the
point where the planner starts refusing -- which is the number that
actually drives the decision.

No TensorFlow needed. That is the point.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import axiomrnn as ax

GIB = 1 << 30
WIDTHS = (1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768)


def build(width, horizon=128, layers=2, classes=1000):
    s = ax.Model()
    for _ in range(layers):
        s.add_spiking(width, horizon=horizon)
    s.add_dense(classes)
    return s


def main():
    print("=" * 78)
    print("ONE ARCHITECTURE, EVERY BUDGET")
    print("=" * 78)
    print(f"  {'TensorFlow installed:':<28} {ax._HAS_TF}")
    print("  2 spiking layers, T=128, output into 1000 classes, batch 32")
    print()

    budgets = [
        ax.Budget.consumer_6gb(),
        ax.Budget.consumer_8gb(),
        ax.Budget.pro_24gb(),
        ax.Budget(48.0, host_ram_gb=128.0, name="pro 48 GB"),
        ax.Budget.datacenter_80gb(),
    ]

    # ── 1. usable memory, which is not the marketing number ─────────────────
    print("  1. USABLE MEMORY")
    print("  " + "-" * 74)
    print(f"  {'profile':<20}{'vram':>8}{'usable':>11}{'overhead':>11}")
    for b in budgets:
        over = b.vram_gb * GIB - b.usable_bytes()
        print(f"  {b.name:<20}{b.vram_gb:>7.0f}G"
              f"{b.usable_bytes() / GIB:>9.2f} GiB"
              f"{over / GIB:>9.2f} GiB")
    print()
    print("  The overhead is the CUDA context, cuBLAS workspaces and")
    print("  allocator fragmentation. It is real memory, and a planner")
    print("  that ignores it promises a number you will hit at 2am.")
    print()

    # ── 2. the whole matrix ──────────────────────────────────────────────────
    print("  2. THE MATRIX")
    print("  " + "-" * 74)
    head = f"  {'width':>7}{'peak':>10}"
    for b in budgets:
        head += f"{b.name.replace(' ', '_')[:11]:>13}"
    print(head)
    print("  " + "-" * (7 + 10 + 13 * len(budgets)))

    fits = {b.name: [] for b in budgets}
    for width in WIDTHS:
        spec = build(width)
        row = f"  {width:>7}{spec.fit_budget(budgets[0], batch=32).peak_bytes / GIB:>9.2f}G"
        for b in budgets:
            p = spec.fit_budget(b, batch=32)
            if not p.infeasible:
                fits[b.name].append(width)
                row += f"{'yes':>13}"
            else:
                row += f"{'NO':>13}"
        print(row)
    print()
    print("  'NO' is not a failure of the tool. It is the tool answering.")
    print()

    # ── 3. the largest that fits, per budget ────────────────────────────────
    print("  3. THE LARGEST WIDTH THAT FITS")
    print("  " + "-" * 74)
    print(f"  {'profile':<20}{'usable':>11}{'max width':>12}{'headroom':>12}")
    for b in budgets:
        if not fits[b.name]:
            print(f"  {b.name:<20}{b.usable_bytes() / GIB:>9.2f} GiB"
                  f"{'none tested':>12}{'—':>12}")
            continue
        w = max(fits[b.name])
        p = build(w).fit_budget(b, batch=32)
        head = b.usable_bytes() / GIB - p.peak_bytes / GIB
        print(f"  {b.name:<20}{b.usable_bytes() / GIB:>9.2f} GiB"
              f"{w:>12}{head:>10.2f} GiB")
    print()
    print("  Read the headroom column, not the width. A width that fits")
    print("  with 0.1 GiB to spare is one allocator decision away from")
    print("  failing, and 'fits on paper' is not the same as 'runs'.")
    print()

    # ── 4. what happens when you ignore the answer ──────────────────────────
    print("  4. WHAT THE PLANNER PREVENTS")
    print("  " + "-" * 74)
    width = 24576
    spec = build(width)
    print(f"  a {width}-wide, 2-layer network at T=128:")
    print()
    txt = spec.explain(ax.Budget.consumer_6gb(), batch=32, verbose=False)
    for ln in txt.splitlines():
        s = ln.strip()
        if not s:
            continue
        if any(k in s for k in ("does not fit", "NO FIT", "peak:", "nearest",
                                "regime", "reason", "exceed", "instead")):
            print(f"    {s}")
    print()
    print("  That is the entire value proposition in one place: the")
    print("  failure arrives as a number at design time, not as an")
    print("  OutOfMemoryError an hour into training.")
    print()

    # ── 5. the knob that always works ───────────────────────────────────────
    print("  5. THE KNOB THAT ALWAYS WORKS, AND WHAT IT COSTS")
    print("  " + "-" * 74)
    print(f"  {'trade':<38}{'peak':>10}{'vs naive':>10}")
    print("  " + "-" * 58)
    b6 = ax.Budget.consumer_6gb()
    base = build(24576).fit_budget(b6, batch=32).peak_bytes
    for label, s in [
        ("width 24576, batch 32", build(24576)),
        ("width 16384, batch 32", build(16384)),
        ("width 24576, batch 4", build(24576)),
        ("width 24576, T=32", build(24576, horizon=32)),
        ("width 24576, batch 32, 1 layer", build(24576, layers=1)),
    ]:
        p = s.fit_budget(b6, batch=32)
        ratio = p.peak_bytes / base if base else 0
        tag = f"{ratio:.2f}x" if p.peak_bytes < base else "---"
        print(f"  {label:<38}{p.peak_bytes / GIB:>8.2f}G{tag:>10}")
    print()
    print("  The horizon is usually the cheapest knob, because memory")
    print("  grows with T under BPTT while the weights do not change.")
    print("  Which one you pick depends on whether your problem can")
    print("  actually live with a shorter horizon -- a question no planner")
    print("  can answer for you.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())