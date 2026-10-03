"""
03_credit_rule.py -- the one lever that actually moves memory.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/03_credit_rule.py

WHAT THIS TEACHES
=================
Bit-packing is the lever everyone expects, and it is nearly worthless on a
GPU: measured 16.8x on its own line, but that line is 0.16-0.32% of memory,
so the overall effect is 2.6-5.1%.

The reason is one sentence: binary activations do NOT make the gradient
binary. The backward tape is always dense, and it is 30x larger than the
forward tape.

The real lever is the CREDIT RULE:

    BPTT   -> Theta(T * A), memory grows with the horizon
    e-prop -> Theta(S),      T-independent

This example measures the ratio across scales, and shows where the quality
price lands. No TensorFlow needed for the ratio table.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import axiomrnn as ax
from axplan.credit import (CreditRule, NeuronModel, credit_bytes,
                           local_quality, crossover_batch)


def main():
    print("=" * 78)
    print("THE CREDIT RULE IS THE LEVER -- NOT THE BIT")
    print("=" * 78)
    print()

    print("  PART 1 -- WHY BIT-PACKING BARELY MATTERS HERE")
    print("  " + "-" * 74)
    print("  QUOTED, not recomputed in this script -- run")
    print("  examples/advanced/04_budget_and_precision.py for the live")
    print("  figures, and trust those over this block:")
    print("""
    bit-packing on its own line     16.8x
    but that line is                0.16-0.32% of memory
    so the overall effect is        2.6-5.1%

  One sentence explains it: the backward tape is much larger than the
  forward tape, and the backward tape is ALWAYS dense, because the
  surrogate derivative is non-zero everywhere. Binary activations do
  not make the gradient binary.
""")

    print("  PART 2 -- THE RATIO THAT ACTUALLY MATTERS")
    print("  " + "-" * 74)
    print(f"  {'config':<26}{'BPTT':>13}{'e-prop':>13}{'ratio':>9}")
    print("  " + "-" * 61)
    for L, h, T, b in [(8, 1024, 32, 8), (16, 2048, 64, 16),
                       (32, 4096, 128, 32)]:
        vals = {}
        for rule in (CreditRule.BPTT, CreditRule.LOCAL):
            p = credit_bytes(rule, NeuronModel.LIF, L * h * h, T, b, h, L)
            vals[rule] = p.bytes_total
        cfg = f"L={L} h={h} T={T} b={b}"
        print(f"  {cfg:<26}{vals[CreditRule.BPTT] / 2**20:>10.1f} MiB"
              f"{vals[CreditRule.LOCAL] / 2**20:>10.1f} MiB"
              f"{vals[CreditRule.BPTT] / vals[CreditRule.LOCAL]:>8.1f}x")
    print()
    print("  Note how the ratio FALLS at T=128 (17x -> 8.7x). Do not carry")
    print("  the headline number over to long horizons without re-measuring.")
    print()

    print("  PART 3 -- WHEN THE LOCAL RULE IS WORTH IT")
    print("  " + "-" * 74)
    print("  The crossover batch FALLS as the horizon grows:")
    print()
    print(f"  {'T':>6}{'crossover batch':>20}   meaning")
    for T in (16, 32, 64, 128, 256):
        xb = crossover_batch(T, hidden=768, n_layers=12)
        meaning = ("local rule wins at almost any batch" if xb <= 4
                   else "local rule needs a decent batch"
                   if xb <= 32 else "BPTT is cheaper for most batches")
        print(f"  {T:>6}{xb:>20.0f}   {meaning}")
    print()
    print("  Below the crossover, the local rule is extra complexity with no")
    print("  memory win. That is a decision, not a detail -- and it moves")
    print("  the opposite way from what people assume.")
    print()

    print("  PART 4 -- WHAT IT COSTS IN QUALITY")
    print("  " + "-" * 74)
    print("  Measured (FPTT, arXiv:2112.11231v2):")
    print("    T=60  -> 85.5% of full BPTT")
    print("    T=500 -> 38.9% of full BPTT")
    print("    a 47-point gap")
    print()
    print(f"  {'T':>6}{'quality kept':>16}   note")
    for T in (16, 60, 200, 500, 2000):
        q = local_quality(T)
        note = ("no data -- we do NOT extrapolate" if q is None
                else f"~{q:.1%} of BPTT")
        print(f"  {T:>6}{(f'{q:.1%}' if q is not None else 'None'):>16}   {note}")
    print()
    print("  Beyond T=500 the function returns None rather than a guess.")
    print("  A linear extrapolation would read 0.0 at T=1024, which would")
    print("  look like a claim that the local rule is useless. We simply")
    print("  do not know, and the API says so.")
    print()

    print("  PART 5 -- WHAT TO DO ABOUT IT IN YOUR PROJECT")
    print("  " + "-" * 74)
    spec = ax.Model()
    spec.add_spiking(1024, horizon=64)
    spec.add_spiking(1024, horizon=64)
    spec.add_dense(1000)
    txt = spec.explain(ax.Budget.consumer_6gb(), batch=32, verbose=False)
    lines = txt.splitlines()
    start = next(i for i, l in enumerate(lines)
                 if "THE MAIN MEMORY LEVER" in l)
    stop = next((i for i, l in enumerate(lines)
                 if "VERIFIED, AND NOT" in l and i > start), len(lines))
    for line in lines[start:stop]:
        if line.strip():
            print("  " + line.rstrip())
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())