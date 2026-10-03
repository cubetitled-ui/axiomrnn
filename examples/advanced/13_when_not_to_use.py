"""
13_when_not_to_use.py -- the honest case against this framework.

Run:  python3 examples/advanced/13_when_not_to_use.py     (no TensorFlow)

WHAT THIS TEACHES
=================
A README that only lists strengths is a sales page. This one is the other
half: where axiomrnn is the wrong tool, what it costs, and what we would
use instead.

Everything below is a measured number or a stated limitation. Nothing
here is hedging, and nothing here is a promise to fix it later.

THE SHORT VERSION
=================
  Use it when       you know your memory budget and it constrains you.
  Use it when       you want a spiking neuron without forking a kernel.
  Use it when       you want to plan on a machine with no GPU.

  Do NOT use it when you want the best accuracy on a GPU. We measured
                   parity with a dense MLP, not superiority. If accuracy
                   is the goal, a dense MLP is simpler and no worse.
  Do NOT use it when you want low energy. There is no neuromorphic
                   hardware on our test machine, so the energy claim is
                   UNMEASURED. Spikes are cheap only where the hardware
                   makes them cheap.
  Do NOT use it when you want speed. run_eagerly=True is required for
                   correctness, and it costs. Not benchmarked either way.
  Do NOT use it when your problem has no temporal structure. Then there
                   is nothing for the recurrence to carry.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import axiomrnn as ax
from axplan.credit import CreditRule, NeuronModel, credit_bytes, crossover_batch


def main():
    print("=" * 78)
    print("WHEN NOT TO USE THIS")
    print("=" * 78)

    # ── 1. accuracy: parity, not superiority ─────────────────────────────────
    print()
    print("  1. IF YOU WANT ACCURACY, USE A DENSE MLP")
    print("  " + "-" * 74)
    print("""
      MEASURED, same task, same width, same budget (example 04):

        dense MLP                          0.9981
        spiking, readout flatten           0.9912
        spiking, readout mean              0.7444

      The spiking net is 0.0069 behind. On this task that is noise, and
      the honest reading is PARITY. It is not a win, and we do not
      describe it as one.

      A dense MLP is also:
        - simpler, with fewer things that can go wrong;
        - faster, because there is no loop over time in the forward pass;
        - supported by every tool in the ecosystem.

      Pick spiking when the DEPLOYMENT is neuromorphic, or when the memory
      model is the binding constraint and you have measured that it helps.
      Not because the name sounds better.
    """)

    # ── 2. energy: not measured ──────────────────────────────────────────────
    print("  2. IF YOU WANT LOW ENERGY, WE CANNOT HELP YOU YET")
    print("  " + "-" * 74)
    print("""
      The entire motivation for spiking networks is energy on neuromorphic
      hardware. We have a GPU and a CPU on this machine. There is no
      neuromorphic device here.

      So the energy claim is NOT MEASURED, and we will not put a number
      next to it. Anyone who quotes you an energy figure for a spiking
      network running on a GPU has not measured it either.

      What is true: a binary activation can be stored in one bit, and one
      bit is cheaper than 16. What is also true: the BACKWARD tape is
      dense regardless, and it is about 30x larger than the forward tape.
      Those two facts point in opposite directions, and the net result is
      the bit-packing measurement below.
    """)

    # ── 3. the bit-packing disappointment, with numbers ──────────────────────
    print("  3. BIT-PACKING, MEASURED AT FIVE SCALES")
    print("  " + "-" * 74)
    print(f"  {'config':<26}{'BPTT':>14}{'e-prop':>12}{'ratio':>8}")
    print("  " + "-" * 60)
    for L, h, T, b in [(8, 1024, 32, 8), (12, 1536, 48, 12),
                       (16, 2048, 64, 16), (24, 3072, 96, 24),
                       (32, 4096, 128, 32)]:
        vals = {}
        for rule in (CreditRule.BPTT, CreditRule.LOCAL):
            vals[rule] = credit_bytes(rule, NeuronModel.LIF,
                                      L * h * h, T, b, h, L).bytes_total
        cfg = f"L={L} h={h} T={T} b={b}"
        print(f"  {cfg:<26}{vals[CreditRule.BPTT] / 2**20:>11.1f} MiB"
              f"{vals[CreditRule.LOCAL] / 2**20:>9.1f} MiB"
              f"{vals[CreditRule.BPTT] / vals[CreditRule.LOCAL]:>7.1f}x")
    print()
    print("  The ratio peaks and then FALLS: 17x at T=32, under 9x at")
    print("  T=128. Do not carry the headline number to long horizons")
    print("  without re-measuring. It is a function of T, not a constant.")
    print()

    # ── 4. what it DOES cost ─────────────────────────────────────────────────
    print("  4. WHAT THE FRAMEWORK COSTS YOU")
    print("  " + "-" * 74)
    print("""
      run_eagerly=True is mandatory. Keras 3 caches a forward pass that
      contains a Python loop over time, and the gradient then lands on
      STALE activations. We measured it with a call counter: the forward
      ran 2 times where 4 were needed, and 0 times on a repeat fit.

      The proper fix is tf.while_loop over tf.Variable state, which would
      make the layer graph-native and let the cache be correct. NOT DONE.

      What it costs in practice, measured on this machine:
        the same model trains to the same accuracy with and without a GPU,
        and the GPU does not make the Python loop faster, because the loop
        is the bottleneck, not the matmul.

      So: expect to be slower than a dense net of the same size, and do
      not expect the GPU to rescue you.
    """)

    # ── 5. tasks with no temporal structure ──────────────────────────────────
    print("  5. IF YOUR PROBLEM HAS NO TEMPORAL STRUCTURE")
    print("  " + "-" * 74)
    print("""
      A recurrent network carries state across time. On tabular data, or
      on a fixed image, there is no time axis to carry anything along, and
      the recurrence is pure overhead.

      There is a measurable symptom and it is not obvious: with a
      mean-over-time readout on a task that does NOT depend on WHEN
      something happened, you lose 0.25 accuracy. That gap is not about
      spiking at all -- it is the readout averaging away information. On a
      time-independent task the information was never there to begin with.

      Check: does your label depend on order? If you shuffle the time axis
      and the answer does not change, a dense MLP is the right model.
    """)

    # ── 6. what to use instead ───────────────────────────────────────────────
    print("  6. WHAT TO USE INSTEAD")
    print("  " + "-" * 74)
    print(f"""
      best accuracy on a GPU           a plain dense MLP. Measured 0.9981
                                      against our 0.9912.
      lowest energy, neuromorphic      whatever the vendor's SDK provides,
                                      not this. We cannot measure it.
      fastest training                 anything without a Python loop over
                                      time. We are not that.
      planning only, no GPU            keep this: axplan does not import
                                      TensorFlow at all. That part is real.
      your own neuron                  keep this too: the extension point
                                      is verified, and the derivative hooks
                                      are checkable against finite
                                      differences.

      THE PLANNER IS THE PRODUCT. The spiking layer is a way to have
      something real to plan for. If planning is not your problem, this
      framework is not solving your problem.
    """)

    # ── 7. the crossover, for the case that IS ours ──────────────────────────
    print("  7. AND WHEN IT IS THE RIGHT TOOL")
    print("  " + "-" * 74)
    print(f"  {'T':>6}{'crossover batch':>20}   reading")
    for T in (32, 64, 128, 256, 512):
        xb = crossover_batch(T, hidden=1024, n_layers=16)
        reading = ("BPTT cheaper for most batches" if xb > 32
                   else "worth considering" if xb > 8
                   else "the local rule pays at almost any batch")
        print(f"  {T:>6}{xb:>20.0f}   {reading}")
    print()
    print("  If your horizon is long AND your batch is small, you are in")
    print("  the case this exists for. Measure the crossover yourself; it")
    print("  moves with hidden size and layer count.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())