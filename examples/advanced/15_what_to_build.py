"""
What should I build? -- the inverse question.

Every other tool answers "what does THIS network cost?". This one answers
"I have this much memory; what is the largest network I can train, and what
exactly stops me from going one step further?"

Runs on a bare python3. No TensorFlow, no GPU: the whole answer comes from the
memory model in axplan, which is the part of this package that does not need a
neural network to be useful.

WHAT IT IS NOT
--------------
It is not a quality optimiser. This package has no verified quality model, and
inventing one would be the worst thing it could do. So the answer is a
CAPACITY answer, and the difference matters: "2.1 billion synapses at 2 bytes
fit in 6 GiB" is arithmetic; "this network will learn better" is a claim
nobody here has measured.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import axiomrnn as ax   # noqa: E402

GIB = 2 ** 30


def header(t):
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


def main():
    header("1. THE FORWARD QUESTION (what the rest of the package does)")
    m = ax.Model()
    m.add_spiking(1024, horizon=128)
    m.add_spiking(1024, horizon=128)
    plan = m.fit_budget(ax.Budget.consumer_6gb(), batch=16)
    print(plan.explain())
    print("  -> a verdict on ONE architecture. Which knob to turn is your job.")

    header("2. THE INVERSE QUESTION (what this example is about)")
    print(ax.capacity_report(6.0, hidden=32768, n_layers=32,
                             timesteps=256, batch=8))
    print()
    print("Read it as two numbers:")
    print("  * how large a network fits;")
    print("  * the smallest step up that does NOT fit, and by how much.")
    print("The second line is the one a forward-only report can never give you.")

    header("3. THE BINDING TERM TELLS YOU WHICH KNOB IS WORTH TURNING")
    best, fail = ax.largest_that_fits(6.0, hidden=32768, n_layers=32,
                                      timesteps=256, batch=8)
    for label, c in (("fits", best), ("overflows", fail)):
        if c is None:
            print(f"  {label}: none in the searched space")
            continue
        print(f"  {label}: hidden={c.hidden} layers={c.n_layers} "
              f"T={c.timesteps} batch={c.batch}")
        print(f"           binding term = {c.binding}  "
              f"({c.peak_bytes/GIB:.2f} GiB peak of {c.vram_bytes/GIB:.2f})")
    if best is not None and best.binding == "static":
        print()
        print("  binding term = static means the WEIGHTS dominate, so the")
        print("  credit rule and the spike coding buy nothing here. Raising")
        print("  width or depth is the only lever that moves the number.")
    if best is not None and best.binding in ("tape", "backward"):
        print()
        print("  binding term is on the tape: the credit rule now decides")
        print("  whether this budget grows with the horizon.")

    header("4. THE SAME BUDGET UNDER A LOCAL CREDIT RULE")
    best_bptt, _ = ax.largest_that_fits(6.0, hidden=32768, n_layers=32,
                                        timesteps=256, batch=8,
                                        credit_rule="bptt")
    best_local, _ = ax.largest_that_fits(6.0, hidden=32768, n_layers=32,
                                         timesteps=256, batch=8,
                                         credit_rule="local")
    print(f"  BPTT   : {best_bptt.connections:,} synapses "
          f"at T={best_bptt.timesteps}")
    print(f"  local  : {best_local.connections:,} synapses "
          f"at T={best_local.timesteps}")
    ratio = best_local.connections / max(best_bptt.connections, 1)
    print(f"  capacity ratio: {ratio:.2f}x")
    if ratio < 1.01:
        print()
        print("  1.00x, and that is the finding: at THIS budget the weights")
        print("  dominate (binding term = static), so the credit rule has")
        print("  nothing to save. It pays only where the tape dominates.")
        print("  A capacity tool that always showed a win for the local")
        print("  rule would be advertising, not measuring.")

    q = ax.local_quality(best_local.timesteps)
    print()
    print("  And the price, from the only quality figure in the package:")
    print(f"    local rule at T={best_local.timesteps} keeps "
          + ("NO DATA (measured only to T=500)" if q is None
             else f"{q:.1%} of BPTT"))
    print("  Capacity without the quality price would be a sales pitch.")
    print("  Both numbers together are an answer.")

    header("5. WHEN NOTHING FITS, IT SAYS SO")
    nothing, why = ax.largest_that_fits(0.4, hidden=128, n_layers=64,
                                        timesteps=512, batch=32)
    print(f"  solution: {nothing}")
    print(f"  smallest searched configuration that overflows: "
          f"{why.hidden if why else None} wide, "
          f"{why.n_layers if why else None} layers, "
          f"T={why.timesteps if why else None}, "
          f"batch={why.batch if why else None}")
    print("  A planner that always returns an answer is not a planner.")

    header("6. THE SEARCH CEILING IS NOT A RESULT")
    at_ceiling, phantom = ax.largest_that_fits(6.0, hidden=8192, n_layers=32,
                                               timesteps=256, batch=8)
    print(f"  fit {at_ceiling.connections:,} synapses "
          f"(hidden={at_ceiling.hidden})")
    print(f"  reported overflow: {phantom}")
    print("  The answer sits ON the width ceiling, so no boundary exists")
    print("  above it inside the searched space. None is invented.")


if __name__ == "__main__":
    main()