# axiomrnn

**Design for the memory budget you have, not the memory you wish you had.**

```bash
pip install axiomrnn           # core: numpy only
pip install axiomrnn[tf]       # + Keras/TF for training
pip install axiomrnn[tf-gpu]   # + CUDA
```

```python
import axiomrnn as ax

model = ax.Model()
model.add_spiking(256, horizon=64)
model.add_dense(10)

print(model.explain(ax.Budget.consumer_6gb()))
```

`explain()` answers a question no other tool asks. Not "does it fit", but
**"what can you learn on this budget"** — and it does so without
TensorFlow installed.

---

## What this is

A budget is an input. You say how much memory there is; the planner tells
you what that buys. Nothing in the core touches TensorFlow, so planning
runs on a laptop with no GPU.

Measured on an RTX 3050, 6 GiB, CUDA 12.5, TF 2.21. What is **not** claimed
is stated in bold wherever it applies, and listed under
[Known limitations](#known-limitations).

## Three lines of honest positioning

**1. Quality: parity, not superiority.** A spiking network matches a dense
MLP on a task with temporal structure — 0.9919 against 0.9994 in the run
recorded in [`examples/README.md`](examples/README.md). The figures move
between runs, so the gap is noise on a task this easy. Spiking has never been
shown to win on accuracy here.

**2. Memory: bit-packing is nearly worthless on a GPU.** It gives 16.8x on
its own line, but that line is 0.16–0.32% of memory, so the real effect is
**2.6–5.1%**. One sentence explains it: *binary activations do not make the
gradient binary.* The backward tape is 30x larger and always dense.

**3. The lever is the credit rule.** BPTT → e-prop gives **17x** on the
line that holds ~97% of memory. At T=128 that falls to 8.7x.

So the honest value proposition is the **planner**, not the bit. The word
"spiking" in the name is not what earns the project its keep.

## The neuron is an abstraction, not a model of biology

Stated plainly, because the name invites the wrong assumption:

- The cell is a **leaky integrator with a threshold**. It is not
  Hodgkin-Huxley. There are no ion channels, no gating variables, no
  reversal potentials, and no spike-initiation mechanism.
- Adaptive thresholds and drive-gated leaks are still **LIF-family**. Calling
  them ALIF or adaptive-LIF in an example is a claim this interface cannot
  honour, so it is not done here.
- "Spiking" means the signal is a **binary event** and the gradient is an
  **explicit surrogate**, verified against finite differences on the smoothed
  model. The biological reading is a design analogy, nothing more.
- Nothing in this repository is a neuroscience claim. The honest summary of
  the neuron is: *an integer-valued activation with a hand-written derivative,
  chosen because it makes the backward pass cheap to check.*

## Quick start

```bash
python3 examples/01_minimal.py            # train a network      (needs TF)
python3 examples/02_budget.py             # plan a budget        (no TF)
python3 examples/03_custom_neuron.py       # your own neuron      (needs TF)
python3 examples/04_equal_budget.py        # compare honestly     (needs TF)
python3 examples/05_t9.py                  # next-word prediction (needs TF)
```

`examples/advanced/` adds fifteen more: embedding stems, horizon sweeps,
the credit rule, codec selection, the segmentation planner, CPU/GPU parity,
dead-neuron diagnosis, reproducibility, an API tour, save/load, the
derivative hooks, failure modes, when *not* to use this, a budget matrix, and
the inverse question — *what is the largest network this budget can train?*

Full list with measured results: [`examples/README.md`](examples/README.md).

## The inverse question

Every other tool answers *"what does this network cost?"*. This one answers
the question you actually have — *"I have this much memory; what is the
largest network I can train, and what stops me from one step further?"*

```python
import axiomrnn as ax

best, too_big = ax.largest_that_fits(6.0)      # vram_gb, in GiB
print(best.explain())        # largest that fits
print(too_big.explain())     # the boundary, verified not assumed
```

It is pure arithmetic over the verified memory model, so it runs with **no
TensorFlow installed**. It reports three things a forward-only report cannot:

- **the binding term** — weights, tape, backward, logits or workspace. That
  tells you which knob is worth turning; if the weights dominate, the credit
  rule and the spike coding buy you nothing, and the tool says so;
- **the failure boundary** — the smallest step up that does *not* fit, with
  the overshoot factor. A tool that only reports successes cannot be told
  apart from a tool that is simply optimistic;
- **refusal** — `None` when nothing in the searched space fits, and no
  invented boundary when the answer sits on the search ceiling.

One caveat it states in its own output: this is a **capacity** answer, not a
quality one. This package has no verified quality model, and the one quality
figure it does carry — what the local credit rule costs — is printed next to
the capacity gain, because a capacity number without its price is a sales
pitch.

`examples/advanced/15_what_to_build.py` walks through all of it, on CPU, with
no GPU and no TensorFlow.

## Documentation

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | what we deliberately do NOT build, and the four packages |
| [`docs/UX_RULES.md`](docs/UX_RULES.md) | 20 rules, each grown from a specific failure |
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | install, first model, first budget |
| [`docs/ATTACKING_THE_QUADRATIC.md`](docs/ATTACKING_THE_QUADRATIC.md) | an independent attack on the premise: replacing the KV cache with a spike-gated fast-weight state |
| [`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md) | the original design record (Russian, unedited) |

## Verify it yourself

```bash
python3 tests/run_all.py
```

```
  [  OK  ] GATE (numpy core, gradient proven)
  [  OK  ] PLANNER (DP optimal vs brute force)
  [  OK  ] MEMORY MODEL (31 tests, Korthikanti)
  [  OK  ] CREDIT RULES (e-prop vs BPTT, quality price)
  [  OK  ] TF CORE (gradient, normalisation, custom cell)
  [  OK  ] KERAS ASSEMBLY + training via keras.fit
  [  OK  ] API AUDIT (every advertised function called)
  [  OK  ] C++ KERNELS (bit-packing, layout)
  [  OK  ] EXAMPLE 02 (budget, NO TensorFlow)
  [  OK  ] EXAMPLE 01 (train a network)
  [  OK  ] EXAMPLE 03 (custom neuron, no fork)
  [  OK  ] EXAMPLE 05 (T9 next-word prediction)
  [  OK  ] ADV 01 (embedding + stacked layers)
  [  OK  ] ADV 02 (horizon sweep)
  [  OK  ] ADV 03 (credit rule, the real lever)
  [  OK  ] ADV 04 (budget, codec, precision)
  [  OK  ] ADV 05 (segmentation planner)
  [  OK  ] ADV 06 (devices: CPU and GPU)
  [  OK  ] CAPACITY SOLVER (axplan.solve)
  [  OK  ] ADV 15 (what to build: the inverse question)
  ALL GREEN. Results may be published.
```

Twenty suites. The last two are the newest: the capacity solver, whose
boundary claim is checked by re-evaluating the memory model independently and
by refusing to invent a boundary when the answer sits on the search ceiling.

The gate is not optional. No result is published until it is green — a rule
we adopted after three consecutive experiments looked convincing and were
wrong.

## Extending without forking

Neuron dynamics are an **object**, so a professional can replace them
without touching the kernel, the backward pass, or the planner:

```python
import tensorflow as tf
from axtf.cells import SpikingCell

class SaturatingCell(SpikingCell):
    """Membrane with a nonlinearity, so du/dstate is not the leak."""

    def membrane(self, state, drive, thr):
        return self.leak * state + drive - tf.square(state)

    def membrane_dstate(self, state, drive, thr):
        return self.leak - 2.0 * state

spec = ax.Model()
spec.add_spiking(192, horizon=64, cell=SaturatingCell())
spec.add_spiking(192, horizon=64)          # a plain LIF, same network
```

Override a derivative hook **only** when you replace the function it belongs
to. `spike_fn` needs `spike_d1`; `membrane` needs `membrane_dstate`. If you
override one and not the other, the core builds a backward pass with a
derivative that does not match the forward pass and reports nothing. That is
the one thing it will not do: guess.

This exact trap was live in the framework until recently — the kernel read
`cell.leak` as the membrane's state derivative, which is right for a linear
membrane and silently wrong for every other one. It is break #4 below.

## The four places this codebase silently broke

Found by the gate, invisible to the eye, documented so they cannot return:

**1. The 1/T normalisation.** With mean readout the maths requires
`dL/ds_t = (1/T)·dL/dmean`. Without it the gradient grows linearly with the
horizon and training diverges. It was *completely absent* from the TF
kernel: the parameter was threaded through four signatures and never
applied. The test that should have caught it re-implemented the backward
pass inline and compared it against itself.

**2. The time axis in `compute_output_shape`.** `_as_shape` mistook a shape
tuple `(None, 16, 24)` for a list of layer inputs, because the batch dim is
`None`. Since the batch dim is *always* `None`, this was the common case,
not an edge case. Every Functional model broke.

**3. Model input is not layer input.** Calibration read `model.inputs[0]`,
which for an embedding stem is the integer token tensor, not the
activations the layer receives.

**4. The backward pass guessed a derivative.** `_lif_backward` read
`cell.leak` as `du/dstate`. For a linear membrane that is accidentally
correct; for any other membrane it is wrong, and a user's cell had no way to
say so without forking the kernel. Fixed by adding
`SpikingCell.membrane_dstate(state, drive, thr)`, which the kernel now calls.

The verification for that fix is the interesting part: finite differences are
**invalid** here, because a hard spike is a step function. The first check
reported `rel.err = 0.98` on correct code. The check now verifies the smooth
membrane separately, counts that the hook is actually called, and compares
gradients element by element — after sabotaging the kernel in the opposite
direction and confirming the test fails.

Full account, including the mistakes made while finding all four:
[`docs/UX_RULES.md`](docs/UX_RULES.md).

## Known limitations

- `run_eagerly=True` is the default because Keras 3 caches a forward pass
  that contains a Python loop over time, and the gradient then lands on
  stale activations. The proper fix is a `tf.while_loop`; **not done**.
- The segmentation planner is optimal but almost always answers "no cuts",
  because its peak is `stored + max_inter` where `max_inter` is a floor.
  That is a property of the conservative model, not a bug.
- The gradient rule cannot cut *inside* a segment; a segment larger than
  the budget is simply infeasible.
- Not measured: the sparse/bit crossover (by anyone), energy savings (no
  neuromorphic device on this machine), and whether spiking is *faster*.
- **Modelled energy and measured wall-clock dissociate.** An independent audit
  of the five most-cited "spiking beats dense" results found them inside a
  single run on a single network: modelled energy 11.90x better, batch-1
  latency 5.94x worse. Every modelled energy claim — this package's bit-packing
  result included — is a statement about operation counts, and operation counts
  are not latency. Only one audited claim survived on **measured** silicon
  power, and its accuracy claim did not survive with it. See
  `../neuroarch/B13_fair_comparison_protocol/REPORT.md`.

## License

MIT.