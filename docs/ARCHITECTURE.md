# axiomrnn architecture

## The main decision: what is NOT built

The most common failure in a project like this is building everything. So
first, the list of what is absent here and will stay absent:

| not built | why |
|---|---|
| an architecture DSL | TF/Keras is mature, debugged, and tested by someone else |
| layers | `Dense`, `Flatten`, `Conv` exist in Keras; a private layer set is over-engineering |
| optimizers | Adam exists; the contribution here is *when to switch credit rule*, not how to step |
| the training loop | `keras.fit` works; only the settings are intervened with |
| automatic differentiation | `custom_gradient` + `GradientTape` |
| weight serialisation | `save_weights` works |
| data preprocessing | a separate project, not a framework |

None of that is the value. A user who writes `model.fit(...)` gets working
training without this package.

**What is built — exactly six things:**

1. a **memory model** — what a network costs under a budget;
2. a **planner** — what to choose so it fits (DP over the Pareto front);
3. **diagnostics** — `explain()` answers "what can you learn on this";
4. a **kernel** — a leaky integrator whose neuron is a replaceable object;
5. **rules** — codec, precision, credit rule;
6. a **capacity solver** — the inverse question: given a budget, the largest
   network that fits, the smallest step up that does not, and which term of
   the model binds. Pure memory arithmetic; it makes no quality claim.

---

## Four packages and the boundary between them

```
axiomrnn/
├── axiomrnn.py          public API: Model, Budget, explain()
│
├── axplan/              does NOT know TensorFlow exists
│   ├── memory.py        memory model: Korthikanti, backward, codec, regime
│   ├── planner.py       Pareto-front DP, optimality checked by brute force
│   ├── solve.py         the inverse question: what is the largest model
│   │                    that fits, and what exactly stops the next step up
│   └── credit.py        credit rules: BPTT versus e-prop
│
├── axtf/                the ONLY layer that knows about TF
│   ├── cells.py         SpikingCell (extendable) + SpikingRNNCell
│   └── build.py         Keras assembly, calibration, readout
│
├── axon/                numpy reference + the gate
│   ├── lif.py           gradient proven against finite differences
│   └── gate.py          12 checks, mandatory before any conclusion
│
├── axcore/              C++ only where a win was measured
│   └── kernels.cpp      bit-packing (bit-exact), blocked layout (1.68x)
│
├── docs/
├── examples/
└── tests/
```

**The boundary between `axplan` and `axtf` is functional, not stylistic.**

It is verified by one fact: `axplan` imports and runs on a Python with no
TensorFlow installed.

```bash
python3 -c "import axiomrnn as ax; \
  print(ax.Model().explain(ax.Budget.consumer_6gb(), verbose=False)[:80])"
```

That runs without TF. So the planner can be used and tested on a machine with
no GPU stack, and it will not break when the TF version changes.

If JAX or PyTorch is ever needed, `axplan` ports whole and `axtf` is
rewritten once.

---

## Architectural wounds that are already inflicted

They cannot be erased, but they can be named.

### `run_eagerly=True` is the price of a Python loop over time

The forward pass is a `for t in range(T)` Python loop inside
`tf.custom_gradient`. Keras 3 wraps the graph in `tf.function` and **reuses**
the result. Measured with a call counter: on 1 batch over 4 epochs,
`_lif_forward` was called **2 times instead of 4**; on repeated `fit`,
**0 times**. The gradient applied to stale activations and loss bottomed out
at `ln(C) = 1.386`.

**Worked around** by `run_eagerly=True` — slow, but correct. The cost was
measured.

**Properly fixed** by rewriting the loop as a `tf.while_loop` over
`tf.Variable` state. The layer becomes graph-native, the cache stops being
wrong, and `run_eagerly` can be dropped. **Not done**, and stated plainly.

### The surrogate gradient is not `1[u>thr]`

A hard spike has an identically zero derivative. It cannot be checked with
finite differences: the differences come out as exactly 0 on 18 of 20
coordinates.

So only the **smoothed** model is verified, and the gate demands
`rel.err ≤ 1e-8`. The corollary matters for anyone extending the neuron: a
hard forward pass means finite differences are the wrong instrument, and a
check built on them can fail correct code. See `UX_RULES.md` §17, where that
happened — a check reported `rel.err = 0.98` on a correct implementation.

---

## The extension point: your neuron, no fork

The requirement is that a professional can override the dynamics without
forking. So neuron dynamics are an **object**, not a bag of kwargs:

```python
import tensorflow as tf
from axtf.cells import SpikingCell

class SaturatingCell(SpikingCell):
    """A membrane with a nonlinearity, so du/dstate is not the leak."""

    def membrane(self, state, drive, thr):
        return self.leak * state + drive - tf.square(state)

    def membrane_dstate(self, state, drive, thr):
        return self.leak - 2.0 * state
```

Two methods are overridden; the kernel, the backward pass, and training are
untouched. This is the device used to isolate the cause of a quality
regression in `examples/04_equal_budget.py` (see `UX_RULES.md` §9).

**Derivatives are an extension point too.** Overriding `spike_fn` means
overriding `spike_d1`; overriding `membrane` means overriding
`membrane_dstate`. Otherwise the kernel **cannot build** backprop — and it
says so instead of silently constructing a wrong gradient.

The hook must exist **even when the default is currently correct**. The
leak-as-derivative default was correct only for the one membrane this project
happened to ship, which is exactly why every new cell would have needed a
fork. See `UX_RULES.md` §16.

---

## What is verified, and what is not

Verified by measurement — reproduce with:

| claim | command |
|---|---|
| gradient equals the derivative of the smoothed model | `axon/gate.py` |
| DP planner optimal against exhaustive search | `tests/run_all.py` |
| memory model vs measured GPU allocation | `tests/run_all.py` |
| spiking vs dense, honestly compared | `examples/04_equal_budget.py` |
| bit-packing counted against total memory | `examples/advanced/04_budget_and_precision.py` |
| credit rule as the real memory lever | `examples/advanced/03_credit_rule.py` |
| the four silent breaks, with their tests | `tests/test_tf_cells.py` |

The figures change between runs and are deliberately not transcribed here.
Each script prints its own numbers; `examples/README.md` records the current
ones together with the hardware that produced them.

**Not measured:**

- the sparse/bit crossover — by nobody, including this project;
- energy savings: that is neuromorphic hardware, and none was used;
- whether spiking is faster on a GPU;
- any claim of superiority over dense networks. What was measured is
  **parity**, and the gap that did appear was traced to this project's own
  readout, not to spikes.