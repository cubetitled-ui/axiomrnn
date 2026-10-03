# Getting started

Ten minutes from nothing to a budgeted model. No neuromorphic background
assumed.

---

## Install

```bash
pip install axiomrnn            # core, numpy only -- planning works
pip install axiomrnn[tf]        # + TensorFlow/Keras for training
pip install axiomrnn[tf-gpu]    # + CUDA
```

Check what landed:

```bash
python3 -c "import axiomrnn as ax; print('TF:', ax._HAS_TF)"
```

`False` is fine and expected. The planner does not need TensorFlow.

---

## 1. Your first budget, with no TensorFlow

```python
import axiomrnn as ax

spec = ax.Model()
spec.add_spiking(1024, horizon=256)
spec.add_spiking(1024, horizon=256)
spec.add_dense(1000)

print(spec.explain(ax.Budget.consumer_6gb()))
```

You get, among other things:

```
  WHAT IT COSTS IN MEMORY
    regime:        activation
    peak:          0.78 GiB of 6 GiB

  THE MAIN MEMORY LEVER: THE CREDIT RULE
    now: BPTT, 0.016 GiB
    e-prop LIF: 0.008 GiB (0.50x Adam -- cheaper than momentum)
    batch crossover: batch* = 16
```

That is arithmetic. Nothing imported TensorFlow to produce it.

## 1b. The other direction: what should you build?

Section 1 answers "what does this cost?". The inverse question is usually the
one you have:

```python
import axiomrnn as ax

best, too_big = ax.largest_that_fits(6.0)     # vram_gb in GiB
print(best.explain())                        # largest that fits
print(too_big.explain())                     # the boundary, if one exists
```

Three fields earn their keep:

- `binding` names which term dominates — `static`, `tape`, `backward`,
  `logits`, `workspace`. If it says `static`, the **weights** dominate, so
  the credit rule and the spike coding save you nothing on this budget and
  width is the only lever. That single word saves a wasted afternoon.
- The second object is the smallest configuration that does **not** fit, so
  you can see the overshoot before you spend the day discovering it.
- `bytes_per_param` is printed next to the synapse count, because
  "2.1 billion synapses" means nothing without it.

Both may be `None`, and that is not an error: `None` for `best` means nothing
in the searched space fits; `None` for `too_big` means the answer sits on the
search ceiling and there is no boundary above it inside the space. Neither
case is papered over.

It is a **capacity** answer. This package has no verified quality model, so
it does not rank architectures by accuracy — `ax.local_quality()` is the one
quality figure it carries, it is read from a cited measurement, and it
returns `None` outside its measured range instead of extrapolating.

## 2. Your first network

```python
import numpy as np, tensorflow as tf, keras
import axiomrnn as ax

T, C = 16, 4
rng = np.random.default_rng(0)
pat = rng.normal(0, 0.4, (C, T, 12))
for k in range(C):
    pat[k, k * (T // C):(k + 1) * (T // C)] += 2.0
y = rng.integers(0, C, 4000)
X = (pat[y] * (rng.random((4000, T, 12)) < 0.3)
     + rng.normal(0, 0.6, (4000, T, 12))).astype("float32")

spec = ax.Model()
spec.add_spiking(192, horizon=T)
spec.add_spiking(192, horizon=T)
spec.add_dense(C)

model = spec.keras_model(
    inputs=keras.Input(batch_shape=(None, T, 12)), outputs=C,
    compile_model=True)
model.fit(X, y, epochs=12, batch_size=64)
```

Measured on this machine: accuracy **0.9967** against a 0.25 baseline.

### Three Keras 3 things that will bite you

**Use `batch_shape`, not `shape`.** `Input(shape=(T, D))` leaves the time
axis `None`, and our layer cannot derive its horizon from it. The error is
"horizon T is unknown", and the fix is `batch_shape=(None, T, D)`.

**`from_logits=True` is mandatory.** Our output *is* logits. Keras defaults
to expecting probabilities and applies softmax itself. Measured: a double
softmax costs you 2.4341 against the correct 1.4414.

**`run_eagerly=True` is not optional.** Our forward pass is a Python loop
over time, and Keras 3 caches the result instead of recomputing it. The
gradient then applies to stale activations and the loss saturates at
`ln(C) = 1.386`. `compile_model=True` sets it for you.

## 3. Calibrate, or it will not train

A fresh spiking network is **silent**: potentials stay below the default
threshold of 1.0, nothing spikes, no gradient flows.

```python
from axtf.build import calibrate_model
calibrate_model(model, X[:512], target_rate=0.2)
```

You can see the problem before you fix it:

```python
from axtf.build import measure_model
measure_model(model, X[:512])
# [{'layer': 'spiking_0', 'firing': 0.0, 'dead': 1.0}, ...]
```

`dead: 1.0` means every neuron is silent. After calibration the firing rate
lands near your target.

## 4. Swap in your own neuron

```python
from axtf.cells import SpikingCell, _atan_surrogate

class SmoothCell(SpikingCell):
    def spike_fn(self, u, thr, smooth=True):
        return _atan_surrogate(u, thr, self.alpha)
    def reset_fn(self, u, spike, smooth=True):
        return u * (1.0 - spike)

spec.add_spiking(192, horizon=T, cell=SmoothCell())
```

Different layers of one network may hold different neurons. The backward
pass, calibration and training loop stay shared, which is the point: they
stay *verified*.

## 5. Check the work before you believe it

```bash
python3 tests/run_all.py
```

The gate compares the analytic gradient against finite differences on the
smooth model and demands `rel.err ≤ 1e-8`. It is not ceremony — it caught
five real backward-pass bugs, and later caught a `norm_t` parameter that
had been threaded through four signatures and never applied.

## Where to go next

| you want | read |
|---|---|
| plan for real hardware | `examples/02_budget.py`, `examples/advanced/04_budget_and_precision.py` |
| understand the memory lever | `examples/advanced/03_credit_rule.py` |
| a working language model | `examples/05_t9.py` |
| not fool yourself when comparing | `examples/04_equal_budget.py` |
| why this codebase looks like this | `docs/ARCHITECTURE.md`, `docs/UX_RULES.md` |