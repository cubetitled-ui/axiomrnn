# Changelog

All notable changes. Dates are ISO.

## [0.1.0] — 2026-10-02

First public release. Everything below was found by measurement, and
several items are recorded as bugs we shipped and then fixed.

### Added

- **Budget planner** (`axplan`): memory model, Pareto-front DP for graph
  segmentation, credit rules. Pure Python — no TensorFlow import anywhere
  in the package, so planning runs on a machine with no GPU stack.
- **Spiking kernel** (`axtf`): `SpikingRNNCell`, an explicitly backwards
  layer whose gradient is verified against finite differences to
  `rel.err ≤ 1e-8`.
- **Neuron as an object**: `SpikingCell` with four overridable dynamics
  methods and three derivative hooks, so a user's neuron drops into the
  framework without forking it.
- **`explain()`**: answers "what can you learn on this budget", which no
  other tool does. Also available as the `axiomrnn-explain` console
  command.
- **Verification harness**: `axon/gate.py`, 12 checks, mandatory before
  publishing any result, wrapped by `tests/run_all.py` — 20 suites covering
  the numpy core, the planner, the capacity solver, the memory model, the
  credit rules, the TF kernel, Keras assembly, the API audit, the C++
  kernels, and 12 of the 20 examples.
- **API audit** (`tests/test_api_surface.py`): every advertised symbol is
  called for real, with values asserted — not merely introspected.
- **Capacity solver** (`axplan/solve.py`): the inverse question — given a
  budget, the largest configuration that fits, the smallest step up that does
  not, and which term of the memory model binds. It refuses to answer when
  nothing fits, and reports no boundary when the answer sits on the search
  ceiling. It makes **no quality claim**: this package has no verified quality
  model, and says so in its own output.
- **Twenty examples** (5 core, 15 advanced), 12 of which run inside the
  gate. Each prints what it measured; no example prints a number it did not
  compute.
- **pip packaging** (PEP 621): core needs only numpy; `[tf]` and
  `[tf-gpu]` extras.

### Measured, and worth stating plainly

- Spiking vs dense on a temporal task: **0.9938 vs 0.9988** — parity, not
  superiority. The 0.26 gap we first reported was **our own** readout, not
  a property of spikes.
- Bit-packing: 16.8x on its own line, but that line is 0.16–0.32% of
  memory, so the real effect is **2.6–5.1%**. Binary activations do not
  make the gradient binary.
- Credit rule: **17x** (8.7x at T=128) on the line holding ~97% of memory.
  This is the real lever.
- Memory model vs measured allocation on GPU: **0.99**.

### Fixed

- **`norm_t` was dead.** The 1/T gradient normalisation was threaded
  through four signatures and never applied. The test that should have
  caught it re-implemented the backward pass inline and compared it
  against itself. Caught by attempting to break the code and observing
  that the test stayed green.
- **Time axis lost in `compute_output_shape`.** `_as_shape` mistook a
  shape tuple for a list of inputs because the batch dimension is `None` —
  which made this the common case, not an edge case. Every Functional
  model was affected.
- **Model input taken for layer input.** Calibration read
  `model.inputs[0]`, which for an embedding stem is the integer token
  tensor rather than the activations the layer receives.
- **`add_spiking(cell=...)` raised `TypeError`.** The advertised ability to
  extend the system without forking did not work through the public API;
  it only worked by importing `axtf` directly.
- **The backward pass guessed a derivative.** `_lif_backward` read
  `cell.leak` as `du/dstate`. That is accidentally correct for a linear
  membrane and silently wrong for every nonlinear one, with no way for a
  user's cell to correct it short of forking the kernel. Added
  `SpikingCell.membrane_dstate(state, drive_next)`; the kernel now asks the
  cell instead of assuming. Verified by element-wise gradient comparison and
  a call counter, **not** by finite differences — a hard spike is
  discontinuous, and a finite-difference check on the layer reported
  `rel.err = 0.98` on correct code.
- **Default readout was wrong.** Mean-over-time readout cost 0.24 accuracy
  on temporal tasks and had been the default. Now `flatten`.
- **Old `examples/` were stale** and disagreed with the core. Replaced.

### Documented limitations

- `run_eagerly=True` is the default: Keras 3 caches a forward pass that
  contains a Python loop over time, so gradients would otherwise land on
  stale activations. The real fix is `tf.while_loop`, **not done**.
- The segmentation planner is optimal but almost always returns zero
  cuts: its peak is `stored + max_inter`, and `max_inter` is a floor.
- The planner cannot cut *inside* a segment.

### Not measured

- The sparse/bit crossover — by nobody, us included.
- Energy cost: no neuromorphic hardware on the test machine.
- Whether spiking is faster than dense on a GPU.
