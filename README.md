# axiomrnn

**Design for the memory budget you have, not the memory you wish you had.**

This file was rewritten on **2026-10-04**. Every number below carries a tag
and the command that produced it. Numbers tagged `FROM LITERATURE` were not
re-run by the author of this file; numbers tagged `MEASURED` were.

---

## Read this first: what this project does not claim

A reader who skips the limitations section will feel lied to at scale. Here
they are, up front, and they are findings rather than fine print.

**1. Neuron-style dynamics do not beat transformers in general.** The
project's own research reached that conclusion. For rate coding — one binary
event per neuron per step, the spike rate carrying the value — the crossover
against a transformer is astronomically far away: `n ≥ 2^(2^b − 1)`, which for
int8 (`b = 8`, so `T = 2⁸−1 = 255` serial steps) is `n ≥ 2²⁵⁵ = 10^76.8`
tokens. `FROM LITERATURE` —
`../neuroarch/A02_impossibility_hunt/REPORT.md` §O11, labelled
`[MEASURED + DERIVED: exp4_out.txt, E4d]`, derived from
`exp4_prec.py`. **I did not re-run it and there is no single command in this
repository that prints it.**

**2. That `10^76.8` figure is one closed form under two assumptions, and the
project's own theorist has flagged both.** `FROM LITERATURE` —
`../theorist/NOTES.md` §1, which reaches this conclusion:

- the derivation assumes **one** neuron (`S = 1`). With `S` neurons in
  parallel, capacity is `S·log₂(T+1)` bits and `T = 2^(p/S) − 1`. At
  `n = 10¹²`, `S = 32` already gives `T = 1.37` and `S = 64` gives `T = 0.54`;
- it assumes the transformer needs **`log₂ n`** dependent stages. One layer at
  width `d` carries `b_t·d` bits, so the stages needed to extract `p` bits is
  `p/(b_t·d)` — at `n = 10¹²`, `d = 768`, `b_t = 16` that is `0.0032` stages,
  not `39.9`. The original figure overstates the transformer by about five
  orders of magnitude.

So: the *direction* is a finding; the *magnitude* is one model of it. Read the
direction, not the digits.

**3. There is a bounded regime where it does win, and the number usually quoted
for it says something else entirely.** `22.73×` is **not** a margin of
victory. It is the ratio of membrane-state traffic to spike traffic —
`2T/(11·d·N)` at `N = 4096`, `T = 512`, `d = 0.001` — in the one corner where
event-driven execution beats dense. `FROM LITERATURE` —
`../neuroarch/A07_event_driven_energy/REPORT.md` §5.2 and its
`results_derived.json` field `state_over_spike_traffic = 22.72727272727273`.
The same report lists it as criterion **K9, status KILLED**: state traffic
*must* be under `1×` spike traffic for the win to be real, and it is `5.68×`
at `T = 128` and `22.73×` at `T = 512`. The honest sentence is the report's
own: *the regime where spiking wins is the regime where its own dense state
dominates it by more than an order of magnitude.*

**4. Modelled energy and measured latency point in opposite directions.** In
one run, on one network: the modelled 45 nm energy advantage is `11.90×`
(SNN `1.640 nJ` vs ANN `19.509 nJ`) while measured batch-1 latency is
`5.94×` **worse** (`89.5 ms` vs `15.1 ms`). `FROM LITERATURE` —
`../neuroarch/B13_fair_comparison_protocol/REPORT.md`. A dense accelerator
does not skip work because the values are zeros, and no operation-count table
contains that fact. Every energy figure in this project, bit-packing
included, is a statement about operation counts.

**5. The strong form of claim 1 is not certifiable.** Proving that rate-coded
dynamics cannot win requires a `TC⁰`-versus-`NC¹` separation. It is not
assumed quietly here; it is named as unavailable.

**6. There is no neuromorphic hardware in this measurement.** Energy is not
measured by anyone on this project. There is no such device on this machine.

---

## Install

**Nothing is published.** There is no `axiomrnn` on PyPI, and the name is
unclaimed — which means `pip install axiomrnn` will fail or install somebody
else's project. Do not run it.

A wheel has been built locally and is in the tree:

```
dist/axiomrnn-0.1.0-py3-none-any.whl    58,298 bytes, 18 files
dist/axiomrnn-0.1.0.tar.gz             396,507 bytes
```

`MEASURED` — `python -c "import zipfile; print(len(zipfile.ZipFile('dist/axiomrnn-0.1.0-py3-none-any.whl').namelist()))"`
run in the repository root. The wheel ships `axiomrnn.py`, `axon/`, `axplan/`
and `axtf/`, declares `Requires-Python >=3.10` and `numpy>=1.4`, offers extras
`tf`, `tf-gpu`, `dev`, and installs one console script, `axiomrnn-explain`. It
does **not** ship `axcore/`, `tests/` or `examples/` — `axcore` holds one C++
file with no `__init__.py`, nothing imports it, and it travels in the sdist
instead.

### The wheel install, verified

```bash
python3 -m venv /tmp/v && /tmp/v/bin/python -m pip install dist/axiomrnn-0.1.0-py3-none-any.whl
/tmp/v/bin/axiomrnn-explain --budget 6gb --layers 2 --width 1024 --horizon 256
```

`MEASURED` 2026-10-04, re-run by this file's author against the environment
Release built:

```
Package  Version
-------- -------
axiomrnn 0.1.0
numpy    2.5.3
pip      25.0.1

find_spec(tensorflow) = None
find_spec(keras)      = None
ax._HAS_TF            = False
ax._TF_ERR            = No module named 'tensorflow'
ax.SpikingCell        = None
site-packages path    = .../site-packages/axiomrnn.py
any sys.path entry in /home/cune/axiomrnn: False
Budget                = Budget(consumer 6 GB: 6.0 GB VRAM, host 32.0 GB, offload=True)
```

Three things follow, and they matter:

1. **The planning path works with no TensorFlow installed at all.** Exactly
   three packages land: `axiomrnn`, `numpy`, `pip`.
2. **The soft-fail is a soft-fail.** `ax.SpikingCell` is `None` and
   `ax._TF_ERR` explains why, instead of an `ImportError` at package import.
3. **`axplan` and `axon` are TensorFlow-free; `axtf` is not.** `MEASURED` —
   module by module in that venv: `axplan`, `axplan.memory`, `axplan.planner`,
   `axplan.solve`, `axplan.credit`, `axon`, `axon.lif`, `axon.gate` and the
   `axtf` *namespace* all import; `axtf.cells` and `axtf.build` both raise
   `ModuleNotFoundError`. So do not read "the wheel works without TensorFlow"
   as "`axtf` works without TensorFlow". It does not.

`MEASURED` — the console script exits 0 and prints
`tape share: 98.6%`, `peak: 0.78 GiB of 6 GiB`, `HEADROOM: 3.33 GiB unused`,
independently agreeing with `examples/02_budget.py`.

### What works today from a checkout

```bash
cd axiomrnn
python3 examples/02_budget.py          # works, no TensorFlow
```

`MEASURED` — `env -u PYTHONPATH /usr/bin/python3 -c "import axiomrnn as ax;
print(ax._HAS_TF)"` in the repository root prints `import ok, _HAS_TF = False`.

---

## The five-minute path

### What TensorFlow is, and why it is unavoidable here

TensorFlow is Google's numerical library; **Keras** is its model-building
layer, and you meet Keras whether or not you asked for it. A Keras model is
not a Python object full of layers you call yourself — it is a *graph*: you
declare the shape of the data going in, list the layers, and Keras wires them
into one callable that moves the data through. This project needs it because
a spiking layer's memory cost is a function of the graph the layer sits in, and
because the layer has to be differentiable by a framework that owns the
backward pass. There is no way to get a *trained* neuron here without it.
**Planning, however, needs nothing but NumPy** — see the third step below,
which is the part that runs in under a second.

### Step 1 — get the code, no install (0 min)

```bash
git clone <this repository>
cd axiomrnn
```

Nothing to install. The planning path runs on a bare `python3` with NumPy.
`MEASURED` — the system interpreter here is Python **3.14.4** with NumPy
**2.5.0** and **no TensorFlow**, and `python3 examples/02_budget.py` exits 0
on it.

If you would rather install the built wheel than use the checkout, the
[Install section](#install) has the verified transcript. Either way, step 2
works with no TensorFlow anywhere.

### Step 2 — the first budget, and it needs no TensorFlow (about 1 second)

```bash
python3 examples/02_budget.py
```

`MEASURED` — exit code 0 in **0.11 s**, **15 MB** peak RSS
(`/usr/bin/time -f "%e s %M KB peakRSS" python3 examples/02_budget.py`). Its
first line of output is the claim being tested:

```
  TensorFlow in this environment: ABSENT, and not needed
```

and the core of it is this table:

```
  budget                    usable      needed     fits?
  consumer 6 GB           4.12 GiB    0.78 GiB       yes
  pro 24 GB              17.96 GiB    0.78 GiB       yes
  datacenter 80 GB       61.04 GiB    0.78 GiB       yes
```

The same architecture at width `20480` needs `7.58 GiB` and the planner says
`NO` on a 6 GB card rather than promising and failing later. This is the
product: **the budget is an input, and the tool tells you what it buys.**

### Step 3 — the inverse question (about 1 second)

```bash
python3 examples/advanced/15_what_to_build.py
```

Still no TensorFlow. Every other tool answers *"what does this network
cost?"*. This one answers the question you actually have. `MEASURED` — exit
code 0; at a 6 GiB budget:

```
  LARGEST THAT FITS
  hidden 8192, layers 32, horizon 256, batch 1
  connections 2,147,745,792   bytes per synapse 2.00
  peak 5.604 GiB of 6.000 GiB   headroom 0.396 GiB   binding term static

  FIRST THAT DOES NOT FIT
  hidden 32768, layers 4, horizon 1
  peak 8.509 GiB of 6.000 GiB   overshoot 1.42x
```

Three fields earn their keep:

- **the binding term** — `static` means the *weights* dominate, so the credit
  rule and the spike coding save you nothing on this budget and width is the
  only lever. That one word saves a wasted afternoon.
- **the failure boundary** — the smallest step up that does not fit, with the
  overshoot. A tool that only reports successes cannot be told apart from a
  tool that is simply optimistic.
- **refusal** — `None` when nothing in the searched space fits, and no
  invented boundary when the answer sits on the search ceiling. The same run
  prints both refusals on demand: with a budget too small for anything it
  prints `solution: None`, and at the width ceiling it prints `reported
  overflow: None`.

One caveat it states in its own output: this is a **capacity** answer, not a
quality one. This package has no verified quality model. The one quality
figure it carries — what the local credit rule costs — is printed next to the
capacity gain, because a capacity number without its price is a sales pitch.
`MEASURED` — same run: capacity ratio under the local rule is `1.00x`, and
`local rule at T=256 keeps 64.7% of BPTT`.

### Step 4 — a neuron that actually fires (75 s of compute, plus a 1.9 GB download)

**Read this before step 4.** TensorFlow has no build for Python 3.14. The
system interpreter on this machine is 3.14.4, so `pip install tensorflow`
into it will not work; you need an environment on Python 3.12 or older.
`MEASURED` — `/usr/bin/python3 -V` is `Python 3.14.4`; the TensorFlow
environment here is `/home/cune/.venvs/ax/bin/python`, `Python 3.12.13`.
(`tests/run_all.py:25` carries the same note in a comment.)

TensorFlow is a large dependency: `1.9 GB` for the `tensorflow` package,
`21 MB` for `keras`, `6.3 GB` for the whole environment including the
interpreter. `MEASURED` — `du -sh` on each. **That download, not the code, is
what decides whether this step fits in five minutes.**

```bash
# with a Python 3.12-or-older environment that has TensorFlow 2.x:
python examples/01_minimal.py
```

`MEASURED` — TensorFlow `2.21.0`, Keras `3.15.1`, CPU only
(`CUDA_VISIBLE_DEVICES=`), exit code 0 in **75.2 s**, **641 MB** peak RSS,
running standalone. Held-out accuracy **0.9983** against a **0.2500** chance
baseline, after 15 epochs at batch 64; final training loss `3.7285e-04`, final
validation loss `0.0071`; 51,844 total parameters.

`MEASURED` — the same script as a suite of the gate took **190.9 s** and
reported **0.9975**. The wall clock differs because the gate run had other work
on the machine; the accuracy differs by run-to-run variance on a 1,200-example
held-out set. Both numbers are from this session, and the difference is why
this file quotes a range rather than a constant.

**One warning about this file, `MEASURED`:** it used to run the entire test
gate as a subprocess at the end (`examples/01_minimal.py:165-175`), so running
it trained the network, then ran the whole gate, which trained it a second time.
Observed with `pgrep` during this documentation pass: `01_minimal.py` spawned
`run_all.py`, whose `EXAMPLE 01` suite spawned `01_minimal.py` *again*.
**Fixed at 23:58:59 on 2026-10-03**, during this pass, by the code owner — the
subprocess is now behind `AX_RUN_GATE=1` and the script otherwise prints the
gate command for you to run yourself. This is the one number in this file that
was correct when written and became wrong an hour later.

### Step 5 — check the work before you believe it

```bash
python3 tests/run_all.py
```

`MEASURED` — this is **21 min 12 s**, not five minutes. It is deliberately
outside the five-minute path: the five minutes are for getting a neuron to
fire, and the gate is for deciding whether to believe anything. See *Verify it
yourself* below for the real output, its date, and its per-suite inventory.

---

### The five minutes, honestly totalled

`MEASURED`:

| step | wall clock | blocking download |
|---|---:|---|
| clone | seconds | the repository |
| step 2 — budget as an input | **0.04 s** | none |
| step 3 — the inverse question | **0.06 s** | none |
| step 4 — a neuron that fires | **75.2 s** | **1.9 GB** of TensorFlow, on a Python ≤ 3.12 environment |

**What you can genuinely do in under a minute, with no install at all:** run
`python3 examples/02_budget.py` and `python3
examples/advanced/15_what_to_build.py` on a bare interpreter with NumPy, and
get the entire planning surface — hardware profiles, the codec decision, the
precision arithmetic, the segmentation answer, the largest network a budget
buys, and the boundary that stops it.

**What you cannot do in five minutes:** train. Not because the code is slow —
75 seconds of compute — but because TensorFlow is a 1.9 GB download that needs
a different Python than the one already on your machine. That download is the
entire cost of the five-minute path, and no amount of documentation removes
it.

---

## The neuron is an abstraction, not a model of biology

Stated plainly, because the name invites the wrong assumption:

- The cell is a **leaky integrator with a threshold**. It is not
  Hodgkin–Huxley. There are no ion channels, no gating variables, no reversal
  potentials, and no spike-initiation mechanism.
- Adaptive thresholds and drive-gated leaks are still **LIF-family**. Calling
  them ALIF or adaptive-LIF would be a claim this interface cannot honour, so
  it is not done here.
- "Spiking" means the signal is a **binary event** and the gradient is an
  **explicit surrogate**, checked against finite differences on the smoothed
  model. The biological reading is a design analogy, nothing more.
- Nothing in this repository is a neuroscience claim. The honest summary:
  *an integer-valued activation with a hand-written derivative, chosen because
  it makes the backward pass cheap to check.*

---

## Extending without forking

Neuron dynamics are an **object**, so a professional can replace them without
touching the kernel, the backward pass, or the planner:

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

`MEASURED` — the gate's `TF CORE` suite checks `membrane_dstate` against
finite differences for four different membranes at `rel.err` between `1.06e-05`
and `1.24e-04`, and separately checks that the kernel *calls the hook*
(`membrane_dstate called 7 times for T=7`) and that a wrong
`membrane_dstate` breaks gradient entries (`18/30 entries differ by more than
50%, worst 95.6%`). See the gate block below.

---

## The four places this codebase silently broke

Found by the gate, invisible to the eye, documented so they cannot return:

**1. The 1/T normalisation.** With a mean readout the maths requires
`dL/ds_t = (1/T)·dL/dmean`. Without it the gradient grows linearly with the
horizon and training diverges. It was *completely absent* from the TF kernel:
the parameter was threaded through four signatures and never applied. The
test that should have caught it re-implemented the backward pass inline and
compared it against itself.
`MEASURED` — the gate now checks both directions:
`without 1/T grows 0.44x, with it 0.11x (same weights, so this is norm_t and
nothing else)`, and `T=4,8,16 -> 0.0186, 0.0094, 0.0021`.

**2. The time axis in `compute_output_shape`.** `_as_shape` mistook a shape
tuple `(None, 16, 24)` for a list of layer inputs, because the batch dim is
`None`. Since the batch dim is *always* `None`, this was the common case, not
an edge case. Every Functional model broke.

**3. Model input is not layer input.** Calibration read `model.inputs[0]`,
which for an embedding stem is the integer token tensor, not the activations
the layer receives.

**4. The backward pass guessed a derivative.** `_lif_backward` read
`cell.leak` as `du/dstate`. For a linear membrane that is accidentally
correct; for any other membrane it is wrong, and a user's cell had no way to
say so without forking the kernel. Fixed by adding
`SpikingCell.membrane_dstate(state, drive, thr)`, which the kernel now calls.

The verification for that fix is the interesting part: finite differences are
**invalid** on the hard spike, because it is a step function. The check
therefore verifies the smooth membrane separately, counts that the hook is
actually called, and compares gradients element by element — after sabotaging
the kernel in the opposite direction and confirming the test fails.

Full account, including the mistakes made while finding all four:
[`docs/UX_RULES.md`](docs/UX_RULES.md).

`MEASURED` note on the count: this list is **four** silent breaks of the
kind "no error, just a quietly wrong model". `docs/UX_RULES.md:209` separately
enumerates **five** backward-pass and algorithmic errors the gate caught — a
`(1−s)` factor on the wrong term, an inverted time recursion, a `dL/dthr` with
no path through the reset, an early `return` in the DP, and a wrong tuple field
in the Pareto filter. Two lists, two scopes, nine bugs in total.

---

## What is not built

Stated as absence, not as a roadmap.

| not built | what that means for you |
|---|---|
| a PyPI release | `pip install axiomrnn` does not work; use the checkout, or the local wheel by path |
| a byte-reproducible build | rebuilding the sdist yields a wheel of the same size but a different sha256 (zip timestamps, no `SOURCE_DATE_EPOCH`); two builds are not the same bits |
| a CPU-graph-native layer | `run_eagerly=True` is mandatory; `tf.while_loop` is not done |
| energy measurement | no neuromorphic device exists here; no energy number is measured by anyone on this project |
| speed measurement | not measured. Operation counts are not latency |
| the sparse/bit crossover point | measured by nobody, this project included |
| large-vocabulary quality | the T9 corpus is a smoke test, not a benchmark |
| a quality model | `largest_that_fits` is a **capacity** answer; the package has no verified accuracy model and says so |
| sub-segment graph partitioning | `Segment` has no notion of internal structure, so memopt-style splitting is not expressible |
| gradient rules that cut inside a segment | a segment larger than the budget is simply infeasible |

Known code-level limitations, all stated by the framework's own output:

- `run_eagerly=True` is the default because Keras 3 caches a forward pass that
  contains a Python loop over time, and the gradient then lands on stale
  activations. The proper fix is a `tf.while_loop`; **not done**.
- The segmentation planner is optimal but almost always answers "no cuts",
  because its peak is `stored + max_inter` where `max_inter` is a floor. That
  is a property of the conservative model, not a bug. `MEASURED` —
  `python3 examples/advanced/05_segmentation_planner.py`: `(0,4)` → `3.50G`,
  0 cuts; `(0,2,4)` → `4.00G`, 1 cut; `(0,1,3,4)` → `4.50G`, 2 cuts;
  `(0,1,2,3,4)` → `5.00G`, 3 cuts. Cutting the graph makes memory *worse*, and
  the planner agrees with brute force on 5 of 5 budgets.
- Spike density is not a constant. `MEASURED` — same run as the budget table:
  `between tasks: 0.175 .. 0.324` but `within a net: 0.00 .. 0.83`. A single
  mean is useless; the planner budgets the worst case.

---

## Known limitations

These are the ones a reader will hit while trying this, as opposed to the
ones they will hit at scale.

- **Bit-packing buys much less than the headline number suggests, and I could
  not reproduce the headline number.** `FROM LITERATURE` — `16.8×` on its own
  line, that line being `0.16–0.32%` of memory, so `2.6–5.1%` overall. These
  three figures appear only inside a block that
  `examples/advanced/03_credit_rule.py` explicitly labels *"QUOTED, not
  recomputed in this script"*, and **no command I ran prints `16.8×`**. What
  `examples/advanced/04_budget_and_precision.py` does print live is the codec
  table (reproduced in [`examples/README.md`](examples/README.md)), from which
  the shape of the claim is checkable: at firing rate `0.25` the bit codec
  uses `124,780` bytes, the sparse codec `1,999,646`, and ANS `219,494` — so
  *sparsity* is what costs, and past rate ≈ `0.5` the bit codec wins because
  sparsity stops paying. One sentence explains the whole thing: *binary
  activations do not make the gradient binary.* The backward tape is much
  larger than the forward tape and is **always dense**, because the surrogate
  derivative is non-zero everywhere.
- **The credit rule is a function of the horizon and the batch size, not a
  constant.** `MEASURED` —
  `python3 examples/advanced/03_credit_rule.py` prints BPTT against e-prop at
  three scales and the ratio **rises toward 1.0 as `T` grows**:

  ```
    config                        BPTT       e-prop    ratio
    L=8  h=1024 T=32  b=8       8.1 MiB     32.0 MiB     0.3x
    L=16 h=2048 T=64  b=16    129.3 MiB    256.0 MiB     0.5x
    L=32 h=4096 T=128 b=32   2069.3 MiB   2048.0 MiB     1.0x
  ```

  Read carefully: at `T=32, b=8` the local rule is **more** expensive than
  BPTT. It pays only above a crossover batch that *falls* as the horizon
  grows — `48, 24, 12, 6, 3` for `T = 16, 32, 64, 128, 256` (`MEASURED`, same
  run). Any single "e-prop is Nx cheaper" figure is a number without its `T`
  and `b`, and this file will not print one.
- **The local rule's quality price is real and large.** `MEASURED` — same run:
  `T=60 -> 85.5%` of full BPTT, `T=500 -> 38.9%`, a 47-point gap. Beyond the
  measured range the API returns `None` rather than extrapolating, because a
  linear extrapolation would read `0.0` at `T=1024` and look like a claim that
  the local rule is useless. It simply is not known.
- **A readout that averages over time destroys temporal tasks.** This is not
  about spiking; it is the readout. `MEASURED` —
  `python3 examples/04_equal_budget.py` compares four readouts on one task;
  the numbers are in [`examples/README.md`](examples/README.md) with the
  command.
- **A Keras model is not reproducible across two builds** unless you assign
  seeds explicitly; the NumPy core and the planner are bit-exact under a fixed
  seed. `MEASURED` —
  `python examples/advanced/08_reproducibility.py`.
- **The gate proves less than its length suggests, and here is exactly where.**
  See the gate block below.

---

## Verify it yourself

```bash
python3 tests/run_all.py
```

Real output, produced **2026-10-04 at 00:09 MSK**, 21 min 12 s wall clock,
CPU only (`CUDA_VISIBLE_DEVICES=`), TensorFlow 2.21.0, exit code 0:

```
SUMMARY
==========================================================================
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
  [  OK  ] ADV 07 (silence diagnosis)
  [  OK  ] ADV 08 (reproducibility)
  [  OK  ] ADV 09 (API tour)
  [  OK  ] ADV 10 (save and load)
  [  OK  ] ADV 11 (derivative hooks)
  [  OK  ] ADV 12 (failure modes)
  [  OK  ] ADV 13 (when not to use)
  [  OK  ] ADV 14 (budget matrix)
==========================================================================
  ALL GREEN. Results may be published.
```

**Twenty-eight suites. An earlier version of this file said twenty, and an
earlier version still said eighteen inside `examples/02_budget.py` and
`examples/advanced/02_horizon_sweep.py`. All three were wrong. The count is
28 and the date is above.**

### What the gate actually examined, and what "OK" is worth

Rule 6: a green that verified nothing is a failure, so here is the inventory
per suite rather than the word "green". `MEASURED` — parsed from the same
run:

| suite | declared checks | wall |
|---|---:|---:|
| GATE (numpy core, gradient proven) | 12 | 1.7 s |
| PLANNER (DP optimal vs brute force) | 12 | 0.2 s |
| MEMORY MODEL (Korthikanti) | **38 printed, label says 31** | 0.0 s |
| CREDIT RULES (e-prop vs BPTT, quality price) | 23 | 0.1 s |
| TF CORE (gradient, normalisation, custom cell) | 20 | 6.7 s |
| KERAS ASSEMBLY + training via `keras.fit` | 45 | 73.9 s |
| API AUDIT (every advertised function called) | 119 (`SKIPPED 0`) | 4.4 s |
| C++ KERNELS (bit-packing, layout) | 5 printed, no aggregate | 2.8 s |
| EXAMPLE 02 · 01 · 03 · 05 | **0 surfaced** | 0.1 · 190.9 · 84.6 · **770.1 s** |
| ADV 01 · 02 · 03 · 04 · 05 · 06 | **0 surfaced** | 41.3 · 0.2 · 0.1 · 0.1 · 0.0 · 5.7 s |
| CAPACITY SOLVER · ADV 15 | **0 surfaced** | 0.1 · 0.1 s |
| ADV 07 · 08 · 09 · 10 · 11 · 12 · 13 · 14 | **0 surfaced** | 54.3 · 2.2 · 3.2 · 23.2 · 2.1 · 3.9 · 0.0 · 0.0 s |

**269 declared checks** across the seven suites that report a count, and
**1272 s** of suite time. One suite, `EXAMPLE 05`, is 61% of the whole gate.

**One artefact of running it, so nobody is confused by it.** The gate writes its
pid into `tests/.gate.lock` and holds an `flock` on that file for the duration.
`flock` is tied to the file descriptor, so the kernel releases it even if the
process is `SIGKILL`ed — the lock cannot go stale the way a lockfile normally
can. The *text* in the file can still name a dead pid, and after that run it
reads `pid 1411367`. **That string is harmless and blocks nothing**; a second
gate refuses only if another process genuinely holds the descriptor, and then it
exits 2 and says so.

Three honest caveats, all `MEASURED` from that run:

1. **The four `EXAMPLE` suites and the fifteen `ADV` suites print zero
   check lines.** `tests/run_all.py:150-154` filters suite output down to
   lines containing `PASS`/`ok]`/`Error`/`Traceback`/`SKIPPED`, and the
   examples print none of those. For those nineteen suites an `[  OK  ]`
   attests **only that the process exited 0**. It does not surface the
   example's own numbers, and it cannot catch a wrong number — only a
   non-zero exit.
2. **One example is not in the gate at all.** `examples/04_equal_budget.py`
   — the honest spiking-versus-dense comparison, the single most-quoted
   measurement in this repository — is **not** a suite, even though
   `tests/run_all.py:86-89` says *"every example is now a suite"*. Nineteen of
   the twenty examples are covered; `04` is the exception. Reported to the
   owner; documented here rather than papered over.
3. **The MEMORY MODEL suite's label undercounts itself.** It is named
   `(31 tests, Korthikanti)` and prints 38 `[ok]` lines. The label is stale.

The gate is not optional. No result here is published until it is green — a
rule adopted after three consecutive experiments looked convincing and were
wrong. But a green is only worth its inventory size, so the inventory size is
printed above.

---

## Documentation

| | |
|---|---|
| [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) | the same five-minute path, expanded |
| [`examples/README.md`](examples/README.md) | all 20 examples, each with its real output and the command that produced it |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | what we deliberately do **not** build, and the four packages |
| [`docs/UX_RULES.md`](docs/UX_RULES.md) | 20 rules, each grown from a specific failure |
| [`docs/ATTACKING_THE_QUADRATIC.md`](docs/ATTACKING_THE_QUADRATIC.md) | an independent attack on the premise: replacing the KV cache with a spike-gated fast-weight state |
| [`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md) | the original design record (Russian, unedited) |

---

## License

Apache-2.0. See [LICENSE](LICENSE).