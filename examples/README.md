# Examples

Twelve runnable examples. Each teaches one thing and prints what it
measured. Nothing here prints a number it did not compute.

## Core set

| # | file | teaches | TF |
|---|---|---|---|
| 01 | `01_minimal.py` | train a spiking network | yes |
| 02 | `02_budget.py` | what can you learn under a budget | **no** |
| 03 | `03_custom_neuron.py` | replace the neuron, no fork | yes |
| 04 | `04_equal_budget.py` | compare two nets honestly | yes |
| 05 | `05_t9.py` | next-word prediction, T9 style | yes |

## Advanced set

| # | file | teaches | TF |
|---|---|---|---|
| 01 | `advanced/01_embedding_and_layers.py` | embedding stem, stacked layers, widths that differ | yes |
| 02 | `advanced/02_horizon_sweep.py` | what each horizon costs, and what it buys | partly |
| 03 | `advanced/03_credit_rule.py` | the credit rule is the lever, not the bit | **no** |
| 04 | `advanced/04_budget_and_precision.py` | hardware profiles, codec choice, precision | **no** |
| 05 | `advanced/05_segmentation_planner.py` | keep-or-recompute, and why cuts usually lose | **no** |
| 06 | `advanced/06_devices.py` | CPU and GPU, same code | yes |

Files marked **no** run on a plain `python3` with no TensorFlow installed.
That is not a trick: `axplan` does not import TF, so budget planning works
on a machine with no GPU stack at all.

---

## Running them

```bash
cd axiomrnn

# no TensorFlow needed
python3 examples/02_budget.py
python3 examples/advanced/03_credit_rule.py
python3 examples/advanced/04_budget_and_precision.py
python3 examples/advanced/05_segmentation_planner.py

# needs TensorFlow (this repo keeps it in a separate venv)
PY=/home/cune/.venvs/ax/bin/python
$PY examples/01_minimal.py
$PY examples/05_t9.py
$PY examples/advanced/06_devices.py
```

`06_devices.py` is designed to be run twice:

```bash
$PY examples/advanced/06_devices.py
CUDA_VISIBLE_DEVICES= $PY examples/advanced/06_devices.py
```

---

## Measured results

Taken from actual runs on an RTX 3050, TF 2.21.

**01 — minimal**
```
accuracy 0.9967   (chance 0.2500)
loss     2.3758e-04
```

**02 — budget (no TF)**
```
consumer 6 GB        available 4.12 GiB   needs 0.78 GiB   fits
pro 24 GB            available 17.96 GiB  needs 0.78 GiB   fits
datacenter 80 GB     available 61.04 GiB  needs 0.78 GiB   fits
```
The same architecture at width 20480 stops fitting on 6 GB, and the
planner says so rather than promising and failing later.

**03 — custom neuron**
```
ALIF (yours)   val_accuracy 0.9975
gradient norm  0.001069   (non-zero: the layer will train)
```

**04 — honest comparison**
```
dense (MLP)                     0.9988
spiking, readout=flatten        0.9919
spiking, readout=mean           0.7525
spiking smooth, readout=mean    0.7550
```
The gap is the readout, not the spikes. See `04_equal_budget.py` for the
five rules this teaches.

**05 — T9**
```
corpus           Pride and Prejudice, 114,959 words, 1,807 types
loss             5.9384 -> 4.8147
held-out acc     0.0968
majority-word    0.0379
most-common-bigram 0.0041
"i will"  ->  not 0.19  have 0.16  be 0.07  think 0.05  know 0.02
```
Beats both baselines. Treat the absolute number as a smoke test, not a
benchmark: the corpus is small and the tokenisation is deliberately naive.

**adv 01 — embedding**
```
before calibration   spiking_0 firing 0.0000  dead 1.0000
                     spiking_1 firing 0.0000  dead 1.0000
after calibration    spiking_0 firing 0.0468  dead 0.0000
                     spiking_1 firing 0.0000  dead 1.0000   <- still dead
held-out accuracy    0.9413   (chance 0.5037)
```
The second layer stays dead. That is reported rather than hidden: a wider
first layer does not guarantee a live second layer, and you should measure
it.

**adv 02 — horizon**
```
     T   peak GiB  credit k      regime   batches   fits
     8       0.51        66  activation        32    yes
    32       0.54        66  activation         8    yes
   128       0.57        66  activation         2    yes
```
The crossover batch *falls* as the horizon grows — the opposite of what
people assume about when local rules pay off.

**adv 03 — credit rule**
```
L= 8 h=1024 T= 32 b= 8      136.3 MiB     8.1 MiB   16.9x
L=16 h=2048 T= 64 b=16     2178.5 MiB   129.3 MiB   16.8x
L=32 h=4096 T=128 b=32    17962.3 MiB  2069.3 MiB    8.7x
```
And at T=2000, `local_quality` returns `None` rather than extrapolating.

**adv 04 — precision**
```
fp32 weights + fp32 master + fp32 Adam        16.0 B/param
fp16 weights + fp32 master + fp32 Adam        14.0 B/param
fp16 weights + fp32 Adam momentum              6.0 B/param
fp16 weights, fused update, no momentum        2.0 B/param
```
fp32 → fp16 on the weights is a 12% saving, not 2x. Adam's fp32 state is
the floor.

**adv 05 — segmentation**
```
partition               exact peak   cuts
(0, 4)                      3.50G      0
(0, 2, 4)                   4.00G      1
(0, 1, 2, 3, 4)             5.00G      3
```
Cuts make memory *worse* here, and the example explains why rather than
picking friendlier data.

**adv 06 — devices**
```
GPU:  loss 0.8758 -> 0.0310   accuracy 0.9995   2.4s   peak 29.9 MiB
CPU:  loss 0.7914 -> 0.0245   accuracy 0.9995
```

---

## What these examples do not show

- **energy savings** — there is no neuromorphic device on this machine;
- **superiority over dense networks** — we measured parity, not a win;
- **large-vocabulary quality** — the T9 corpus is a smoke test;
- **faster execution** — not measured, and `run_eagerly` is a known cost.

The honest framing, repeated in every report the framework prints: the
spiking choice costs nothing in quality here. It is worth it only on
hardware where spikes are cheap. On a normal GPU it is a curiosity.