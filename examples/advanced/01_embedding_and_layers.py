"""
Embedding + stacked spiking layers -- the shape every real project starts from.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/01_embedding_and_layers.py

WHAT THIS TEACHES
=================
Two things every practical model needs and the earlier examples skipped:

  1. An EMBEDDING stem. Tokens are integers; the layer needs vectors.
     The embedding sits in front of the spiking stack, and the spiking
     layer must be built for a KNOWN time length (see note below).

  2. More than one spiking layer, with widths that differ. Stacking is
     where the recurrent state starts to matter.

THE NOTE THAT WILL BITE YOU
===========================
SpikingRNNCell derives its horizon T from the input shape. So:

    Input(shape=(T, D))   -> the time axis is None -> it cannot tell
    Input(batch_shape=(None, T, D))  -> T is known -> fine

If you see "horizon T is unknown", that is the cause. It is the same
family of bug as the shape-axis one in docs/UX_RULES.md section 1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow import keras

import axiomrnn as ax
from axtf.build import calibrate_model, measure_model

T, VOCAB, EMB, W1, W2, C = 12, 40, 64, 128, 96, 4


def synthetic_sequences(n=4000, seed=0):
    """Sequences where the label depends on the ORDER of the tokens.

    A bag-of-words model cannot solve this, which is the point: it forces
    the network to carry state across time.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, VOCAB, (n, T))
    # label = which half had the higher mean, so position matters
    left = x[:, :T // 2].mean(1)
    right = x[:, T // 2:].mean(1)
    y = (left > right).astype(np.int32)
    return x, y


def build(vocab, context, emb, w1, w2, classes):
    """Assemble through axiomrnn where possible; the embedding is Keras."""
    tokens = keras.Input(batch_shape=(None, context), dtype="int32",
                         name="tokens")
    x = keras.layers.Embedding(vocab, emb, name="embed")(tokens)

    spec = ax.Model()
    spec.add_spiking(w1, horizon=context)
    spec.add_spiking(w2, horizon=context)

    from axtf.cells import SpikingRNNCell
    for i, layer in enumerate(spec.layers):
        x = SpikingRNNCell(
            layer.units, cell=layer.cell, return_sequences=True,
            norm_t=True, horizon_hint=context, name=f"spiking_{i}",
        )(x)

    x = keras.layers.Flatten(name="readout")(x)
    out = keras.layers.Dense(classes, name="logits")(x)

    model = keras.Model(tokens, out, name="embedded_snn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        # Mandatory: Keras 3 caches the forward pass, and ours is a Python
        # loop over time. See the note in axtf/cells.py.
        run_eagerly=True,
    )
    return model


def main():
    print("=" * 74)
    print("EMBEDDING STEM + STACKED SPIKING LAYERS")
    print("=" * 74)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  device: {'GPU' if gpus else 'CPU'}")
    print(f"  task: which half of {T} tokens has the higher mean (binary)")
    print(f"  a bag-of-words model cannot do this: order is the whole signal")
    print()

    x, y = synthetic_sequences()
    cut = int(0.8 * len(x))
    xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
    print(f"  {len(x):,} sequences, class balance {y.mean():.2f}")

    model = build(VOCAB, T, EMB, W1, W2, C)

    # A fresh spiking net is silent until the threshold is calibrated.
    print("\n  before calibration:")
    rows = measure_model(model, xtr[:512])
    for r in rows:
        print(f"    {r['layer']:<10} firing {r['firing']:.4f}  "
              f"dead {r['dead']:.4f}")

    info = calibrate_model(model, xtr[:512], target_rate=0.2, iters=6)
    print("\n  after calibration:")
    for name, fr, dead in info:
        print(f"    {name:<10} firing {fr:.4f}  dead {dead:.4f}")

    h = model.fit(xtr, ytr, epochs=10, batch_size=128, verbose=0,
                  validation_data=(xte, yte))
    pred = model.predict(xte, verbose=0).argmax(1)
    acc = float((pred == yte).mean())

    print()
    print(f"  loss        {h.history['loss'][0]:.4f} -> "
          f"{h.history['loss'][-1]:.4f}")
    print(f"  train acc   {h.history['accuracy'][-1]:.4f}")
    print(f"  held-out    {acc:.4f}   (chance {max(y.mean(), 1-y.mean()):.4f})")
    print()

    from axtf.cells import SpikingRNNCell
    n_spiking = sum(1 for l in model.layers if isinstance(l, SpikingRNNCell))
    print(f"  spiking layers in the graph: {n_spiking}")
    widths = [l.units for l in model.layers if isinstance(l, SpikingRNNCell)]
    print(f"  widths (differ on purpose):   {widths}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
