"""
11_derivative_hooks.py -- the seven hooks, and how to check yours.

Run:  /home/cune/.venvs/ax/bin/python examples/advanced/11_derivative_hooks.py

WHAT THIS TEACHES
=================
The neuron is an object with seven hooks. Four compute forward values and
three compute derivatives:

    membrane(state, drive, thr)          forward
    spike_fn(u, thr, smooth)             forward
    reset_fn(u, spike, smooth)           forward
    threshold_dthr(u, spike, thr, ds)    derivative

    membrane_dstate(state, drive, thr)   derivative
    spike_d1(u, thr)                     derivative
    reset_dud(u, spike, thr, ds)         derivative

THE RULE IS SHORT: override a forward hook, and you own the matching
derivative. Get it wrong and the kernel builds a backward pass with
somebody else's derivative, silently. Nothing raises. The loss still
falls, just worse than it should, and you conclude your idea is worse
than it is.

This script shows how to check each hook, and -- more importantly -- shows
a WRONG hook producing a wrong gradient that no error message mentions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import tensorflow as tf


def fd_grad(fn, x, eps=1e-3):
    """Central difference of a SMOOTH scalar function. float32 needs care."""
    n = int(np.prod(x.shape))
    flat = tf.reshape(x, [n])
    out = np.zeros(n, np.float32)
    for i in range(n):
        p = flat.numpy().copy(); p[i] += eps
        m = flat.numpy().copy(); m[i] -= eps
        fp = float(tf.reduce_sum(fn(tf.reshape(p, x.shape))))
        fm = float(tf.reduce_sum(fn(tf.reshape(m, x.shape))))
        out[i] = (fp - fm) / (2 * eps)
    return tf.reshape(tf.constant(out.reshape(x.shape)), x.shape)


def main():
    print("=" * 78)
    print("THE SEVEN HOOKS, AND HOW TO CHECK YOURS")
    print("=" * 78)

    from axtf.cells import SpikingCell

    rng = np.random.default_rng(4)
    B, N = 6, 8
    u = tf.constant(rng.normal(0, 1, (B, N)).astype(np.float32))
    s = tf.cast(rng.random((B, N)) > 0.5, np.float32)
    thr = tf.constant(np.abs(rng.normal(0.6, 0.1, N)).astype(np.float32))

    cell = SpikingCell(0.9, 2.0)

    # ── 1. spike_d1 ──────────────────────────────────────────────────────────
    print()
    print("  1. spike_d1  --  d spike / d u")
    print("  " + "-" * 74)
    print("     The hard rule is 1[u > thr]: piecewise constant, derivative")
    print("     zero everywhere. So the DEFAULT is the smooth surrogate,")
    print("     and the derivative hook must match whichever branch the")
    print("     forward hook took.")
    print()
    hard = float(tf.reduce_sum(tf.cast(u > thr, np.float32)))
    ana_hard = float(tf.reduce_sum(tf.zeros_like(u)))
    print(f"     hard 1[u>thr]:        sum {hard:.4f}, d/du sum "
          f"{ana_hard:.4f}")
    print(f"     smooth surrogate:    sum "
          f"{float(tf.reduce_sum(cell.spike_fn(u, thr, smooth=True))):.4f}, "
          f"d/du sum {float(tf.reduce_sum(cell.spike_d1(u, thr))):.4f}")
    num = fd_grad(lambda z: cell.spike_fn(z, thr, smooth=True), u)
    ana = cell.spike_d1(u, thr)
    rel = float(tf.norm(num - ana) / tf.norm(ana))
    print(f"     finite differences vs spike_d1: rel.err {rel:.2e} "
          f"{'OK' if rel < 5e-3 else 'WRONG'}")
    print()
    print("     Mismatch to watch for: overriding spike_fn with something")
    print("     else and leaving spike_d1 as the atan surrogate. The")
    print("     backward then uses a derivative that belongs to a different")
    print("     neuron.")

    # ── 2. reset_dud ─────────────────────────────────────────────────────────
    print()
    print("  2. reset_dud  --  d reset / d u")
    print("  " + "-" * 74)
    dsig = cell.spike_d1(u, thr)
    # BOTH arguments must come from the same branch. Passing a random spike
    # with a surrogate derivative makes reset_dud internally inconsistent,
    # and it reports a mismatch that is not there -- which is exactly the
    # kind of false alarm that teaches people to ignore hook checks.
    sp_surrogate = cell.spike_fn(u, thr, smooth=True)
    # The spike MUST also be recomputed from z inside the function. Holding
    # it fixed differentiates u*(1-s) as (1-s), while reset_dud correctly
    # assumes s = s(u) and returns (1-s) - u*s'(u).
    reset = lambda z: cell.reset_fn(z, cell.spike_fn(z, thr, smooth=True),
                                    smooth=True)
    num_r = fd_grad(reset, u)
    ana_r = cell.reset_dud(u, sp_surrogate, thr, dsig)
    rel_r = float(tf.norm(num_r - ana_r) / tf.maximum(tf.norm(ana_r), 1e-9))
    print(f"     finite differences vs reset_dud: rel.err {rel_r:.2e} "
          f"{'OK' if rel_r < 5e-3 else 'WRONG'}")
    print(f"     note reset_dud takes dspike as an argument, because")
    print(f"     reset = u*(1 - spike) depends on u through BOTH factors.")

    # ── 3. membrane_dstate ───────────────────────────────────────────────────
    print()
    print("  3. membrane_dstate  --  the hook that was missing")
    print("  " + "-" * 74)
    state = tf.constant(rng.normal(0, 1, (B, N)).astype(np.float32))
    drive = tf.constant(rng.normal(0, 1, (B, N)).astype(np.float32))
    num_m = fd_grad(lambda z: cell.membrane(z, drive, thr), state)
    ana_m = cell.membrane_dstate(state, drive, thr)
    rel_m = float(tf.norm(num_m - ana_m) / tf.maximum(tf.norm(ana_m), 1e-9))
    print(f"     default LIF: d membrane/d state = {float(tf.reduce_sum(ana_m)):.4f}"
          f" (that is just leak={cell.leak})")
    print(f"     finite differences agree to rel.err {rel_m:.2e}")
    print()
    print("     This hook did not exist until a custom membrane exposed the")
    print("     gap: the backward pass read cell.leak directly, so a neuron")
    print("     with a non-linear membrane was graded as a plain LIF. No")
    print("     error, just a quietly worse model.")

    # ── 4. threshold_dthr ────────────────────────────────────────────────────
    print()
    print("  4. threshold_dthr  --  only matters with learn_threshold=True")
    print("  " + "-" * 74)
    ana_t = cell.threshold_dthr(u, s, thr, dsig)
    print(f"     sum {float(tf.reduce_sum(ana_t)):.6f}")
    print()
    print("     There is a trap here that cost us a rel.err of 0.1: the")
    print("     direct path already reduced over batch AND time, while the")
    print("     reset path accumulates as (B, units). Broadcasting the two")
    print("     together makes dthr[j] point at the wrong element. Sum over")
    print("     the batch before combining.")

    # ── 5. a wrong hook, demonstrated ───────────────────────────────────────
    print()
    print("  5. WHAT A WRONG HOOK LOOKS LIKE")
    print("  " + "-" * 74)

    class GoodCell(SpikingCell):
        def membrane(self, state, drive, thr):
            return state * (1.0 + 0.6 * tf.tanh(drive)) + drive

        def membrane_dstate(self, state, drive, thr):
            return 1.0 + 0.6 * tf.tanh(drive)

    class SameForwardWrongDeriv(SpikingCell):
        """IDENTICAL forward pass. Only the derivative hook is wrong."""

        def membrane(self, state, drive, thr):
            return state * (1.0 + 0.6 * tf.tanh(drive)) + drive

        def membrane_dstate(self, state, drive, thr):
            return tf.fill(tf.shape(state), tf.cast(self.leak, state.dtype))

    # The two cells produce IDENTICAL spikes. Proved, not asserted.
    from axtf.cells import SpikingRNNCell
    T, DIN = 8, 5
    xin = tf.constant(rng.normal(0, 1, (4, T, DIN)).astype(np.float32))
    W = tf.constant(np.full((DIN, N), 0.9, np.float32))
    THR = tf.constant(np.full(N, 0.5, np.float32))

    def forward_of(c):
        lay = SpikingRNNCell(N, cell=c, return_sequences=True, norm_t=True,
                             horizon_hint=T)
        lay.build((None, T, DIN))
        lay.kernel.assign(W)
        lay.thr.assign(THR)
        return lay

    a = forward_of(GoodCell(0.9, 2.0))
    b = forward_of(SameForwardWrongDeriv(0.9, 2.0))
    same_forward = bool(np.array_equal(a(xin).numpy(), b(xin).numpy()))
    print(f"     the two cells give IDENTICAL spikes: {same_forward}")
    print(f"     (spikes: {int(a(xin).numpy().sum())} in both)")
    print()
    print("     So this bug is invisible in every forward check you might")
    print("     write. Here is the gradient difference:")
    grads = []
    for lay in (a, b):
        with tf.GradientTape() as tape:
            out = lay(xin)
            loss = tf.reduce_mean(out * out)
        grads.append(tape.gradient(loss, lay.kernel).numpy())
    ga, gb = grads
    rel_el = np.abs(ga - gb) / np.maximum(np.abs(ga), 1e-9)
    print(f"     ||g|| correct {float(np.linalg.norm(ga)):.6f}, "
          f"wrong {float(np.linalg.norm(gb)):.6f}")
    print(f"     entries differing by more than 50%: "
          f"{int((rel_el > 0.5).sum())}/{ga.size}, "
          f"worst {rel_el.max():.1%}")
    print()
    print("     CHECK YOUR HOOKS LIKE THIS: compare against finite")
    print("     differences of the FORWARD function, on the smooth branch,")
    print("     with fixed weights. It takes five lines and it is the only")
    print("     thing standing between you and a wrong model.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())