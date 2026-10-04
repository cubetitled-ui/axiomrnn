# Research notes

This file was split out of `README.md` on 2026-10-04. It is the record of what
the project's own research concluded, kept in full because the conclusions are
unflattering and burying them would be dishonest.

Nothing here is needed to use the library. If you want to build a model, read
the README instead.

The cross-references below point at `../neuroarch/` and `../theorist/`, which
are **not shipped in the wheel and the sdist**. They exist only in the source
repository. That is why these findings are `FROM LITERATURE` here: a reader who
installed this package cannot re-run them, and the README says so wherever it
repeats a number from this file.

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

