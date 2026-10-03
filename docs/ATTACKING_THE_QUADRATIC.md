> **Status (2026-10-03).** An independent design analysis, written as a
> design workspace document. It is **not** package documentation, and it was
> produced by a separate investigation, not by the framework's own gate.
>
> Two things a reader should know before quoting it:
>
> 1. Where it cites this project's recorded measurements (for example the
>    spiking-vs-dense parity figures in §7), those are **historical run
>    values**. The figures move between runs — `examples/04_equal_budget.py`
>    prints the current ones, and its threshold sweep in the same run varies
>    by more than the spiking-vs-dense gap. Quote the script, not the number.
> 2. Its own `[MEASURED]` tags mean "measured by that investigation", not
>    "verified by this framework's gate". The derivations are the substance
>    here: the conditional SETH impossibility, the TC⁰ containment, and the
>    event-sparse backward-tape argument in §2.4.

---

# ATTACKING THE QUADRATIC — Event-Gated Delta Memory (EGDM)

Design workspace document, axiomrnn. Mission framing: find the mechanism most likely to remove
the O(L²) of attention, or prove removal impossible. Both are delivered below: removal is
impossible for *exact* attention (conditional proof), possible for bounded-state approximations
(provable expressiveness loss). The mechanism is the approximation judged most likely to survive.

Label legend: **MEASURED** (observed on this machine or in a cited paper), **DERIVED** (proved
here from stated premises), **HYPOTHESIS** (unverified conjecture), **LITERATURE** (cited result,
author+year). Every number carries a label.

---

## 0. VERDICT

1. **The quadratic cannot be removed exactly.** Computing exact softmax attention over L tokens
   requires the L×L score matrix; for entry magnitudes B = Θ(√log L) and d = O(log L), a truly
   subquadratic algorithm would refute SETH (Alman & Song, NeurIPS 2023 — **LITERATURE**). The
   only asymptotic escape is rectangular fast matrix multiplication, O(L^{ω−1}) ≈ O(L^{1.372})
   for L = Θ(d), whose constants exceed practical reach below astronomically large L (**DERIVED**).
2. **Removal is possible only approximately**, by changing the operator class: bounded-state
   recurrences. This costs expressiveness — provably. Log-precision transformers ⊆ uniform TC⁰
   (Merrill & Sabharwal, TACL 2023 — **LITERATURE**); SSMs and linear RNNs are also ⊆ TC⁰ and
   additionally cannot solve permutation composition (A₅) at any depth (Merrill, Petty,
   Sabharwal, ICML 2024 — **LITERATURE**). Bounded state must forget (pigeonhole — **DERIVED**);
   attention avoids forgetting only by paying O(L·d) memory.
3. **The mechanism**: a gated delta-rule fast-weight mixer whose writes are gated by LIF spike
   events under a hard budget. Between events the state is *piecewise constant*, which makes
   exact BPTT event-sparse: the backward tape scales with write density p, not L (**DERIVED**,
   proof in §2.4). This is the one place where the project's memory-budget-as-input axiom becomes
   a literal, measurable quantity: budget → threshold → p → tape size.
4. **Status of the central claim** ("a bounded-state event-gated delta mixer is the best removal
   of the quadratic"): the complexity claim is **DERIVED**; the exact-event-BPTT claim is **DERIVED**
   (theorem); the "best" claim is **HYPOTHESIS** — the closest prior art (Sparse Delta Memory,
   Meta FAIR, Jul 2026) removes a different axis (addressing sparsity), and hybrids currently
   beat pure subquadratic models in production (Jamba, AI21 2024; Zamba, Zyphra 2024).

---

## 1. MECHANISM — Event-Gated Delta Memory (EGDM)

### 1.1 Objects (exact shapes)

Per layer, per head h ∈ {1..H}: d_h = d/H (head width), E ∈ {1,2,4} (state expansion).

| symbol | shape | meaning |
|---|---|---|
| x_t | R^d | input token |
| q_t, k_t | R^{E·d_h} | query, key |
| v_t | R^{d_h} | value |
| β_t | R^{E·d_h} | write rate (delta-rule learning rate) |
| α_t | (0,1) | decay gate |
| S_t | R^{d_h × E·d_h} | fast-weight state (associative memory) |
| a_t | R | LIF membrane potential of the write-gate neuron |
| z_t | {0,1} | write event (spike), z_t = 1[a_t ≥ θ_t] |

### 1.2 Update rules (per token t)

```
u_t     = Conv1D(x_t; kernel=4)                       # depthwise, kernel 4
q_t,k_t,v_t,β_t,α_t = split(W_u · u_t)              # 6 learned projections; β_t = softplus, α_t = sigmoid
# --- LIF write gate (the axiomrnn core) ---
a_t     = a_{t−1} + (W_sal·u_t − a_{t−1})/τ          # leak-integrate, τ learnable
z_t     = 1[a_t ≥ θ_t];   â_t = σ̃(a_t − θ_t)         # â_t = surrogate, e.g. 1/(1+e^{−α(a_t−θ_t)})
a_t     ← a_t·(1 − z_t)                              # reset on event
θ_t     = θ_0 + η·(p̄_t − p_target)                   # adaptive threshold = homeostasis; p_target = budget
# --- gated delta rule (write only on event) ---
S_t     = α_t·S_{t−1} + z_t·β_t·(v_t − α_t·S_{t−1}k_t)·k_tᵀ     # rank-1, gated
y_t     = S_t·q_t;  o_t = W_out·(y_t ⊙ σ(W_g·u_t))               # read + output gate
```

The delta rule erases the old association of k_t before writing (fast-weight programming,
Schmidhuber 1992; DeltaNet, Schlag/Irie/Schmidhuber ICML 2021 — **LITERATURE**); the decay α_t
and input-dependence follow Gated DeltaNet (Yang/Kautz/Hatamizadeh, ICLR 2025 — **LITERATURE**).
The gate z_t is the only new element. **The gate does not reduce compute** — the chunk's dense
intra-chunk pass still runs; it reduces the *tape* and enforces the budget. Stated plainly:
compute saving from z_t is zero; memory saving is Θ(1/p).

### 1.3 Chunked parallel form (training, C = chunk size, typically 64–256)

Within chunk c (positions 1..C), with A_i = I − β_i k_i k_iᵀ (symmetric Householder-like
factors), the state and reads admit the chunkwise form used by parallel DeltaNet (Yang et al.,
NeurIPS 2024, WY representation, Bischof & Van Loan 1985 — **LITERATURE**), extended with the
gate (z_i multiplies β_i, so gated A_i = I − z_iβ_i k_i k_iᵀ):

```
S_c = S_{c−1}·(A_1···A_C) + Σ_{j≤C} β_j v_j k_jᵀ·(A_{j+1}···A_C)        # state carry + write matrix
y_i = S_{c−1}·(A_1···A_i)q_i + Σ_{j≤i} β_j v_j (k_jᵀ·(A_{j+1}···A_i)q_i) # read
```

The intra-chunk coefficients are computed by a triangular solve on the C×C matrix
G[j,l] = −z_lβ_l(k_jᵀk_l) (strictly upper triangular): X = (I − G)⁻¹·tril(KQᵀ), then
y_intra = (X·diag(β)·V). Cost per chunk: O(C²·E·d_h + C·E·d_h²) flops (**DERIVED**; WY form
avoids the naive C³·E·d_h expansion). The recurrence is only across chunks — L/C parallel
chunk-columns on GPU.

### 1.4 Event-driven exact BPTT (the memory lever)

**Theorem (exact event BPTT).** For the gated recurrence above, between events the state map is
the identity: for t ∉ E (event set), S_t = α_t·S_{t−1} — with the gate, and taking α_t = 1
for simplicity of the statement, S_t = S_{t−1}. The chain rule factorizes over events:
∂S_t/∂S_{t′}[M] = M·∏(I − z_iβ_i k_i k_iᵀ) over events i ∈ (t′, t]. Backward pass applies,
per event, the linear map M ↦ M(I − zβkkᵀ) at cost 4·E·d_h² flops per head — the same order
as the forward update (6·E·d_h²). **Proof:** direct differentiation of the rank-1 update; the
map is linear in M, so no per-timestep Jacobian is stored. ∎ (**DERIVED**)

**Tape (per layer, per batch element, bf16):**

```
Tape_EGDM = (L/C)·E·d²/H        # chunk-boundary state checkpoints
          + p·L·(E+2)·d         # per event: k (E·d), βv (d), across H heads
          + O(E·d²/H)           # live state
```

Alternative (recompute-in-chunk, FlashAttention-style): drop the checkpoint term to
O(E·d²/H) and recompute intra-chunk ops in backward — trades ~2x backward compute for
(L/C)·E·d²/H bytes.

Numbers (d=2048, H=16, E=1, C=256, L=4096, p=0.05): tape ≈ 8.4 MB + 2.5 MB + 0.5 MB
≈ **11.4 MB** per layer per batch element (**DERIVED**). Transformer with FlashAttention-2:
activations ≈ 8·L·d floats ≈ **134 MB** (Q,K,V,O,gate; the L² term is recomputed, not stored —
Dao et al., NeurIPS 2022 — **LITERATURE**). Ratio ≈ **12x less activation memory** (**DERIVED**).
At L=32768: 1.07 GB vs 84 MB → **12.7x** (**DERIVED**). At p=0.01, L=4096: 9.4 MB → **14x**.

Contrast with the project core: the dense LIF-BPTT backward tape is **MEASURED** at ~30x the
forward tape and always dense; bit-packing gives **MEASURED** 2.6–5.1% overall and does **not**
binarize gradients. EGDM's lever is orthogonal: event sparsity of the *update schedule*, not
precision. The two compose (event data k, βv can be int8-stored; gradients remain dense, exactly
as the project measured).

### 1.5 Integration with axiomrnn

- The write-gate neuron **is** the existing LIF core (axon/lif.py); its surrogate gradient must
  pass the project gate (axon/gate.py): verified by finite differences **on the smoothed model**,
  since the hard forward pass is piecewise-constant and finite differences return exactly 0
  (project fact (a), **MEASURED** in the project). Gate check #8 (`credit_k=T` ≡ full BPTT)
  transfers: event-truncated BPTT must equal full event BPTT at T = |E|.
- The credit horizon becomes the event count: truncating the backward event chain to the last T
  events bounds backward state cost by T·4·E·d²/H per position — the same truncation lever the
  project measured at 17x, falling to 8.7x at T=128 (**MEASURED** in the project core), now
  expressed in events rather than timesteps.
- The planner (axplan, no TensorFlow) takes the budget as input and outputs (E, C, p_target,
  θ_0) — the Pareto front over state size × chunk size × write density.

---

## 2. MATH — asymptotics, constants, proof status

Parameters: N_mixer ≈ 6d² per layer (six d×d projections; state E·d²/H is a buffer, not a
parameter, unless the initial state is learned as in SDM). FFN (SwiGLU, d_ff = 8d/3) adds
16d²/3 — reported separately, never folded into mixer claims.

### 2.1 Time (per layer, per token, flops) — **DERIVED**

| term | softmax attention (FA2) | EGDM |
|---|---|---|
| projections | 8d² (q,k,v,out) | 12d² (q,k,v,β,α,out) + 4d (conv) |
| mixing | 4Ld (QKᵀ + AV) | 6·E·d²/H (state ops) + C·E·d (intra-chunk) |
| total | 8d² + 4Ld | 12d² + 6Ed²/H + C·E·d |

**Flop crossover:** 4Ld = 4d² + 6Ed²/H + CEd ⟹ **L\* = d + 1.5·E·d/H + C/4**.
For d=2048, H=16, E=1, C=256: **L\* ≈ 2300 tokens** (**DERIVED**). Below L\*, EGDM burns
~1.25x more flops (at L=1024: 52.4M vs 41.9M); above, the ratio grows linearly (at L=8192:
52.4M vs 100.5M → **1.9x cheaper**). With E=4: L\* = d + 6d/H + C/4 ≈ 2800.

**Wall-clock crossover is a different, constant-factor question.** FA2 runs L×d GEMMs at
near-peak MFU; EGDM's d_h×d_h state GEMMs (128×128 at d_h=128) and C×C triangular solves
run at lower MFU, and the chunk recurrence serializes across L/C chunk boundaries.
**HYPOTHESIS:** wall-clock L\* ∈ [1K, 4K] on H100-class hardware; tested by P3.

**Inference:** EGDM is O(E·d²/H) per token, O(1) in L; transformer is O(L·d) per token
(KV cache read) — EGDM wins on decode FLOPs by Θ(L/d) at long contexts (**DERIVED**), but
see §4.2: the KV cache is a *quality* feature, not only a cost.

### 2.2 Memory (training) — **DERIVED**

| | transformer (FA2) | EGDM |
|---|---|---|
| activations | 8·L·d floats | (L/C)·E·d²/H + p·L·(E+2)·d floats |
| at d=2048,H=16,E=1,C=256,L=4096,p=0.05 | 134 MB | 11.4 MB (**12x less**) |
| scaling in L | Θ(L·d) | Θ(L/C · Ed²/H + p·L·d) |

Naive (non-FA) attention stores the L² score matrix: at L=4096, 2·L² bytes = **33.5 MB per
layer per batch element** on top of activations (**DERIVED**); EGDM has no L² term at any L.

### 2.3 Capacity (associative storage) — **DERIVED** + **HYPOTHESIS**

Read: y = S·q = Σᵢ βᵢ vᵢ (kᵢᵀq) (erasure terms ignored). The map k↦v lives in a matrix of
rank R ≤ min(d_h, E·d_k) = E·d_h per head, E·d total across heads. Storing N associations
with learnable (near-orthogonal) keys requires N ≤ E·d — **N\* = Θ(E·d) with learned keys**
(**DERIVED**, rank bound; matches the capacity limitation inferred by Schlag et al., ICML 2021 —
**LITERATURE**). With random keys, crosstalk at q = k_j: off-diagonal |kᵢᵀk_j| ~ σ²√(E·d)
vs diagonal σ²·E·d, so per-association SNR ~ √(E·d)/N and reliable retrieval needs
**N ≲ √(E·d) for random keys** (**DERIVED**, random-matrix crosstalk; the constant is
**HYPOTHESIS**). Prediction P1 tests the learned-key boundary at N = E·d.

### 2.4 Central claims — proof status

| # | claim | status |
|---|---|---|
| A | exact softmax attention cannot be computed in truly subquadratic time (bounded-entry regime) | **PROVEN conditionally** (SETH; Alman & Song 2023 — **LITERATURE**) |
| B | bounded memory M + unbounded context ⟹ forgetting mandatory | **PROVEN** (pigeonhole — **DERIVED**) |
| C | gated delta recurrence admits chunked O(L·(Ed²/H + Cd)) training | **PROVEN** (chunkwise form — **DERIVED**; WY representation **LITERATURE**) |
| D | event-driven BPTT is *exact* and tape scales as Θ(p·L·d + (L/C)·Ed²/H) | **PROVEN** (§1.4 theorem — **DERIVED**) |
| E | capacity N\* = Θ(E·d) (learned keys), Θ(√(E·d)) (random keys) | rank bound **PROVEN**; constants **HYPOTHESIS** |
| F | wall-clock crossover L\* ∈ [1K, 4K] on H100-class | **HYPOTHESIS** (P3) |
| G | EGDM is the *best* removal of the quadratic | **HYPOTHESIS** — competing axis exists (SDM addressing sparsity, 2026); hybrids currently win in production |

---

## 3. FALSIFIABLE PREDICTIONS

Each prediction names the experiment, the configuration, and the numeric threshold that refutes
the associated claim.

**P1 — capacity boundary (tests E, the rank claim).**
MQAR (Arora et al., ICLR 2024 — **LITERATURE**; task: N_kv random (key,value) pairs, then
queries must emit the bound values). Config: d=1024, H=8, E=1, L=4096, C=128, p=1.0
(gate off), N_kv ∈ {64, 128, 256, 512, 1024}. Baseline: matched softmax-attention
transformer.
Prediction: EGDM within **2 points** of softmax at N_kv ≤ 256 (= d/4); drop ≥ **10 points**
by N_kv = 1024 (= d).
**Refutation:** accuracy within 2 points at N_kv = 1024 ⟹ the rank bound is not the binding
constraint; claim E and the capacity story die. Then the state size is not the lever, and the
mechanism reduces to a budget wrapper on GDN.

**P2 — tape scaling (tests D, the event-BPTT theorem).**
Fix L=8192, d=1024, H=8, E=1, C=128; vary θ to set p ∈ {0.01, 0.05, 0.1, 1.0}; measure
peak backward-pass memory (bytes) on identical hardware.
Prediction: memory(p=0.01) / memory(p=1.0) ∈ **[50, 150]** (≈ 1/p, since event data
dominates at small p; exact ratio **DERIVED** from §2.2 formula).
**Refutation:** ratio < **5** ⟹ the tape is not event-sparse in practice (e.g., the gate is
saturated at p≈1, or the checkpoint term dominates); claim D's practical value dies.

**P3 — wall-clock crossover (tests F).**
d=2048, H=16, E=1, C=256, batch=32, H100-class GPU; tokens/sec vs FlashAttention-2 baseline
at L ∈ {512, 1024, 2048, 4096, 8192}.
Prediction: EGDM slower than FA2 at L=512; crossover L\* ∈ **[1024, 4096]**; EGDM ≥ **1.5x
faster** than FA2 at L=8192.
**Refutation:** EGDM slower than FA2 at L=8192 ⟹ constant-factor model wrong; the asymptotic
win is unreachable at practical widths and the mechanism fails its reason to exist.

**P4 — gate utility (tests the biology story and the budget mechanism).**
Wikitext-103, d=512, L=1024, 340M-parameter-matched models; p=0.1 (θ calibrated) vs p=1.0
(θ = −∞, gate removed).
Prediction: perplexity gap ≤ **0.5 nats**.
**Refutation:** gap ≥ **2 nats** ⟹ the spike gate destroys learning; drop the gate, keep the
delta rule (the mechanism degenerates to Gated DeltaNet with a memory-budget wrapper, and the
[BIO] correspondence is decoration).

**P5 — backward compute ratio (tests the project's 30x pain point).**
Same config as P2; measure backward FLOPs / forward FLOPs for the mixer layer.
Prediction: ratio ∈ **[1, 2]** (the event chain's backward is a linear map application —
§1.4).
**Refutation:** ratio > **5** ⟹ the event-tape story fails to address the project's measured
30x backward-tape blowup; the credit-rule lever must be sought elsewhere.

---

## 4. WHERE IT LOSES — against a transformer, quantitatively

1. **Short-context wall-clock.** Predicted **1.2–3x slower than FA2 at L ≤ 1K** (**HYPOTHESIS**):
   the 12d² vs 8d² projection overhead plus low-MFU small GEMMs and C×C triangular solves.
   At L=512, d=2048: 26.2M vs 22.0M flops per token (**DERIVED**), and the flop penalty
   understates the wall-clock penalty.
2. **Retrieval collapse beyond capacity.** At N_kv > E·d, failure is *hard* (state aliasing,
   §2.3); a transformer degrades *gracefully* — its KV cache grows O(L·d) and softmax argmax
   stays intact. Concrete: L=128K, d=4096, E=1 → capacity ≈ 4096 associations; a 128K-context
   transformer stores all 128K tokens. EGDM cannot match RULER-style multi-needle retrieval at
   128K without E ≥ 16–32 (**DERIVED** from the rank bound; the exact E is **HYPOTHESIS**).
3. **State tracking.** Provably ⊆ TC⁰ like transformers (**LITERATURE**: Merrill & Sabharwal
   2023), but empirically worse on permutation composition (Merrill et al., ICML 2024 —
   **LITERATURE**); Mamba-3's authors report prior linear models fail even parity (Grazzi et
   al., cited in Lahoti et al., ICLR 2026 — **LITERATURE**). EGDM inherits this unless
   hybridized.
4. **Training stability.** The delta rule contracts only when β < 2/‖k‖² (eigenvalue
   1 − β‖k‖² on the k-direction must stay in (−1,1) — **DERIVED**); unbounded learned β
   diverges. Gated DeltaNet's input gate + decay mitigates (**LITERATURE**); EGDM adds the
   spike gate, which can freeze learning if θ drifts high (p → 0) — P4 bounds the damage.
5. **No free lunch at inference.** The transformer's O(L·d) KV cache is a *quality* feature
   (no forgetting); EGDM's O(E·d²/H) state is a *cost* feature (mandatory forgetting, claim B).
   On distribution shift at L ≫ training length, EGDM's forgetting rule, not its compute,
   determines quality (**DERIVED**).
6. **Bit-packing does not transfer.** The project's measured 2.6–5.1% bit-packing win applies to
   activation storage; EGDM's tape is event-sparse, not bit-sparse, and gradients stay dense
   (project fact: binary activations ≠ binary gradients — **MEASURED**). Composable, not
   substitutive.

---

## 5. PRIOR ART — who tried it, year, why it did not take

**The quadratic-attacks (2019–2021) — all failed to displace softmax attention:**
- **Sparse Transformer** (Child et al., 2019): fixed sparse patterns, O(L√L). Hand-designed
  patterns; no learnable routing.
- **Reformer** (Kitaev, Kaiser, Levskaya, ICLR 2020): LSH attention, O(L log L), reversible
  layers; trained to 64K tokens. Why it failed: LSH bucketing is stochastic and
  query-independent per round — accuracy loss at n_rounds=1, needs ≥4 rounds (their own
  duplication-task table — **LITERATURE**); engineering complexity; FlashAttention later made
  the dense quadratic cheaper in practice.
- **Linear Transformers** (Katharopoulos et al., ICML 2020): feature-map attention, recurrent
  form O(L·d²). Why: fixed feature maps (elu, random) lose associative recall — the rank
  bottleneck (§2.3); "linear attention generally underperforms ordinary softmax attention"
  (Yang et al., ICML 2024 — **LITERATURE**).
- **Performer** (Choromanski et al., ICLR 2021): FAVOR+ random features. Why: biased
  estimator, variance at long contexts, never scaled to LLM training.
- **Linformer** (Wang et al., 2020): low-rank projection — projection is position-mixing and
  breaks causal inference.
- **Longformer** (Beltagy et al., 2020), **BigBird** (Zaheer et al., NeurIPS 2020):
  window/dilated + global / random + window + global, O(L·w), O(L^{1.5}). Work, but reduce
  the constant, not the exponent; patterns hand-designed.
- **Synthesizer** (Tay et al., ICLR 2021): learned *static* attention. Underperformed —
  proof that input-dependent routing is load-bearing.
- **Routing Transformer** (Roy et al., ICLR 2021): online k-means clustering of tokens;
  clustering overhead ate the gains.

**The recurrence line (2022–2026) — the current frontier:**
- **S4** (Gu, Goel, Ré, ICLR 2022): diagonal structured SSM. Strong long-range tasks; fixed
  linear dynamics — no input-dependent state.
- **Mamba** (Gu & Dao, 2023), **Mamba-2** (Dao & Gu, ICML 2024, SSD): selective SSM,
  input-dependent decay; Mamba-2 is 2–8x faster than Mamba-1. Why still not the answer:
  state decay is uniform (all associations decay together), and state-tracking failures are
  documented (Merrill et al. 2024; Mamba-3's own citations of parity failures).
- **RetNet** (Sun et al., 2023), **RWKV** (Peng et al., 2023): decay-based linear RNNs; good
  perplexity, weak retrieval; GLA's retrieval table shows the gap (37.7 vs 41.8 for
  Transformer++ at 1.3B/100B tokens — **LITERATURE**).
- **GLA / FlashLinearAttention** (Yang et al., ICML 2024): data-dependent decay + I/O-aware
  chunkwise kernel; faster than FlashAttention-2 as a standalone layer even at 1K sequences
  (**LITERATURE**) — the hardware template EGDM reuses.
- **DeltaNet** (Schlag, Irie, Schmidhuber, ICML 2021): the delta rule as fast-weight
  programming; sequential training only — could not scale.
- **Parallel DeltaNet** (Yang et al., NeurIPS 2024): WY-representation chunkwise algorithm;
  1.3B/100B tokens beats Mamba and GLA; hybrids with sliding-window attention beat strong
  transformer baselines.
- **Gated DeltaNet** (Yang, Kautz, Hatamizadeh, ICLR 2025): decay + delta rule combined;
  beats Mamba2 and DeltaNet on recall, length extrapolation, long-context.
- **Lightning Attention-2** (Qin et al., 2024): intra-block attention + inter-block linear
  recurrence; constant training speed vs L under fixed memory (**LITERATURE**).
- **Hybrids**: **Jamba** (AI21, 2024; ICLR 2025) — 1:7 attention:Mamba, 256K context;
  **Zamba** (Zyphra, 2024) — shared attention every 6 blocks; Poli et al. (2024) found
  ~1/4 attention layers optimal. **Why the field converged here**: pure subquadratic models
  still trail on recall and state tracking; hybrids buy most of the memory win at a small
  attention tax. EGDM is a drop-in for the *subquadratic* layers of such a hybrid.
- **Mamba-3** (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu, ICLR 2026): complex-valued
  state + MIMO; +1.8 points over Gated DeltaNet at 1.5B; explicit state-tracking gains.
  The family is converging on richer *state content*, not sparser writes.
- **Sparse Delta Memory (SDM)** (Cabannes, Mazaré, Szilvasy, Douze, Lomeli, Auzina,
  Carpentier, Synnaeve, Jégou — Meta FAIR, Jul 2026, arXiv 2607.07386): **the closest prior
  art**. Extends Gated DeltaNet with Product-Key-Memory *addressing*: sparse top-W reads and
  writes into an explicit N-slot memory table; ~1000x state scaling at constant FLOPs;
  learned initial state as parametric memory. Why EGDM is not SDM: SDM's sparsity is in
  *addressing* (which slots), with a dense write *schedule* (every token) and a dense
  backward tape; EGDM's sparsity is in the *write schedule* (which timesteps), with an
  event-driven exact backward tape and a hard budget enforced by a homeostatic threshold.
  The two are composable: SDM's slots with EGDM's event gate. SDM's own caveat: unthrottled
  state growth scales memory as O(d³) per layer — a budget mechanism is exactly what it lacks.
- **Fast-Weight Product Key Memory** (Zhao & Jones, 2026, arXiv 2601.00671): sparse PKM for
  test-time-training fast weights; independent confirmation that sparsity is the live axis.

**Why the 2020–2021 attacks did not take, in one line each:** wrong regime (L was small, so
attention's constants won), weak feature maps (rank bottleneck), unstable routing (LSH,
k-means), and — decisive — FlashAttention (Dao et al., NeurIPS 2022; FA2, 2023) made the
quadratic IO-optimal, so any replacement must beat near-peak MFU, not just asymptotics.

**Biology priors for the gate:**
- Delta-rule erase-write: [BIO] — three-factor neoHebbian plasticity; the erase term
  approximates bidirectional (LTP/LTD) calcium-dependent plasticity (Pfister & Gerstner,
  J Neurosci 2006 — triplet STDP, maps to BCM; Frémaux & Gerstner 2016 — three-factor
  review).
- Spike-gated write with adaptive threshold: [BIO] — eligibility traces gated by postsynaptic
  spikes and a neuromodulator (three-factor rules; Yagishita et al. 2014 — dopamine timing
  window); adaptive threshold ≈ BCM sliding threshold (Bienenstock, Cooper, Munro 1982;
  Pfister & Gerstner 2006 show the triplet rule maps to BCM).
- Bounded state (finite E·d²/H per layer): [BIO] — finite synapses.
- Chunking: [DEPARTS] — hardware serialization compromise; biology is continuous-time and has
  no chunk boundary. The chunk size C is chosen for tensor-core efficiency, not function.
- Deterministic threshold + surrogate gradient: [DEPARTS] — biological neurons are stochastic;
  determinism keeps gradients clean and the budget hard. Verified only on the smoothed model
  (project gate fact (a) — **MEASURED**).
- No axonal delays, no inhibition, no stochastic release: [DEPARTS] — each adds compute with
  no measured benefit on the target tasks; energy is **not measured** (no neuromorphic
  hardware on the test machine — established project fact), so no energy claim is made.

---

## 6. WHAT WOULD MAKE THIS WRONG

1. **P1 refuted** (MQAR within 2 points at N_kv = E·d): the rank/capacity claim dies; state
   size is not the long-context lever; the mechanism is a budget wrapper on GDN with no
   expressiveness story.
2. **P2 refuted** (tape ratio < 5x): event-sparsity does not materialize in the backward pass;
   the exact-event-BPTT theorem is true but useless in practice (gate saturates at p ≈ 1).
3. **P3 refuted** (slower than FA2 at L=8192): the constant-factor model is wrong; the
   asymptotic win is unreachable at practical widths.
4. **P4 refuted** (≥ 2 nats at p=0.1): the spike gate is harmful; drop it — the [BIO] story
   dies with it, and the design collapses into Gated DeltaNet + budget wrapper (still a
   legitimate memory-budget mechanism, but not a new one).
5. **P5 refuted** (backward/forward > 5x): the credit-rule lever fails; the project's 30x
   backward-tape problem persists.
6. **SDM-style addressing wins the same budget with better quality** (Meta FAIR, 2026 —
   **LITERATURE**): if sparse *addressing* scales state 1000x at constant FLOPs and beats
   EGDM on RULER at matched memory, then write-schedule sparsity is the wrong axis; the
   correct move is to compose (event gate on top of PKM slots) or to switch axes entirely.
7. **Fast matrix multiplication becomes practical** (structured MM accelerators): exact
   subquadratic attention becomes reachable, killing the approximate-route thesis (claim A's
   premise).
8. **A hybrid beats everything anyway** (already the production pattern — Jamba/Zamba): then
   EGDM's value is only as a component, and its independent contribution must be measured
   against GDN layers in a hybrid, not against transformers.

---

## APPENDIX — axiomrnn established facts, used not inflated

- spiking 0.9938 vs dense MLP 0.9988 → **PARITY, not superiority** (**MEASURED**). The gate's
  quality ceiling is parity-level; P4's 0.5-nat budget is the operative test.
- bit-packing 2.6–5.1% overall; backward tape ~30x forward and always dense; binary
  activations do not make gradients binary (**MEASURED**). EGDM's lever is event sparsity,
  orthogonal and composable.
- credit rule: 17x, falling to 8.7x at T=128 (**MEASURED**). EGDM expresses the same lever
  in events: truncating the backward event chain to T events bounds backward state cost by
  T·4·E·d²/H per position (**DERIVED**).
- energy **NOT measured** (no neuromorphic hardware on the test machine). No energy claims.
- the sparse/bit crossover **has been measured by nobody** — P2 and P4 are exactly those
  measurements for this operator class.
- this is **NOT a biology simulator**. LIF is a crude abstraction of a neuron, not
  Hodgkin-Huxley; the gate is a budget-enforcing engineering device that happens to have a
  biological counterpart.

---

*Prepared for the axiomrnn design workspace. All claims labeled; all six required sections
delivered. The strongest honest statement available: the quadratic is provably removable only
approximately; the delta-rule bounded-state mixer is the best-supported approximation family
as of 2026; the event gate's contribution is exact event-sparse BPTT (proved) and a hard
memory budget (testable in P2/P4); whether it beats addressing-sparsity (SDM) or hybrids is
an open measurement, not a claim.*
