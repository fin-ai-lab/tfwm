# `factor_diagram_clean.png` — caption + what the panels mean

Figure files: `factor_diagram_clean.{png,pdf}` (paper/slides, titles only) and
`factor_diagram.{png,pdf}` (same figure with in-panel captions, internal).

```bash
uv run plots/latent_eval/factors/factor_diagram.py --clean   # this figure
uv run plots/latent_eval/factors/factor_diagram.py           # annotated twin
```

Every number is real: one month (**2020-08**) run through the analysis scripts'
own code paths — nothing is a cartoon. 2020-08 is one of the 31 reported
months: the 6-month SSL embeddings are cached for those months only, so no
held-out month is available.

**The idea in one sentence.** The panel of stocks defines its own latent
factors, and we ask a frozen embedding two separate things — can a readout on
the *whole* embedding recover one loading (panel 3, "is it recoverable?"), and
how many loading directions do its *top-10 PCs* reach (panel 4, "is it
prominent?") — always against an untrained-ViT floor.

## Paste-ready caption

> **Figure N. What the latent-factor analyses ask of an embedding.** One month
> (2020-08), read left to right. **(1)** Realized correlation matrix of 5-minute
> mid-quote log returns for the *N* = 699 stocks present every day of the month
> (ordered by their first loading); its spectrum (inset) is cut by the perturbed
> eigenvalue-ratio rule at $\hat K = 6$ latent factors. **(2)** Each stock is a
> point in loading space $(\lambda_1, \lambda_2)$, coloured by $r^2_{\hat K}$,
> the share of its standardised variance the factors explain. **(3)** *Decode:*
> ridge from a stock's month-mean full-day embedding to its loading
> $\lambda_1$, scored out of fold on disjoint firms; points are out-of-sample
> predictions and *r* is their correlation with the truth — I-JEPA reaches
> +0.77 against +0.62 for an untrained ViT of the same architecture.
> **(4)** *Subspace alignment:* how many of the $\hat K$ loading directions the
> embedding's dominant geometry reaches. $\sum_k \rho_k^2$ over the canonical
> correlations between the top-10 embedding PCs and the loading space is the
> effective number of shared dimensions, drawn as a $\hat K$-wide box filled to
> that count; the filled fraction is the reported
> $\bar\rho = \sum_k \rho_k^2/\hat K$. I-JEPA reaches 1.24 of 6
> ($\bar\rho$ = 0.21) against 0.79 ($\bar\rho$ = 0.13) for the untrained ViT.
> Canonical directions are *rotations* of $\Lambda$, so the widest segments are
> tagged with the raw loading each resembles. Panels 3 and 4 are siblings, not a
> sequence: both read panel 2's loading space, and in both the untrained-ViT
> floor — not chance — is the reference.

## The panels

### 1 — Realized correlation PCA (where the targets come from)

5-minute mid-quote log returns, 09:30–16:00, for the 699 stocks present on
every full day of the month (Pelger's intersection rule). The 09:30→09:35
increment is dropped — one-sided opening books give garbage first marks —
leaving 77/day. Each stock is standardised by the square root of its realized
quadratic variation, so the heatmap is the realized **correlation** matrix
(unit diagonal = the thin dark line). Stocks are sorted by λ₁, which is why it
reads as a gradient: the bottom-right block co-moves with the market factor.

Inset: leading eigenvalues, cut by the perturbed eigenvalue-ratio rule
(Pelger 2018, ε = 0.08) at **K̂ = 6** — one large eigenvalue (0.145, the market)
plus five above the cut. Loadings are Λ = √N·U_K̂, sign-fixed so each factor's
mean loading is positive.

### 2 — Stocks in loading space (what a "loading" is)

One point per stock at its (λ₁, λ₂). Hue is r²_K̂, the share of that stock's
standardised variance the K̂ factors explain (0.00–0.75 here); it rises with λ₁
— names that load hard on the market are the ones the factor model explains.
Panels 3 and 4 both interrogate this space, and both are cross-sectional: the
unit of observation is a stock, not a day.

### 3 — Decoded loadings, out of sample (`decode_loadings.py`)

Feature per stock: the **month-mean full-day embedding** (one trading day is
one view through the frozen encoder, averaged over the month). Model:
`Ridge(alpha=100)` on standardised features, target λ₁. Harness: 5-fold CV
**grouped by gvkey**, so dual listings and ticker changes never straddle a
fold. Every plotted point is an out-of-fold prediction and *r* is the OOS
Pearson correlation — I-JEPA (`ijepa_6mo`) **+0.77**, random-init ViT
(`randvit_s0`) **+0.62**. The y-axis is unticked on purpose: ridge shrinks
predictions, so read the co-movement, not the scale.

The full analysis scores nine targets this way (total loadings 1–4, r²_K̂,
continuous 1–2, jump 1–2) and reports the mean over months with a *t* across
months.

### 4 — Subspace alignment (`subspace_alignment.py`)

Take the same stock-centroid embeddings, demean across stocks, keep the **top
J = 10 PCs** (N×10); take the demeaned loading matrix Λ (N×K̂). Their canonical
correlations ρ₁ ≥ … ≥ ρ_K̂ give Σρ², the effective number of shared dimensions
— drawn as a K̂-wide box filled to that count and segmented by direction. The
filled fraction is **ρ̄ = Σρ²/K̂**, the statistic the tables report. I-JEPA:
**1.24 of 6** (ρ̄ = 0.21). Untrained ViT: **0.79 of 6** (ρ̄ = 0.13, 5 seeds).

* Canonical directions are **rotations** of Λ, not λ₁…λ_K̂ in order, so the
  widest segments are tagged with the raw loading each resembles (≈λ₂, ≈λ₃,
  ≈λ₄ here). Without the tags the bar reads as λ₁, λ₂, … — exactly wrong.
* **The gap is spread across directions.** Per direction I-JEPA gets
  ρ² = 0.46, 0.40, 0.20, 0.09, 0.08, 0.02 against the floor's 0.44, 0.22, 0.07,
  0.04, 0.02, 0.01: directions 1–2 carry 44% of the ρ̄ gap and directions 3–5
  most of the rest. The first direction is a tie.
* **Permutation null: 0.09 of 6** (200 row shuffles of Λ) — computed and
  printed on every run, not drawn (both encoders sit an order of magnitude
  above it). Quote it if anyone asks whether the in-sample CCA manufactures the
  alignment.

## Panel 3 vs panel 4 — not the same question

| | panel 3 | panel 4 |
|---|---|---|
| target | one axis, λ₁ | all K̂ axes, optimally rotated |
| features | the **whole** embedding | only its **top-10 PCs** |
| fit | ridge, out-of-fold | canonical correlation, in-sample |
| asks | is it *recoverable*? | is it *prominent*? |

√ρ₁² = 0.68 is not panel 3's *r* = 0.77. For I-JEPA here the leading
canonical direction resembles λ₂ (|r| = 0.95; λ₁ is 0.86), the second λ₃ (0.72)
and the third λ₄ (0.78). Restricting to the top-10 PCs drops λ₁'s decode from
0.77 to 0.58. So λ₁ is recoverable but **not**
prominent, which is why both panels are drawn. The rotation is
encoder-specific, so it is never printed on a shared axis —
`factor_diagram.py` prints the full nearest-loading table on every run.

## Reading it right

* **Panels 3 and 4 are siblings, not a sequence.** Both read panel 2's loading
  space, from two separate scripts; neither consumes the other. The fork above
  them says so.
* **The floor is the evidential standard.** An untrained ViT already decodes λ₁
  at +0.62 and reaches 0.79 of 6 directions — a random projection of a price
  grid is not orthogonal to market beta. State claims as a margin over the
  random-init floor, never as a raw *r* or a distance from zero. (Panel 3 draws
  one floor seed for legibility; panel 4 averages 5.)
* **This is a representation question, not forecasting.** The encoder never
  trained on this month, but features and targets come from the same price path.
* **Factor identity holds only up to rotation across months** — "factor 2"
  means the second eigenvalue's direction in *that* month.
* **The month is illustrative.** Over the 31 reported months I-JEPA decodes
  the top-4 loadings at mean OOS *r* = 0.61 vs 0.50 for the floor, and reaches
  ρ̄ = 0.20 vs 0.15. Panel-3 folds use a fresh `rng(0)` for this single
  month, so the annotated *r* can differ from the JSON in the third decimal.

## Symbols

| symbol | meaning |
|---|---|
| N | stocks present on every day of the month (699 in 2020-08) |
| K̂ | latent factor count, perturbed eigenvalue-ratio rule, ε = 0.08 (6 here) |
| λ₁, λ₂ | a stock's loadings on the first two latent factors |
| r²_K̂ | share of a stock's standardised variance explained by the K̂ factors |
| r | out-of-sample Pearson correlation of decoded vs true loading |
| ρ_k | k-th canonical correlation between the top-10 embedding PCs and the loading space (a **rotation** of Λ, not λ_k) |
| Σρ² | effective number of loading directions the embedding reaches, out of K̂ |
| ρ̄ | Σρ²/K̂ — the captured share of the loading space |
