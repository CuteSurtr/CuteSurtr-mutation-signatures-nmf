# Mutation Signatures via NMF

A from-scratch implementation of **non-negative matrix factorization (NMF)** applied to somatic mutation data from TCGA, reproducing the **COSMIC single-base-substitution (SBS) mutation signatures** described in Alexandrov et al., *Nature* (2013).

> **Portfolio framing.** This project demonstrates competence in (a) numerical linear algebra (matrix factorization, convergence), (b) cancer genomics (SBS-96 mutation context, TCGA data pipelines), and (c) reproducible research (Docker, CI, benchmarks against a published standard).

---

## What this project does

1. **Ingests** TCGA somatic mutation calls (MAF format) and converts them into the **SBS-96 trinucleotide-context feature matrix** (96 mutation types × N samples).
2. **Factorizes** that matrix as `V ≈ W H` with `W ≥ 0`, `H ≥ 0`, using NMF with **multiplicative updates** (both Frobenius-norm and KL-divergence variants), implemented from scratch.
3. **Selects** the number of signatures via **cophenetic correlation** across random restarts and cosine similarity to COSMIC reference signatures.
4. **Compares** recovered signatures to the **COSMIC v3 reference set** (SBS1, SBS2, SBS5, SBS13, ...) and reports best-matching assignments.
5. **Validates** against `sklearn.decomposition.NMF` for correctness.

---

## The math (short version)

Given a non-negative matrix `V ∈ ℝ^{F × N}` (here `F = 96` mutation channels, `N` = samples), NMF solves

```
minimize   D(V, WH)
subject to W ≥ 0, H ≥ 0,   W ∈ ℝ^{F × K}, H ∈ ℝ^{K × N}
```

where `K` is the number of latent signatures and `D` is a divergence.

### Frobenius-norm updates (Lee & Seung 2001)

Objective: `D_F(V, WH) = ½ ‖V − WH‖_F^2`.

Multiplicative updates (guarantee monotone non-increase of the objective):

```
H ← H ⊙ (Wᵀ V)  ⊘ (Wᵀ W H + ε)
W ← W ⊙ (V Hᵀ)  ⊘ (W H Hᵀ + ε)
```

where ⊙ is elementwise product, ⊘ is elementwise division, and `ε` avoids division by zero.

**Derivation sketch.** At a stationary point of the Lagrangian, KKT conditions yield `W ⊙ ∇_W D = 0`. Writing `∇_W D = −V Hᵀ + W H Hᵀ` and imposing stationarity motivates the update rule as a fixed-point iteration with a diagonal rescaling that preserves non-negativity. The monotonicity proof uses an auxiliary function `G(W, W_t) ≥ D(V, WH)` with equality at `W = W_t`.

### KL-divergence updates

Objective: `D_KL(V ‖ WH) = Σ V log(V / WH) − V + WH`.

```
H ← H ⊙ (Wᵀ (V ⊘ (WH))) ⊘ (Wᵀ 𝟙)
W ← W ⊙ ((V ⊘ (WH)) Hᵀ) ⊘ (𝟙 Hᵀ)
```

KL is the **canonical choice** for count data like mutation counts, since the Poisson log-likelihood reduces to KL divergence up to constants.

---

## Repo layout

```
mutation-signatures-nmf/
├── src/mutsig/
│   ├── nmf.py           # from-scratch NMF (Frobenius + KL)
│   ├── features.py      # SBS-96 feature builder from MAF
│   ├── cosmic.py        # COSMIC signature loading + matching
│   └── plotting.py      # 96-channel barplot
├── tests/
│   └── test_nmf.py      # validates against sklearn + KKT checks
├── notebooks/
│   ├── 01_download.ipynb
│   ├── 02_features.ipynb
│   ├── 03_nmf.ipynb
│   └── 04_compare_cosmic.ipynb
├── data/
│   ├── raw/             # downloaded MAFs + reference genome
│   └── processed/       # SBS-96 matrices
├── figures/             # publication-quality signature plots
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Quickstart

```bash
# 1. install
pip install -e .

# 2. run tests (validates NMF against sklearn)
pytest

# 3. run the full pipeline (once data is downloaded)
python -m mutsig.pipeline --k 6 --loss kl
```

Or with Docker:

```bash
docker build -t mutsig .
docker run --rm -v $PWD/data:/app/data mutsig pytest
```

---

## Data sources

| Source | Purpose | Link |
|---|---|---|
| TCGA MC3 public MAF | somatic mutations | [GDC](https://gdc.cancer.gov/about-data/publications/mc3-2017) |
| COSMIC v3 SBS | reference signatures | [COSMIC](https://cancer.sanger.ac.uk/signatures/sbs/) |
| GRCh38 reference | trinucleotide context | [NCBI](https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/) |

All data is public and non-credentialed.

---

## First real-data result: TCGA-LUAD (lung adenocarcinoma)

Running the full pipeline on 50 TCGA-LUAD cases (14,674 SNVs) recovers the
**tobacco mutagenesis signature SBS4** across all model orders K ∈ {2, ..., 7}:

| K | best KL | SBS4 cosine similarity |
|---|---|---|
| 2 | 2494.0 | **0.971** |
| 3 | 2243.9 | **0.980** |
| 4 | 2108.8 | **0.968** |
| 5 | 1997.5 | **0.964** |
| 6 | 1900.7 | **0.950** |
| 7 | 1797.6 | **0.933** |

SBS4 is the COSMIC signature attributed to tobacco-smoking mutagenesis in
lung adenocarcinoma — recovering it with cosine similarity ≥ 0.93 across
every tested K (and 0.98 at K=3) demonstrates that a from-scratch NMF with
multiplicative updates reproduces the central biological finding of
Alexandrov et al. 2013 on a focused cohort.

Reproduce with:

```bash
python scripts/run_luad.py --n-cases 50
```

See `figures/TCGA-LUAD_K4_signatures.png` for the K=4 panel.

## Synthetic validation

On synthetic data (4 Dirichlet-sampled signatures, 80 samples, 1500 mutations
per sample), our NMF recovers all 4 ground-truth signatures with cosine
similarities {0.962, 0.992, 0.997, 0.996}. See
`figures/synthetic_recovery.png`.

---

## References

- Lee & Seung. *Algorithms for non-negative matrix factorization.* NeurIPS 2001.
- Alexandrov et al. *Signatures of mutational processes in human cancer.* Nature 2013.
- Alexandrov et al. *The repertoire of mutational signatures in human cancer.* Nature 2020.
- COSMIC Mutational Signatures v3.3.
