# ALO Array Modeling

Radio astronomy modeling for the **Astrophysical Lunar Observatory (ALO)** on the lunar far side (Tsiolkovsky crater, φ = −20.38°, λ = 128.97°).

## What this code does

The workflow is split across two scripts:

### `alo_array_modeling.py`  — Steps 1–4
| Step | What it computes |
|------|-----------------|
| 1 | ENU/MCMF element positions for 12 core+outrigger configurations (32×32 and 128×128 core arrays, outrigger distances 0.1–5 km) |
| 2 | 2-D Array Factor |AF(l,m)|² in dB for all configs × 4 frequency sub-bands (1–5, 5–10, 10–20, 20–40 MHz) |
| 3 | Beam metrics: Ω_B, D_peak, G_peak, A_eff, max sidelobe level |
| 4 | Sensitivity model: interferometric radiometer equation + confusion noise, per-target integration times, feasibility classification |

### `alo_extended_analysis.py`  — Extended analysis
- Solar-system reference emitter detection horizons (Jupiter DAM, Earth AKR, Saturn SKR, …)
- Target × config detection matrix heatmap
- Cumulative N_detected vs integration time curves
- Per-band sensitivity curves with target flux scatter overlay
- Weighted configuration ranking/scoring
- Aspirational target deep-dive (100–1000 h targets)
- Best config per target and per-config summary CSVs

## Quick start

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib scipy

# 2. Run Step 1–4 modeling (takes ~5–10 min for 128×128 configs)
python alo_array_modeling.py

# 3. Run extended analysis (reads outputs/csv/sensitivity_results.csv)
python alo_extended_analysis.py
```

All outputs land in `outputs/plots/` (PNG figures) and `outputs/csv/` (data tables).

## Target catalogue

Place your exoplanet CSV at the path configured in `alo_array_modeling.py` (default:
`/home/.../ALL.csv`). Required columns: `Name`, `flux_mJy`, `frequency_MHz`.

## Key results

- **Best configuration**: core 128×128 + 5 km outriggers (17 408 elements, B_max ≈ 10.7 km)
- **Feasible targets** (< 100 h): 30 exoplanets
- **Aspirational targets** (100–1000 h): 21 exoplanets
- Easiest target: HD 38529 b (~0.008 h ≈ 30 s at 30 MHz)

## Physical model

- Sky temperature: T_sky = 180 × (ν/180 MHz)^{−2.6} K
- Interferometric sensitivity: σ = SEFD / √(N_pol · N(N−1) · Δν · t)
- Confusion noise: σ_c = 0.2 mJy · (ν/74 MHz)^{0.8} · (θ_HPBW / 4 arcmin)
- Detection threshold: 5σ

## Repository structure

```
ALO_modeling/
├── alo_array_modeling.py       # Steps 1–4
├── alo_extended_analysis.py    # Extended analysis
├── README.md
└── outputs/                    # Generated (not tracked by Git)
    ├── csv/
    └── plots/
```
