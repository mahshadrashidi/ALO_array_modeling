"""
ALO Core Array + Outrigger Modeling
Astrophysical Lunar Observatory — Tsiolkovsky Crater, Lunar Far Side

Implements Steps 1-4 from the ALO Core Array + Outrigger Modeling specification:
  Step 1 — Array geometry (ENU + MCMF positions, CSV, layout plots)
  Step 2 — Array factor |AF(l,m)|² in dB
  Step 3 — Beam metrics (Ω_B, D, G, A_eff, sidelobe level)
  Step 4 — Sensitivity model + detectability + CSV + all required plots
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
CSV_TARGETS = (
    "/home/mahshad/Documents/Thesis/"
    "Step4_sensitivity&confusion/revise_the_table/ALL.csv"
)
OUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
PLOT_DIR  = os.path.join(OUT_DIR, "plots")
CSV_DIR   = os.path.join(OUT_DIR, "csv")
for d in (PLOT_DIR, CSV_DIR):
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS & GLOBAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
C       = 3e8          # speed of light [m/s]
KB      = 1.38e-23     # Boltzmann constant [J/K]
JY      = 1e-26        # 1 Jansky [W m⁻² Hz⁻¹]
N_POL   = 2            # dual polarisation
NSIGMA  = 5            # detection threshold

# Array hardware
D_SPACE    = 5.15      # element spacing [m]
A_EFF_ELE  = 6.28      # effective area per element [m²]
ETA        = 0.7       # aperture efficiency

# Tsiolkovsky crater
PHI_DEG    = -20.38
LAM_DEG    = 128.97
PHI        = np.radians(PHI_DEG)
LAM        = np.radians(LAM_DEG)

# Configurations
CORE_SIZES  = [32, 128]
OUT_NSIDE   = {32: 4, 128: 16}        # outrigger sub-array side
OUT_DIST_KM = [0.1, 0.25, 0.5, 0.75, 1.0, 5.0]

# Frequency sub-bands
SUBBANDS    = [(1, 5), (5, 10), (10, 20), (20, 40)]   # MHz lo, hi
BAND_LABELS = ["1-5", "5-10", "10-20", "20-40"]
BAND_CTR    = [3.0,   7.5,    15.0,    30.0]          # representative centres

# Bandwidths to test [MHz]
BANDWIDTHS  = [1, 5, 10]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: sky temperature
# ─────────────────────────────────────────────────────────────────────────────
def T_sky(nu_MHz):
    """Galactic sky temperature [K]: T = 180 × (ν/180 MHz)^{−2.6}."""
    return 180.0 * (nu_MHz / 180.0) ** (-2.6)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ARRAY GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
def rotation_matrix():
    """ENU → MCMF rotation matrix (3 × 3)."""
    eE = np.array([-np.sin(LAM),  np.cos(LAM), 0.0])
    eN = np.array([-np.sin(PHI)*np.cos(LAM), -np.sin(PHI)*np.sin(LAM),  np.cos(PHI)])
    eU = np.array([ np.cos(PHI)*np.cos(LAM),  np.cos(PHI)*np.sin(LAM),  np.sin(PHI)])
    return np.column_stack([eE, eN, eU])   # columns are basis vectors


def rect_array_enu(n_side, d, cx=0.0, cy=0.0):
    """
    Return ENU positions of an n_side × n_side rectangular array
    centred at (cx, cy, 0) with element spacing d. Shape: (n², 3).
    """
    offsets = (np.arange(n_side) - (n_side - 1) / 2.0) * d
    xs, ys = np.meshgrid(offsets, offsets)
    pos = np.column_stack([xs.ravel() + cx, ys.ravel() + cy,
                           np.zeros(n_side ** 2)])
    return pos


def build_config(core_n, out_dist_km):
    """
    Build full array ENU positions: core + 4 outrigger sub-arrays.
    Four outrigger centres are placed at angles 0°, 90°, 180°, 270°.
    Returns (pos_enu [N×3], meta dict).
    """
    out_n   = OUT_NSIDE[core_n]
    dist_m  = out_dist_km * 1e3
    angles  = np.linspace(0, 2 * np.pi, 4, endpoint=False)

    parts = [rect_array_enu(core_n, D_SPACE)]
    out_centres = []
    for ang in angles:
        cx = dist_m * np.cos(ang)
        cy = dist_m * np.sin(ang)
        out_centres.append((cx, cy))
        parts.append(rect_array_enu(out_n, D_SPACE, cx, cy))

    pos_enu = np.vstack(parts)
    meta = {
        "core_n":       core_n,
        "out_dist_km":  out_dist_km,
        "out_n":        out_n,
        "N_core":       core_n ** 2,
        "N_out_each":   out_n ** 2,
        "N_total":      len(pos_enu),
        "out_centres":  out_centres,
    }
    return pos_enu, meta


def config_name(core_n, dist_km):
    return f"core{core_n}x{core_n}_out{dist_km}km"


def step1_geometry(R):
    """Run Step 1: build all configurations, save CSVs, save layout plots."""
    print("\n── STEP 1: Array Geometry ──")
    configs = {}

    for cn in CORE_SIZES:
        for dk in OUT_DIST_KM:
            name = config_name(cn, dk)
            pos_enu, meta = build_config(cn, dk)
            pos_mcmf = (R @ pos_enu.T).T

            # ── CSV ──
            df_pos = pd.DataFrame({
                "element":  np.arange(len(pos_enu)),
                "E_m":      pos_enu[:, 0],
                "N_m":      pos_enu[:, 1],
                "U_m":      pos_enu[:, 2],
                "X_mcmf":   pos_mcmf[:, 0],
                "Y_mcmf":   pos_mcmf[:, 1],
                "Z_mcmf":   pos_mcmf[:, 2],
            })
            df_pos.to_csv(os.path.join(CSV_DIR, f"positions_{name}.csv"), index=False)

            # ── layout plot ──
            fig, ax = plt.subplots(figsize=(7, 7))
            Nc = meta["N_core"]
            ax.scatter(pos_enu[:Nc, 0], pos_enu[:Nc, 1],
                       s=0.5, c="steelblue", alpha=0.7, label=f"Core {cn}×{cn}")
            ax.scatter(pos_enu[Nc:, 0], pos_enu[Nc:, 1],
                       s=0.5, c="tomato",    alpha=0.7, label=f"Outriggers {OUT_NSIDE[cn]}×{OUT_NSIDE[cn]}")
            ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
            ax.set_title(f"Array layout: {name}\nN_total = {meta['N_total']}")
            ax.legend(markerscale=6, fontsize=8); ax.set_aspect("equal")
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"layout_{name}.png"), dpi=100)
            plt.close()

            configs[name] = {"pos_enu": pos_enu, "meta": meta}
            print(f"  {name}: {meta['N_total']} elements")

    return configs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ARRAY FACTOR
# Using separable decomposition: each rectangular sub-array → outer product
# of two 1-D array factors, then multiply by a phase-offset term.
# ─────────────────────────────────────────────────────────────────────────────
def af_1d(offsets, k, grid):
    """
    1-D array factor for elements at `offsets` metres, evaluated at
    direction-cosine `grid` (1-D array).  Returns complex 1-D array.
    """
    # phase[elem, grid_pt] = k * offset[elem] * grid[grid_pt]
    return np.sum(np.exp(1j * k * np.outer(offsets, grid)), axis=0)


def af_rect_subarray(n_side, d, cx, cy, k, l_grid, m_grid):
    """
    2-D array factor for one rectangular sub-array centred at (cx, cy).
    Uses separable decomposition → O(n × N_grid) instead of O(n² × N_grid²).
    Returns complex (N_m, N_l) array.
    """
    offsets = (np.arange(n_side) - (n_side - 1) / 2.0) * d
    AFx = af_1d(offsets, k, l_grid)   # (N_l,)
    AFy = af_1d(offsets, k, m_grid)   # (N_m,)
    L, M = np.meshgrid(l_grid, m_grid)
    AF2D = np.outer(AFy, AFx) * np.exp(1j * k * (cx * L + cy * M))
    return AF2D


def compute_af(pos_enu, meta, freq_MHz, n_grid=512):
    """
    Compute |AF(l,m)|² / |AF_max|² in dB for the full array.
    Returns l_arr, m_arr (each length n_grid), AF_dB (n_grid × n_grid).
    """
    k = 2 * np.pi * freq_MHz * 1e6 / C

    l_arr = np.linspace(-1, 1, n_grid)
    m_arr = np.linspace(-1, 1, n_grid)

    cn     = meta["core_n"]
    out_n  = meta["out_n"]
    out_c  = meta["out_centres"]

    # Core
    AF = af_rect_subarray(cn, D_SPACE, 0.0, 0.0, k, l_arr, m_arr)

    # 4 outriggers
    for (cx, cy) in out_c:
        AF += af_rect_subarray(out_n, D_SPACE, cx, cy, k, l_arr, m_arr)

    B     = np.abs(AF) ** 2
    B_dB  = 10 * np.log10(B / B.max() + 1e-20)

    # Mask outside visible hemisphere
    L, M = np.meshgrid(l_arr, m_arr)
    B_dB[L**2 + M**2 > 1] = np.nan

    return l_arr, m_arr, B_dB, B / B.max()


def step2_array_factor(configs):
    """Run Step 2: compute and plot |AF|² for every config × sub-band."""
    print("\n── STEP 2: Array Factor ──")
    af_store = {}

    for name, cfg in configs.items():
        af_store[name] = {}
        for bl, fc in zip(BAND_LABELS, BAND_CTR):
            print(f"  AF: {name}  band {bl} MHz (centre {fc} MHz)")
            l, m, AF_dB, B_norm = compute_af(cfg["pos_enu"], cfg["meta"], fc,
                                              n_grid=512)
            af_store[name][bl] = (l, m, AF_dB, B_norm)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.pcolormesh(l, m, AF_dB, vmin=-60, vmax=0,
                               cmap="inferno", shading="auto")
            plt.colorbar(im, ax=ax, label="|AF|² [dB]")
            ax.set_xlabel("l  (East dir. cosine)")
            ax.set_ylabel("m  (North dir. cosine)")
            ax.set_title(f"|AF|²  {name}\n{bl} MHz")
            ax.set_aspect("equal")
            plt.tight_layout()
            safe_bl = bl.replace("-", "_")
            plt.savefig(os.path.join(PLOT_DIR,
                        f"AF_{name}_band{safe_bl}.png"), dpi=100)
            plt.close()

    return af_store


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BEAM METRICS
# ─────────────────────────────────────────────────────────────────────────────
def beam_metrics(B_norm, l_arr, m_arr, freq_MHz):
    """
    From normalised power beam B_norm (N_m × N_l):
      Ω_B   = ∫ B(s) dΩ   [sr]
      D_peak = 4π / Ω_B
      G_peak = η × D_peak
      A_eff  = (λ² / 4π) × G_peak
      MSL    = max sidelobe [dB]
    """
    dl = l_arr[1] - l_arr[0]
    dm = m_arr[1] - m_arr[0]
    L, M = np.meshgrid(l_arr, m_arr)
    inside = L**2 + M**2 <= 1.0

    # dΩ = dl dm / √(1 − l² − m²)
    sqrt_term = np.sqrt(np.maximum(1 - L**2 - M**2, 1e-12))
    dOmega = np.where(inside, dl * dm / sqrt_term, 0.0)

    Omega_B  = float(np.sum(B_norm * dOmega))
    D_peak   = 4 * np.pi / Omega_B
    G_peak   = ETA * D_peak
    wl       = C / (freq_MHz * 1e6)
    A_eff    = (wl**2 / (4 * np.pi)) * G_peak

    # MSL: peak outside main lobe (> −3 dB from peak defined as B_norm > 0.5)
    main_lobe  = inside & (B_norm >= 0.5)
    sidelobe_r = inside & ~main_lobe
    if sidelobe_r.any():
        MSL_dB = float(10 * np.log10(B_norm[sidelobe_r].max() + 1e-20))
    else:
        MSL_dB = -999.0

    return dict(Omega_B=Omega_B, D_peak=D_peak, G_peak=G_peak,
                A_eff=A_eff, MSL_dB=MSL_dB, wavelength=wl)


def step3_beam_metrics(configs, af_store):
    """Run Step 3: compute metrics, generate all required beam plots."""
    print("\n── STEP 3: Beam Metrics ──")
    metrics = {}   # metrics[name][band_label] = dict

    for name, cfg in configs.items():
        metrics[name] = {}
        for bl, fc in zip(BAND_LABELS, BAND_CTR):
            l, m, _, B_norm = af_store[name][bl]
            mtr = beam_metrics(B_norm, l, m, fc)
            metrics[name][bl] = mtr
            print(f"  {name}  {bl} MHz → "
                  f"Ω_B={mtr['Omega_B']:.3e} sr  "
                  f"G={mtr['G_peak']:.2e}  "
                  f"A_eff={mtr['A_eff']:.2e} m²  "
                  f"MSL={mtr['MSL_dB']:.1f} dB")

    # ── Plot 1: 2-D beam pattern for 1 km and 5 km outrigger configs ──
    for cn in CORE_SIZES:
        for dk in [1.0, 5.0]:
            name = config_name(cn, dk)
            for bl, fc in zip(BAND_LABELS, BAND_CTR):
                l, m, AF_dB, B_norm = af_store[name][bl]
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))

                im0 = axes[0].pcolormesh(l, m, AF_dB, vmin=-60, vmax=0,
                                         cmap="inferno", shading="auto")
                plt.colorbar(im0, ax=axes[0], label="|AF|² [dB]")
                axes[0].set_title(f"2-D beam pattern\n{name}  {bl} MHz")
                axes[0].set_xlabel("l"); axes[0].set_ylabel("m")
                axes[0].set_aspect("equal")

                # 1-D cut through m = 0
                mid = len(m) // 2
                cut = 10 * np.log10(B_norm[mid, :] + 1e-20)
                axes[1].plot(l, cut)
                axes[1].axhline(-3, color="r", ls="--", lw=0.8, label="−3 dB")
                axes[1].set_xlabel("l"); axes[1].set_ylabel("Power [dB]")
                axes[1].set_title(f"1-D beam cut (m = 0)\n{name}  {bl} MHz")
                axes[1].set_ylim(-70, 5); axes[1].grid(True, alpha=0.3)
                axes[1].legend(fontsize=8)

                plt.tight_layout()
                safe_bl = bl.replace("-", "_")
                plt.savefig(os.path.join(PLOT_DIR,
                            f"beam2D1D_{name}_band{safe_bl}.png"), dpi=100)
                plt.close()

    # ── Plot 3: Beam solid angle vs frequency ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for cn in CORE_SIZES:
        for dk in [0.1, 1.0, 5.0]:
            name = config_name(cn, dk)
            omegas = [metrics[name][bl]["Omega_B"] for bl in BAND_LABELS]
            ax.semilogy(BAND_CTR, omegas, marker="o",
                        label=f"{cn}×{cn}, {dk} km")
    ax.set_xlabel("Frequency [MHz]"); ax.set_ylabel("Beam solid angle [sr]")
    ax.set_title("Beam Solid Angle vs Frequency")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "beam_solid_angle_vs_freq.png"), dpi=100)
    plt.close()

    # ── Plot 4: Peak gain vs frequency ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for cn in CORE_SIZES:
        for dk in [0.1, 1.0, 5.0]:
            name = config_name(cn, dk)
            gains = [metrics[name][bl]["G_peak"] for bl in BAND_LABELS]
            ax.semilogy(BAND_CTR, gains, marker="o",
                        label=f"{cn}×{cn}, {dk} km")
    ax.set_xlabel("Frequency [MHz]"); ax.set_ylabel("Peak Gain")
    ax.set_title("Peak Gain vs Frequency")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "peak_gain_vs_freq.png"), dpi=100)
    plt.close()

    # ── Plot 5: Effective area vs N_antennas ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for bl, fc in zip(BAND_LABELS, BAND_CTR):
        Ns, Aeffs = [], []
        for cn in CORE_SIZES:
            for dk in OUT_DIST_KM:
                name = config_name(cn, dk)
                Ns.append(configs[name]["meta"]["N_total"])
                Aeffs.append(metrics[name][bl]["A_eff"])
        idx = np.argsort(Ns)
        ax.loglog(np.array(Ns)[idx], np.array(Aeffs)[idx],
                  marker="o", ms=4, label=f"{bl} MHz")
    ax.set_xlabel("Total number of antennas N")
    ax.set_ylabel("Effective area A_eff [m²]")
    ax.set_title("Effective Area vs Number of Antennas")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "Aeff_vs_N.png"), dpi=100)
    plt.close()

    # ── Plot 6: Max sidelobe level vs baseline length ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for bl, fc in zip(BAND_LABELS, BAND_CTR):
        for cn in CORE_SIZES:
            baselines, msls = [], []
            for dk in OUT_DIST_KM:
                name  = config_name(cn, dk)
                B_max = 2 * dk * 1e3
                baselines.append(B_max)
                msls.append(metrics[name][bl]["MSL_dB"])
            ax.plot(baselines, msls, marker="o", ms=4,
                    label=f"{cn}×{cn} {bl} MHz")
    ax.set_xlabel("Approximate max baseline [m]")
    ax.set_ylabel("Max sidelobe level [dB]")
    ax.set_title("Max Sidelobe Level vs Baseline Length")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "sidelobe_vs_baseline.png"), dpi=100)
    plt.close()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SENSITIVITY MODEL
# ─────────────────────────────────────────────────────────────────────────────
def sefd_element(nu_MHz):
    """SEFD per element [Jy]: 2 k_B T_sys / A_eff,ele."""
    return 2 * KB * T_sky(nu_MHz) / A_EFF_ELE / JY


def sigma_thermal_Jy(N, nu_MHz, bw_MHz, t_hr):
    """
    Interferometric radiometer equation [Jy].
    σ_thermal = SEFD_elem / √(n_pol × N × (N−1) × Δν × t)
    """
    sefd = sefd_element(nu_MHz)
    t_s  = t_hr * 3600.0
    dnu  = bw_MHz * 1e6
    return sefd / np.sqrt(N_POL * N * (N - 1) * dnu * t_s)


def confusion_limit_Jy(nu_MHz, B_max_m):
    """
    Confusion limit [Jy]:
      σ_c = 0.2 mJy × (ν/74 MHz)^{0.8} × (θ_HPBW / 4 arcmin)
      θ_HPBW = λ / B_max  [rad]
    """
    wl = C / (nu_MHz * 1e6)
    theta_rad    = wl / B_max_m
    theta_arcmin = np.degrees(theta_rad) * 60.0
    return 0.2e-3 * (nu_MHz / 74.0) ** 0.8 * (theta_arcmin / 4.0)   # Jy


def required_t_hours(S_mJy, N, nu_MHz, bw_MHz, B_max_m):
    """
    Integration time [h] for a 5σ detection of a source with flux S_mJy.
    Solves: 5 × √(σ_th(t)² + σ_c²) = S  →  t = (A / √(rhs))²
    Returns np.inf when confusion-limited.
    """
    S_Jy   = S_mJy * 1e-3
    sc     = confusion_limit_Jy(nu_MHz, B_max_m)

    if NSIGMA * sc >= S_Jy:
        return np.inf           # confusion-limited for any t

    # σ_th(t) = A / √(t)  where A is sigma at 1 h
    A   = sigma_thermal_Jy(N, nu_MHz, bw_MHz, 1.0)
    rhs = (S_Jy / NSIGMA) ** 2 - sc ** 2
    if rhs <= 0:
        return np.inf
    return (A / np.sqrt(rhs)) ** 2


def feasibility(t_h):
    if t_h < 100:   return "feasible"
    if t_h < 1000:  return "aspirational"
    return "impossible"


def max_baseline(core_n, dk):
    """Approximate maximum baseline [m]."""
    return 2 * dk * 1e3 + core_n * D_SPACE


def load_targets():
    df = pd.read_csv(CSV_TARGETS)
    df = df[df["flux_mJy"] > 0].dropna(subset=["flux_mJy", "frequency_MHz"])
    # Keep only targets whose predicted emission frequency is in ALO band
    df = df[(df["frequency_MHz"] >= 1) & (df["frequency_MHz"] <= 40)]
    print(f"  Loaded {len(df)} targets in 1–40 MHz band")
    return df


def step4_sensitivity(configs, targets):
    """Run Step 4: sensitivity model, detectability, CSV, plots."""
    print("\n── STEP 4: Sensitivity Model ──")

    rows = []
    for name, cfg in configs.items():
        meta  = cfg["meta"]
        cn    = meta["core_n"]
        dk    = meta["out_dist_km"]
        N     = meta["N_total"]
        B_max = max_baseline(cn, dk)

        for bl, (f_lo, f_hi), fc in zip(BAND_LABELS, SUBBANDS, BAND_CTR):
            for bw in BANDWIDTHS:
                eff_bw = min(bw, f_hi - f_lo)
                sc_Jy  = confusion_limit_Jy(fc, B_max)
                st_Jy  = sigma_thermal_Jy(N, fc, eff_bw, 1.0)
                stot   = np.sqrt(st_Jy**2 + sc_Jy**2)

                band_targets = targets[
                    (targets["frequency_MHz"] >= f_lo) &
                    (targets["frequency_MHz"] <  f_hi)
                ]

                detected = []
                for _, row in band_targets.iterrows():
                    t_h = required_t_hours(row["flux_mJy"], N, fc, eff_bw, B_max)
                    fc_cls = feasibility(t_h)
                    if fc_cls == "feasible":
                        detected.append(row["Name"])
                    rows.append({
                        "config_name":        name,
                        "main_array_size":    f"{cn}×{cn}",
                        "outrigger_dist_km":  dk,
                        "N_elements":         N,
                        "bandwidth_MHz":      eff_bw,
                        "frequency_band":     bl,
                        "freq_centre_MHz":    fc,
                        "thermal_sens_Jy":    st_Jy,
                        "confusion_Jy":       sc_Jy,
                        "total_sens_Jy":      stot,
                        "target_name":        row["Name"],
                        "target_flux_mJy":    row["flux_mJy"],
                        "target_freq_MHz":    row["frequency_MHz"],
                        "required_t_h":       t_h if np.isfinite(t_h) else 1e9,
                        "feasibility":        fc_cls,
                        "n_detected_config":  0,          # filled below
                        "detected_exoplanets": ";".join(detected),
                    })

    df_res = pd.DataFrame(rows)

    # fill n_detected_config (unique feasible targets per config)
    feas = df_res[df_res["feasibility"] == "feasible"]
    n_det = feas.groupby("config_name")["target_name"].nunique().rename("_nd")
    df_res = df_res.join(n_det, on="config_name")
    df_res["n_detected_config"] = df_res["_nd"].fillna(0).astype(int)
    df_res.drop(columns=["_nd"], inplace=True)

    out_csv = os.path.join(CSV_DIR, "sensitivity_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}  ({len(df_res)} rows)")

    _sensitivity_plots(df_res, configs, targets)
    return df_res


def _sensitivity_plots(df_res, configs, targets):
    """Generate all 8 required sensitivity/detection plots."""
    t_arr = np.logspace(-1, 4, 300)   # hours

    rep_cfgs = [
        ("core32x32_out0.1km",  "32×32, 0.1 km"),
        ("core32x32_out1.0km",  "32×32, 1 km"),
        ("core128x128_out1.0km","128×128, 1 km"),
        ("core128x128_out5.0km","128×128, 5 km"),
    ]

    # ── Plot 1: Thermal RMS sensitivity vs integration time ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, lbl in rep_cfgs:
        if name not in configs:
            continue
        N   = configs[name]["meta"]["N_total"]
        fc  = 15.0; bw = 5.0
        A   = sigma_thermal_Jy(N, fc, bw, 1.0)
        ax.loglog(t_arr, A / np.sqrt(t_arr) * 1e3, label=lbl)
    ax.set_xlabel("Integration time [h]")
    ax.set_ylabel("Thermal RMS sensitivity [mJy]")
    ax.set_title("Thermal RMS Sensitivity vs Integration Time\n(10–20 MHz, Δν=5 MHz)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "s1_thermal_sens_vs_time.png"), dpi=100)
    plt.close()

    # ── Plot 2: Total sensitivity vs integration time ──
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, lbl in rep_cfgs:
        if name not in configs:
            continue
        meta  = configs[name]["meta"]
        N     = meta["N_total"]
        cn    = meta["core_n"]
        dk    = meta["out_dist_km"]
        fc    = 15.0; bw = 5.0
        Bm    = max_baseline(cn, dk)
        sc    = confusion_limit_Jy(fc, Bm)
        A     = sigma_thermal_Jy(N, fc, bw, 1.0)
        st    = A / np.sqrt(t_arr)
        stot  = np.sqrt(st**2 + sc**2)
        ax.loglog(t_arr, stot * 1e3, label=lbl)
    ax.set_xlabel("Integration time [h]")
    ax.set_ylabel("Total RMS sensitivity [mJy]")
    ax.set_title("Total Sensitivity vs Integration Time\n(10–20 MHz, Δν=5 MHz)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "s2_total_sens_vs_time.png"), dpi=100)
    plt.close()

    # ── Plot 3: Confusion limit vs frequency ──
    nu_plot = np.linspace(1, 40, 300)
    fig, ax = plt.subplots(figsize=(9, 5))
    for cn in CORE_SIZES:
        for dk in [0.1, 1.0, 5.0]:
            Bm  = max_baseline(cn, dk)
            sc  = [confusion_limit_Jy(f, Bm) * 1e3 for f in nu_plot]
            ax.semilogy(nu_plot, sc, label=f"{cn}×{cn}, {dk} km")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Confusion limit [mJy]")
    ax.set_title("Confusion Limit vs Frequency")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "s3_confusion_vs_freq.png"), dpi=100)
    plt.close()

    # ── Plot 4: Confusion limit vs maximum baseline ──
    B_plot = np.logspace(2, 5, 300)
    fig, ax = plt.subplots(figsize=(9, 5))
    for fc, bl in zip(BAND_CTR, BAND_LABELS):
        sc = [confusion_limit_Jy(fc, B) * 1e3 for B in B_plot]
        ax.loglog(B_plot, sc, label=f"{bl} MHz")
    ax.set_xlabel("Maximum baseline [m]")
    ax.set_ylabel("Confusion limit [mJy]")
    ax.set_title("Confusion Limit vs Maximum Baseline")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "s4_confusion_vs_baseline.png"), dpi=100)
    plt.close()

    # ── Plot 5: Required integration time per target (top 30 feasible) ──
    feas = df_res[df_res["feasibility"] == "feasible"].copy()
    if len(feas) > 0:
        top = (feas.drop_duplicates("target_name")
                   .nsmallest(30, "required_t_h"))
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top["target_name"], top["required_t_h"], color="steelblue")
        ax.axvline(100, color="r", ls="--", lw=1, label="100 h")
        ax.set_xlabel("Required integration time [h]")
        ax.set_title("Top 30 feasible targets (shortest integration time)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "s5_t_int_per_target.png"), dpi=100)
        plt.close()

    # ── Plot 6: N_detected vs configuration ──
    grp = (df_res[df_res["feasibility"] == "feasible"]
           .groupby("config_name")["target_name"].nunique()
           .sort_values(ascending=False).reset_index())
    grp.columns = ["config_name", "n_det"]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(range(len(grp)), grp["n_det"], color="teal")
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(grp["config_name"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Unique feasible detections")
    ax.set_title("Number of Detected Exoplanets vs Configuration")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "s6_ndet_vs_config.png"), dpi=100)
    plt.close()

    # ── Plot 7: N_detected vs bandwidth ──
    grp_bw = (df_res[df_res["feasibility"] == "feasible"]
              .groupby("bandwidth_MHz")["target_name"].nunique()
              .reset_index())
    grp_bw.columns = ["bw", "n_det"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grp_bw["bw"].astype(str), grp_bw["n_det"], color="darkorange")
    ax.set_xlabel("Bandwidth [MHz]"); ax.set_ylabel("Unique feasible detections")
    ax.set_title("Number of Detected Exoplanets vs Bandwidth")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "s7_ndet_vs_bandwidth.png"), dpi=100)
    plt.close()

    # ── Plot 8: Predicted flux vs detection threshold ──
    # Use 128×128 + 1 km, 10–20 MHz, 5 MHz BW, 100 h as representative
    name_rep = "core128x128_out1.0km"
    if name_rep in configs:
        meta  = configs[name_rep]["meta"]
        N_rep = meta["N_total"]
        Bm    = max_baseline(128, 1.0)
        fc    = 15.0; bw = 5.0; t_h = 100.0

        sc  = confusion_limit_Jy(fc, Bm)
        st  = sigma_thermal_Jy(N_rep, fc, bw, t_h)
        thr = NSIGMA * np.sqrt(st**2 + sc**2) * 1e3   # mJy

        band_t = targets[(targets["frequency_MHz"] >= 10) &
                         (targets["frequency_MHz"] <  20)].copy()
        flx_sorted = np.sort(band_t["flux_mJy"].values)[::-1]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogy(range(len(flx_sorted)), flx_sorted, "o", ms=3,
                    color="steelblue", label="Target flux")
        ax.axhline(thr, color="r", ls="--",
                   label=f"5σ threshold = {thr:.3f} mJy")
        ax.set_xlabel("Target index (sorted by flux)")
        ax.set_ylabel("Predicted flux [mJy]")
        ax.set_title(f"Predicted Flux vs Detection Threshold\n"
                     f"({name_rep}, 10–20 MHz, 5 MHz BW, 100 h)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "s8_flux_vs_threshold.png"), dpi=100)
        plt.close()

    print("  All 8 sensitivity plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL ANALYSIS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def final_summary(df_res, configs):
    print("\n═══════════════════════════════════════════════════════════")
    print("  FINAL ANALYSIS SUMMARY")
    print("═══════════════════════════════════════════════════════════")

    # 1. Outrigger distance effect
    print("\n1. Effect of outrigger distance on sensitivity & confusion noise")
    grp = (df_res.groupby(["main_array_size", "outrigger_dist_km"])
           .agg(mean_thermal_Jy=("thermal_sens_Jy", "mean"),
                mean_confusion_Jy=("confusion_Jy", "mean"),
                n_feasible=("feasibility",
                            lambda x: (x == "feasible").sum()))
           .reset_index())
    print(grp.to_string(index=False))

    # 2. 1 km vs 5 km beam quality
    print("\n2. Feasible detections: 1 km vs 5 km outrigger")
    for cn in CORE_SIZES:
        for dk in [1.0, 5.0]:
            name = config_name(cn, dk)
            sub  = df_res[df_res["config_name"] == name]
            nd   = sub[sub["feasibility"] == "feasible"]["target_name"].nunique()
            print(f"   {name:35s}  {nd:3d} unique feasible targets")

    # 3. Best overall configuration
    det = (df_res[df_res["feasibility"] == "feasible"]
           .groupby("config_name")["target_name"].nunique()
           .reset_index())
    det.columns = ["config", "n"]
    if len(det):
        best = det.loc[det["n"].idxmax()]
        print(f"\n3. Best configuration: {best['config']}  →  {best['n']} detectable targets")

    # 4. Bandwidth effect
    print("\n4. Unique detections vs bandwidth")
    grp_bw = (df_res[df_res["feasibility"] == "feasible"]
              .groupby("bandwidth_MHz")["target_name"].nunique()
              .reset_index())
    print(grp_bw.to_string(index=False))

    # 5. Which bands are productive
    print("\n5. Unique detections vs frequency band")
    grp_b = (df_res[df_res["feasibility"] == "feasible"]
             .groupby("frequency_band")["target_name"].nunique()
             .reset_index())
    print(grp_b.to_string(index=False))

    print("\n6. For best compromise see plots s6_ndet_vs_config.png "
          "and sidelobe_vs_baseline.png")
    print("═══════════════════════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("ALO Array Modeling — Starting")
    print(f"Output directory: {OUT_DIR}\n")

    R = rotation_matrix()
    print(f"Rotation matrix (ENU → MCMF):\n{R}\n")

    # Step 1
    configs = step1_geometry(R)

    # Step 2
    af_store = step2_array_factor(configs)

    # Step 3
    metrics = step3_beam_metrics(configs, af_store)

    # Load targets
    print("\n── Loading target catalogue ──")
    targets = load_targets()

    # Step 4
    df_res = step4_sensitivity(configs, targets)

    # Summary
    final_summary(df_res, configs)

    print(f"Done. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
