"""
ALO Geometry Trade-off Study
=============================
Compares 4 outrigger layout geometries:
  Ring | Y-shape | Random | Line
across 4 core + distance variants:
  32×32 core + 4×4 outriggers  @ 1 km / 5 km
  128×128 core + 16×16 outriggers @ 1 km / 5 km
= 16 configurations total

For every configuration computes:
  - Array layout (ENU positions, plot)
  - Beam pattern |AF(l,m)|² at all 4 ALO sub-bands
  - Beam quality metrics (Ω_B, HPBW, MSL, directivity, A_eff)
  - Interferometric sensitivity + confusion noise
  - Detectability against the exoplanet target catalogue

Final outputs:
  - Beam quality comparison plot (all 16 configs)
  - Sensitivity comparison plot  (all 16 configs)
  - geometry_tradeoff_metrics.csv
  - geometry_tradeoff_sensitivity.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

# ── import shared infrastructure from the main script ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alo_array_modeling import (
    rect_array_enu, compute_af, beam_metrics,
    sigma_thermal_Jy, confusion_limit_Jy, required_t_hours,
    feasibility as classify_feasibility, load_targets,
    core_edge_m, A_EFF_ELE,
    D_SPACE, C, NSIGMA, ETA,
    PHI_DEG, LAM_DEG,
    BAND_LABELS, BAND_CTR, SUBBANDS, BANDWIDTHS,
    PLOT_DIR, CSV_DIR,
)

# ── output directories ─────────────────────────────────────────────────────────
GT_PLOT = os.path.join(PLOT_DIR, "geometry_tradeoff")
GT_CSV  = os.path.join(CSV_DIR,  "geometry_tradeoff")
os.makedirs(GT_PLOT, exist_ok=True)
os.makedirs(GT_CSV,  exist_ok=True)

# ── study parameters ───────────────────────────────────────────────────────────
CORE_OUT   = {32: 4, 128: 16}      # core_n → outrigger sub-array side
DISTANCES  = [1.0, 5.0]            # outrigger distance [km]
GEOMETRIES = ["ring", "y_shape", "random", "line"]
N_GRID     = 512                   # AF grid resolution
SEED       = 42                    # fixed seed for random layout
REF_FREQ   = 30.0                  # MHz  – reference for comparison metrics
REF_BW     = 20.0                  # MHz
REF_T_H    = 100.0                 # hours – reference integration time

GEOM_LABEL = {"ring": "Ring", "y_shape": "Y-shape",
               "random": "Random", "line": "Line"}
GEOM_COLOR = {"ring": "#2196F3", "y_shape": "#4CAF50",
               "random": "#FF9800", "line": "#E91E63"}
DIST_STYLE = {1.0: "-", 5.0: "--"}
CORE_MARKER= {32: "o", 128: "s"}

# ── UV coverage: 1 MHz channels per sub-band ──────────────────────────────────
SUBBAND_CH = {
    "1-5":   np.arange(1.0,  6.0, 1.0),    # 5 channels
    "5-10":  np.arange(5.0, 11.0, 1.0),    # 6 channels
    "10-20": np.arange(10.0, 21.0, 1.0),   # 11 channels
    "20-40": np.arange(20.0, 41.0, 2.0),   # 11 channels at 2 MHz step
}
SUBBAND_CH_COLOR = {"1-5":   "#E53935", "5-10": "#FB8C00",
                    "10-20": "#1E88E5", "20-40": "#8E24AA"}

# ── Contour levels (fraction of peak power, linear) ───────────────────────────
CONTOUR_LVL = [0.10, 0.30, 0.50]
CONTOUR_COL = ["cyan", "lime", "white"]

# ── Cross configuration parameters ────────────────────────────────────────────
# Irregular arm distances break the aliasing rings that equal-arm arrays produce.
# cross_ew: wider E-W baseline  →  better E-W angular resolution
# cross_ns: 90° rotation       →  wider N-S baseline, better for low-dec targets
CROSS_N       = 128
CROSS_ARMS_EW = {"N": 1.0, "S": 1.0, "E": 5.0, "W": 3.0}  # km
CROSS_ARMS_NS = {"N": 5.0, "S": 3.0, "E": 1.0, "W": 1.0}  # km
CROSS_GEOM_LABEL = {
    "cross_ew": (f"Cross E-W  (N={CROSS_ARMS_EW['N']}, S={CROSS_ARMS_EW['S']},"
                 f" E={CROSS_ARMS_EW['E']}, W={CROSS_ARMS_EW['W']} km)"),
    "cross_ns": (f"Cross N-S  (N={CROSS_ARMS_NS['N']}, S={CROSS_ARMS_NS['S']},"
                 f" E={CROSS_ARMS_NS['E']}, W={CROSS_ARMS_NS['W']} km)"),
}
CROSS_COLOR = {"cross_ew": "#009688", "cross_ns": "#FF5722"}
CROSS_PLOT  = os.path.join(GT_PLOT, "cross_config")
CROSS_CSV_D = os.path.join(GT_CSV,  "cross_config")


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def layout_ring(core_n, out_n, dist_km):
    """4 outrigger sub-arrays at 0°, 90°, 180°, 270° – ring symmetry."""
    d       = dist_km * 1e3
    angles  = np.linspace(0, 2*np.pi, 4, endpoint=False)
    centres = [(d*np.cos(a), d*np.sin(a)) for a in angles]
    parts   = [rect_array_enu(core_n, D_SPACE)]
    for cx, cy in centres:
        parts.append(rect_array_enu(out_n, D_SPACE, cx, cy))
    return np.vstack(parts), centres


def layout_y_shape(core_n, out_n, dist_km):
    """
    Y-shape: 3 arms at 90° (N), 210° (SW), 330° (SE).
    4 outriggers: N tip (d), SW tip (d), SE tip (d), SW mid (d/2).

    The 4th station is on the SW arm, NOT the N arm.  Placing it on the N arm
    (as was done previously) left SW and SE as exact left-right mirrors, giving
    the array an unintended N-S mirror symmetry and making |AF|² look identical
    to the ring pattern.  Moving it to SW means SW has two stations while SE
    has one, which breaks all mirror symmetry.  Only the fundamental centrosymmetry
    |AF(l,m)|² = |AF(-l,-m)|² remains — a mathematical property that holds for
    every array with uniform real weights and cannot be removed.
    """
    d       = dist_km * 1e3
    centres = [
        ( d    * np.cos(np.radians( 90)),  d    * np.sin(np.radians( 90))),  # N  tip
        ( d    * np.cos(np.radians(210)),  d    * np.sin(np.radians(210))),  # SW tip
        ( d    * np.cos(np.radians(330)),  d    * np.sin(np.radians(330))),  # SE tip
        ( d/2  * np.cos(np.radians(210)),  d/2  * np.sin(np.radians(210))),  # SW mid
    ]
    parts = [rect_array_enu(core_n, D_SPACE)]
    for cx, cy in centres:
        parts.append(rect_array_enu(out_n, D_SPACE, cx, cy))
    return np.vstack(parts), centres


def layout_random(core_n, out_n, dist_km, seed=SEED):
    """
    4 outrigger sub-arrays at random positions within radius dist_km.
    Rejection-sampled to maintain minimum separation of
    max(0.2·dist_km, 3·outrigger_footprint).
    Fixed seed for reproducibility.
    """
    rng     = np.random.default_rng(seed)
    d       = dist_km * 1e3
    min_sep = max(0.20 * d, 3 * out_n * D_SPACE)
    r_min   = max(0.25 * d, out_n * D_SPACE)     # keep outriggers off-core

    centres  = []
    attempts = 0
    while len(centres) < 4 and attempts < 20000:
        r     = rng.uniform(r_min, d)
        theta = rng.uniform(0, 2*np.pi)
        cx, cy = r*np.cos(theta), r*np.sin(theta)
        if all(np.hypot(cx - ex, cy - ey) >= min_sep for ex, ey in centres):
            centres.append((cx, cy))
        attempts += 1

    # Robust fallback: perturb ring positions if rejection fails
    if len(centres) < 4:
        base = np.linspace(0, 2*np.pi, 4, endpoint=False)
        rng2 = np.random.default_rng(seed + 99)
        centres = [
            (d * np.cos(a + rng2.uniform(-0.3, 0.3)),
             d * np.sin(a + rng2.uniform(-0.3, 0.3)))
            for a in base
        ]

    parts = [rect_array_enu(core_n, D_SPACE)]
    for cx, cy in centres:
        parts.append(rect_array_enu(out_n, D_SPACE, cx, cy))
    return np.vstack(parts), centres


def layout_line(core_n, out_n, dist_km):
    """
    Linear East-West baseline.
    4 outriggers at −d, −d/3, +d/3, +d along the East axis.
    Maximum distance from core centre = dist_km; max baseline = 2·dist_km.
    """
    d       = dist_km * 1e3
    centres = [(-d, 0.0), (-d/3, 0.0), (d/3, 0.0), (d, 0.0)]
    parts   = [rect_array_enu(core_n, D_SPACE)]
    for cx, cy in centres:
        parts.append(rect_array_enu(out_n, D_SPACE, cx, cy))
    return np.vstack(parts), centres


LAYOUT_FN = {
    "ring":    layout_ring,
    "y_shape": layout_y_shape,
    "random":  layout_random,
    "line":    layout_line,
}


def layout_cross_ew(core_n, out_n, dist_km=None):
    """
    Cross with irregular arm lengths — wider E-W baseline.
    Arms (km): N=1, S=1, E=5, W=3  (from CROSS_ARMS_EW).
    dist_km is ignored; distances are fixed to break equal-arm aliasing.
    Max E-W baseline = 8 km, N-S = 2 km.
    """
    arms    = CROSS_ARMS_EW
    centres = [
        (0.0,               arms["N"] * 1e3),   # North
        (0.0,              -arms["S"] * 1e3),   # South
        (arms["E"] * 1e3,   0.0),               # East
        (-arms["W"] * 1e3,  0.0),               # West
    ]
    parts = [rect_array_enu(core_n, D_SPACE)]
    for cx, cy in centres:
        parts.append(rect_array_enu(out_n, D_SPACE, cx, cy))
    return np.vstack(parts), centres


def layout_cross_ns(core_n, out_n, dist_km=None):
    """
    Cross rotated 90° — wider N-S baseline.
    Arms (km): N=5, S=3, E=1, W=1  (from CROSS_ARMS_NS).
    Same arm lengths as cross_ew, rotated so the long arms point N-S.
    Improves beam quality for low-declination exoplanet host stars.
    """
    arms    = CROSS_ARMS_NS
    centres = [
        (0.0,               arms["N"] * 1e3),
        (0.0,              -arms["S"] * 1e3),
        (arms["E"] * 1e3,   0.0),
        (-arms["W"] * 1e3,  0.0),
    ]
    parts = [rect_array_enu(core_n, D_SPACE)]
    for cx, cy in centres:
        parts.append(rect_array_enu(out_n, D_SPACE, cx, cy))
    return np.vstack(parts), centres


def cfg_name(geom, core_n, dist_km):
    return f"{geom}_core{core_n}x{core_n}_{dist_km}km"


def max_baseline_m(out_centres, core_n):
    """Maximum pairwise distance between outrigger centres + core half-extent."""
    pts       = np.array(out_centres)
    core_half = core_n * D_SPACE / 2
    if len(pts) < 2:
        return 2 * core_half
    max_d = max(
        np.linalg.norm(pts[i] - pts[j])
        for i in range(len(pts)) for j in range(i+1, len(pts))
    )
    return max_d + core_half


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BUILD ALL 16 CONFIGURATIONS + LAYOUT PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def build_all_configs():
    print("\n── STEP 1: Build configurations & layout plots ──")
    configs = {}

    # Composite layout overview: 4 rows (geometry) × 4 cols (core+dist)
    variant_labels = ["32×32\n1 km", "32×32\n5 km", "128×128\n1 km", "128×128\n5 km"]
    fig_ov, axes_ov = plt.subplots(4, 4, figsize=(16, 16))

    for gi, geom in enumerate(GEOMETRIES):
        vi = 0
        for core_n in [32, 128]:
            out_n = CORE_OUT[core_n]
            for dist_km in DISTANCES:
                name = cfg_name(geom, core_n, dist_km)
                pos_enu, centres = LAYOUT_FN[geom](core_n, out_n, dist_km)
                B_max = max_baseline_m(centres, core_n)
                N_total = len(pos_enu)
                N_core  = core_n**2

                meta = dict(core_n=core_n, out_n=out_n, out_centres=centres,
                            N_core=N_core, N_out_each=out_n**2, N_total=N_total,
                            dist_km=dist_km, geom=geom, B_max_m=B_max)
                configs[name] = dict(pos_enu=pos_enu, meta=meta)
                print(f"  {name}: {N_total} elements, B_max = {B_max/1e3:.2f} km")

                # Individual layout plot
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(pos_enu[:N_core, 0], pos_enu[:N_core, 1],
                           s=0.4, c="steelblue", alpha=0.7,
                           label=f"Core {core_n}×{core_n}")
                ax.scatter(pos_enu[N_core:, 0], pos_enu[N_core:, 1],
                           s=2.0, c="tomato", alpha=0.9,
                           label=f"Outrigger {out_n}×{out_n}")
                for cx, cy in centres:
                    ax.plot(cx, cy, "r+", ms=10, mew=1.5)
                ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
                ax.set_title(f"{GEOM_LABEL[geom]} — core {core_n}×{core_n}, {dist_km} km\n"
                             f"N = {N_total}, B_max = {B_max/1e3:.2f} km")
                ax.legend(markerscale=6, fontsize=8); ax.set_aspect("equal")
                ax.grid(True, alpha=0.25)
                plt.tight_layout()
                plt.savefig(os.path.join(GT_PLOT, f"layout_{name}.png"), dpi=100)
                plt.close()

                # Overview subplot
                ax_ov = axes_ov[gi, vi]
                ax_ov.scatter(pos_enu[:N_core, 0]/1e3, pos_enu[:N_core, 1]/1e3,
                              s=0.2, c="steelblue", alpha=0.6)
                ax_ov.scatter(pos_enu[N_core:, 0]/1e3, pos_enu[N_core:, 1]/1e3,
                              s=1.0, c="tomato", alpha=0.9)
                ax_ov.set_aspect("equal")
                ax_ov.set_title(f"{GEOM_LABEL[geom]}\n{variant_labels[vi]}", fontsize=8)
                ax_ov.tick_params(labelsize=6)
                if gi == 3: ax_ov.set_xlabel("East [km]", fontsize=7)
                if vi == 0: ax_ov.set_ylabel("North [km]", fontsize=7)
                vi += 1

    fig_ov.suptitle("ALO Geometry Trade-off: Array Layout Overview\n"
                    "(blue = core, red = outrigger sub-arrays)", fontsize=13)
    plt.tight_layout()
    fig_ov.savefig(os.path.join(GT_PLOT, "layout_overview.png"), dpi=120)
    plt.close(fig_ov)
    print("  Saved layout_overview.png")
    return configs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — ARRAY FACTOR FOR ALL CONFIGS × 4 BANDS
# ══════════════════════════════════════════════════════════════════════════════
def compute_all_af(configs):
    print("\n── STEP 2: Array Factor ──")
    af_store = {}

    for name, cfg in configs.items():
        af_store[name] = {}
        geom = cfg["meta"]["geom"]
        for bl, fc in zip(BAND_LABELS, BAND_CTR):
            print(f"  AF: {name}  {bl} MHz")
            l, m, AF_dB, B_norm = compute_af(cfg["pos_enu"], cfg["meta"], fc, N_GRID)
            af_store[name][bl] = (l, m, AF_dB, B_norm)

            # Individual AF plot
            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.pcolormesh(l, m, AF_dB, vmin=-60, vmax=0,
                               cmap="inferno", shading="auto")
            plt.colorbar(im, ax=ax, label="|AF|² [dB]")
            ax.set_xlabel("l"); ax.set_ylabel("m")
            ax.set_title(f"|AF|²  {name}\n{bl} MHz", fontsize=8)
            ax.set_aspect("equal")
            plt.tight_layout()
            safe = bl.replace("-", "_")
            plt.savefig(os.path.join(GT_PLOT, f"AF_{name}_band{safe}.png"), dpi=100)
            plt.close()

    # ── Comparative AF panel at REF_FREQ (30 MHz) for all 16 configs ──
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    vi_map = {(32,1.0):0, (32,5.0):1, (128,1.0):2, (128,5.0):3}
    ref_bl = "20-40"

    for gi, geom in enumerate(GEOMETRIES):
        for core_n in [32, 128]:
            for dist_km in DISTANCES:
                vi  = vi_map[(core_n, dist_km)]
                name = cfg_name(geom, core_n, dist_km)
                l, m, AF_dB, _ = af_store[name][ref_bl]
                ax = axes[gi, vi]
                ax.pcolormesh(l, m, AF_dB, vmin=-60, vmax=0,
                              cmap="inferno", shading="auto")
                ax.set_title(f"{GEOM_LABEL[geom]}\n{core_n}×{core_n} {dist_km}km",
                             fontsize=8)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=6)
                if gi == 3: ax.set_xlabel("l", fontsize=7)
                if vi == 0: ax.set_ylabel("m", fontsize=7)

    fig.suptitle(f"Array Factor |AF|² at {REF_FREQ} MHz — All 16 Configurations\n"
                 "(columns: 32×32@1km | 32×32@5km | 128×128@1km | 128×128@5km)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(GT_PLOT, "AF_overview_30MHz.png"), dpi=120)
    plt.close(fig)
    print("  Saved AF_overview_30MHz.png")
    return af_store


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — BEAM METRICS + BEAM QUALITY PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def compute_all_metrics(configs, af_store):
    print("\n── STEP 3: Beam Metrics ──")
    all_metrics = {}

    for name, cfg in configs.items():
        all_metrics[name] = {}
        for bl, fc in zip(BAND_LABELS, BAND_CTR):
            l, m, _, B_norm = af_store[name][bl]
            mtr = beam_metrics(B_norm, l, m, fc)
            # HPBW from beam solid angle (assuming approximately circular beam)
            hpbw_rad = 2 * np.sqrt(mtr["Omega_B"] / np.pi)
            mtr["HPBW_arcmin"] = np.degrees(hpbw_rad) * 60.0
            all_metrics[name][bl] = mtr

    # ── 2-D + 1-D beam pattern plots for representative configs ──
    ref_bl = "20-40"
    for geom in GEOMETRIES:
        for core_n in [32, 128]:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            for di, dist_km in enumerate(DISTANCES):
                name = cfg_name(geom, core_n, dist_km)
                l, m, AF_dB, B_norm = af_store[name][ref_bl]
                fc = BAND_CTR[BAND_LABELS.index(ref_bl)]

                ax2d = axes[di, 0]
                im = ax2d.pcolormesh(l, m, AF_dB, vmin=-60, vmax=0,
                                     cmap="inferno", shading="auto")
                plt.colorbar(im, ax=ax2d, label="|AF|² [dB]")
                ax2d.set_title(f"2-D beam — {dist_km} km outrigger", fontsize=9)
                ax2d.set_xlabel("l"); ax2d.set_ylabel("m")
                ax2d.set_aspect("equal")

                ax1d = axes[di, 1]
                mid  = N_GRID // 2
                cut  = 10 * np.log10(B_norm[mid, :] + 1e-20)
                ax1d.plot(l, cut, color=GEOM_COLOR[geom], lw=1.5)
                ax1d.axhline(-3,  color="red",  ls="--", lw=0.9, label="−3 dB")
                ax1d.axhline(-10, color="orange", ls=":", lw=0.9, label="−10 dB")
                mtr = all_metrics[name][ref_bl]
                ax1d.set_title(f"1-D cut (m=0) — {dist_km} km\n"
                               f"HPBW={mtr['HPBW_arcmin']:.1f}′  "
                               f"MSL={mtr['MSL_dB']:.1f} dB", fontsize=9)
                ax1d.set_xlabel("l"); ax1d.set_ylabel("Power [dB]")
                ax1d.set_ylim(-70, 5)
                ax1d.legend(fontsize=8); ax1d.grid(True, alpha=0.3)

            fig.suptitle(f"Beam Pattern — {GEOM_LABEL[geom]} | Core {core_n}×{core_n} | "
                         f"{ref_bl} MHz ({REF_FREQ} MHz)", fontsize=11)
            plt.tight_layout()
            fig.savefig(os.path.join(GT_PLOT,
                        f"beam_{geom}_core{core_n}x{core_n}_30MHz.png"), dpi=100)
            plt.close(fig)
    print("  Saved beam pattern plots")

    return all_metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
def compute_all_sensitivity(configs, targets):
    print("\n── STEP 4: Sensitivity ──")
    rows = []

    for name, cfg in configs.items():
        meta    = cfg["meta"]
        core_n  = meta["core_n"]
        dist_km = meta["dist_km"]
        N       = meta["N_total"]
        B_max   = meta["B_max_m"]
        geom    = meta["geom"]

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

                for _, trow in band_targets.iterrows():
                    t_h  = required_t_hours(trow["flux_mJy"], N, fc, eff_bw, B_max)
                    fc_s = classify_feasibility(t_h)
                    rows.append(dict(
                        config_name      = name,
                        geometry         = geom,
                        core_size        = f"{core_n}×{core_n}",
                        outrigger_dist_km= dist_km,
                        N_elements       = N,
                        max_baseline_m   = round(B_max, 1),
                        bandwidth_MHz    = eff_bw,
                        frequency_band   = bl,
                        freq_centre_MHz  = fc,
                        thermal_sens_Jy  = st_Jy,
                        confusion_Jy     = sc_Jy,
                        total_sens_Jy    = stot,
                        target_name      = trow["Name"],
                        target_flux_mJy  = trow["flux_mJy"],
                        target_freq_MHz  = trow["frequency_MHz"],
                        required_t_h     = t_h if np.isfinite(t_h) else 1e9,
                        feasibility      = fc_s,
                    ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(GT_CSV, "geometry_tradeoff_sensitivity.csv"), index=False)
    print(f"  Saved geometry_tradeoff_sensitivity.csv  ({len(df)} rows)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# COMPILE SUMMARY METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════
def compile_summary(configs, all_metrics, df_sens):
    print("\n── Compiling summary metrics ──")
    ref_bl = "20-40"   # 20–40 MHz band contains REF_FREQ=30 MHz
    rows   = []

    for name, cfg in configs.items():
        meta    = cfg["meta"]
        mtr     = all_metrics[name][ref_bl]
        N       = meta["N_total"]
        B_max   = meta["B_max_m"]

        # Sensitivity at reference conditions
        sc_ref = confusion_limit_Jy(REF_FREQ, B_max)
        st_ref = sigma_thermal_Jy(N, REF_FREQ, REF_BW, REF_T_H)
        stot_ref = np.sqrt(st_ref**2 + sc_ref**2)

        # Detection counts (best t_req across all bands and bandwidths per target)
        sub = df_sens[df_sens["config_name"] == name]
        best_t = sub.groupby("target_name")["required_t_h"].min()
        n_feas = (best_t < 100).sum()
        n_asp  = ((best_t >= 100) & (best_t < 1000)).sum()

        rows.append(dict(
            config_name         = name,
            geometry            = meta["geom"],
            geom_label          = GEOM_LABEL[meta["geom"]],
            core_size           = f"{meta['core_n']}×{meta['core_n']}",
            outrigger_dist_km   = meta["dist_km"],
            N_elements          = N,
            max_baseline_km     = round(B_max/1e3, 3),
            # beam quality at 30 MHz
            Omega_B_sr          = mtr["Omega_B"],
            HPBW_arcmin         = mtr["HPBW_arcmin"],
            MSL_dB              = mtr["MSL_dB"],
            D_peak              = mtr["D_peak"],
            G_peak              = mtr["G_peak"],
            A_eff_m2            = mtr["A_eff"],
            # sensitivity at 30 MHz, 20 MHz BW, 100 h
            thermal_sens_100h_mJy = st_ref * 1e3,
            confusion_mJy         = sc_ref * 1e3,
            total_sens_5sig_mJy   = NSIGMA * stot_ref * 1e3,
            # detection yield
            n_feasible          = n_feas,
            n_aspirational      = n_asp,
            n_detectable        = n_feas + n_asp,
        ))

    df_sum = pd.DataFrame(rows).sort_values(
        ["geometry", "core_size", "outrigger_dist_km"])
    df_sum.to_csv(os.path.join(GT_CSV, "geometry_tradeoff_metrics.csv"), index=False)
    print(f"  Saved geometry_tradeoff_metrics.csv  ({len(df_sum)} rows)")
    print(df_sum[["config_name","MSL_dB","HPBW_arcmin",
                  "total_sens_5sig_mJy","n_feasible","n_aspirational"]].to_string(index=False))
    return df_sum


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def comparison_plots(df_sum):
    print("\n── Generating comparison plots ──")

    # ── helpers ──────────────────────────────────────────────────────────────
    variants      = ["32×32\n1 km", "32×32\n5 km", "128×128\n1 km", "128×128\n5 km"]
    variant_keys  = [(32, 1.0), (32, 5.0), (128, 1.0), (128, 5.0)]
    x             = np.arange(len(variants))
    bar_w         = 0.18
    offsets       = np.linspace(-(1.5)*bar_w, 1.5*bar_w, 4)

    def get_val(geom, core_n, dist_km, col):
        row = df_sum[
            (df_sum["geometry"] == geom) &
            (df_sum["core_size"] == f"{core_n}×{core_n}") &
            (df_sum["outrigger_dist_km"] == dist_km)
        ]
        return float(row[col].values[0]) if len(row) else np.nan

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT A: BEAM QUALITY COMPARISON
    # Three sub-panels: (1) HPBW, (2) Max Sidelobe Level, (3) Beam Solid Angle
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=False)

    metrics_info = [
        ("HPBW_arcmin",   "HPBW [arcmin]",               "lower", True),
        ("MSL_dB",        "Max Sidelobe Level [dB]",       "lower", False),
        ("Omega_B_sr",    "Beam Solid Angle Ω_B [sr]",     "lower", True),
    ]

    for ax, (col, ylabel, best_dir, lower_better) in zip(axes, metrics_info):
        for gi, geom in enumerate(GEOMETRIES):
            vals = [get_val(geom, cn, dk, col) for cn, dk in variant_keys]
            bars = ax.bar(x + offsets[gi], vals, bar_w,
                          color=GEOM_COLOR[geom], label=GEOM_LABEL[geom],
                          edgecolor="white", lw=0.5, alpha=0.90)

            # value labels
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() * 1.02,
                            f"{v:.1f}", ha="center", va="bottom",
                            fontsize=5.5, rotation=90)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=9)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)

        # best-value annotation
        note = "lower = better" if lower_better else "less negative = better"
        if col == "MSL_dB":
            note = "more negative = better (lower sidelobes)"
        ax.set_title(f"{ylabel}  [{note}]  @ 30 MHz", fontsize=10)

    fig.suptitle("ALO Geometry Trade-off: Beam Quality Comparison\n"
                 "(all 16 configurations, 20–40 MHz band, reference freq = 30 MHz)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(GT_PLOT, "comparison_beam_quality.png"), dpi=150)
    plt.close(fig)
    print("  Saved comparison_beam_quality.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT B: SENSITIVITY COMPARISON
    # Three sub-panels: (1) 5σ total sensitivity, (2) Confusion limit,
    # (3) N detectable targets
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=False)

    sens_info = [
        ("total_sens_5sig_mJy",  "5σ Total Sensitivity [mJy]\n(30 MHz, 20 MHz BW, 100h)", True),
        ("confusion_mJy",        "Confusion Noise σ_c [mJy]\n(at 30 MHz)", True),
        ("n_detectable",         "N Detectable Targets\n(feasible + aspirational, best band)", False),
    ]

    for ax, (col, ylabel, log_scale) in zip(axes, sens_info):
        for gi, geom in enumerate(GEOMETRIES):
            vals = [get_val(geom, cn, dk, col) for cn, dk in variant_keys]
            bars = ax.bar(x + offsets[gi], vals, bar_w,
                          color=GEOM_COLOR[geom], label=GEOM_LABEL[geom],
                          edgecolor="white", lw=0.5, alpha=0.90)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() * 1.02,
                            f"{v:.2g}", ha="center", va="bottom",
                            fontsize=5.5, rotation=90)

        if log_scale and col != "n_detectable":
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, fontsize=9)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, axis="y", alpha=0.3, which="both")
        note = "lower = better" if col != "n_detectable" else "higher = better"
        ax.set_title(f"{ylabel.split(chr(10))[0]}  [{note}]", fontsize=10)

    fig.suptitle("ALO Geometry Trade-off: Sensitivity Comparison\n"
                 "(all 16 configurations — thermal sensitivity, confusion noise, detection yield)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(GT_PLOT, "comparison_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("  Saved comparison_sensitivity.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT C: COMBINED TRADE-OFF SCATTER — MSL vs Sensitivity (bubble = N_det)
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for col_idx, (core_n_sel, ax) in enumerate(zip([32, 128], axes)):
        sub = df_sum[df_sum["core_size"] == f"{core_n_sel}×{core_n_sel}"]
        for _, row in sub.iterrows():
            geom = row["geometry"]
            sz   = max(30, row["n_detectable"] * 15)
            sc = ax.scatter(
                row["MSL_dB"],
                np.log10(row["total_sens_5sig_mJy"] + 1e-9),
                s=sz, color=GEOM_COLOR[geom],
                alpha=0.85, edgecolors="black", lw=0.5,
                label=f"{GEOM_LABEL[geom]} {row['outrigger_dist_km']}km",
                zorder=5
            )
            ax.annotate(
                f"{GEOM_LABEL[geom][:3]}\n{row['outrigger_dist_km']}km",
                (row["MSL_dB"], np.log10(row["total_sens_5sig_mJy"] + 1e-9)),
                fontsize=6.5, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points"
            )

        ax.set_xlabel("Max Sidelobe Level [dB]  (more negative = better)", fontsize=10)
        ax.set_ylabel("log₁₀(5σ Sensitivity [mJy])  (lower = better)", fontsize=10)
        ax.set_title(f"Core {core_n_sel}×{core_n_sel}: Sidelobe vs Sensitivity\n"
                     f"(bubble size ∝ N detectable targets)", fontsize=10)
        ax.grid(True, alpha=0.3)

        # best quadrant annotation
        ax.annotate("◄ better sidelobes\n▼ better sensitivity",
                    xy=(0.97, 0.03), xycoords="axes fraction",
                    fontsize=8, ha="right", va="bottom", color="grey",
                    style="italic")

    # custom legend
    handles = [mpatches.Patch(color=GEOM_COLOR[g], label=GEOM_LABEL[g])
               for g in GEOMETRIES]
    axes[1].legend(handles=handles, fontsize=9, title="Geometry", loc="upper right")

    fig.suptitle("Trade-off: Beam Sidelobe Level vs 5σ Sensitivity (30 MHz, 100h)\n"
                 "Ideal configuration is bottom-left with large bubble",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(GT_PLOT, "comparison_tradeoff_scatter.png"), dpi=150)
    plt.close(fig)
    print("  Saved comparison_tradeoff_scatter.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT D: SENSITIVITY vs INTEGRATION TIME for all 16 configs at 30 MHz
    # ══════════════════════════════════════════════════════════════════════════
    t_arr = np.logspace(-2, 4, 400)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for col_idx, core_n_sel in enumerate([32, 128]):
        ax = axes[col_idx]
        sub = df_sum[df_sum["core_size"] == f"{core_n_sel}×{core_n_sel}"]

        for _, row in sub.iterrows():
            geom    = row["geometry"]
            dk      = row["outrigger_dist_km"]
            N       = row["N_elements"]
            B_max   = row["max_baseline_km"] * 1e3
            ls      = DIST_STYLE[dk]

            sc  = confusion_limit_Jy(REF_FREQ, B_max)
            st1 = sigma_thermal_Jy(N, REF_FREQ, REF_BW, 1.0)
            stot = np.sqrt((st1 / np.sqrt(t_arr))**2 + sc**2)

            ax.loglog(t_arr, stot * 1e3,
                      color=GEOM_COLOR[geom], ls=ls, lw=1.8,
                      label=f"{GEOM_LABEL[geom]} {dk}km")

        ax.set_xlabel("Integration time [h]", fontsize=10)
        ax.set_ylabel("5σ Total sensitivity [mJy]", fontsize=10)
        ax.set_title(f"Core {core_n_sel}×{core_n_sel} — Sensitivity vs Time\n"
                     f"(30 MHz, BW={REF_BW} MHz)", fontsize=10)
        ax.legend(fontsize=7.5, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

        # shade confusion-dominated region
        worst_sc = df_sum[df_sum["core_size"]==f"{core_n_sel}×{core_n_sel}"]["confusion_mJy"].max()
        ax.axhline(NSIGMA * worst_sc, color="grey", ls=":", lw=0.9, alpha=0.6)

    # distance linestyle legend
    leg_lines = [plt.Line2D([0],[0], color="k", ls="-",  lw=1.5, label="1 km outrigger"),
                 plt.Line2D([0],[0], color="k", ls="--", lw=1.5, label="5 km outrigger")]
    axes[1].legend(handles=leg_lines, fontsize=9, loc="lower left")

    fig.suptitle("Sensitivity vs Integration Time — All 16 Geometry Configurations\n"
                 "(solid = 1 km outrigger, dashed = 5 km outrigger)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(GT_PLOT, "comparison_sens_vs_time.png"), dpi=150)
    plt.close(fig)
    print("  Saved comparison_sens_vs_time.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PLOT E: DETECTION YIELD — grouped bar per geometry, stacked feas/asp
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for col_idx, core_n_sel in enumerate([32, 128]):
        ax    = axes[col_idx]
        sub   = df_sum[df_sum["core_size"] == f"{core_n_sel}×{core_n_sel}"]
        x_pos = np.arange(len(GEOMETRIES))
        w     = 0.32

        for di, (dist_km, alpha) in enumerate([(1.0, 1.0), (5.0, 0.55)]):
            feas_vals = []
            asp_vals  = []
            for geom in GEOMETRIES:
                row = sub[(sub["geometry"]==geom) & (sub["outrigger_dist_km"]==dist_km)]
                feas_vals.append(float(row["n_feasible"].values[0])   if len(row) else 0)
                asp_vals.append( float(row["n_aspirational"].values[0]) if len(row) else 0)

            offset = (di - 0.5) * w
            b1 = ax.bar(x_pos + offset, feas_vals, w,
                        color="#4CAF50", alpha=alpha, edgecolor="white",
                        label=f"Feasible (<100h), {dist_km}km")
            b2 = ax.bar(x_pos + offset, asp_vals, w, bottom=feas_vals,
                        color="#FF9800", alpha=alpha, edgecolor="white",
                        label=f"Aspirational (100–1000h), {dist_km}km")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([GEOM_LABEL[g] for g in GEOMETRIES], fontsize=10)
        ax.set_ylabel("Number of Detectable Targets", fontsize=10)
        ax.set_title(f"Core {core_n_sel}×{core_n_sel} — Detection Yield by Geometry", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Detection Yield (Exoplanet Targets) — All Geometries\n"
                 "(green = feasible <100h | orange = aspirational 100–1000h)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(GT_PLOT, "comparison_detection_yield.png"), dpi=150)
    plt.close(fig)
    print("  Saved comparison_detection_yield.png")


# ══════════════════════════════════════════════════════════════════════════════
# SYMMETRY DIAGNOSTIC PLOT
# Shows quantitatively that different geometries have different symmetry levels
# ══════════════════════════════════════════════════════════════════════════════
def symmetry_diagnostic_plot(configs, af_store):
    """
    For each of the 16 configurations, compute three symmetry error metrics
    on the normalised power beam at 30 MHz:

      err_180 = max |B(l,m) − B(−l,−m)|   ← centrosymmetry  (must be ~0)
      err_LR  = max |B(l,m) − B(−l, m)|   ← left-right mirror
      err_UD  = max |B(l,m) − B( l,−m)|   ← up-down mirror

    For a ring:  all three ≈ 0  (full D4 symmetry)
    For a line:  all three ≈ 0  (mirror-symmetric E-W array)
    For fixed Y: err_180 ≈ 0, err_LR and err_UD > 0  (only centrosymmetry)
    For random:  err_180 ≈ 0, err_LR and err_UD > 0  (only centrosymmetry)

    Also shows a side-by-side |B| vs mirror(|B|) difference image for the
    most asymmetric configuration to make the break in symmetry visible.
    """
    print("\n── Symmetry diagnostic ──")
    ref_bl = "20-40"
    rows   = []

    for name, cfg in configs.items():
        _, _, _, B_norm = af_store[name][ref_bl]
        err_180 = float(np.nanmax(np.abs(B_norm - B_norm[::-1, ::-1])))
        err_LR  = float(np.nanmax(np.abs(B_norm - B_norm[:,   ::-1])))
        err_UD  = float(np.nanmax(np.abs(B_norm - B_norm[::-1,   :])))
        rows.append(dict(config=name, geom=cfg["meta"]["geom"],
                         core_n=cfg["meta"]["core_n"],
                         dist_km=cfg["meta"]["dist_km"],
                         err_180=err_180, err_LR=err_LR, err_UD=err_UD))

    df_sym = pd.DataFrame(rows)
    df_sym.to_csv(os.path.join(GT_CSV, "symmetry_errors.csv"), index=False)

    # ── Bar chart: LR and UD errors per config ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x   = np.arange(len(df_sym))
    w   = 0.4
    colors = [GEOM_COLOR[g] for g in df_sym["geom"]]

    for ax, col, title in zip(axes,
            ["err_LR",  "err_UD"],
            ["Left-Right Mirror Error\n(max|B(l,m)−B(−l,m)|)",
             "Up-Down Mirror Error\n(max|B(l,m)−B(l,−m)|)"]):
        ax.bar(x - w/2, df_sym[col], w, color=colors, edgecolor="white", lw=0.4)
        ax.axhline(0.01, color="grey", ls="--", lw=0.9,
                   label="1% threshold — visually perceptible asymmetry")
        ax.set_xticks(x)
        ax.set_xticklabels(df_sym["config"], rotation=50, ha="right", fontsize=6.5)
        ax.set_ylabel("Max absolute error in B_norm", fontsize=10)
        ax.set_title(title + "\n(0 = perfectly symmetric, >0 = asymmetric)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # geometry legend
    handles = [mpatches.Patch(color=GEOM_COLOR[g], label=GEOM_LABEL[g])
               for g in GEOMETRIES]
    axes[0].legend(handles=handles + [
        plt.Line2D([0],[0], color="grey", ls="--", lw=0.9,
                   label="1% threshold")], fontsize=8)

    fig.suptitle("Beam Pattern Symmetry Analysis — All 16 Configurations\n"
                 "Centrosymmetry (180° rotation) is always ≈0 by physics;\n"
                 "Mirror symmetry breaks only for Y-shape and Random.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(GT_PLOT, "symmetry_errors.png"), dpi=150)
    plt.close(fig)
    print("  Saved symmetry_errors.png")

    # ── Difference images: B(l,m) − B(−l,m) for one representative config ──
    # Show ring (should be 0), Y-shape, random (should be non-zero)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    check_cfgs = [
        ("ring_core128x128_5.0km",    "Ring 128×128 + 5 km\n(expected: ~0 difference)"),
        ("y_shape_core128x128_5.0km", "Y-shape 128×128 + 5 km\n(expected: asymmetric difference)"),
        ("random_core128x128_5.0km",  "Random 128×128 + 5 km\n(expected: asymmetric difference)"),
        ("line_core128x128_5.0km",    "Line 128×128 + 5 km\n(expected: ~0 difference)"),
    ]
    row_labels = ["Beam B(l,m)", "Mirror B(−l,m)", "Difference B − mirror"]

    for ci, (cname, ctitle) in enumerate(check_cfgs):
        l_arr, m_arr, _, B_norm = af_store[cname][ref_bl]
        B_mirror = B_norm[:, ::-1]
        diff     = B_norm - B_mirror
        vmax_b   = 1.0
        vmax_d   = max(0.02, np.nanpercentile(np.abs(diff), 99))

        im0 = axes[0, ci].pcolormesh(l_arr, m_arr, B_norm,
                                      vmin=0, vmax=vmax_b, cmap="inferno",
                                      shading="auto")
        axes[0, ci].set_title(ctitle, fontsize=8)
        plt.colorbar(im0, ax=axes[0, ci], fraction=0.046)

        im1 = axes[1, ci].pcolormesh(l_arr, m_arr, B_mirror,
                                      vmin=0, vmax=vmax_b, cmap="inferno",
                                      shading="auto")
        plt.colorbar(im1, ax=axes[1, ci], fraction=0.046)

        im2 = axes[2, ci].pcolormesh(l_arr, m_arr, diff,
                                      vmin=-vmax_d, vmax=vmax_d,
                                      cmap="RdBu_r", shading="auto")
        plt.colorbar(im2, ax=axes[2, ci], fraction=0.046)

    for ri, rl in enumerate(row_labels):
        axes[ri, 0].set_ylabel(rl, fontsize=9, fontweight="bold")

    for ax in axes.ravel():
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)

    fig.suptitle("Left-Right Mirror Symmetry Test: B(l,m) vs B(−l,m)\n"
                 "Row 1: beam  |  Row 2: its left-right mirror  |  Row 3: difference\n"
                 "Non-zero Row 3 confirms broken mirror symmetry (Y-shape & Random)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(GT_PLOT, "symmetry_difference_images.png"), dpi=130)
    plt.close(fig)
    print("  Saved symmetry_difference_images.png")

    print(f"\n  Symmetry summary (at 30 MHz, max |ΔB|):")
    print(f"  {'Config':40s}  {'err_LR':>8s}  {'err_UD':>8s}  {'err_180':>8s}")
    print(f"  {'-'*68}")
    for _, r in df_sym.iterrows():
        print(f"  {r['config']:40s}  {r['err_LR']:>8.4f}  {r['err_UD']:>8.4f}  {r['err_180']:>8.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL TEXT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(df_sum):
    print("\n" + "="*72)
    print("  GEOMETRY TRADE-OFF: FINAL SUMMARY")
    print("="*72)

    print("\n  Best 5σ sensitivity (30 MHz, 100h) — top 4 configurations:")
    top_sens = df_sum.nsmallest(4, "total_sens_5sig_mJy")[
        ["config_name","total_sens_5sig_mJy","confusion_mJy","MSL_dB","n_detectable"]]
    print(top_sens.to_string(index=False))

    print("\n  Best sidelobe suppression (most negative MSL) — top 4:")
    top_msl = df_sum.nsmallest(4, "MSL_dB")[
        ["config_name","MSL_dB","HPBW_arcmin","total_sens_5sig_mJy","n_detectable"]]
    print(top_msl.to_string(index=False))

    print("\n  Most detectable targets — top 4:")
    top_det = df_sum.nlargest(4, "n_detectable")[
        ["config_name","n_detectable","n_feasible","n_aspirational",
         "total_sens_5sig_mJy","MSL_dB"]]
    print(top_det.to_string(index=False))

    # Geometry-level average
    print("\n  Average metrics by geometry (all distances and core sizes):")
    avg = df_sum.groupby("geom_label")[
        ["total_sens_5sig_mJy","MSL_dB","HPBW_arcmin","n_detectable"]
    ].mean().round(3)
    print(avg.to_string())

    print("\n  Key findings:")
    best_s = df_sum.loc[df_sum["total_sens_5sig_mJy"].idxmin()]
    best_m = df_sum.loc[df_sum["MSL_dB"].idxmin()]
    best_d = df_sum.loc[df_sum["n_detectable"].idxmax()]
    print(f"  • Best sensitivity : {best_s['config_name']}  "
          f"({best_s['total_sens_5sig_mJy']:.4f} mJy)")
    print(f"  • Best sidelobes   : {best_m['config_name']}  "
          f"(MSL = {best_m['MSL_dB']:.1f} dB)")
    print(f"  • Most detections  : {best_d['config_name']}  "
          f"({best_d['n_detectable']} targets)")
    print("="*72)


# ══════════════════════════════════════════════════════════════════════════════
# BEAM CONTOUR PLOTS  (10 % / 30 % / 50 % of peak)
# ══════════════════════════════════════════════════════════════════════════════
def beam_contour_plots(configs, af_store):
    """
    Overlay contours at 10 %, 30 %, 50 % of normalised peak onto the dB
    colour map.  Uses the best configurations: 128×128 + 5 km outriggers.

    Outputs
    -------
    beam_contours_overview.png  — 4 geometries × 4 bands
    beam_contours_30MHz.png     — 2×2 panel at 20–40 MHz (≈ 30 MHz)
    """
    print("\n── Beam Contour Plots (10 % / 30 % / 50 %) ──")
    ref = {g: cfg_name(g, 128, 5.0) for g in GEOMETRIES}

    # ── overview: 4 rows (geometry) × 4 cols (band) ───────────────────────────
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    for gi, geom in enumerate(GEOMETRIES):
        for ci, (bl, fc) in enumerate(zip(BAND_LABELS, BAND_CTR)):
            l, m, AF_dB, B_norm = af_store[ref[geom]][bl]
            ax = axes[gi, ci]
            im = ax.pcolormesh(l, m, AF_dB, vmin=-30, vmax=0,
                               cmap="inferno", shading="auto")
            ax.contour(l, m, B_norm, levels=CONTOUR_LVL,
                       colors=CONTOUR_COL, linewidths=[0.7, 0.9, 1.2])
            ax.set_aspect("equal"); ax.tick_params(labelsize=6)
            if gi == 0: ax.set_title(f"{bl} MHz", fontsize=9)
            if ci == 0: ax.set_ylabel(f"{GEOM_LABEL[geom]}\nm", fontsize=8)
            if gi == 3: ax.set_xlabel("l", fontsize=8)

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=c, lw=1.5, label=f"{int(lv*100)} % of peak")
           for lv, c in zip(CONTOUR_LVL, CONTOUR_COL)]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Beam Patterns with Contours at 10 %, 30 %, 50 % of Peak\n"
                 "(128×128 core + 5 km outriggers  |  colour scale −30 to 0 dB)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(os.path.join(GT_PLOT, "beam_contours_overview.png"), dpi=120)
    plt.close()

    # ── 2×2 at 20–40 MHz with inline contour labels ───────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    for ax, geom in zip(axes2.ravel(), GEOMETRIES):
        l, m, AF_dB, B_norm = af_store[ref[geom]]["20-40"]
        im = ax.pcolormesh(l, m, AF_dB, vmin=-30, vmax=0,
                           cmap="inferno", shading="auto")
        plt.colorbar(im, ax=ax, label="|AF|² [dB]", fraction=0.046)
        cs = ax.contour(l, m, B_norm, levels=CONTOUR_LVL,
                        colors=CONTOUR_COL, linewidths=[1.0, 1.2, 1.5])
        ax.clabel(cs, fmt={lv: f"{int(lv*100)}%" for lv in CONTOUR_LVL},
                  fontsize=7, inline=True)
        ax.set_title(f"{GEOM_LABEL[geom]}  (20–40 MHz, 128×128 + 5 km)", fontsize=9)
        ax.set_xlabel("l"); ax.set_ylabel("m")
        ax.set_aspect("equal")
    fig2.suptitle("Beam Contours at 10 %, 30 %, 50 % of Peak  —  20–40 MHz band",
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(GT_PLOT, "beam_contours_30MHz.png"), dpi=120)
    plt.close()
    print("  Saved beam_contours_overview.png, beam_contours_30MHz.png")


# ══════════════════════════════════════════════════════════════════════════════
# UV COVERAGE  (station-level, multi-frequency synthesis)
# ══════════════════════════════════════════════════════════════════════════════
def _station_xy(cfg):
    """
    Return (N_stations, 2) array of station ENU (x, y) positions.
    Station 0 = core at (0, 0); stations 1..4 = outrigger sub-array centres.
    """
    centres = cfg["meta"]["out_centres"]
    return np.array([(0.0, 0.0)] + [(cx, cy) for cx, cy in centres])


def _uv_from_stations(sxy, freq_MHz_arr):
    """
    UV coordinates [wavelengths] for all station pairs at each frequency.
    Returns (u, v) arrays including conjugate baselines.
    """
    u, v = [], []
    n = len(sxy)
    for fmhz in freq_MHz_arr:
        lam = C / (fmhz * 1e6)
        for i in range(n):
            for j in range(i + 1, n):
                dx = sxy[i, 0] - sxy[j, 0]
                dy = sxy[i, 1] - sxy[j, 1]
                u += [dx / lam, -dx / lam]
                v += [dy / lam, -dy / lam]
    return np.array(u), np.array(v)


def plot_uv_coverage(configs, cross_cfgs=None):
    """
    UV coverage for all four best original configs + optional cross configs.
    Each sub-band is a distinct colour; each baseline traces a radial track
    as frequency varies across the sub-band (multi-frequency synthesis).
    """
    print("\n── UV Coverage Plots ──")

    labels, sxy_list, col_list = [], [], []
    for g in GEOMETRIES:
        labels.append(f"{GEOM_LABEL[g]}\n(128×128 + 5 km)")
        sxy_list.append(_station_xy(configs[cfg_name(g, 128, 5.0)]))
        col_list.append(GEOM_COLOR[g])
    if cross_cfgs:
        for cg, cfg in cross_cfgs.items():
            short = CROSS_GEOM_LABEL[cg].split("(")[0].strip()
            labels.append(short + "\n" + CROSS_GEOM_LABEL[cg].split("(")[1].rstrip(")"))
            sxy_list.append(np.array([(0.0, 0.0)] +
                                     [(cx, cy) for cx, cy in cfg["meta"]["out_centres"]]))
            col_list.append(CROSS_COLOR[cg])

    n    = len(labels)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.5, nrow * 4.8))
    axes = np.array(axes).ravel()

    for idx, (lbl, sxy) in enumerate(zip(labels, sxy_list)):
        ax = axes[idx]
        handles = []
        for bl, ch in SUBBAND_CH.items():
            u, v = _uv_from_stations(sxy, ch)
            sc   = ax.scatter(u / 1e3, v / 1e3, s=5, alpha=0.7,
                              color=SUBBAND_CH_COLOR[bl], linewidths=0)
            handles.append(sc)
        ax.set_xlabel("u  [kλ]", fontsize=9)
        ax.set_ylabel("v  [kλ]", fontsize=9)
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.4, alpha=0.5)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(handles=handles,
                      labels=[f"{bl} MHz" for bl in SUBBAND_CH],
                      fontsize=8, markerscale=3, loc="upper right")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "UV Coverage  —  Station-Level, Multi-Frequency Synthesis\n"
        "Each sub-band divided into 1 MHz channels  |  "
        "radial tracks show baseline scaling with frequency",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(GT_PLOT, "uv_coverage.png"), dpi=120)
    plt.close()
    print("  Saved uv_coverage.png")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS CONFIGURATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def build_cross_configs():
    """
    Build cross_ew and cross_ns — each a 128×128 core + 4 outrigger sub-arrays
    in N/S/E/W directions with irregular arm distances to reduce aliasing rings.
    """
    print("\n── Build cross configurations ──")
    out_n = CORE_OUT[CROSS_N]
    cross_cfgs = {}
    for cg, arms in [("cross_ew", CROSS_ARMS_EW), ("cross_ns", CROSS_ARMS_NS)]:
        centres = [
            (0.0,              arms["N"] * 1e3),
            (0.0,             -arms["S"] * 1e3),
            (arms["E"] * 1e3,  0.0),
            (-arms["W"] * 1e3, 0.0),
        ]
        pos   = np.vstack([rect_array_enu(CROSS_N, D_SPACE)] +
                          [rect_array_enu(out_n, D_SPACE, cx, cy)
                           for cx, cy in centres])
        B_max = max_baseline_m(centres, CROSS_N)
        meta  = dict(core_n=CROSS_N, out_n=out_n, out_centres=centres,
                     N_core=CROSS_N**2, N_out_each=out_n**2,
                     N_total=len(pos), dist_km=None, geom=cg, B_max_m=B_max)
        cross_cfgs[cg] = dict(pos_enu=pos, meta=meta)
        print(f"  {cg}: {len(pos)} elements  |  "
              f"B_max = {B_max/1e3:.2f} km  |  "
              f"arms N/S/E/W = {arms['N']}/{arms['S']}/{arms['E']}/{arms['W']} km")
    return cross_cfgs


def run_cross_analysis(cross_cfgs, orig_configs, targets):
    """
    Full analysis for cross_ew and cross_ns compared against all four original
    128×128 + 5 km configurations.

    Produces
    --------
    cross_config/layout_cross_overview.png
    cross_config/beam_contours_cross.png     — 6 rows × 4 bands with contours
    cross_config/uv_coverage_cross.png       — UV coverage for all 6 configs
    cross_config/metrics_cross_vs_originals.png
    cross_config/sensitivity_cross_vs_originals.png
    cross_config/cross_metrics.csv
    cross_config/cross_sensitivity.csv
    """
    os.makedirs(CROSS_PLOT, exist_ok=True)
    os.makedirs(CROSS_CSV_D, exist_ok=True)

    print("\n" + "=" * 70)
    print("  CROSS CONFIGURATION ANALYSIS")
    print(f"  Core {CROSS_N}×{CROSS_N} + {CORE_OUT[CROSS_N]}×{CORE_OUT[CROSS_N]} outriggers")
    print("=" * 70)

    # Combined set: 4 original best + 2 cross
    orig_best  = {g: orig_configs[cfg_name(g, 128, 5.0)] for g in GEOMETRIES}
    all_cfgs   = {**orig_best, **cross_cfgs}
    order      = GEOMETRIES + ["cross_ew", "cross_ns"]
    bar_colors = [GEOM_COLOR.get(g, CROSS_COLOR.get(g)) for g in order]
    bar_labels = [GEOM_LABEL.get(g, CROSS_GEOM_LABEL.get(g, g).split("(")[0].strip())
                  for g in order]

    # ── A: Layout plots ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (cg, cfg) in zip(axes, cross_cfgs.items()):
        pos = cfg["pos_enu"]
        N_c = CROSS_N ** 2
        ax.scatter(pos[:N_c, 0] / 1e3, pos[:N_c, 1] / 1e3,
                   s=0.3, c="steelblue", alpha=0.7, label=f"Core {CROSS_N}×{CROSS_N}")
        ax.scatter(pos[N_c:, 0] / 1e3, pos[N_c:, 1] / 1e3,
                   s=2.0, c="tomato",    alpha=0.9, label="Outriggers")
        arms = CROSS_ARMS_EW if cg == "cross_ew" else CROSS_ARMS_NS
        for cx, cy in cfg["meta"]["out_centres"]:
            ax.plot(cx / 1e3, cy / 1e3, "r+", ms=10, mew=1.5)
        ax.annotate(f"N {arms['N']} km", (0,  arms["N"] + 0.15), ha="center", fontsize=8)
        ax.annotate(f"S {arms['S']} km", (0, -arms["S"] - 0.28), ha="center", fontsize=8)
        ax.annotate(f"E {arms['E']} km", ( arms["E"] + 0.15, 0), ha="left",   fontsize=8)
        ax.annotate(f"W {arms['W']} km", (-arms["W"] - 0.15, 0), ha="right",  fontsize=8)
        ax.set_xlabel("East [km]"); ax.set_ylabel("North [km]")
        ax.set_title(CROSS_GEOM_LABEL[cg], fontsize=9)
        ax.set_aspect("equal")
        ax.legend(markerscale=8, fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Cross Array Layouts  |  {CROSS_N}×{CROSS_N} core + "
                 f"{CORE_OUT[CROSS_N]}×{CORE_OUT[CROSS_N]} outrigger sub-arrays", fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(CROSS_PLOT, "layout_cross_overview.png"), dpi=120)
    plt.close()
    print("  Saved layout_cross_overview.png")

    # ── B: Compute AF for all 6 configs ───────────────────────────────────────
    print("  Computing AF for 6 configurations × 4 bands …")
    all_af = {}
    for name, cfg in all_cfgs.items():
        all_af[name] = {}
        for bl, fc in zip(BAND_LABELS, BAND_CTR):
            l, m, AF_dB, B_norm = compute_af(cfg["pos_enu"], cfg["meta"], fc, N_GRID)
            all_af[name][bl] = (l, m, AF_dB, B_norm)

    # ── C: Beam contour plot: 6 rows × 4 cols ─────────────────────────────────
    fig, axes = plt.subplots(6, 4, figsize=(18, 26))
    for ri, name in enumerate(order):
        row_lbl = GEOM_LABEL.get(name, CROSS_GEOM_LABEL.get(name, name))
        for ci, (bl, fc) in enumerate(zip(BAND_LABELS, BAND_CTR)):
            l, m, AF_dB, B_norm = all_af[name][bl]
            ax = axes[ri, ci]
            im = ax.pcolormesh(l, m, AF_dB, vmin=-30, vmax=0,
                               cmap="inferno", shading="auto")
            ax.contour(l, m, B_norm, levels=CONTOUR_LVL,
                       colors=CONTOUR_COL, linewidths=[0.7, 0.9, 1.2])
            ax.set_aspect("equal"); ax.tick_params(labelsize=6)
            if ri == 0: ax.set_title(f"{bl} MHz", fontsize=9)
            if ci == 0:
                ax.set_ylabel(row_lbl.replace("(", "\n("), fontsize=7)
            if ri == 5: ax.set_xlabel("l", fontsize=8)

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=c, lw=1.5, label=f"{int(lv*100)}%")
           for lv, c in zip(CONTOUR_LVL, CONTOUR_COL)]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Beam Patterns with 10 % / 30 % / 50 % Contours\n"
        "Rows 1–4: original configs (128×128 + 5 km)  |  Rows 5–6: cross configs",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(os.path.join(CROSS_PLOT, "beam_contours_cross.png"), dpi=120)
    plt.close()
    print("  Saved beam_contours_cross.png")

    # ── D: UV coverage for all 6 configs ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    axes = axes.ravel()
    for idx, name in enumerate(order):
        cfg = all_cfgs[name]
        sxy = np.array([(0.0, 0.0)] +
                       [(cx, cy) for cx, cy in cfg["meta"]["out_centres"]])
        ax  = axes[idx]
        for bl, ch in SUBBAND_CH.items():
            u, v = _uv_from_stations(sxy, ch)
            ax.scatter(u / 1e3, v / 1e3, s=5, alpha=0.7,
                       color=SUBBAND_CH_COLOR[bl], linewidths=0,
                       label=f"{bl} MHz")
        lbl = GEOM_LABEL.get(name, CROSS_GEOM_LABEL.get(name, name))
        ax.set_title(lbl.replace("(", "\n("), fontsize=9, fontweight="bold")
        ax.set_xlabel("u  [kλ]", fontsize=9); ax.set_ylabel("v  [kλ]", fontsize=9)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
        ax.axvline(0, color="gray", lw=0.4, alpha=0.5)
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, markerscale=3, loc="upper right")
    fig.suptitle(
        "UV Coverage — Original Configs vs Cross Configs  (station-level, multi-band)\n"
        "Radial tracks = one baseline at multiple frequencies  |  colours = sub-bands",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(CROSS_PLOT, "uv_coverage_cross.png"), dpi=120)
    plt.close()
    print("  Saved uv_coverage_cross.png")

    # ── E: Beam metrics comparison ────────────────────────────────────────────
    mtr_rows = []
    for name in order:
        for bl, fc in zip(BAND_LABELS, BAND_CTR):
            l, m, _, B_norm = all_af[name][bl]
            mtr = beam_metrics(B_norm, l, m, fc)
            hpbw = 2 * np.sqrt(mtr["Omega_B"] / np.pi)
            mtr_rows.append(dict(
                config=name, band=bl, freq_MHz=fc,
                HPBW_arcmin=np.degrees(hpbw) * 60.0,
                MSL_dB=mtr["MSL_dB"],
                D_peak=mtr["D_peak"],
                Omega_B_sr=mtr["Omega_B"],
            ))
    df_mtr = pd.DataFrame(mtr_rows)
    df_mtr.to_csv(os.path.join(CROSS_CSV_D, "cross_metrics.csv"), index=False)

    df_ref = df_mtr[df_mtr.band == "20-40"].set_index("config")
    x = np.arange(len(order))
    fig, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, ylabel, fmt in [
        (axes3[0], "HPBW_arcmin", "HPBW [arcmin]",           ".1f"),
        (axes3[1], "MSL_dB",      "Max sidelobe level [dB]",  ".1f"),
        (axes3[2], "D_peak",      "Peak directivity",          ".2e"),
    ]:
        vals = [df_ref.loc[g, col] if g in df_ref.index else np.nan for g in order]
        ax.bar(x, vals, color=bar_colors, alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for xi, val in enumerate(vals):
            if not np.isnan(val):
                ax.text(xi, val * 1.02 if val > 0 else val * 0.98,
                        f"{val:{fmt}}", ha="center", fontsize=7)
    fig.suptitle(
        "Beam Metrics at 20–40 MHz  —  Original (128×128+5km) vs Cross Configs",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(CROSS_PLOT, "metrics_cross_vs_originals.png"), dpi=120)
    plt.close()
    print("  Saved metrics_cross_vs_originals.png, cross_metrics.csv")

    # ── F: Sensitivity / exoplanet detection yield ────────────────────────────
    sens_rows = []
    for name in order:
        cfg   = all_cfgs[name]
        N     = cfg["meta"]["N_total"]
        B_max = cfg["meta"]["B_max_m"]
        for bl, (f_lo, f_hi), fc in zip(BAND_LABELS, SUBBANDS, BAND_CTR):
            bw   = f_hi - f_lo
            band_tgts = targets[
                (targets["frequency_MHz"] >= f_lo) &
                (targets["frequency_MHz"] <  f_hi)
            ]
            for _, trow in band_tgts.iterrows():
                t_h  = required_t_hours(trow["flux_mJy"], N, fc, bw, B_max)
                feas = classify_feasibility(t_h)
                sens_rows.append(dict(
                    config=name, target=trow["Name"],
                    band=bl, t_h=t_h if np.isfinite(t_h) else 1e9,
                    feasibility=feas,
                ))
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(os.path.join(CROSS_CSV_D, "cross_sensitivity.csv"), index=False)

    n_det = {name: int(df_sens[(df_sens.config == name) &
                                (df_sens.t_h < 100)]["target"].nunique())
             for name in order}

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(x, [n_det[g] for g in order], color=bar_colors,
            alpha=0.85, edgecolor="k", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Exoplanet targets detectable in < 100 h", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    for xi, name in enumerate(order):
        ax2.text(xi, n_det[name] + 0.3, str(n_det[name]),
                 ha="center", fontsize=9, fontweight="bold")
    fig2.suptitle(
        "Exoplanet Detection Yield  (t < 100 h)  —  Cross vs Original Configs",
        fontsize=11,
    )
    plt.tight_layout()
    fig2.savefig(os.path.join(CROSS_PLOT, "sensitivity_cross_vs_originals.png"), dpi=120)
    plt.close()
    print("  Saved sensitivity_cross_vs_originals.png, cross_sensitivity.csv")

    # ── G: Print summary table ─────────────────────────────────────────────────
    print(f"\n  {'Config':32s}  N_elem  B_max km  n_det(<100h)  "
          f"HPBW′  MSL dB")
    for name in order:
        cfg  = all_cfgs[name]
        lbl  = GEOM_LABEL.get(name, CROSS_GEOM_LABEL.get(name, name)).split("\n")[0]
        hpbw = df_ref.loc[name, "HPBW_arcmin"] if name in df_ref.index else float("nan")
        msl  = df_ref.loc[name, "MSL_dB"]      if name in df_ref.index else float("nan")
        print(f"  {lbl:32s}  {cfg['meta']['N_total']:6d}  "
              f"{cfg['meta']['B_max_m']/1e3:8.2f}  {n_det[name]:12d}  "
              f"{hpbw:6.1f}  {msl:6.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# ASYMMETRIC BEAM STUDY
# Demonstrates how |AF|² breaks centrosymmetry under realistic conditions:
#   1. Phase steering   — complex per-element weights for off-zenith pointing
#   2. Calibration errors — random phase (σ_φ=12°) + amplitude (σ_a=5%) errors
#   3. E-W dipole element pattern  — P(l,m) = 1 − l²
#   4. Survey sensitivity map — max sensitivity over 5×5 hemisphere grid
# Focus: core 128×128 + 5 km outriggers (best configuration)
# ══════════════════════════════════════════════════════════════════════════════

# ── asymmetric beam parameters ────────────────────────────────────────────────
AB_CORE      = 128           # best core size for demonstration
AB_DIST_KM   = 5.0           # best outrigger distance
AB_N_GRID    = 256           # grid resolution (reduced for full per-element computation)
AB_SIGMA_PHI = 12.0          # phase error RMS [degrees]
AB_SIGMA_AMP = 0.05          # amplitude error RMS [fractional]
AB_SEED      = 42

# Survey pointing grid: 5×5, trimmed to visible hemisphere
_g  = np.linspace(-0.55, 0.55, 5)
_L0, _M0 = np.meshgrid(_g, _g)
_mask     = _L0**2 + _M0**2 < 0.85**2
AB_POINTINGS = list(zip(_L0[_mask].ravel(), _M0[_mask].ravel()))

# Output subdirectory
AB_PLOT = os.path.join(GT_PLOT, "asymmetric_beam")
AB_CSV  = os.path.join(GT_CSV,  "asymmetric_beam")
os.makedirs(AB_PLOT, exist_ok=True)
os.makedirs(AB_CSV,  exist_ok=True)


def ab_element_pattern(L, M):
    """E-W dipole power pattern: P(l,m) = 1 − l²."""
    return np.where(L**2 + M**2 <= 1.0, 1.0 - L**2, np.nan)


def ab_steering_weights(pos_enu, l0, m0, freq_MHz):
    """Per-element steering weights: w_k = exp(−ik·(x_k·l0 + y_k·m0))."""
    k = 2 * np.pi * freq_MHz * 1e6 / C
    return np.exp(-1j * k * (pos_enu[:, 0] * l0 + pos_enu[:, 1] * m0))


def ab_calibration_weights(N, sigma_phi_deg=AB_SIGMA_PHI,
                           sigma_amp=AB_SIGMA_AMP, seed=AB_SEED):
    """
    Per-element hardware errors (cable + receiver tolerances):
      phase offset δφ_k ~ N(0, σ_φ²)   breaks Hermitian symmetry → true asymmetry
      amplitude     a_k ~ N(1, σ_a²)   real errors alone preserve centrosymmetry
    Combined weight: w_k = a_k · exp(iδφ_k)
    """
    rng = np.random.default_rng(seed)
    phi = rng.normal(0.0, np.radians(sigma_phi_deg), N)
    amp = np.clip(rng.normal(1.0, sigma_amp, N), 0.5, 1.5)
    return amp * np.exp(1j * phi)


def ab_compute_af(pos_enu, weights, freq_MHz, n_grid=AB_N_GRID):
    """
    Per-element weighted AF using the two-step matrix product:
        AF[l,m] = (W·Φ_x)ᵀ · Φ_y
    where Φ_x[k,i]=exp(ik·x_k·l_i), Φ_y[k,j]=exp(ik·y_k·m_j).
    Memory ≈ 2 × N × n_grid × 16 B  (≈142 MB for N=17408, n_grid=256).
    Returns l_arr, m_arr, B_norm where B_norm has shape (n_grid, n_grid).
    """
    k     = 2 * np.pi * freq_MHz * 1e6 / C
    l_arr = np.linspace(-1.0, 1.0, n_grid)
    m_arr = np.linspace(-1.0, 1.0, n_grid)
    Phi_x = np.exp(1j * k * np.outer(pos_enu[:, 0], l_arr))  # (N, N_l)
    Phi_y = np.exp(1j * k * np.outer(pos_enu[:, 1], m_arr))  # (N, N_m)
    AF    = (weights[:, None] * Phi_x).T @ Phi_y              # (N_l, N_m)
    B     = np.abs(AF) ** 2
    B_norm = B / (B.max() + 1e-30)
    L, M  = np.meshgrid(l_arr, m_arr, indexing="ij")
    B_norm[L**2 + M**2 > 1.0] = np.nan
    return l_arr, m_arr, B_norm                               # (N_l, N_m)


def ab_build_configs():
    """Build one 128×128 + 5 km config for each geometry."""
    out_n   = CORE_OUT[AB_CORE]
    cfgs    = {}
    for geom in GEOMETRIES:
        pos, centres = LAYOUT_FN[geom](AB_CORE, out_n, AB_DIST_KM)
        cfgs[geom]   = dict(pos_enu=pos, N=len(pos))
    return cfgs


def ab_beam_evolution(cfgs):
    """
    Plot A — for each geometry: 3 rows × 4 columns.
      Row 1: ideal zenith beam  (no steering, no errors)
      Row 2: steered to (l0=0.30, m0=0.20)  [steering only]
      Row 3: steered + calibration errors    [full realistic model]
    All multiplied by E-W dipole element pattern.
    """
    print("\n── AB Plot A: Beam evolution (zenith → steered → errors) ──")
    l0d, m0d = 0.30, 0.20

    for geom, cfg in cfgs.items():
        pos = cfg["pos_enu"]; N = cfg["N"]
        cal = ab_calibration_weights(N)
        fig, axes = plt.subplots(3, 4, figsize=(18, 13))
        row_titles = [
            "Ideal — zenith pointing  (centrosymmetric, no steering)",
            f"Steered to (l₀={l0d}, m₀={m0d})  — no errors",
            f"Steered + hardware errors  (σ_φ={AB_SIGMA_PHI}°, σ_a={AB_SIGMA_AMP*100:.0f}%)",
        ]
        for ci, (bl, fc) in enumerate(zip(BAND_LABELS, BAND_CTR)):
            w_z = np.ones(N, dtype=complex)
            w_s = ab_steering_weights(pos, l0d, m0d, fc)
            w_e = cal * w_s
            for ri, w in enumerate([w_z, w_s, w_e]):
                l, m, B = ab_compute_af(pos, w, fc)
                L, M    = np.meshgrid(l, m, indexing="ij")
                EP      = ab_element_pattern(L, M)
                B_full  = B * EP
                B_dB    = 10 * np.log10(B_full / np.nanmax(B_full) + 1e-20)
                ax = axes[ri, ci]
                im = ax.pcolormesh(l, m, B_dB.T, vmin=-50, vmax=0,
                                   cmap="inferno", shading="auto")
                if ci == 3:
                    plt.colorbar(im, ax=ax, label="dB", fraction=0.046)
                ax.set_aspect("equal"); ax.tick_params(labelsize=6)
                if ri == 0: ax.set_title(f"{bl} MHz", fontsize=8)
                if ci == 0: ax.set_ylabel(row_titles[ri], fontsize=7.5)
                if ri > 0:
                    ax.plot(l0d, m0d, "w+", ms=8, mew=1.5)
        fig.suptitle(f"Beam Pattern: Ideal → Steered → Errors\n"
                     f"{GEOM_LABEL[geom]} | {AB_CORE}×{AB_CORE} + {AB_DIST_KM} km  "
                     f"× E-W dipole  (white '+' = pointing dir.)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(AB_PLOT, f"beam_evolution_{geom}.png"), dpi=120)
        plt.close(fig)
        print(f"  Saved beam_evolution_{geom}.png")


def ab_geometry_comparison(cfgs):
    """
    Plot B — 4 rows (geometries) × 4 cols (bands).
    Two versions: clean steered beam, and steered + calibration errors.
    Both include the E-W dipole element pattern.
    """
    print("\n── AB Plot B: All-geometry comparison (off-zenith, all bands) ──")
    l0, m0 = 0.30, 0.20
    for scenario, use_err in [("clean_steered", False), ("errors_steered", True)]:
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        for ri, geom in enumerate(GEOMETRIES):
            pos = cfgs[geom]["pos_enu"]; N = cfgs[geom]["N"]
            cal = ab_calibration_weights(N) if use_err else np.ones(N, dtype=complex)
            for ci, (bl, fc) in enumerate(zip(BAND_LABELS, BAND_CTR)):
                w       = cal * ab_steering_weights(pos, l0, m0, fc)
                l, m, B = ab_compute_af(pos, w, fc)
                L, M    = np.meshgrid(l, m, indexing="ij")
                EP      = ab_element_pattern(L, M)
                BEP     = B * EP
                B_dB    = 10 * np.log10(BEP / np.nanmax(BEP) + 1e-20)
                ax = axes[ri, ci]
                im = ax.pcolormesh(l, m, B_dB.T, vmin=-50, vmax=0,
                                   cmap="inferno", shading="auto")
                if ci == 3:
                    plt.colorbar(im, ax=ax, fraction=0.046)
                ax.plot(l0, m0, "w+", ms=7, mew=1.4)
                ax.set_aspect("equal"); ax.tick_params(labelsize=6)
                if ri == 0: ax.set_title(f"{bl} MHz", fontsize=9, fontweight="bold")
                if ci == 0:
                    ax.set_ylabel(GEOM_LABEL[geom], fontsize=9,
                                  color=GEOM_COLOR[geom], fontweight="bold")
        err_note = (f" + σ_φ={AB_SIGMA_PHI}°, σ_a={AB_SIGMA_AMP*100:.0f}%"
                    if use_err else "")
        fig.suptitle(f"All-Geometry Comparison — Steered (l₀=0.3, m₀=0.2){err_note}\n"
                     f"{AB_CORE}×{AB_CORE} + {AB_DIST_KM} km  ×  E-W dipole",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(AB_PLOT, f"geometry_comparison_{scenario}.png"), dpi=120)
        plt.close(fig)
        print(f"  Saved geometry_comparison_{scenario}.png")


def ab_survey_map(cfgs):
    """
    Plot C — survey sensitivity map.
    For each geometry, sweep {AB_POINTINGS} and record max(B × element_pattern)
    at each sky pixel. Shows the sky coverage achievable with 30 MHz beam.
    """
    print(f"\n── AB Plot C: Survey sensitivity map ({len(AB_POINTINGS)} pointings, 30 MHz) ──")
    fc  = 30.0
    n_s = 128
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    axes = axes.ravel()

    for gi, (geom, cfg) in enumerate(cfgs.items()):
        pos = cfg["pos_enu"]; N = cfg["N"]
        cal = ab_calibration_weights(N)
        l_arr = np.linspace(-1, 1, n_s)
        m_arr = np.linspace(-1, 1, n_s)
        L, M  = np.meshgrid(l_arr, m_arr, indexing="ij")
        EP    = ab_element_pattern(L, M)
        smap  = np.zeros((n_s, n_s))

        for l0, m0 in AB_POINTINGS:
            w       = cal * ab_steering_weights(pos, l0, m0, fc)
            _, _, B = ab_compute_af(pos, w, fc, n_s)
            smap    = np.maximum(smap, np.nan_to_num(B * EP))

        smap[L**2 + M**2 > 1] = np.nan
        cov = np.sum(np.nan_to_num(smap) > 0.1) / np.sum(L**2 + M**2 < 1)
        rows.append(dict(geometry=geom, n_pointings=len(AB_POINTINGS),
                         mean_sensitivity=float(np.nanmean(smap)),
                         coverage_fraction=float(cov)))

        ax = axes[gi]
        im = ax.pcolormesh(l_arr, m_arr, smap.T, vmin=0, vmax=1,
                           cmap="plasma", shading="auto")
        plt.colorbar(im, ax=ax, label="Max norm. sensitivity")
        for l0p, m0p in AB_POINTINGS:
            ax.plot(l0p, m0p, "w.", ms=3, alpha=0.5)
        ax.set_title(f"{GEOM_LABEL[geom]}", fontsize=10,
                     color=GEOM_COLOR[geom], fontweight="bold")
        ax.set_xlabel("l"); ax.set_ylabel("m")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)

    fig.suptitle(f"Survey Sensitivity Map — Max over {len(AB_POINTINGS)} pointings\n"
                 f"(30 MHz, {AB_CORE}×{AB_CORE} + {AB_DIST_KM} km, "
                 f"σ_φ={AB_SIGMA_PHI}°, σ_a={AB_SIGMA_AMP*100:.0f}%  ×  E-W dipole)\n"
                 "Dots = pointing centres",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(AB_PLOT, "survey_sensitivity_map.png"), dpi=130)
    plt.close(fig)
    print("  Saved survey_sensitivity_map.png")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AB_CSV, "survey_metrics.csv"), index=False)
    print("  Saved survey_metrics.csv")
    return df


def ab_symmetry_vs_pointing(cfgs):
    """
    Plot D — left-right symmetry error vs pointing offset from zenith.
    Shows how quickly centrosymmetry breaks with steering angle,
    and how calibration errors add an irreducible asymmetry floor.
    """
    print("\n── AB Plot D: Symmetry error vs pointing offset ──")
    fc      = 30.0
    offsets = np.linspace(0.0, 0.70, 12)
    rows    = []
    fig, ax = plt.subplots(figsize=(10, 6))

    for geom, cfg in cfgs.items():
        pos = cfg["pos_enu"]; N = cfg["N"]
        cal = ab_calibration_weights(N)
        e_ideal, e_err = [], []

        for off in offsets:
            l0 = off * np.cos(np.radians(45))
            m0 = off * np.sin(np.radians(45))
            _, _, B_id = ab_compute_af(pos, ab_steering_weights(pos, l0, m0, fc), fc, 128)
            _, _, B_er = ab_compute_af(pos, cal * ab_steering_weights(pos, l0, m0, fc), fc, 128)
            B_id = np.nan_to_num(B_id); B_er = np.nan_to_num(B_er)
            ei = float(np.max(np.abs(B_id - B_id[::-1, :])))
            ee = float(np.max(np.abs(B_er - B_er[::-1, :])))
            e_ideal.append(ei); e_err.append(ee)
            rows.append(dict(geometry=geom, pointing_offset=float(off),
                              err_LR_ideal=ei, err_LR_errors=ee))

        theta = np.degrees(np.arcsin(np.clip(offsets, 0, 0.9999)))
        ax.plot(theta, e_ideal, color=GEOM_COLOR[geom], ls="-",  lw=1.8,
                label=f"{GEOM_LABEL[geom]} — ideal")
        ax.plot(theta, e_err,   color=GEOM_COLOR[geom], ls="--", lw=1.8,
                label=f"{GEOM_LABEL[geom]} + errors")

    ax.axhline(0.05, color="grey", ls=":", lw=1.1, alpha=0.7,
               label="5% — visually perceptible threshold")
    ax.set_xlabel("Pointing offset from zenith [°]", fontsize=11)
    ax.set_ylabel("Left-right symmetry error  max|B(l,m)−B(−l,m)|", fontsize=11)
    ax.set_title(f"Symmetry Breaking vs Pointing Offset  (30 MHz, {AB_CORE}×{AB_CORE} + {AB_DIST_KM} km)\n"
                 f"Solid = steering only  |  Dashed = steering + σ_φ={AB_SIGMA_PHI}°, σ_a={AB_SIGMA_AMP*100:.0f}%",
                 fontsize=10)
    ax.legend(fontsize=7.5, ncol=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(AB_PLOT, "symmetry_vs_pointing.png"), dpi=130)
    plt.close(fig)
    print("  Saved symmetry_vs_pointing.png")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AB_CSV, "symmetry_vs_pointing.csv"), index=False)
    print("  Saved symmetry_vs_pointing.csv")


def ab_element_pattern_effect(cfgs):
    """
    Plot E — isotropic vs E-W dipole element pattern, for Y-shape and random.
    Two rows: isotropic (|AF|² only) vs multiplied by P(l,m)=1−l².
    """
    print("\n── AB Plot E: Element pattern effect ──")
    l0, m0 = 0.30, 0.20
    for geom in ["y_shape", "random"]:
        pos = cfgs[geom]["pos_enu"]; N = cfgs[geom]["N"]
        cal = ab_calibration_weights(N)
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for ci, (bl, fc) in enumerate(zip(BAND_LABELS, BAND_CTR)):
            w       = cal * ab_steering_weights(pos, l0, m0, fc)
            l, m, B = ab_compute_af(pos, w, fc)
            L, M    = np.meshgrid(l, m, indexing="ij")
            EP      = ab_element_pattern(L, M)
            panels  = [
                (B,      "Isotropic element  (|AF|² only)"),
                (B * EP, "× E-W dipole  P(l,m) = 1−l²"),
            ]
            for ri, (data, title) in enumerate(panels):
                B_dB = 10 * np.log10(data / np.nanmax(data) + 1e-20)
                ax   = axes[ri, ci]
                im   = ax.pcolormesh(l, m, B_dB.T, vmin=-50, vmax=0,
                                     cmap="inferno", shading="auto")
                if ci == 3:
                    plt.colorbar(im, ax=ax, fraction=0.046, label="dB")
                ax.plot(l0, m0, "w+", ms=7, mew=1.4)
                ax.set_aspect("equal"); ax.tick_params(labelsize=6)
                if ri == 0: ax.set_title(f"{bl} MHz", fontsize=9, fontweight="bold")
                if ci == 0: ax.set_ylabel(title, fontsize=8)
        fig.suptitle(f"Element Pattern Effect — {GEOM_LABEL[geom]}\n"
                     f"Row 1: isotropic  |  Row 2: ×E-W dipole  "
                     f"(steered l₀=0.3,m₀=0.2 + errors)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(os.path.join(AB_PLOT, f"element_pattern_effect_{geom}.png"), dpi=120)
        plt.close(fig)
        print(f"  Saved element_pattern_effect_{geom}.png")


def run_asymmetric_beam_study():
    """Run the full asymmetric beam study on the best configuration."""
    print("\n" + "="*70)
    print("  ASYMMETRIC BEAM STUDY")
    print(f"  Core {AB_CORE}×{AB_CORE} + {AB_DIST_KM} km outriggers  |  "
          f"σ_φ={AB_SIGMA_PHI}°  σ_a={AB_SIGMA_AMP*100:.0f}%  |  "
          f"{len(AB_POINTINGS)} survey pointings")
    print("="*70)

    cfgs = ab_build_configs()
    ab_beam_evolution(cfgs)
    ab_geometry_comparison(cfgs)
    df_sv = ab_survey_map(cfgs)
    ab_symmetry_vs_pointing(cfgs)
    ab_element_pattern_effect(cfgs)

    print("\n  Survey metrics:")
    print(df_sv.to_string(index=False))
    print(f"\n  Outputs: {AB_PLOT}")


# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED SOURCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
# Use 32×32+5km: HPBW ≈ 8 pixels at 256-grid (main beam clearly resolved)
# compared to 128×128+5km where HPBW ≈ 2 pixels (too small for dirty images).
# Geometry differences between ring/Y/random/line are the same at any core size.
ES_CORE     = 32            # core size for ES demonstration
ES_DIST_KM  = 5.0           # outrigger distance [km]
ES_FREQ_MHZ = 30.0          # representative frequency
ES_N_GRID   = 256           # AF grid resolution
ES_ASPECT   = 0.25          # minor/major ratio for elongated source test
ES_SIZES    = np.logspace(-1.3, 0.8, 40)   # source sizes in units of HPBW: 0.05→6
ES_PAS      = np.linspace(0, 165, 24)      # position angles for orientation test [deg]
ES_PLOT     = os.path.join(GT_PLOT, "extended_source")
ES_CSV      = os.path.join(GT_CSV,  "extended_source")


def _es_psf(pos_enu, freq_MHz=ES_FREQ_MHZ, n_grid=ES_N_GRID):
    """Normalized PSF (beam) with NaN outside unit circle replaced by 0."""
    l_arr, m_arr, B = ab_compute_af(
        pos_enu, np.ones(len(pos_enu), dtype=complex), freq_MHz, n_grid
    )
    return l_arr, m_arr, np.where(np.isnan(B), 0.0, B)


def _es_hpbw(l_arr, B):
    """HPBW [rad] from half-power beam solid angle (B must be NaN-free)."""
    dl = l_arr[1] - l_arr[0]
    omega_b = float((B >= 0.5 * B.max()).sum()) * dl * dl
    return 2.0 * np.sqrt(omega_b / np.pi)


def _es_source(l_arr, m_arr, theta_s_rad, pa_deg=0.0, aspect=1.0):
    """
    Elliptical Gaussian source (peak=1) centred at (0,0).
    theta_s_rad : Gaussian sigma along major axis [direction cosines].
    pa_deg      : position angle CCW from East (l-axis).
    aspect      : sigma_minor / sigma_major  (1 = circular).
    Zeroed outside the unit circle.
    """
    L, M = np.meshgrid(l_arr, m_arr, indexing="ij")
    pa   = np.radians(pa_deg)
    Lr   =  L * np.cos(pa) + M * np.sin(pa)
    Mr   = -L * np.sin(pa) + M * np.cos(pa)
    sig_maj = max(theta_s_rad, 1e-9)
    sig_min = max(theta_s_rad * aspect, 1e-9)
    I    = np.exp(-0.5 * (Lr**2 / sig_maj**2 + Mr**2 / sig_min**2))
    I[L**2 + M**2 > 1.0] = 0.0
    return I


def _es_dirty_image(I_true, B):
    """
    Dirty image = I_true convolved with PSF B, normalized to peak=1.
    FFT-based: I_D = IFFT{ FFT{I_true} × FFT{B} }.
    """
    from scipy.signal import fftconvolve
    B_n = B / (B.sum() + 1e-30)
    I_D = fftconvolve(I_true, B_n, mode="same")
    pk  = I_D.max()
    return I_D / (pk + 1e-30)


def _es_eta(I_true, B):
    """
    Beam-weighted flux recovery fraction.
      η = ΣΣ I_true(l,m)·B(l,m) / ΣΣ I_true(l,m)
    For a point source at beam centre  → η = B(0,0) = 1  for ALL arrays.
    For an extended source             → η < 1, depends on PSF shape.
    """
    denom = I_true.sum()
    return float((I_true * B).sum() / denom) if denom > 0 else 0.0


def es_build_configs():
    """Build one {ES_CORE}×{ES_CORE} + {ES_DIST_KM} km config per geometry."""
    out_n = CORE_OUT[ES_CORE]
    cfgs  = {}
    for geom in GEOMETRIES:
        pos, cen = LAYOUT_FN[geom](ES_CORE, out_n, ES_DIST_KM)
        cfgs[geom] = dict(pos_enu=pos, centres=cen)
    return cfgs


def es_plot_psf_and_images(cfgs):
    """
    Plot A — 4 rows (geometry) × 5 columns:
      col 0: Full PSF in dB  (shows sidelobe geometry clearly)
      col 1: Zoomed PSF (linear) — main beam region only
      col 2: Point-source dirty image  (= PSF, all arrays identical here)
      col 3: Extended-source dirty image  θ_s = 2×HPBW
      col 4: Imaging residual  |dirty_image − true_source|  (θ_s = 2×HPBW)
    """
    print("── ES Plot A: PSF and dirty images ──")
    os.makedirs(ES_PLOT, exist_ok=True)

    col_titles = [
        "Full PSF  (dB)\nsidelobe geometry",
        "Main-lobe zoom\n(linear, 10×HPBW)",
        "Point-source\nresponse  (all same)",
        "Extended  (2×HPBW)\ndirty image",
        "Imaging residual\n|dirty − true|",
    ]
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=9, fontweight="bold")

    for gi, geom in enumerate(GEOMETRIES):
        l_arr, m_arr, B = _es_psf(cfgs[geom]["pos_enu"])
        hpbw = _es_hpbw(l_arr, B)
        dl   = l_arr[1] - l_arr[0]
        L, M = np.meshgrid(l_arr, m_arr, indexing="ij")
        circ = L**2 + M**2 <= 1.0

        B_dB   = 10 * np.log10(np.where(circ, np.clip(B, 1e-6, None), np.nan))
        B_lin  = np.where(circ, B, np.nan)

        # Extended source (2×HPBW, circular)
        I_two    = _es_source(l_arr, m_arr, 2.0 * hpbw)
        D_two    = _es_dirty_image(I_two, B)
        I_two_n  = I_two / (I_two.max() + 1e-30)
        residual = np.abs(D_two - I_two_n)
        D_two_m  = np.where(circ, D_two, np.nan)
        resid_m  = np.where(circ, residual, np.nan)

        # Zoom window = ±5×HPBW
        zoom_r   = 5.0 * hpbw
        zoom_idx = np.where(np.abs(l_arr) <= zoom_r)[0]
        if len(zoom_idx) < 3:
            zoom_idx = np.arange(len(l_arr))
        l_z = l_arr[zoom_idx]
        m_z = m_arr[zoom_idx]
        B_z = B[np.ix_(zoom_idx, zoom_idx)]
        L_z, M_z = np.meshgrid(l_z, m_z, indexing="ij")
        B_z_m = np.where(L_z**2 + M_z**2 <= 1.0, B_z, np.nan)

        # Point source: dirty image = PSF (pointed at source = beam peak)
        D_pt_m = B_lin

        eta_tiny = _es_eta(_es_source(l_arr, m_arr, 0.05 * hpbw), B)
        eta_two  = _es_eta(I_two, B)
        print(f"  {geom}: HPBW={np.degrees(hpbw)*60:.1f}′  "
              f"η_point≈{eta_tiny:.3f}  η_2×HPBW={eta_two:.3f}  "
              f"diff={eta_tiny - eta_two:.3f}")

        datasets = [
            (l_arr, m_arr, B_dB,   {"vmin": -30, "vmax": 0,   "cmap": "inferno"}),
            (l_z,   m_z,   B_z_m,  {"vmin": 0,   "vmax": 1,   "cmap": "inferno"}),
            (l_arr, m_arr, D_pt_m, {"vmin": 0,   "vmax": 1,   "cmap": "inferno"}),
            (l_arr, m_arr, D_two_m,{"vmin": 0,   "vmax": 1,   "cmap": "inferno"}),
            (l_arr, m_arr, resid_m,{"vmin": 0,   "vmax": 0.5, "cmap": "magma"}),
        ]
        for ci, (lx, mx, data, kwargs) in enumerate(datasets):
            ax = axes[gi, ci]
            im = ax.pcolormesh(lx, mx, data.T, shading="auto", **kwargs)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if ci == 0:
                ax.set_ylabel(f"{GEOM_LABEL[geom]}\nm", fontsize=9)
            else:
                ax.set_yticklabels([])
            if gi == len(GEOMETRIES) - 1:
                ax.set_xlabel("l", fontsize=8)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)

    fig.suptitle(
        f"PSF and Extended-Source Response  "
        f"({ES_CORE}×{ES_CORE} + {ES_DIST_KM} km  |  {ES_FREQ_MHZ} MHz)\n"
        "cols 2 & 3 show point vs extended source: geometry only matters for extended emission",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(ES_PLOT, "psf_and_dirty_images.png"), dpi=120)
    plt.close()
    print("  Saved psf_and_dirty_images.png")


def es_flux_recovery_vs_size(cfgs):
    """
    Plot B — beam-weighted flux recovery η vs source angular size in units of HPBW.
    Left panel: circular source.  Right panel: elongated source (aspect=ES_ASPECT).
    All arrays converge to η=1 for point sources; they diverge for extended sources.
    """
    print("── ES Plot B: Flux recovery vs source size ──")
    records = []
    psfs    = {}
    hpbws   = {}
    for geom in GEOMETRIES:
        l_arr, m_arr, B = _es_psf(cfgs[geom]["pos_enu"])
        psfs[geom]  = (l_arr, m_arr, B)
        hpbws[geom] = _es_hpbw(l_arr, B)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for geom in GEOMETRIES:
        l_arr, m_arr, B = psfs[geom]
        hpbw = hpbws[geom]
        eta_c, eta_e = [], []
        for sz in ES_SIZES:
            I_c = _es_source(l_arr, m_arr, sz * hpbw, aspect=1.0)
            I_e = _es_source(l_arr, m_arr, sz * hpbw, aspect=ES_ASPECT)
            ec = _es_eta(I_c, B); ee = _es_eta(I_e, B)
            eta_c.append(ec); eta_e.append(ee)
            records.append(dict(geom=geom, size_over_hpbw=sz,
                                eta_circular=ec, eta_elongated=ee))
        col = GEOM_COLOR[geom]
        ax1.plot(ES_SIZES, eta_c, lw=2, color=col, label=GEOM_LABEL[geom])
        ax2.plot(ES_SIZES, eta_e, lw=2, color=col, label=GEOM_LABEL[geom])

    for ax, title in [(ax1, "Circular source  (aspect = 1)"),
                      (ax2, f"Elongated source  (aspect = {ES_ASPECT})")]:
        ax.axvline(1.0, color="gray", ls="--", lw=1.2, label="θ_s = HPBW")
        ax.set_xlabel("Source angular size  θ_s / HPBW", fontsize=10)
        ax.set_ylabel("Beam-weighted flux recovery  η", fontsize=10)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(title, fontsize=10)
        # Annotate the convergence region
        ax.annotate("All arrays identical\n(point-source limit)",
                    xy=(0.07, 0.97), xytext=(0.07, 0.80),
                    arrowprops=dict(arrowstyle="->", color="dimgray"),
                    fontsize=7.5, ha="center", color="dimgray")

    fig.suptitle(
        f"Why Point-Source Detection Is Geometry-Independent\n"
        f"η = ∫ I_true·B / ∫ I_true  —  "
        f"{ES_CORE}×{ES_CORE} + {ES_DIST_KM} km  |  {ES_FREQ_MHZ} MHz",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(ES_PLOT, "flux_recovery_vs_size.png"), dpi=120)
    plt.close()

    df_rec = pd.DataFrame(records)
    df_rec.to_csv(os.path.join(ES_CSV, "flux_recovery.csv"), index=False)
    print("  Saved flux_recovery_vs_size.png, flux_recovery.csv")
    return df_rec


def es_orientation_test(cfgs):
    """
    Plot C — flux recovery η vs source position angle (PA) for elongated source.
    Source: θ_s = 1×HPBW, aspect = ES_ASPECT.
    Symmetric arrays (ring, line) → flat η(PA).
    Asymmetric arrays (y_shape, random) → modulated η(PA).
    Includes an explanation panel on what this means for science.
    """
    print("── ES Plot C: Orientation dependence ──")
    records = []

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig_p, ax_p = plt.subplots(1, 1, figsize=(7, 7),
                                subplot_kw={"projection": "polar"})

    for geom in GEOMETRIES:
        l_arr, m_arr, B = _es_psf(cfgs[geom]["pos_enu"])
        hpbw = _es_hpbw(l_arr, B)
        etas = []
        for pa in ES_PAS:
            I  = _es_source(l_arr, m_arr, 1.0 * hpbw, pa_deg=pa, aspect=ES_ASPECT)
            ec = _es_eta(I, B)
            etas.append(ec)
            records.append(dict(geom=geom, pa_deg=pa, eta=ec))
        etas = np.array(etas)
        eta_range = etas.max() - etas.min()
        col = GEOM_COLOR[geom]
        lbl = f"{GEOM_LABEL[geom]}  (Δη = {eta_range:.4f})"
        axes[0].plot(ES_PAS, etas, lw=2, color=col, label=lbl, marker="o", ms=4)
        # Polar: close the loop by reflecting 0→165 to 180→345
        pa_full  = np.radians(np.concatenate([ES_PAS, ES_PAS + 180]))
        eta_full = np.concatenate([etas, etas])
        ax_p.plot(pa_full, eta_full, lw=2, color=col, label=lbl)

    axes[0].set_xlabel("Source position angle PA [deg]", fontsize=10)
    axes[0].set_ylabel("Beam-weighted flux recovery  η", fontsize=10)
    axes[0].set_title(f"η vs Source Orientation  (θ_s = HPBW, aspect = {ES_ASPECT})",
                      fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 165)

    # Explanation panel
    ax2 = axes[1]
    ax2.axis("off")
    explanation = (
        "Why extended sources reveal beam asymmetry\n"
        "─────────────────────────────────────────\n\n"
        "POINT SOURCE  (θ_s → 0)\n"
        "  η = B(0,0) = 1  for every array\n"
        "  Only the beam PEAK matters\n"
        "  → All configs with same N are equivalent\n"
        "  → Geometry is invisible to detection\n\n"
        "EXTENDED SOURCE  (θ_s ≳ HPBW)\n"
        "  η = ∫ I_true(l,m) · B(l,m) dl dm\n"
        "      ─────────────────────────────\n"
        "          ∫ I_true(l,m) dl dm\n\n"
        "  Source now samples the FULL PSF shape\n"
        "  → Sidelobe asymmetry → η depends on PA\n"
        "  Symmetric beam  → η(PA) ≈ constant\n"
        "  Asymmetric beam → η varies with PA\n\n"
        "CONSEQUENCE FOR SCIENCE\n"
        "  • Asymmetric beam → orientation-biased\n"
        "    flux recovery → apparent source\n"
        "    asymmetry in dirty images\n"
        "  • Symmetric beam → consistent recovery\n"
        "    for any source orientation\n"
        "  • Harder deconvolution with asymmetric\n"
        "    PSF → larger CLEAN residuals"
    )
    ax2.text(0.04, 0.97, explanation, transform=ax2.transAxes,
             va="top", ha="left", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    fig.suptitle(
        f"Orientation-Dependent Flux Recovery  "
        f"({ES_CORE}×{ES_CORE} + {ES_DIST_KM} km  |  {ES_FREQ_MHZ} MHz)\n"
        f"Elongated source: θ_s = 1×HPBW, minor/major = {ES_ASPECT}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(ES_PLOT, "orientation_dependence.png"), dpi=120)
    plt.close(fig)

    ax_p.set_title(
        f"Flux Recovery vs Source PA  ({ES_FREQ_MHZ} MHz)\n"
        f"θ_s = HPBW, aspect = {ES_ASPECT}",
        pad=15, fontsize=10,
    )
    ax_p.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8)
    fig_p.tight_layout()
    fig_p.savefig(os.path.join(ES_PLOT, "orientation_polar.png"), dpi=120)
    plt.close(fig_p)

    df_or = pd.DataFrame(records)
    df_or.to_csv(os.path.join(ES_CSV, "orientation_dependence.csv"), index=False)
    print("  Saved orientation_dependence.png, orientation_polar.png, "
          "orientation_dependence.csv")
    return df_or


def es_sidelobe_sectors(cfgs):
    """
    Plot D — sidelobe power in 8 azimuthal sectors (polar bar charts).
    Symmetric arrays → equal power per sector.
    Asymmetric arrays → uneven distribution → direction-dependent contamination
    for extended sources whose emission extends into the sidelobe region.
    """
    print("── ES Plot D: Sidelobe power by azimuthal sector ──")

    n_sectors = 8
    sector_edges = np.linspace(0, 2 * np.pi, n_sectors + 1)
    sector_mids  = 0.5 * (sector_edges[:-1] + sector_edges[1:])
    sector_width = 2 * np.pi / n_sectors
    # CCW from East: E, NE, N, NW, W, SW, S, SE
    sector_labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]

    records = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 10),
                             subplot_kw={"projection": "polar"})
    axes = axes.ravel()

    for gi, geom in enumerate(GEOMETRIES):
        l_arr, m_arr, B = _es_psf(cfgs[geom]["pos_enu"])
        hpbw = _es_hpbw(l_arr, B)
        L, M = np.meshgrid(l_arr, m_arr, indexing="ij")
        r_arr     = np.sqrt(L**2 + M**2)
        theta_arr = np.arctan2(M, L) % (2 * np.pi)   # map [-π,π] → [0, 2π]

        # Sidelobe region: beyond main beam, within unit circle
        sl_mask = (r_arr > hpbw) & (r_arr <= 1.0)
        power   = []
        for si in range(n_sectors):
            ang_mask = (theta_arr >= sector_edges[si]) & (theta_arr < sector_edges[si + 1])
            sel = sl_mask & ang_mask
            pw  = float(B[sel].sum()) if sel.any() else 0.0
            power.append(pw)
            records.append(dict(geom=geom, sector=sector_labels[si], power=pw))

        power      = np.array(power)
        total      = power.sum()
        # Normalise so that 1.0 = perfectly uniform distribution
        power_norm = (power / (total / n_sectors + 1e-30)) if total > 0 else np.ones(n_sectors)
        asym_idx   = power_norm.std() / (power_norm.mean() + 1e-30)

        ax = axes[gi]
        bars = ax.bar(sector_mids, power_norm, width=sector_width * 0.82,
                      color=GEOM_COLOR[geom], alpha=0.75, align="center")
        ax.axhline(1.0, color="gray", lw=1.2, ls="--", alpha=0.8)
        ax.set_thetagrids(np.degrees(sector_mids), sector_labels, fontsize=9)
        ax.set_ylim(0, max(2.5, power_norm.max() * 1.15))
        ax.set_title(f"{GEOM_LABEL[geom]}", pad=15, fontsize=11, fontweight="bold")
        ax.text(0.5, -0.10,
                f"Asymmetry index σ/μ = {asym_idx:.3f}",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="darkred" if asym_idx > 0.05 else "navy")

    fig.suptitle(
        f"Sidelobe Power per Azimuthal Sector  "
        f"({ES_CORE}×{ES_CORE} + {ES_DIST_KM} km  |  {ES_FREQ_MHZ} MHz)\n"
        "Dashed circle = perfectly uniform distribution (σ/μ = 0)\n"
        "Extended source emission entering the sidelobe zone is recovered unevenly "
        "when σ/μ > 0",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(ES_PLOT, "sidelobe_sectors.png"), dpi=120)
    plt.close(fig)

    df_sl = pd.DataFrame(records)
    df_sl.to_csv(os.path.join(ES_CSV, "sidelobe_sectors.csv"), index=False)
    print("  Saved sidelobe_sectors.png, sidelobe_sectors.csv")
    return df_sl


def es_summary_comparison(cfgs, df_rec, df_or):
    """
    Plot E — two-panel summary.
    Left:  grouped bar chart — η(point) vs η(2×HPBW) for each geometry.
           Shows convergence at small sizes and divergence at large sizes.
    Right: grouped bar chart — orientation spread Δη for each geometry.
           Shows which arrays have PA-biased extended-source response.
    """
    print("── ES Plot E: Point vs extended summary ──")
    geom_order = GEOMETRIES
    x = np.arange(len(geom_order))
    w = 0.35
    colors = [GEOM_COLOR[g] for g in geom_order]

    # eta at ~0 (point) and ~2×HPBW (extended)
    pt_mask  = df_rec["size_over_hpbw"] < 0.08
    ext_mask = (df_rec["size_over_hpbw"] >= 1.8) & (df_rec["size_over_hpbw"] <= 2.3)

    eta_pt_c  = [df_rec[pt_mask  & (df_rec.geom == g)]["eta_circular"].mean()
                 for g in geom_order]
    eta_ex_c  = [df_rec[ext_mask & (df_rec.geom == g)]["eta_circular"].mean()
                 for g in geom_order]
    eta_pt_e  = [df_rec[pt_mask  & (df_rec.geom == g)]["eta_elongated"].mean()
                 for g in geom_order]
    eta_ex_e  = [df_rec[ext_mask & (df_rec.geom == g)]["eta_elongated"].mean()
                 for g in geom_order]

    # PA spread Δη per geometry (from orientation test)
    delta_eta = []
    for g in geom_order:
        sub = df_or[df_or.geom == g]["eta"]
        delta_eta.append(float(sub.max() - sub.min()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel — circular source
    ax1.bar(x - w/2, eta_pt_c, w, color=colors, alpha=0.45, hatch="//",
            edgecolor="k", linewidth=0.6, label="Point source  (θ→0)")
    ax1.bar(x + w/2, eta_ex_c, w, color=colors, alpha=0.90,
            edgecolor="k", linewidth=0.6, label="Extended  (θ_s = 2×HPBW)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([GEOM_LABEL[g] for g in geom_order], fontsize=10)
    ax1.set_ylabel("Beam-weighted flux recovery  η", fontsize=10)
    ax1.set_ylim(0, 1.15)
    ax1.axhline(1.0, color="gray", ls="--", lw=1)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Circular source: η does not depend on orientation", fontsize=9)
    for xi, (ep, ee) in enumerate(zip(eta_pt_c, eta_ex_c)):
        ax1.annotate(f"Δ={ep-ee:.3f}", xy=(xi, ee + 0.02), ha="center",
                     fontsize=7.5, color="darkred")

    # Right panel — orientation spread Δη
    bar_cols = [GEOM_COLOR[g] for g in geom_order]
    bars = ax2.bar(x, delta_eta, 0.55, color=bar_cols, alpha=0.85,
                   edgecolor="k", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels([GEOM_LABEL[g] for g in geom_order], fontsize=10)
    ax2.set_ylabel("Δη = max η − min η  over all source PA", fontsize=10)
    ax2.set_ylim(0, max(delta_eta) * 1.35 + 1e-5)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title(f"PA-orientation spread  (elongated source, aspect={ES_ASPECT})",
                  fontsize=9)
    for xi, de in enumerate(delta_eta):
        ax2.text(xi, de + max(delta_eta) * 0.03, f"{de:.4f}",
                 ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(0, color="gray", lw=1)
    ax2.text(0.02, 0.95,
             "Larger Δη → more orientation-biased\nflux recovery for extended sources",
             transform=ax2.transAxes, fontsize=8.5, va="top",
             color="darkred",
             bbox=dict(boxstyle="round", facecolor="mistyrose", alpha=0.7))

    fig.suptitle(
        f"Point Source vs Extended Source  —  Summary\n"
        f"({ES_CORE}×{ES_CORE} + {ES_DIST_KM} km  |  {ES_FREQ_MHZ} MHz  |  "
        f"θ_s = 2×HPBW for extended case)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(ES_PLOT, "point_vs_extended_summary.png"), dpi=120)
    plt.close(fig)
    print("  Saved point_vs_extended_summary.png")


def run_extended_source_analysis():
    """
    Compare symmetric (ring, line) vs asymmetric (y_shape, random) array responses
    to point sources and extended sources.
    Key result: all arrays are equivalent for point-source detection;
    geometry matters only when observing resolved/extended emission.
    """
    print("\n" + "=" * 70)
    print("  EXTENDED SOURCE ANALYSIS")
    print(f"  Core {ES_CORE}×{ES_CORE} + {ES_DIST_KM} km  |  {ES_FREQ_MHZ} MHz")
    print("=" * 70)
    os.makedirs(ES_PLOT, exist_ok=True)
    os.makedirs(ES_CSV,  exist_ok=True)

    cfgs   = es_build_configs()
    es_plot_psf_and_images(cfgs)
    df_rec = es_flux_recovery_vs_size(cfgs)
    df_or  = es_orientation_test(cfgs)
    es_sidelobe_sectors(cfgs)
    es_summary_comparison(cfgs, df_rec, df_or)

    # Print key results
    pt_mask  = df_rec["size_over_hpbw"] < 0.08
    ext_mask = (df_rec["size_over_hpbw"] >= 1.8) & (df_rec["size_over_hpbw"] <= 2.3)
    print("\n  Flux recovery summary (circular source):")
    for g in GEOMETRIES:
        ep = df_rec[pt_mask  & (df_rec.geom == g)]["eta_circular"].mean()
        ee = df_rec[ext_mask & (df_rec.geom == g)]["eta_circular"].mean()
        deta = df_or[df_or.geom == g]["eta"].max() - df_or[df_or.geom == g]["eta"].min()
        print(f"    {GEOM_LABEL[g]:8s}  η_point={ep:.4f}  η_2xHPBW={ee:.4f}  "
              f"Δη(PA)={deta:.4f}")

    print(f"\n  Outputs: {ES_PLOT}")


# ══════════════════════════════════════════════════════════════════════════════
# NEW CONFIGURATIONS  (a – f)
#
# Distances are measured from the farthest element at the edge of the core,
# not from the core centre.  For a core_n×core_n array with spacing d:
#   edge distance from centre = (core_n − 1) / 2 × d
#
# Outrigger sub-array sizes: 2×2 for 32×32 core, 4×4 for 128×128 core.
#
# Configurations:
#   a  Ring 32×32:  4 arms × 4 outriggers each at 250/500/750/1000 m from edge
#   b  Ring 32×32:  4 arms × 4 outriggers each at 1250/2500/3750/5000 m from edge
#   c  Ring 128×128:  4 arms × 4 outriggers at 250/500/750/1000 m from edge
#   d  Ring 128×128:  4 arms × 4 outriggers at 1250/2500/3750/5000 m from edge
#   e  Cross 32×32:   N short, E long, S intermediate, W intermediate
#   f  Cross 128×128: N short, E long, S intermediate, W intermediate
# ══════════════════════════════════════════════════════════════════════════════

NEW_OUT_NSIDE   = {32: 2, 128: 4}              # outrigger sub-array side
NEW_SHORT_DISTS = [250,  500,  750,  1000]     # short-arm distances from edge [m]
NEW_LONG_DISTS  = [1250, 2500, 3750, 5000]    # long-arm distances from edge [m]
NEW_INT_DISTS   = [750,  1500, 2250, 3000]    # intermediate-arm distances [m]

NEW_PLOT = os.path.join(GT_PLOT, "new_configs")
NEW_CSV  = os.path.join(GT_CSV,  "new_configs")
os.makedirs(NEW_PLOT, exist_ok=True)
os.makedirs(NEW_CSV,  exist_ok=True)

# Arm direction angles (radians): N, E, S, W
_ARM_N, _ARM_E, _ARM_S, _ARM_W = np.pi/2, 0.0, 3*np.pi/2, np.pi


def layout_ring_multi_arm(core_n, out_n, arm_dists_from_edge_m):
    """
    Ring: 4 symmetric arms (N/E/S/W), each arm containing outriggers
    at arm_dists_from_edge_m measured from the core edge.
    Total outriggers = 4 arms × len(arm_dists_from_edge_m).
    """
    edge   = core_edge_m(core_n)
    parts  = [rect_array_enu(core_n, D_SPACE)]
    centres = []
    for ang in [_ARM_N, _ARM_E, _ARM_S, _ARM_W]:
        for d in arm_dists_from_edge_m:
            dc = edge + d
            cx = dc * np.cos(ang)
            cy = dc * np.sin(ang)
            centres.append((cx, cy))
            parts.append(rect_array_enu(out_n, D_SPACE, cx, cy))
    return np.vstack(parts), centres


def layout_cross_asymmetric(core_n, out_n, arm_specs):
    """
    Asymmetric cross: 4 arms, each with its own outrigger spacing.

    arm_specs : list of (angle_rad, [distances_from_edge_m]) — one per arm.

    Configuration (e/f):
      N arm (π/2): short   — 250, 500, 750, 1000 m from edge
      E arm (0):   long    — 1250, 2500, 3750, 5000 m from edge
      S arm (3π/2): intermediate — 750, 1500, 2250, 3000 m from edge
      W arm (π):   intermediate — 750, 1500, 2250, 3000 m from edge
    """
    edge   = core_edge_m(core_n)
    parts  = [rect_array_enu(core_n, D_SPACE)]
    centres = []
    for ang, dists in arm_specs:
        for d in dists:
            dc = edge + d
            cx = dc * np.cos(ang)
            cy = dc * np.sin(ang)
            centres.append((cx, cy))
            parts.append(rect_array_enu(out_n, D_SPACE, cx, cy))
    return np.vstack(parts), centres


def build_new_configs():
    """
    Build 6 new configurations (a–f) and return as an ordered dict.

    Ring configs (a–d): each arm has 4 outriggers at different distances.
      All 4 arms share the same spacing → symmetric ring.
    Cross configs (e–f): asymmetric cross with 4 different arm spacings.
      N = short (max 1 km), E = long (max 5 km),
      S = W = intermediate (max 3 km).
    """
    print("\n── Build new configurations (a–f) ──")
    configs = {}

    # ── Ring configurations a, b, c, d ──────────────────────────────────────
    ring_specs = [
        ("a", 32,  NEW_SHORT_DISTS, "short", "250/500/750/1000 m from edge"),
        ("b", 32,  NEW_LONG_DISTS,  "long",  "1.25/2.5/3.75/5 km from edge"),
        ("c", 128, NEW_SHORT_DISTS, "short", "250/500/750/1000 m from edge"),
        ("d", 128, NEW_LONG_DISTS,  "long",  "1.25/2.5/3.75/5 km from edge"),
    ]
    for cfg_id, core_n, dists, case, dist_str in ring_specs:
        out_n = NEW_OUT_NSIDE[core_n]
        name  = f"ring_{cfg_id}_c{core_n}x{core_n}_out{out_n}x{out_n}_{case}"
        pos, centres = layout_ring_multi_arm(core_n, out_n, dists)
        B_max  = max_baseline_m(centres, core_n)
        N_core = core_n ** 2
        n_out  = len(centres)   # 4 arms × 4 outriggers = 16
        meta = dict(
            core_n=core_n, out_n=out_n, out_centres=centres,
            N_core=N_core, N_out_each=out_n**2, N_total=len(pos),
            geom="ring_new", B_max_m=B_max, cfg_id=cfg_id,
            label=(f"Config {cfg_id.upper()}: Ring {core_n}×{core_n}  "
                   f"+ {n_out}×{out_n}×{out_n} outriggers  ({dist_str})"),
        )
        configs[name] = dict(pos_enu=pos, meta=meta)
        print(f"  [{cfg_id.upper()}] {name}: "
              f"{len(pos)} elem, B_max={B_max/1e3:.2f} km, "
              f"{n_out} outriggers (4 arms × 4)")

    # ── Cross configurations e, f ────────────────────────────────────────────
    cross_arm_specs = [
        (_ARM_N, NEW_SHORT_DISTS),   # N arm: short
        (_ARM_E, NEW_LONG_DISTS),    # E arm: long
        (_ARM_S, NEW_INT_DISTS),     # S arm: intermediate
        (_ARM_W, NEW_INT_DISTS),     # W arm: intermediate
    ]
    arm_labels = {
        _ARM_N: f"N short ({NEW_SHORT_DISTS[-1]/1e3:.2g}km)",
        _ARM_E: f"E long  ({NEW_LONG_DISTS[-1]/1e3:.2g}km)",
        _ARM_S: f"S int   ({NEW_INT_DISTS[-1]/1e3:.2g}km)",
        _ARM_W: f"W int   ({NEW_INT_DISTS[-1]/1e3:.2g}km)",
    }
    for cfg_id, core_n in [("e", 32), ("f", 128)]:
        out_n = NEW_OUT_NSIDE[core_n]
        name  = f"cross_{cfg_id}_c{core_n}x{core_n}_out{out_n}x{out_n}_asym"
        pos, centres = layout_cross_asymmetric(core_n, out_n, cross_arm_specs)
        B_max  = max_baseline_m(centres, core_n)
        N_core = core_n ** 2
        n_out  = len(centres)   # 4 arms × 4 outriggers = 16
        meta = dict(
            core_n=core_n, out_n=out_n, out_centres=centres,
            N_core=N_core, N_out_each=out_n**2, N_total=len(pos),
            geom="cross_new", B_max_m=B_max, cfg_id=cfg_id,
            arm_labels=arm_labels, cross_arm_specs=cross_arm_specs,
            label=(f"Config {cfg_id.upper()}: Cross {core_n}×{core_n}  "
                   f"+ {n_out}×{out_n}×{out_n} outriggers  "
                   f"(N≤1km / E≤5km / S≤3km / W≤3km from edge)"),
        )
        configs[name] = dict(pos_enu=pos, meta=meta)
        print(f"  [{cfg_id.upper()}] {name}: "
              f"{len(pos)} elem, B_max={B_max/1e3:.2f} km, "
              f"{n_out} outriggers (4 arms × 4, asymmetric)")

    print(f"  Total new configurations: {len(configs)}")
    return configs


# ── Per-configuration individual plots ───────────────────────────────────────

def _new_config_savedir(name):
    d = os.path.join(NEW_PLOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def plot_schematic_layout(name, cfg, savedir=None, return_fig=False):
    """
    Generate a schematic layout diagram matching the reference design images:
      - White background
      - Core drawn as a labelled rectangle (steelblue)
      - Arm lines in steelblue
      - Outrigger sub-arrays as small red-bordered rectangles
      - Distance-from-edge labels in dark green
      - Sub-array size labels in red
    """
    if savedir is None:
        savedir = _new_config_savedir(name)
    meta   = cfg["meta"]
    core_n = meta["core_n"]
    out_n  = meta["out_n"]
    edge_m = core_edge_m(core_n)
    B_max  = meta["B_max_m"]

    use_km = B_max > 1500
    scale  = 1e3 if use_km else 1.0
    unit   = "km" if use_km else "m"

    def fmt(d_m):
        v = d_m / scale
        if use_km:
            return f"{v:.3g}km" if v != round(v) else f"{int(round(v))}km"
        return f"{int(round(d_m))}m"

    edge_s  = edge_m / scale

    # Group outrigger centres by arm direction (snap to 0/90/180/270°)
    arm_groups = {}
    for cx, cy in meta["out_centres"]:
        r   = np.hypot(cx, cy)
        ang = round(np.degrees(np.arctan2(cy, cx)) / 90) * 90
        ang = ang % 360
        arm_groups.setdefault(ang, []).append((cx / scale, cy / scale, r))

    max_reach_m = max(r for vals in arm_groups.values() for _, _, r in vals)
    max_s       = max_reach_m / scale

    # Display geometry
    out_box = max_s * 0.035           # outrigger box half-size in display units
    arm_lw  = 6
    fig_w   = 13 if use_km else 11
    fig_h   = 9  if use_km else 8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#e8e8e8")
    ax.set_facecolor("white")

    # Draw arms + outrigger boxes
    for ang_deg, olist in arm_groups.items():
        ang_rad = np.radians(ang_deg)
        dx, dy  = np.cos(ang_rad), np.sin(ang_rad)
        olist_s = sorted(olist, key=lambda x: x[2])   # nearest → farthest
        far_s   = olist_s[-1][2] / scale

        # Arm line: from core edge to farthest outrigger centre
        ax.plot([dx * edge_s, dx * far_s],
                [dy * edge_s, dy * far_s],
                color="steelblue", lw=arm_lw, zorder=2,
                solid_capstyle="butt")

        for cx_s, cy_s, r_m in olist_s:
            # Outrigger box
            rect = plt.Rectangle(
                (cx_s - out_box, cy_s - out_box),
                2 * out_box, 2 * out_box,
                lw=2, edgecolor="#cc0000", facecolor="white", zorder=5
            )
            ax.add_patch(rect)

            # Sub-array size label (red, beside box)
            ax.text(cx_s + out_box * 1.2, cy_s,
                    f"{out_n}×{out_n}",
                    ha="left", va="center", fontsize=8,
                    color="#cc0000", fontweight="bold", zorder=6)

            # Distance-from-edge label (green, above box)
            d_from_edge_m = r_m - edge_m
            ax.text(cx_s, cy_s + out_box * 1.5,
                    fmt(d_from_edge_m),
                    ha="center", va="bottom", fontsize=8,
                    color="darkgreen", fontweight="bold", zorder=6)

    # Core rectangle
    cr = plt.Rectangle((-edge_s, -edge_s), 2 * edge_s, 2 * edge_s,
                        lw=2, edgecolor="steelblue", facecolor="#d0e8f8",
                        zorder=4)
    ax.add_patch(cr)
    ax.text(0, 0, f"{core_n}×{core_n}",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="steelblue", zorder=7)

    lim_x = max_s * 1.35
    lim_y = max_s * 1.05
    ax.set_xlim(-lim_x, lim_x)
    ax.set_ylim(-lim_y, lim_y)
    ax.set_xlabel(f"East  [{unit}]", fontsize=11)
    ax.set_ylabel(f"North  [{unit}]", fontsize=11)
    ax.set_aspect("equal")
    ax.set_title(
        f"{meta['label']}\n"
        f"N_total = {meta['N_total']}  |  B_max = {meta['B_max_m']/1e3:.2f} km",
        fontsize=10, fontweight="bold"
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    fpath = os.path.join(savedir, "layout_schematic.png")
    plt.savefig(fpath, dpi=130, bbox_inches="tight", facecolor="white")
    if return_fig:
        return fig, ax
    plt.close()
    return fpath


def plot_new_layout(name, cfg):
    """Separate array geometry plot for one new configuration."""
    savedir = _new_config_savedir(name)
    pos    = cfg["pos_enu"]
    meta   = cfg["meta"]
    N_core = meta["N_core"]
    is_cross = meta["geom"] == "cross_new"

    scale  = 1e3 if meta["B_max_m"] > 2000 else 1.0
    unit   = "km" if scale == 1e3 else "m"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(pos[:N_core, 0] / scale, pos[:N_core, 1] / scale,
               s=0.5, c="steelblue", alpha=0.8,
               label=f"Core {meta['core_n']}×{meta['core_n']}")
    ax.scatter(pos[N_core:, 0] / scale, pos[N_core:, 1] / scale,
               s=6, c="tomato", alpha=0.9,
               label=f"Outrigger {meta['out_n']}×{meta['out_n']}")
    for cx, cy in meta["out_centres"]:
        ax.plot(cx / scale, cy / scale, "r+", ms=8, mew=1.5)

    if is_cross and "arm_labels" in meta:
        arm_dir_labels = {
            _ARM_N: ("N arm\n(short)",  (0,  1), "center", "bottom"),
            _ARM_E: ("E arm\n(long)",   (1,  0), "left",   "center"),
            _ARM_S: ("S arm\n(int.)",   (0, -1), "center", "top"),
            _ARM_W: ("W arm\n(int.)",   (-1, 0), "right",  "center"),
        }
        # Find max reach of each arm for annotation position
        for ang, dists in meta.get("cross_arm_specs", []):
            dx, dy = np.cos(ang), np.sin(ang)
            edge = core_edge_m(meta["core_n"])
            far  = (edge + max(dists)) / scale
            label_kw = arm_dir_labels.get(ang)
            if label_kw:
                lbl, _, ha, va = label_kw
                ax.annotate(lbl, xy=(dx * far * 1.05, dy * far * 1.05),
                            ha=ha, va=va, fontsize=8, color="darkred",
                            fontweight="bold")

    ax.set_xlabel(f"East [{unit}]"); ax.set_ylabel(f"North [{unit}]")
    ax.set_title(f"{meta['label']}\nN = {meta['N_total']}, "
                 f"B_max = {meta['B_max_m']/1e3:.2f} km",
                 fontsize=9)
    ax.legend(markerscale=5, fontsize=8)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "layout.png"), dpi=100)
    plt.close()

    # Also save the reference-image-style schematic
    plot_schematic_layout(name, cfg, savedir=savedir)


def plot_new_beam(name, cfg, af_data):
    """2-D + 1-D beam pattern at all 4 sub-bands for one new configuration."""
    savedir = _new_config_savedir(name)
    fig, axes = plt.subplots(4, 2, figsize=(11, 18))
    for ri, (bl, fc) in enumerate(zip(BAND_LABELS, BAND_CTR)):
        l, m, AF_dB, B_norm = af_data[bl]
        mid = len(m) // 2

        im = axes[ri, 0].pcolormesh(l, m, AF_dB, vmin=-30, vmax=0,
                                     cmap="inferno", shading="auto")
        plt.colorbar(im, ax=axes[ri, 0], label="|AF|² [dB]", fraction=0.046)
        cs = axes[ri, 0].contour(l, m, B_norm, levels=CONTOUR_LVL,
                                  colors=CONTOUR_COL, linewidths=[0.7, 0.9, 1.2])
        axes[ri, 0].clabel(cs, fmt={lv: f"{int(lv*100)}%"
                                    for lv in CONTOUR_LVL}, fontsize=7)
        axes[ri, 0].set_aspect("equal")
        axes[ri, 0].set_title(f"2-D beam  {bl} MHz", fontsize=9)
        axes[ri, 0].set_xlabel("l"); axes[ri, 0].set_ylabel("m")

        cut = 10 * np.log10(B_norm[mid, :] + 1e-20)
        axes[ri, 1].plot(l, cut, lw=1.5)
        axes[ri, 1].axhline(-3,  color="red",    ls="--", lw=0.9, label="−3 dB")
        axes[ri, 1].axhline(-10, color="orange",  ls=":",  lw=0.9, label="−10 dB")
        hpbw_rad = 2 * np.sqrt(float(np.sum(B_norm >= 0.5)) *
                                ((l[1]-l[0])**2) / np.pi)
        mtr = beam_metrics(B_norm, l, m, fc)
        axes[ri, 1].set_title(f"1-D cut (m=0)  {bl} MHz\n"
                               f"HPBW={np.degrees(hpbw_rad)*60:.1f}′  "
                               f"MSL={mtr['MSL_dB']:.1f} dB", fontsize=9)
        axes[ri, 1].set_xlabel("l"); axes[ri, 1].set_ylabel("Power [dB]")
        axes[ri, 1].set_ylim(-35, 2)
        axes[ri, 1].legend(fontsize=8); axes[ri, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Beam Pattern — {cfg['meta']['label']}", fontsize=10,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "beam_pattern.png"), dpi=100)
    plt.close()


def plot_new_uv(name, cfg):
    """Station-level UV coverage with multi-frequency synthesis for one config."""
    savedir = _new_config_savedir(name)
    sxy = np.array([(0.0, 0.0)] +
                   [(cx, cy) for cx, cy in cfg["meta"]["out_centres"]])
    fig, ax = plt.subplots(figsize=(7, 7))
    handles = []
    for bl, ch in SUBBAND_CH.items():
        u, v = _uv_from_stations(sxy, ch)
        sc = ax.scatter(u / 1e3, v / 1e3, s=5, alpha=0.7,
                        color=SUBBAND_CH_COLOR[bl], linewidths=0, label=f"{bl} MHz")
        handles.append(sc)
    ax.set_xlabel("u  [kλ]"); ax.set_ylabel("v  [kλ]")
    ax.set_title(f"UV Coverage — {cfg['meta']['label']}", fontsize=9)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.4, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.5)
    ax.grid(True, alpha=0.2)
    ax.legend(handles=handles, labels=[f"{bl} MHz" for bl in SUBBAND_CH],
              fontsize=8, markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "uv_coverage.png"), dpi=100)
    plt.close()


def plot_new_sensitivity(name, cfg, t_arr=None):
    """Sensitivity vs integration time plot for one new configuration."""
    savedir = _new_config_savedir(name)
    if t_arr is None:
        t_arr = np.logspace(-2, 4, 400)
    meta  = cfg["meta"]
    N     = meta["N_total"]
    B_max = meta["B_max_m"]

    fig, ax = plt.subplots(figsize=(8, 5))
    band_colors = {"1-5": "#E53935", "5-10": "#FB8C00",
                   "10-20": "#1E88E5", "20-40": "#8E24AA"}
    for bl, (f_lo, f_hi), fc in zip(BAND_LABELS, SUBBANDS, BAND_CTR):
        bw  = f_hi - f_lo
        sc  = confusion_limit_Jy(fc, B_max)
        st1 = sigma_thermal_Jy(N, fc, bw, 1.0)
        stot = np.sqrt((st1 / np.sqrt(t_arr)) ** 2 + sc ** 2)
        ax.loglog(t_arr, NSIGMA * stot * 1e3,
                  color=band_colors[bl], lw=2, label=f"{bl} MHz")
        ax.axhline(NSIGMA * sc * 1e3, color=band_colors[bl],
                   ls=":", lw=0.8, alpha=0.6)

    ax.set_xlabel("Integration time [h]", fontsize=10)
    ax.set_ylabel("5σ Total Sensitivity [mJy]", fontsize=10)
    ax.set_title(f"Sensitivity vs Integration Time\n{meta['label']}", fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
    ax.text(0.97, 0.03, "Dotted = confusion floor",
            transform=ax.transAxes, fontsize=8, ha="right", color="gray",
            style="italic")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "sensitivity.png"), dpi=100)
    plt.close()


def plot_new_detections(name, cfg, df_sens_cfg):
    """Bar chart + target list for one new configuration."""
    savedir = _new_config_savedir(name)
    meta = cfg["meta"]

    # Count feasible per band
    band_feas = {}
    feas_names = set()
    for bl in BAND_LABELS:
        sub = df_sens_cfg[(df_sens_cfg["frequency_band"] == bl) &
                          (df_sens_cfg["feasibility"] == "feasible")]
        unames = sub["target_name"].unique()
        band_feas[bl] = len(unames)
        feas_names.update(unames)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    band_colors = {"1-5": "#E53935", "5-10": "#FB8C00",
                   "10-20": "#1E88E5", "20-40": "#8E24AA"}

    # Left: bar chart
    ax1.bar(BAND_LABELS, [band_feas[b] for b in BAND_LABELS],
            color=[band_colors[b] for b in BAND_LABELS],
            edgecolor="white", lw=0.5)
    for i, bl in enumerate(BAND_LABELS):
        ax1.text(i, band_feas[bl] + 0.1, str(band_feas[bl]),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Frequency band [MHz]", fontsize=10)
    ax1.set_ylabel("Feasible targets (t < 100 h)", fontsize=10)
    ax1.set_title(f"Detection yield per band\nTotal unique: {len(feas_names)}",
                  fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Right: scatter of flux vs required t for feasible targets
    if len(df_sens_cfg):
        best_t = df_sens_cfg.groupby("target_name")["required_t_h"].min().reset_index()
        feas_df = df_sens_cfg[df_sens_cfg["target_name"].isin(feas_names)].drop_duplicates(
            subset=["target_name"])
        feas_merged = feas_df.merge(best_t, on="target_name")
        if len(feas_merged):
            ax2.scatter(feas_merged["target_flux_mJy"],
                        feas_merged["required_t_h_y"],
                        c="steelblue", alpha=0.7, edgecolors="k", lw=0.4)
            ax2.axhline(100, color="red", ls="--", lw=1.2, label="100 h threshold")
            ax2.set_xscale("log"); ax2.set_yscale("log")
            ax2.set_xlabel("Target flux [mJy]", fontsize=10)
            ax2.set_ylabel("Required integration time [h]", fontsize=10)
            ax2.set_title("Feasible targets: flux vs required time", fontsize=9)
            ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, which="both")
            for _, row in feas_merged.iterrows():
                if row["required_t_h_y"] < 100:
                    ax2.annotate(row["target_name"][:12],
                                 (row["target_flux_mJy"], row["required_t_h_y"]),
                                 fontsize=5.5, alpha=0.7,
                                 xytext=(2, 2), textcoords="offset points")

    fig.suptitle(f"Detection Results — {meta['label']}", fontsize=10,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, "detections.png"), dpi=100)
    plt.close()

    # Save target list
    if feas_names:
        with open(os.path.join(savedir, "feasible_targets.txt"), "w") as fh:
            fh.write(f"# Feasible targets (<100h) for {name}\n")
            for t in sorted(feas_names):
                fh.write(t + "\n")


def compute_new_sensitivity(configs, targets):
    """Sensitivity and detection analysis for all new configurations."""
    print("\n── Sensitivity: new configurations ──")
    rows = []
    for name, cfg in configs.items():
        meta  = cfg["meta"]
        N     = meta["N_total"]
        B_max = meta["B_max_m"]
        for bl, (f_lo, f_hi), fc in zip(BAND_LABELS, SUBBANDS, BAND_CTR):
            bw = f_hi - f_lo
            sc_Jy = confusion_limit_Jy(fc, B_max)
            st_Jy = sigma_thermal_Jy(N, fc, bw, 1.0)
            band_tgts = targets[
                (targets["frequency_MHz"] >= f_lo) &
                (targets["frequency_MHz"] <  f_hi)
            ]
            for _, trow in band_tgts.iterrows():
                t_h  = required_t_hours(trow["flux_mJy"], N, fc, bw, B_max)
                fc_s = classify_feasibility(t_h)
                rows.append(dict(
                    config_name=name,
                    label=meta["label"],
                    N_elements=N,
                    max_baseline_m=round(B_max, 1),
                    frequency_band=bl,
                    freq_centre_MHz=fc,
                    thermal_sens_Jy=st_Jy,
                    confusion_Jy=sc_Jy,
                    target_name=trow["Name"],
                    target_flux_mJy=trow["flux_mJy"],
                    target_freq_MHz=trow["frequency_MHz"],
                    required_t_h=t_h if np.isfinite(t_h) else 1e9,
                    feasibility=fc_s,
                ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(NEW_CSV, "new_configs_sensitivity.csv"), index=False)
    print(f"  Saved new_configs_sensitivity.csv  ({len(df)} rows)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SYMMETRIC CONFIGURATION COMPARISON: OLD vs CORRECTED
# Explains exactly what was wrong with the original "ring" configurations and
# quantifies how the corrected geometry changes the results.
# ══════════════════════════════════════════════════════════════════════════════

SYMCOMP_PLOT = os.path.join(GT_PLOT, "symmetric_comparison")
os.makedirs(SYMCOMP_PLOT, exist_ok=True)

_WHAT_WAS_WRONG = """
WHAT WAS WRONG WITH THE PREVIOUS SYMMETRIC CONFIGURATIONS
==========================================================

Original "ring" configs (ring_core32x32_1.0km etc.):
  × Distance measured from the CENTER of the core, not from the EDGE
  × Only ONE outrigger sub-array per arm (4 sub-arrays total)
  × Large sub-array size: 4×4=16 elem (32×32 core), 16×16=256 elem (128×128 core)
  × Result: all 4 baselines identical → single sidelobe ring, no UV-coverage spread

Corrected symmetric configs (A–D, matching reference images):
  ✓ Distance measured from the EDGE (farthest element) of the core
  ✓ FOUR outrigger sub-arrays per arm (16 sub-arrays total)
  ✓ Smaller sub-array size: 2×2=4 elem (32×32 core), 4×4=16 elem (128×128 core)
  ✓ Result: 4 distinct baseline lengths per arm → richer UV coverage,
            multiple sidelobe rings at different angular scales

Key numerical differences for the 32×32 case (approx. 1 km baseline):
  Old: 4 × 16-element clusters at 1000 m from centre → 64 outrigger elements, 1 baseline length
  New: 16 × 4-element clusters at 330/580/830/1080 m from centre → 64 outrigger elements, 4 baselines
  → Same total element count and collecting area, but richer UV coverage

Key numerical differences for the 128×128 case (approx. 5 km baseline):
  Old: 4 × 256-element clusters → 1024 outrigger elements, total N=17408
  New: 16 × 16-element clusters →  256 outrigger elements, total N=16640
  → Fewer outrigger elements, but 4× more distinct baselines per arm
"""


def run_symmetric_comparison(old_configs, af_store_old, new_cfgs, af_new,
                              df_sens_old, df_sens_new, targets):
    """
    Comprehensive comparison of old (incorrect) vs new (corrected) symmetric configs.

    Outputs
    -------
    symmetric_comparison/schematic_comparison_32x32.png  — side-by-side layouts
    symmetric_comparison/schematic_comparison_128x128.png
    symmetric_comparison/beam_comparison_30MHz.png       — AF side by side
    symmetric_comparison/uv_comparison.png               — UV coverage
    symmetric_comparison/sensitivity_comparison.png      — σ vs t
    symmetric_comparison/detection_comparison.png        — detection yield
    symmetric_comparison/what_changed.txt                — text summary
    """
    print("\n" + "=" * 72)
    print("  SYMMETRIC CONFIGURATION COMPARISON: OLD vs CORRECTED")
    print("=" * 72)
    print(_WHAT_WAS_WRONG)

    # Save text summary
    with open(os.path.join(SYMCOMP_PLOT, "what_changed.txt"), "w") as fh:
        fh.write(_WHAT_WAS_WRONG)

    # Pairs to compare (old_key, new_key, label)
    pairs = [
        ("ring_core32x32_1.0km",    "ring_a_c32x32_out2x2_short",
         "32×32 core, ~1 km baseline\n(short-distance case)"),
        ("ring_core32x32_5.0km",    "ring_b_c32x32_out2x2_long",
         "32×32 core, ~5 km baseline\n(long-distance case)"),
        ("ring_core128x128_1.0km",  "ring_c_c128x128_out4x4_short",
         "128×128 core, ~1 km baseline\n(short-distance case)"),
        ("ring_core128x128_5.0km",  "ring_d_c128x128_out4x4_long",
         "128×128 core, ~5 km baseline\n(long-distance case)"),
    ]

    # ── A: Schematic layout comparison ───────────────────────────────────────
    for core_n, pair_subset in [(32, pairs[:2]), (128, pairs[2:])]:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f"Symmetric Configuration: Old (incorrect) vs Corrected  "
                     f"—  {core_n}×{core_n} core\n"
                     f"Left: old design (1 outrigger/arm, from centre)  |  "
                     f"Right: corrected design (4 outriggers/arm, from edge)",
                     fontsize=11, fontweight="bold")

        for ri, (ok, nk, lbl) in enumerate(pair_subset):
            # ── Old layout ──
            ax_old = axes[ri, 0]
            old_pos  = old_configs[ok]["pos_enu"]
            old_meta = old_configs[ok]["meta"]
            old_Nc   = old_meta["N_core"]
            old_dist = old_meta["dist_km"] * 1e3   # from centre [m]
            out_n_old = old_meta["out_n"]

            sc = 1e3 if old_meta["B_max_m"] > 1500 else 1.0
            unit_old = "km" if sc == 1e3 else "m"
            ax_old.scatter(old_pos[:old_Nc, 0]/sc, old_pos[:old_Nc, 1]/sc,
                           s=0.5 if core_n == 128 else 2, c="steelblue", alpha=0.6)
            ax_old.scatter(old_pos[old_Nc:, 0]/sc, old_pos[old_Nc:, 1]/sc,
                           s=6, c="tomato", alpha=0.9)
            for cx, cy in old_meta["out_centres"]:
                ax_old.plot(cx/sc, cy/sc, "r+", ms=10, mew=2)
            ax_old.set_aspect("equal"); ax_old.grid(True, alpha=0.2)
            ax_old.set_xlabel(f"East [{unit_old}]", fontsize=9)
            ax_old.set_ylabel(f"North [{unit_old}]", fontsize=9)
            ax_old.set_title(f"OLD: {ok}\n"
                              f"N={old_meta['N_total']}, "
                              f"4×{out_n_old}×{out_n_old} outriggers, "
                              f"dist from CENTRE={old_meta['dist_km']} km",
                              fontsize=8, color="tomato")

            # ── New layout ──
            ax_new = axes[ri, 1]
            new_pos  = new_cfgs[nk]["pos_enu"]
            new_meta = new_cfgs[nk]["meta"]
            new_Nc   = new_meta["N_core"]
            out_n_new = new_meta["out_n"]

            sc2    = 1e3 if new_meta["B_max_m"] > 1500 else 1.0
            unit_n = "km" if sc2 == 1e3 else "m"
            ax_new.scatter(new_pos[:new_Nc, 0]/sc2, new_pos[:new_Nc, 1]/sc2,
                           s=0.5 if core_n == 128 else 2, c="steelblue", alpha=0.6)
            ax_new.scatter(new_pos[new_Nc:, 0]/sc2, new_pos[new_Nc:, 1]/sc2,
                           s=6, c="tomato", alpha=0.9)
            for cx, cy in new_meta["out_centres"]:
                ax_new.plot(cx/sc2, cy/sc2, "r+", ms=6, mew=1.5)
            ax_new.set_aspect("equal"); ax_new.grid(True, alpha=0.2)
            ax_new.set_xlabel(f"East [{unit_n}]", fontsize=9)
            ax_new.set_ylabel(f"North [{unit_n}]", fontsize=9)
            ax_new.set_title(f"CORRECTED: Config {new_meta['cfg_id'].upper()}\n"
                              f"N={new_meta['N_total']}, "
                              f"16×{out_n_new}×{out_n_new} outriggers, "
                              f"dist from EDGE",
                              fontsize=8, color="steelblue")

        plt.tight_layout()
        fname = f"schematic_comparison_{core_n}x{core_n}.png"
        fig.savefig(os.path.join(SYMCOMP_PLOT, fname), dpi=120)
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── B: Beam pattern comparison at 30 MHz ─────────────────────────────────
    ref_bl = "20-40"
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Beam Pattern Comparison at 30 MHz: Old vs Corrected Symmetric Configs\n"
                 "Rows: 32×32 (top), 128×128 (bottom)  |  "
                 "Cols: old@1km | new@~1km | old@5km | new@~5km",
                 fontsize=11, fontweight="bold")
    old_new_order = [
        ("ring_core32x32_1.0km",   "ring_a_c32x32_out2x2_short"),
        ("ring_core32x32_5.0km",   "ring_b_c32x32_out2x2_long"),
        ("ring_core128x128_1.0km", "ring_c_c128x128_out4x4_short"),
        ("ring_core128x128_5.0km", "ring_d_c128x128_out4x4_long"),
    ]
    col_titles = ["Old 1km", "New ~1km", "Old 5km", "New ~5km"]
    for ri, core_n_s in enumerate([32, 128]):
        for ci, (ok, nk) in enumerate([(old_new_order[ri*2][0], old_new_order[ri*2][1]),
                                        (old_new_order[ri*2+1][0], old_new_order[ri*2+1][1])]):
            for si, (key, store, clr) in enumerate([(ok, af_store_old, "old"),
                                                     (nk, af_new,       "new")]):
                ax  = axes[ri, ci * 2 + si]
                l, m, AF_dB, B_n = store[key][ref_bl]
                ax.pcolormesh(l, m, AF_dB, vmin=-30, vmax=0,
                              cmap="inferno", shading="auto")
                ax.set_aspect("equal"); ax.tick_params(labelsize=6)
                border = "tomato" if clr == "old" else "steelblue"
                for spine in ax.spines.values():
                    spine.set_edgecolor(border); spine.set_linewidth(2)
                tag = "OLD" if clr == "old" else "CORR"
                ax.set_title(f"{tag}: {key.split('_')[0]}\n"
                             f"N={store[key][ref_bl][0].shape[0]}",
                             fontsize=7,
                             color="tomato" if clr == "old" else "steelblue")
        # Column titles
        if ri == 0:
            for ci2, ct in enumerate(col_titles):
                axes[ri, ci2].text(0.5, 1.15, ct,
                                   transform=axes[ri, ci2].transAxes,
                                   ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(os.path.join(SYMCOMP_PLOT, "beam_comparison_30MHz.png"), dpi=120)
    plt.close(fig)
    print("  Saved beam_comparison_30MHz.png")

    # ── C: UV coverage comparison ─────────────────────────────────────────────
    comp_pairs_uv = [
        ("ring_core128x128_1.0km", old_configs, "Old: 128×128 + 1km (centre)"),
        ("ring_c_c128x128_out4x4_short", new_cfgs, "New: Config C (4/arm, edge)"),
        ("ring_core128x128_5.0km", old_configs, "Old: 128×128 + 5km (centre)"),
        ("ring_d_c128x128_out4x4_long",  new_cfgs, "New: Config D (4/arm, edge)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, (key, store, lbl) in zip(axes, comp_pairs_uv):
        sxy = np.array([(0.0, 0.0)] +
                       [(cx, cy) for cx, cy in store[key]["meta"]["out_centres"]])
        for bl, ch in SUBBAND_CH.items():
            u, v = _uv_from_stations(sxy, ch)
            ax.scatter(u/1e3, v/1e3, s=4, alpha=0.7,
                       color=SUBBAND_CH_COLOR[bl], linewidths=0,
                       label=f"{bl} MHz")
        ax.set_xlabel("u [kλ]"); ax.set_ylabel("v [kλ]")
        ax.set_title(lbl, fontsize=8, fontweight="bold")
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
        ax.grid(True, alpha=0.2)
        n_stations = len(sxy)
        ax.text(0.02, 0.98, f"{n_stations} stations",
                transform=ax.transAxes, fontsize=8, va="top")
    axes[0].legend(fontsize=7, markerscale=3)
    fig.suptitle("UV Coverage: Old vs Corrected Symmetric Configs  (128×128 core)\n"
                 "Old = 5 stations (core + 4 outriggers)  |  "
                 "New = 17 stations (core + 16 outriggers)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(SYMCOMP_PLOT, "uv_comparison.png"), dpi=120)
    plt.close(fig)
    print("  Saved uv_comparison.png")

    # ── D: Sensitivity vs time comparison ─────────────────────────────────────
    t_arr = np.logspace(-2, 4, 400)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    comp_pairs_sens = [
        [("ring_core32x32_1.0km", old_configs,
          "Old 32×32 @1km", "tomato", "-"),
         ("ring_a_c32x32_out2x2_short", new_cfgs,
          "New 32×32 (A)", "steelblue", "-"),
         ("ring_core32x32_5.0km", old_configs,
          "Old 32×32 @5km", "tomato", "--"),
         ("ring_b_c32x32_out2x2_long", new_cfgs,
          "New 32×32 (B)", "steelblue", "--")],
        [("ring_core128x128_1.0km", old_configs,
          "Old 128×128 @1km", "tomato", "-"),
         ("ring_c_c128x128_out4x4_short", new_cfgs,
          "New 128×128 (C)", "steelblue", "-"),
         ("ring_core128x128_5.0km", old_configs,
          "Old 128×128 @5km", "tomato", "--"),
         ("ring_d_c128x128_out4x4_long", new_cfgs,
          "New 128×128 (D)", "steelblue", "--")],
    ]
    for ax, cset, core_n_s in zip(axes, comp_pairs_sens, [32, 128]):
        for key, store, lbl, col, ls in cset:
            meta  = store[key]["meta"]
            N     = meta["N_total"]
            B_max = meta["B_max_m"]
            sc    = confusion_limit_Jy(REF_FREQ, B_max)
            st1   = sigma_thermal_Jy(N, REF_FREQ, REF_BW, 1.0)
            stot  = np.sqrt((st1 / np.sqrt(t_arr)) ** 2 + sc ** 2)
            ax.loglog(t_arr, NSIGMA * stot * 1e3, color=col, ls=ls, lw=2,
                      label=lbl)
        ax.set_xlabel("Integration time [h]", fontsize=10)
        ax.set_title(f"{core_n_s}×{core_n_s} core — Old vs Corrected",
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("5σ Sensitivity [mJy]  (30 MHz, 20 MHz BW)", fontsize=10)
    fig.suptitle("Sensitivity vs Integration Time: Old vs Corrected Symmetric Configs",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(SYMCOMP_PLOT, "sensitivity_comparison.png"), dpi=120)
    plt.close(fig)
    print("  Saved sensitivity_comparison.png")

    # ── E: Detection yield comparison ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    comp_pairs_det = [
        ("ring_core32x32_1.0km",   "ring_a_c32x32_out2x2_short"),
        ("ring_core32x32_5.0km",   "ring_b_c32x32_out2x2_long"),
        ("ring_core128x128_1.0km", "ring_c_c128x128_out4x4_short"),
        ("ring_core128x128_5.0km", "ring_d_c128x128_out4x4_long"),
    ]
    xlabels     = ["32×32\n@1km", "32×32\n@5km", "128×128\n@1km", "128×128\n@5km"]
    x_pos       = np.arange(len(comp_pairs_det))
    bar_w       = 0.35

    n_feas_old, n_feas_new = [], []
    for ok, nk in comp_pairs_det:
        sub_o = df_sens_old[df_sens_old["config_name"] == ok]
        best_o = sub_o.groupby("target_name")["required_t_h"].min()
        n_feas_old.append(int((best_o < 100).sum()))

        sub_n = df_sens_new[df_sens_new["config_name"] == nk]
        best_n = sub_n.groupby("target_name")["required_t_h"].min()
        n_feas_new.append(int((best_n < 100).sum()))

    for ax, (nf_o, nf_n) in zip(axes, [(n_feas_old[:2], n_feas_new[:2]),
                                         (n_feas_old[2:], n_feas_new[2:])]):
        xl  = xlabels[:2] if ax is axes[0] else xlabels[2:]
        xp  = np.arange(len(xl))
        b1  = ax.bar(xp - bar_w/2, nf_o, bar_w, color="tomato", alpha=0.85,
                     edgecolor="white", label="Old (1 outrigger/arm, from centre)")
        b2  = ax.bar(xp + bar_w/2, nf_n, bar_w, color="steelblue", alpha=0.85,
                     edgecolor="white", label="Corrected (4 outriggers/arm, from edge)")
        for b, n in zip(list(b1) + list(b2), list(nf_o) + list(nf_n)):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                    str(n), ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(xp)
        ax.set_xticklabels(xl, fontsize=10)
        ax.set_ylabel("Feasible detections (t < 100 h)", fontsize=10)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        core_s = "32×32" if ax is axes[0] else "128×128"
        ax.set_title(f"{core_s} core — Detection Yield Comparison", fontsize=10)

    fig.suptitle("Detection Yield: Old vs Corrected Symmetric Configurations\n"
                 "Red = old design  |  Blue = corrected design",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(SYMCOMP_PLOT, "detection_comparison.png"), dpi=120)
    plt.close(fig)
    print("  Saved detection_comparison.png")

    # ── F: Quantitative summary table ─────────────────────────────────────────
    print("\n  ┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("  │  QUANTITATIVE COMPARISON: OLD vs CORRECTED SYMMETRIC CONFIGS                   │")
    print("  ├─────────────┬───────────┬──────────┬───────────┬──────────┬────────────────────┤")
    print("  │ Config pair │ N (old)   │ N (new)  │ B_max(old)│ B_max(n) │ n_det<100h(old/new)│")
    print("  ├─────────────┼───────────┼──────────┼───────────┼──────────┼────────────────────┤")
    for (ok, nk, lbl), nfo, nfn in zip(pairs, n_feas_old, n_feas_new):
        om = old_configs[ok]["meta"]
        nm = new_cfgs[nk]["meta"]
        short = lbl.split("\n")[0]
        print(f"  │ {short:12s}│ {om['N_total']:9d}│ {nm['N_total']:8d}│"
              f" {om['B_max_m']/1e3:8.2f} km│ {nm['B_max_m']/1e3:7.2f} km│"
              f" {nfo:9d} / {nfn:8d}     │")
    print("  └─────────────┴───────────┴──────────┴───────────┴──────────┴────────────────────┘")

    print("\n  KEY CHANGES IN RESULTS:")
    print("  • 32×32 configs: N_total unchanged (1088); B_max slightly larger (edge vs centre);")
    print("    4× more distinct baselines → richer UV coverage")
    print("  • 128×128 configs: N_total smaller (16640 vs 17408); smaller outrigger sub-arrays")
    print("    (4×4=16 vs 16×16=256 elem each) but 4× more baselines; comparable sensitivity")
    print("  • Detection yield: numbers reflect the corrected A_eff=6.28 m² and edge-based distances")
    print(f"  Outputs saved to: {SYMCOMP_PLOT}")


def run_new_configs_analysis(targets):
    """
    Full analysis pipeline for new configurations a–f:
    1. Build configurations
    2. Compute AF for all configs × 4 bands
    3. Produce individual plots (layout, beam, UV, sensitivity, detections) per config
    4. Compute sensitivity and detection yield
    5. Print summary comparison table
    """
    print("\n" + "=" * 72)
    print("  NEW CONFIGURATIONS ANALYSIS  (a–f)")
    print("=" * 72)

    configs = build_new_configs()

    # Compute AF for every config × band
    print("\n── Computing AF for new configurations ──")
    af_store_new = {}
    for name, cfg in configs.items():
        af_store_new[name] = {}
        for bl, fc in zip(BAND_LABELS, BAND_CTR):
            l, m, AF_dB, B_norm = compute_af(cfg["pos_enu"], cfg["meta"], fc, N_GRID)
            af_store_new[name][bl] = (l, m, AF_dB, B_norm)

    # Per-configuration individual plots
    print("\n── Per-configuration plots ──")
    for name, cfg in configs.items():
        plot_new_layout(name, cfg)
        plot_new_beam(name, cfg, af_store_new[name])
        plot_new_uv(name, cfg)
        plot_new_sensitivity(name, cfg)
        print(f"  Layout/beam/UV/sensitivity plots → {name}/")

    # Sensitivity + detection
    df_sens_new = compute_new_sensitivity(configs, targets)

    for name, cfg in configs.items():
        df_cfg = df_sens_new[df_sens_new["config_name"] == name]
        plot_new_detections(name, cfg, df_cfg)
        print(f"  Detection plot → {name}/detections.png")

    # Summary table
    print("\n── New configurations summary ──")
    print(f"  {'Config':50s}  {'N_elem':>7}  {'B_max km':>9}  "
          f"{'n_det(<100h)':>12}  {'n_det(100-1000h)':>16}")
    print("  " + "-" * 100)

    summary_rows = []
    for name, cfg in configs.items():
        meta  = cfg["meta"]
        N     = meta["N_total"]
        B_max = meta["B_max_m"]
        df_cfg = df_sens_new[df_sens_new["config_name"] == name]
        best_t = df_cfg.groupby("target_name")["required_t_h"].min()
        n_feas = int((best_t < 100).sum())
        n_asp  = int(((best_t >= 100) & (best_t < 1000)).sum())
        feasible_names = list(best_t[best_t < 100].index)
        print(f"  {meta['label']:50s}  {N:>7d}  {B_max/1e3:>9.2f}  "
              f"{n_feas:>12d}  {n_asp:>16d}")
        summary_rows.append(dict(
            config_name=name, label=meta["label"],
            N_elements=N, B_max_km=round(B_max/1e3, 3),
            n_feasible=n_feas, n_aspirational=n_asp,
            feasible_targets="; ".join(feasible_names),
        ))

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(NEW_CSV, "new_configs_summary.csv"), index=False)
    print(f"\n  Saved new_configs_summary.csv")
    return configs, af_store_new, df_sens_new, df_sum


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS 1: WEIGHTED BEAMFORMING (HANNING + TAYLOR TAPER)
# ══════════════════════════════════════════════════════════════════════════════
ADV_PLOT = os.path.join(GT_PLOT, "advanced")
ADV_CSV  = os.path.join(GT_CSV,  "advanced")
os.makedirs(ADV_PLOT, exist_ok=True)
os.makedirs(ADV_CSV,  exist_ok=True)


def _radial_weights(pos_enu, taper="hanning", n_side_bar=None, sll_db=-20.0):
    """
    Compute scalar weights for each element based on its distance from
    the array centre.  Taper is applied to the 2-D radial coordinate,
    normalised to [0, 1] within the core footprint.

    taper : 'uniform' | 'hanning' | 'taylor'
    n_side_bar : only needed for 'taylor'; number of equal-sidelobe bars
                 (ignored if scipy.signal.taylor is unavailable)
    """
    r     = np.sqrt(pos_enu[:, 0]**2 + pos_enu[:, 1]**2)
    r_max = r.max()
    r_n   = r / (r_max + 1e-12)   # normalised [0, 1]

    if taper == "uniform":
        return np.ones(len(pos_enu))

    if taper == "hanning":
        return 0.5 + 0.5 * np.cos(np.pi * r_n)   # Hanning (von Hann) taper

    if taper == "taylor":
        try:
            from scipy.signal.windows import taylor as scipy_taylor
            n_pts  = 1024
            win_1d = scipy_taylor(n_pts, nbar=n_side_bar or 5, sll=abs(sll_db))
            # Map normalised radius [0, 1] → index [0, n_pts-1]
            idx = np.clip((r_n * (n_pts - 1)).astype(int), 0, n_pts - 1)
            return win_1d[idx]
        except ImportError:
            # Fallback: analytic Taylor approximation
            return (1.0 - r_n ** 2) ** 2   # Blackman-like

    return np.ones(len(pos_enu))


def _weighted_af(pos_enu, weights, freq_MHz, n_grid=256):
    """Compute normalised |AF|² using per-element weights."""
    k     = 2 * np.pi * freq_MHz * 1e6 / C
    l_arr = np.linspace(-1, 1, n_grid)
    m_arr = np.linspace(-1, 1, n_grid)
    Phi_x = np.exp(1j * k * np.outer(pos_enu[:, 0], l_arr))
    Phi_y = np.exp(1j * k * np.outer(pos_enu[:, 1], m_arr))
    AF    = (weights[:, None] * Phi_x).T @ Phi_y
    B     = np.abs(AF) ** 2
    B_n   = B / (B.max() + 1e-30)
    L, M  = np.meshgrid(l_arr, m_arr, indexing="ij")
    B_n[L**2 + M**2 > 1] = np.nan
    return l_arr, m_arr, B_n


def _msl_hpbw_from_norm(B_n, l_arr, m_arr, freq_MHz):
    """Return (MSL_dB, HPBW_arcmin) from a normalised beam."""
    L, M   = np.meshgrid(l_arr, m_arr, indexing="ij")
    inside = (L**2 + M**2) <= 1.0
    main   = inside & (B_n >= 0.5)
    side   = inside & ~main
    msl    = float(10 * np.log10(np.nanmax(B_n[side]) + 1e-20)) if side.any() else -999.0
    dl     = l_arr[1] - l_arr[0]
    omega  = float(np.sum(np.where(inside, B_n, 0.0)) * dl * dl)
    hpbw   = np.degrees(2 * np.sqrt(max(omega, 1e-20) / np.pi)) * 60.0
    return msl, hpbw


def run_weighted_beamforming(new_configs):
    """
    Apply Hanning and Taylor (n=5, SLL=-20 dB) tapers to the cross config
    (128×128 core) and compare with uniform weighting.

    For each taper at 30 MHz:
    – 2-D beam pattern
    – 1-D cut (m = 0)
    – MSL and HPBW metrics

    Explains the trade-off between sidelobe suppression and main-lobe broadening.
    """
    print("\n" + "=" * 72)
    print("  ADVANCED ANALYSIS 1: WEIGHTED BEAMFORMING")
    print("=" * 72)

    cfg_key = "cross_new_c128x128_out4x4_4arm"
    if cfg_key not in new_configs:
        cfg_key = next(k for k in new_configs if "cross" in k and "128" in k)
    cfg   = new_configs[cfg_key]
    pos   = cfg["pos_enu"]
    meta  = cfg["meta"]
    fc    = 30.0
    n_g   = 256

    tapers = [
        ("Uniform",       "uniform", None),
        ("Hanning taper", "hanning", None),
        ("Taylor n=5 (−20 dB)", "taylor", 5),
    ]
    taper_colors = {"Uniform": "steelblue",
                    "Hanning taper": "tomato",
                    "Taylor n=5 (−20 dB)": "seagreen"}

    fig, axes = plt.subplots(len(tapers), 2, figsize=(13, 5 * len(tapers)))
    rows = []
    for ri, (label, ttype, nbar) in enumerate(tapers):
        w     = _radial_weights(pos, taper=ttype, n_side_bar=nbar)
        l, m, B_n = _weighted_af(pos, w.astype(complex), fc, n_g)
        msl, hpbw = _msl_hpbw_from_norm(B_n, l, m, fc)
        rows.append(dict(taper=label, MSL_dB=msl, HPBW_arcmin=hpbw))
        print(f"  {label:28s}  MSL={msl:+.1f} dB  HPBW={hpbw:.1f}′")

        B_dB = 10 * np.log10(B_n.T + 1e-20)
        im   = axes[ri, 0].pcolormesh(l, m, B_dB, vmin=-30, vmax=0,
                                       cmap="inferno", shading="auto")
        plt.colorbar(im, ax=axes[ri, 0], label="dB", fraction=0.046)
        axes[ri, 0].set_title(f"{label}  —  2-D beam at 30 MHz", fontsize=9)
        axes[ri, 0].set_xlabel("l"); axes[ri, 0].set_ylabel("m")
        axes[ri, 0].set_aspect("equal")

        mid  = n_g // 2
        cut  = 10 * np.log10(B_n[:, mid] + 1e-20)
        axes[ri, 1].plot(l, cut, color=taper_colors.get(label, "k"), lw=2)
        axes[ri, 1].axhline(-3,  color="red",    ls="--", lw=0.9, label="−3 dB")
        axes[ri, 1].axhline(-20, color="purple",  ls=":",  lw=1.0, label="−20 dB")
        axes[ri, 1].set_ylim(-35, 2)
        axes[ri, 1].set_title(f"1-D cut  HPBW={hpbw:.1f}′  MSL={msl:.1f} dB",
                               fontsize=9)
        axes[ri, 1].set_xlabel("l"); axes[ri, 1].set_ylabel("Power [dB]")
        axes[ri, 1].legend(fontsize=8); axes[ri, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Weighted Beamforming — {meta['label']}\n"
        "Taper reduces sidelobes at the cost of a broader main lobe",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ADV_PLOT, "adv1_weighted_beamforming.png"), dpi=120)
    plt.close()

    # 1-D overlay comparison
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for label, ttype, nbar in tapers:
        w      = _radial_weights(pos, taper=ttype, n_side_bar=nbar)
        l, m, B_n = _weighted_af(pos, w.astype(complex), fc, n_g)
        mid    = n_g // 2
        cut    = 10 * np.log10(B_n[:, mid] + 1e-20)
        msl, hpbw = _msl_hpbw_from_norm(B_n, l, m, fc)
        ax2.plot(l, cut, lw=2, color=taper_colors.get(label, "k"),
                 label=f"{label}  (MSL={msl:.1f} dB, HPBW={hpbw:.1f}′)")
    ax2.axhline(-3,  color="red", ls="--", lw=1, label="−3 dB")
    ax2.axhline(-20, color="k",   ls=":",  lw=1, label="−20 dB")
    ax2.set_ylim(-35, 2)
    ax2.set_xlabel("l (East direction cosine)", fontsize=10)
    ax2.set_ylabel("Normalised power [dB]", fontsize=10)
    ax2.set_title("Taper comparison — 1-D beam cut at 30 MHz", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ADV_PLOT, "adv1_taper_comparison_1D.png"), dpi=120)
    plt.close()

    df_taper = pd.DataFrame(rows)
    df_taper.to_csv(os.path.join(ADV_CSV, "adv1_taper_metrics.csv"), index=False)
    print("  Saved adv1_weighted_beamforming.png, adv1_taper_comparison_1D.png")
    print("  Saved adv1_taper_metrics.csv")
    print("\n  Trade-off summary:")
    print(f"  {'Taper':28s}  {'MSL [dB]':>10s}  {'HPBW [arcmin]':>15s}")
    for r in rows:
        print(f"  {r['taper']:28s}  {r['MSL_dB']:>+10.1f}  {r['HPBW_arcmin']:>15.1f}")
    ref_hpbw = rows[0]["HPBW_arcmin"]
    for r in rows[1:]:
        brd = (r["HPBW_arcmin"] / ref_hpbw - 1) * 100
        print(f"  → {r['taper']}: HPBW broadened by {brd:.1f}%  "
              f"vs uniform (target ≤20%)")


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS 2: ELEMENT PATTERN (DIPOLE)
# ══════════════════════════════════════════════════════════════════════════════

def run_element_pattern_study(new_configs):
    """
    Multiply the normalised array factor by the short-dipole element radiation
    pattern.  For a horizontal E-W dipole:
      P(l, m) = 1 − l²       (null in E-W horizon, max at zenith)
    For a horizontal N-S dipole:
      P(l, m) = 1 − m²

    Shows how the effective beam shrinks toward the horizon,
    modifying the sensitivity at large zenith angles.
    """
    print("\n" + "=" * 72)
    print("  ADVANCED ANALYSIS 2: ELEMENT PATTERN (DIPOLE)")
    print("=" * 72)

    cfg_key = "cross_new_c128x128_out4x4_4arm"
    if cfg_key not in new_configs:
        cfg_key = next(k for k in new_configs if "cross" in k and "128" in k)
    cfg   = new_configs[cfg_key]
    pos   = cfg["pos_enu"]
    meta  = cfg["meta"]
    fc    = 30.0
    n_g   = 256

    l_arr = np.linspace(-1, 1, n_g)
    m_arr = np.linspace(-1, 1, n_g)
    L, M  = np.meshgrid(l_arr, m_arr, indexing="ij")
    inside = L**2 + M**2 <= 1.0

    w    = np.ones(len(pos), dtype=complex)
    _, _, B_iso = _weighted_af(pos, w, fc, n_g)

    P_ew = np.where(inside, 1.0 - L**2, np.nan)   # E-W dipole
    P_ns = np.where(inside, 1.0 - M**2, np.nan)   # N-S dipole

    B_ew = B_iso * P_ew
    B_ew = B_ew / (np.nanmax(B_ew) + 1e-30)
    B_ns = B_iso * P_ns
    B_ns = B_ns / (np.nanmax(B_ns) + 1e-30)

    # Aeff modification at various zenith angles
    zenith_angles = np.degrees(np.arcsin(np.sqrt(np.maximum(L**2 + M**2, 0))))
    results = []
    for za_thr in [0, 15, 30, 45, 60]:
        mask_z = (zenith_angles <= za_thr) & inside
        a_iso  = float(np.nanmean(B_iso[mask_z])) if mask_z.any() else 0.0
        a_ew   = float(np.nanmean(B_ew[mask_z]))  if mask_z.any() else 0.0
        results.append(dict(zenith_deg=za_thr,
                            mean_B_iso=a_iso,
                            mean_B_ew=a_ew,
                            eff_area_ratio=a_ew / (a_iso + 1e-30)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (B_iso.T, "Isotropic (no element pattern)", "inferno"),
        (P_ew.T,  "E-W dipole pattern P(l,m) = 1 − l²", "viridis"),
        (B_ew.T,  "AF × E-W element pattern (normalised)", "inferno"),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.pcolormesh(l_arr, m_arr, data, vmin=0, vmax=1,
                           cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("l"); ax.set_ylabel("m")
        ax.set_aspect("equal")

    fig.suptitle(f"Element Pattern Effect — {meta['label']}\n"
                 "Dipole creates a null toward the E-W horizon (l = ±1); "
                 "beam shrinks at large zenith angles",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ADV_PLOT, "adv2_element_pattern.png"), dpi=120)
    plt.close()

    # Sensitivity penalty as function of zenith angle
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    za_vals = [r["zenith_deg"] for r in results]
    ratio   = [r["eff_area_ratio"] for r in results]
    ax2.plot(za_vals, ratio, "o-", lw=2, ms=6, color="steelblue")
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.set_xlabel("Zenith angle [deg]", fontsize=10)
    ax2.set_ylabel("Mean beam ratio  (with / without element pattern)", fontsize=10)
    ax2.set_title("Effective sensitivity penalty from dipole element pattern",
                  fontsize=10)
    ax2.set_ylim(0, 1.2); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ADV_PLOT, "adv2_zenith_sensitivity.png"), dpi=120)
    plt.close()

    pd.DataFrame(results).to_csv(os.path.join(ADV_CSV, "adv2_element_pattern.csv"),
                                  index=False)
    print("  Saved adv2_element_pattern.png, adv2_zenith_sensitivity.png")
    print("  Saved adv2_element_pattern.csv")
    print("\n  Effective-area penalty at horizon zenith angles:")
    for r in results:
        print(f"    z ≤ {r['zenith_deg']:2d}°  → mean(B×P) / mean(B) = "
              f"{r['eff_area_ratio']:.3f}  "
              f"({'no penalty' if r['eff_area_ratio'] > 0.95 else 'significant reduction'})")


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS 3: LUNAR ROTATION AND SOURCE TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def run_lunar_rotation_uv(new_configs):
    """
    Simulate UV-coverage accumulation over a full 14-day lunar sidereal day.
    The Moon rotates once per ~27.3 days (sidereal period), but the far side
    is visible to a fixed target for ~11 hours out of each ~24.8-hour synodic
    day when the source is above the horizon.

    We model this as: for each UTC hour in a 14-day window,
    – compute the projected baseline in the (u, v) frame for each station pair
      using the rotation of the ENU baseline vector under the Moon's rotation
    – mark as 'visible' if the target elevation is above 0° (simplified model)

    Source: 51 Peg b at (RA, Dec) = (344.4°, +20.5°).
    """
    print("\n" + "=" * 72)
    print("  ADVANCED ANALYSIS 3: LUNAR ROTATION AND UV TRACKING")
    print("=" * 72)

    # Use the 128×128 cross config
    cfg_key = "cross_new_c128x128_out4x4_4arm"
    if cfg_key not in new_configs:
        cfg_key = next(k for k in new_configs if "cross" in k and "128" in k)
    cfg = new_configs[cfg_key]
    meta = cfg["meta"]

    # Station positions (core + outrigger centres)
    stations = np.array([(0.0, 0.0)] +
                        [(cx, cy) for cx, cy in meta["out_centres"]])

    # Tsiolkovsky crater parameters
    lat_rad = np.radians(PHI_DEG)   # -20.38°
    lon_rad = np.radians(LAM_DEG)   # 128.97°

    # 51 Peg b sky coordinates
    src_ra_rad  = np.radians(344.4)
    src_dec_rad = np.radians(20.5)

    # Time grid: 14 days at 30-minute steps
    dt_hr    = 0.5   # hours per step
    n_days   = 14
    n_steps  = int(n_days * 24 / dt_hr)
    # Lunar sidereal rotation rate: 2π / (27.3 × 24) rad/hour
    omega_moon = 2 * np.pi / (27.3 * 24.0)   # rad / hour

    # Reference frequency: 30 MHz
    freq_MHz = 30.0
    lam      = C / (freq_MHz * 1e6)

    uv_snapshot = {"u": [], "v": []}   # instant UV at t=0
    uv_tracked  = {"u": [], "v": []}   # accumulated UV over 14 days
    visibility_hrs = 0.0

    for step in range(n_steps):
        t_hr    = step * dt_hr
        # Rotation angle of the lunar far-side array in inertial frame
        psi     = omega_moon * t_hr   # cumulative rotation [rad]

        # Hour angle of source at Tsiolkovsky crater
        # (simplified: source RA sets an effective hour angle)
        # HA = LST - RA.  LST = lon + omega_moon * t (in sidereal terms)
        lst     = lon_rad + omega_moon * t_hr
        ha      = lst - src_ra_rad

        # Elevation of source (simplified flat-Moon approximation)
        sin_el = (np.sin(lat_rad) * np.sin(src_dec_rad) +
                  np.cos(lat_rad) * np.cos(src_dec_rad) * np.cos(ha))
        if sin_el <= 0.0:
            continue   # source below horizon

        visibility_hrs += dt_hr
        cos_el  = np.sqrt(max(1.0 - sin_el**2, 0.0))

        # Rotation matrix (ENU basis rotated by psi around vertical axis)
        Rmat = np.array([[np.cos(psi), -np.sin(psi)],
                          [np.sin(psi),  np.cos(psi)]])

        n_st = len(stations)
        for i in range(n_st):
            for j in range(i + 1, n_st):
                dxy     = stations[i] - stations[j]   # ENU (East, North) diff [m]
                dxy_rot = Rmat @ dxy                   # rotated baseline
                # UV from projected baseline
                u_val = dxy_rot[0] / lam * cos_el
                v_val = dxy_rot[1] / lam
                uv_tracked["u"] += [u_val / 1e3, -u_val / 1e3]
                uv_tracked["v"] += [v_val / 1e3, -v_val / 1e3]

        # Snapshot at t = 0
        if step == 0:
            for i in range(n_st):
                for j in range(i + 1, n_st):
                    dxy = stations[i] - stations[j]
                    u_s = dxy[0] / lam * cos_el / 1e3
                    v_s = dxy[1] / lam / 1e3
                    uv_snapshot["u"] += [u_s, -u_s]
                    uv_snapshot["v"] += [v_s, -v_s]

    print(f"  Total source visibility over 14 days: {visibility_hrs:.1f} h "
          f"({visibility_hrs/(14*24)*100:.1f}% of time)")

    n_snap = len(uv_snapshot["u"]) // 2
    n_track = len(uv_tracked["u"]) // 2
    labels_uv = [
        f"Snapshot UV (t = 0)\n{n_snap} baselines",
        f"Tracked UV over 14 days\n({visibility_hrs:.0f} h visibility, {n_track:,} points)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (ud, label) in zip(axes, [
        (uv_snapshot, labels_uv[0]),
        (uv_tracked,  labels_uv[1]),
    ]):
        ax.scatter(ud["u"], ud["v"], s=0.5, alpha=0.4, linewidths=0, c="steelblue")
        ax.set_xlabel("u  [kλ]", fontsize=10); ax.set_ylabel("v  [kλ]", fontsize=10)
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"UV Coverage: Snapshot vs 14-Day Tracked  —  {meta['label']}\n"
        f"Target: 51 Peg b  (RA=344.4°, Dec=+20.5°)  |  30 MHz",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ADV_PLOT, "adv3_lunar_rotation_uv.png"), dpi=120)
    plt.close()
    print("  Saved adv3_lunar_rotation_uv.png")
    print(f"  Snapshot UV points: {len(uv_snapshot['u'])//2}")
    print(f"  Tracked  UV points: {len(uv_tracked['u'])//2:,}  "
          f"(×{len(uv_tracked['u'])//max(1,len(uv_snapshot['u'])):.0f} improvement)")


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS 4: OPTIMAL OUTRIGGER SUB-ARRAY SIZE FOR CROSS CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def run_outrigger_size_study(targets):
    """
    Vary the outrigger sub-array side from N_o = 1 to 8 for the cross
    configuration (128×128 core) at 5 km arm distance.
    For each N_o compute:
      – MSL, HPBW (beam quality)
      – N_det (detection yield)
      – mass proxy: N_elem × element_mass_kg
    """
    print("\n" + "=" * 72)
    print("  ADVANCED ANALYSIS 4: OPTIMAL OUTRIGGER SIZE FOR CROSS")
    print("=" * 72)

    ELEM_MASS_KG = 1.5   # kg per dipole element (estimated)
    core_n = 128
    # Use the asymmetric cross layout matching config f
    cross_arm_specs_so = [
        (_ARM_N, NEW_SHORT_DISTS),
        (_ARM_E, NEW_LONG_DISTS),
        (_ARM_S, NEW_INT_DISTS),
        (_ARM_W, NEW_INT_DISTS),
    ]

    rows = []
    for n_o in range(1, 9):
        parts    = [rect_array_enu(core_n, D_SPACE)]
        centres  = []
        edge     = core_edge_m(core_n)
        for ang, dists in cross_arm_specs_so:
            for d in dists:
                dc = edge + d
                cx = dc * np.cos(ang)
                cy = dc * np.sin(ang)
                centres.append((cx, cy))
                parts.append(rect_array_enu(n_o, D_SPACE, cx, cy))
        pos   = np.vstack(parts)
        N_tot = len(pos)
        B_max = max_baseline_m(centres, core_n)

        meta = dict(core_n=core_n, out_n=n_o, out_centres=centres,
                    N_total=N_tot, B_max_m=B_max)

        l, m, AF_dB, B_n = compute_af(pos, meta, 30.0, 256)
        msl, hpbw = _msl_hpbw_from_norm(B_n, l, m, 30.0)

        # Detection yield
        n_feas = 0
        for bl, (f_lo, f_hi), fc in zip(BAND_LABELS, SUBBANDS, BAND_CTR):
            bw = f_hi - f_lo
            band_tgts = targets[(targets["frequency_MHz"] >= f_lo) &
                                 (targets["frequency_MHz"] <  f_hi)]
            for _, trow in band_tgts.iterrows():
                t_h = required_t_hours(trow["flux_mJy"], N_tot, fc, bw, B_max)
                if t_h < 100:
                    n_feas += 1
                    break   # count target once

        mass_kg  = N_tot * ELEM_MASS_KG
        rows.append(dict(N_o=n_o, N_elements=N_tot,
                         MSL_dB=msl, HPBW_arcmin=hpbw,
                         n_det=n_feas, mass_kg=mass_kg))
        print(f"  N_o={n_o}: N={N_tot:6d}  MSL={msl:+.1f} dB  "
              f"HPBW={hpbw:.1f}′  n_det={n_feas}  mass={mass_kg:.0f} kg")

    df_sz = pd.DataFrame(rows)
    df_sz.to_csv(os.path.join(ADV_CSV, "adv4_outrigger_size.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col, ylabel, invert in [
        (axes[0, 0], "MSL_dB",        "MSL [dB] (more negative = better)", True),
        (axes[0, 1], "HPBW_arcmin",   "HPBW [arcmin]  (smaller = better)", True),
        (axes[1, 0], "n_det",         "Feasible detections (<100 h)", False),
        (axes[1, 1], "mass_kg",       "Total mass proxy [kg]", True),
    ]:
        ax.plot(df_sz["N_o"], df_sz[col], "o-", lw=2, ms=7, color="steelblue")
        for i, row in df_sz.iterrows():
            ax.annotate(f"N_o={row['N_o']}", (row["N_o"], row[col]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Outrigger sub-array side N_o", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split("[")[0].strip(), fontsize=10)
        ax.grid(True, alpha=0.3)
        if invert and col != "HPBW_arcmin":
            ax.invert_yaxis()

    fig.suptitle(
        "Outrigger Size Optimization — 128×128 Cross Config\n"
        "(4 arms × 4 outriggers; arm distances 1.25/2.5/3.75/5 km from edge)",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ADV_PLOT, "adv4_outrigger_size_study.png"), dpi=120)
    plt.close()
    print("  Saved adv4_outrigger_size_study.png, adv4_outrigger_size.csv")

    best = df_sz.loc[df_sz["n_det"].idxmax()]
    print(f"\n  Best N_o for detection yield: N_o = {int(best['N_o'])}  "
          f"(n_det={int(best['n_det'])}, mass≈{best['mass_kg']:.0f} kg)")


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS 5: CALIBRATION ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def run_calibration_architecture():
    """
    Analytical assessment of calibration from the lunar far side.
    Compute: sky temperature contribution of bright calibrators,
    expected SNR per calibration cycle, coherence time estimate.
    """
    print("\n" + "=" * 72)
    print("  ADVANCED ANALYSIS 5: CALIBRATION ARCHITECTURE")
    print("=" * 72)

    # Calibrator flux densities (approximate, at 10 MHz, Jy)
    calibrators = {
        "Cas A":  6e6,
        "Cyg A":  1.5e6,
        "Vir A":  6e5,
        "Tau A":  2.5e5,
        "Her A":  1.1e5,
    }
    # Approximate spectral index for scaling to other frequencies
    alpha = -0.7   # S ∝ ν^α

    # Array parameters for 128×128 + 4×4 outriggers (5 km)
    N        = 128**2 + 4 * 4**2   # total elements
    B_max_m  = 5327.0               # m (from new cross config)
    fc_cal   = 10.0                 # calibration at 10 MHz
    bw_cal   = 1.0                  # 1 MHz calibration bandwidth
    t_cal_s  = 60.0                 # 60-second calibration cycle

    # Thermal noise at calibration frequency
    st_cal  = sigma_thermal_Jy(N, fc_cal, bw_cal, t_cal_s / 3600.0)

    print(f"\n  Array: {N} elements, B_max = {B_max_m/1e3:.2f} km")
    print(f"  Calibration: {fc_cal} MHz, BW = {bw_cal} MHz, t = {t_cal_s:.0f} s")
    print(f"  Thermal noise: {st_cal*1e-3:.3f} kJy  ({st_cal:.3e} Jy)")
    snr_hdr = f"SNR in {t_cal_s:.0f}s"
    print(f"\n  {'Source':8s}  {'S(10MHz) [kJy]':>16s}  {snr_hdr:>18s}  "
          f"{'Usable?':>8s}")
    print("  " + "-" * 60)

    rows = []
    for src, S_10 in calibrators.items():
        # Scale to calibration frequency (already at 10 MHz)
        S_fc  = S_10 * (fc_cal / 10.0) ** alpha
        snr   = S_fc / (NSIGMA * st_cal)
        usable = "YES" if snr > 10 else "marginal" if snr > 3 else "NO"
        print(f"  {src:8s}  {S_fc/1e3:>16.0f}  {snr:>18.1f}  {usable:>8s}")
        rows.append(dict(source=src, flux_Jy=S_fc, SNR=snr, usable=usable))

    # Ionospheric / exospheric coherence time (lunar plasma)
    # Below ~1 MHz, dispersion measure ~10⁻⁴ pc cm⁻³ (lunar exosphere estimate)
    DM_est  = 1e-4     # pc cm⁻³
    for nu_MHz in [0.5, 1.0, 5.0, 10.0, 30.0]:
        # Dispersion delay at ν vs reference ν_ref = 100 MHz
        nu_ref  = 100.0
        dt_disp = 4150.0 * DM_est * (1/nu_MHz**2 - 1/nu_ref**2)   # seconds
        print(f"  Plasma dispersion at {nu_MHz:5.1f} MHz:  Δt ≈ {dt_disp:.2f} s")

    pd.DataFrame(rows).to_csv(os.path.join(ADV_CSV, "adv5_calibrators.csv"),
                               index=False)
    print("\n  Saved adv5_calibrators.csv")
    print("\n  Key findings:")
    print("  • Cas A and Cyg A provide >1000× SNR → excellent gain calibrators")
    print("  • Phase calibration requires short-baseline correlations (coherent on core)")
    print("  • Calibration cycle ≤ 60 s prevents ionospheric/plasma phase drift")
    print("  • Below 1 MHz, dispersion delays require per-channel phase correction")


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS 6: PLASMA / EXOSPHERE EFFECTS BELOW 1 MHz
# ══════════════════════════════════════════════════════════════════════════════

def run_plasma_effects():
    """
    Model dispersive phase delay and phase noise from the lunar exosphere.
    The Moon has a tenuous plasma with electron density n_e ~ 10³–10⁴ cm⁻³
    near the surface (daytime).  This introduces a dispersive group delay
    τ_group = (e²/8π²m_e ε₀) × DM / ν²  ≈  4150 × DM / ν²  [s, pc cm⁻³, MHz]
    and a phase noise bandwidth ~ ν_p (plasma frequency) = 8.98 √n_e  [Hz].
    """
    print("\n" + "=" * 72)
    print("  ADVANCED ANALYSIS 6: PLASMA EFFECTS BELOW 1 MHz")
    print("=" * 72)

    # Lunar exosphere electron density estimates
    n_e_day   = 1e3    # cm⁻³  (daytime, near surface)
    n_e_night = 1e1    # cm⁻³  (nighttime, much lower)
    DM_day    = 3e-5   # pc cm⁻³  (rough estimate for path through exosphere)
    DM_night  = 3e-7

    nu_arr = np.logspace(-1, 2, 300)   # 0.1 to 100 MHz

    # Plasma cutoff frequency
    nu_p_day   = 8.98e-3 * np.sqrt(n_e_day)    # MHz
    nu_p_night = 8.98e-3 * np.sqrt(n_e_night)

    print(f"\n  Daytime  plasma cutoff: {nu_p_day:.3f} MHz  (n_e={n_e_day:.0f} cm⁻³)")
    print(f"  Nighttime plasma cutoff: {nu_p_night:.4f} MHz  (n_e={n_e_night:.0f} cm⁻³)")

    # Dispersion delay relative to 1 GHz reference
    dt_day   = 4150.0 * DM_day   * (1.0/nu_arr**2 - 1.0/1000.0**2)
    dt_night = 4150.0 * DM_night * (1.0/nu_arr**2 - 1.0/1000.0**2)

    # Phase noise: thermal noise + dispersive smearing (simplified)
    phase_noise_day   = np.degrees(2 * np.pi * nu_arr * 1e6 * dt_day)
    phase_noise_night = np.degrees(2 * np.pi * nu_arr * 1e6 * dt_night)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.loglog(nu_arr, np.abs(dt_day),   lw=2, label=f"Daytime  (n_e={n_e_day:.0f} cm⁻³)")
    ax1.loglog(nu_arr, np.abs(dt_night), lw=2, ls="--", label=f"Nighttime (n_e={n_e_night:.0f} cm⁻³)")
    ax1.axvline(1.0,  color="red", ls=":", lw=1.5, label="1 MHz boundary")
    ax1.axvline(nu_p_day, color="tomato", ls="--", lw=1, alpha=0.7,
                label=f"Day plasma cutoff {nu_p_day:.2f} MHz")
    ax1.set_xlabel("Frequency [MHz]", fontsize=10)
    ax1.set_ylabel("Dispersive group delay [s]", fontsize=10)
    ax1.set_title("Lunar exosphere group delay", fontsize=10)
    ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.3)

    ax2.loglog(nu_arr, np.abs(phase_noise_day),   lw=2, label="Daytime")
    ax2.loglog(nu_arr, np.abs(phase_noise_night), lw=2, ls="--", label="Nighttime")
    ax2.axhline(10, color="gray", ls=":", lw=1.2, label="10° phase noise threshold")
    ax2.axvline(1.0,  color="red", ls=":", lw=1.5)
    ax2.set_xlabel("Frequency [MHz]", fontsize=10)
    ax2.set_ylabel("Phase noise [degrees]", fontsize=10)
    ax2.set_title("Phase noise from dispersive delay", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Lunar Exosphere Plasma Effects\n"
        "Significant below ~1 MHz; nighttime operations reduce dispersion by ×100",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ADV_PLOT, "adv6_plasma_effects.png"), dpi=120)
    plt.close()
    print("  Saved adv6_plasma_effects.png")

    # Usable frequency range
    for nu_thr in [0.5, 1.0, 5.0]:
        idx = np.argmin(np.abs(nu_arr - nu_thr))
        print(f"  At {nu_thr} MHz  | day delay={dt_day[idx]:.2f}s "
              f"phase={phase_noise_day[idx]:.1f}°  |  "
              f"night delay={dt_night[idx]:.4f}s phase={phase_noise_night[idx]:.2f}°")
    print("\n  Recommendation: operate below 1 MHz only during lunar night,")
    print("  with per-channel DM correction using known pulsar timing solutions.")


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS 7: COMPARISON WITH FARSIDE / DAPPER / DSL
# ══════════════════════════════════════════════════════════════════════════════

def run_farside_comparison(targets):
    """
    Benchmark ALO against FARSIDE, DAPPER, and DSL using known/published
    array parameters.  Compute sensitivity, confusion, and detection yield
    for each concept using the same target catalogue.
    """
    print("\n" + "=" * 72)
    print("  ADVANCED ANALYSIS 7: COMPARISON WITH FARSIDE / DAPPER / DSL")
    print("=" * 72)

    # Published / estimated parameters for each concept
    # (N_elem, B_max_m, freq_range_MHz, deployment_site)
    concepts = {
        "ALO (128×128 + cross, 5km)": {
            "N": 128**2 + 4*4*(4**2),   # 128^2 core + 4 arms × 4 × 4×4
            "B_max_m": 5327.0,
            "site": "Tsiolkovsky crater, lunar far side (surface)",
            "freq_MHz": (1, 40),
        },
        "FARSIDE (proposed)": {
            "N": 128,       # 128 antennas in lunar orbit + surface nodes
            "B_max_m": 10e3,
            "site": "Lunar farside surface, Aitken Basin",
            "freq_MHz": (0.1, 40),
        },
        "DAPPER (proposed)": {
            "N": 1,         # single spacecraft in lunar orbit (no interferometry)
            "B_max_m": 1e3,
            "site": "Lunar orbit (cislunar)",
            "freq_MHz": (0.1, 40),
        },
        "DSL (proposed, China)": {
            "N": 8,         # 1 mother + 5-7 daughter spacecraft (linear array)
            "B_max_m": 100e3,
            "site": "Lunar orbit, mother-daughter formation",
            "freq_MHz": (1, 30),
        },
    }

    print(f"\n  {'Concept':35s}  {'N_elem':>7s}  {'B_max km':>9s}  "
          f"{'θ_HPBW@10MHz (arcmin)':>24s}  {'n_det(<100h)':>13s}")
    print("  " + "-" * 100)

    rows = []
    for name, params in concepts.items():
        N     = params["N"]
        B_max = params["B_max_m"]
        # HPBW at 10 MHz using λ/B_max
        wl_10    = C / (10e6)
        hpbw_rad = wl_10 / B_max if B_max > 0 else np.pi
        hpbw_arcmin = np.degrees(hpbw_rad) * 60.0

        n_feas = 0
        if N > 1:
            for bl, (f_lo, f_hi), fc in zip(BAND_LABELS, SUBBANDS, BAND_CTR):
                bw = f_hi - f_lo
                p_lo, p_hi = params["freq_MHz"]
                if fc < p_lo or fc > p_hi:
                    continue
                band_tgts = targets[(targets["frequency_MHz"] >= f_lo) &
                                     (targets["frequency_MHz"] <  f_hi)]
                counted = set()
                for _, trow in band_tgts.iterrows():
                    if trow["Name"] in counted:
                        continue
                    t_h = required_t_hours(trow["flux_mJy"], N, fc, bw, B_max)
                    if t_h < 100:
                        n_feas += 1
                        counted.add(trow["Name"])

        print(f"  {name:35s}  {N:>7d}  {B_max/1e3:>9.1f}  "
              f"{hpbw_arcmin:>24.1f}  {n_feas:>13d}")
        rows.append(dict(concept=name, N=N, B_max_km=B_max/1e3,
                         HPBW_arcmin_at_10MHz=hpbw_arcmin,
                         n_feasible_det=n_feas,
                         site=params["site"],
                         freq_range=f"{params['freq_MHz'][0]}–{params['freq_MHz'][1]} MHz"))

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(os.path.join(ADV_CSV, "adv7_farside_comparison.csv"), index=False)

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cols_plot = ["N", "B_max_km", "n_feasible_det"]
    labels_p  = ["Number of elements N",
                 "Max baseline B_max [km]",
                 "Feasible detections (<100h)"]
    concept_labels = [r["concept"].split("(")[0].strip() for r in rows]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for ax, col, ylabel in zip(axes, cols_plot, labels_p):
        vals = df_comp[col].values
        ax.bar(range(len(rows)), vals, color=colors[:len(rows)],
               edgecolor="white", lw=0.5)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(concept_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel.split("[")[0].strip(), fontsize=9)
        if col == "N":
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02 if v > 0 else 1, f"{v:.0f}",
                    ha="center", fontsize=7)

    fig.suptitle(
        "Lunar Far-Side Array Concept Comparison\n"
        "ALO vs FARSIDE vs DAPPER vs DSL  (same 373-target catalogue)",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ADV_PLOT, "adv7_farside_comparison.png"), dpi=120)
    plt.close()
    print("  Saved adv7_farside_comparison.png, adv7_farside_comparison.csv")


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED ANALYSIS 8: DEPLOYMENT FEASIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def run_deployment_study():
    """
    Engineering trade study for the optimal configuration:
    128×128 core + cross 4×4×4 outriggers at 5 km arm distances.

    Estimates: mass, power, thermal environment, deployment strategy.
    """
    print("\n" + "=" * 72)
    print("  ADVANCED ANALYSIS 8: DEPLOYMENT FEASIBILITY")
    print("=" * 72)

    # Element parameters
    elem_mass_kg  = 1.5      # kg per folded dipole
    elem_power_W  = 0.3      # W per active receive element (ADC + LNA)
    wire_mass_kg_per_m = 0.05  # kg/m for signal cable

    # Configuration: 128×128 core + 4 arms × 4 × 4×4 outriggers
    N_core  = 128**2
    N_out   = 4 * 4 * 4**2   # 4 arms × 4 sub-arrays × 16 elem
    N_total = N_core + N_out

    # Mass budget
    mass_elems_kg  = N_total * elem_mass_kg
    core_footprint = (128 - 1) * D_SPACE               # 654 m core side
    # Cable mass: Manhattan distance approximation for core wiring
    cable_core_m   = N_core * core_footprint / 2        # rough estimate
    # Outrigger cable: 4 arms × 4 positions, arm distances from edge
    arm_dists      = [1329 + d for d in [0, 1250, 2500, 3750]]  # m from centre
    cable_out_m    = 4 * sum(arm_dists)
    mass_cable_kg  = (cable_core_m + cable_out_m) * wire_mass_kg_per_m
    # Electronics (LNA + digitiser + hub): ~2 kg per sub-array
    mass_elec_kg   = (1 + 4 * 4) * 2.0   # core hub + 16 sub-array hubs
    mass_total_kg  = mass_elems_kg + mass_cable_kg + mass_elec_kg

    # Power budget
    power_rx_W     = N_total * elem_power_W
    power_proc_W   = 500.0   # data processing + correlation
    power_comms_W  = 200.0   # RF link to Earth relay / orbiter
    power_thermal_W = 100.0  # heater during lunar night
    power_total_W  = power_rx_W + power_proc_W + power_comms_W + power_thermal_W

    # Solar power availability at Moon (1 AU): solar constant 1361 W/m²
    # Lunar day ≈ 14 Earth days → solar energy available half the time
    solar_efficiency = 0.22   # panel efficiency
    panel_area_m2    = power_total_W / (1361 * solar_efficiency * 0.5)

    # RTG alternative: 1 MMRTG ≈ 110 W
    n_rtg  = int(np.ceil(power_total_W / 110.0))

    # Thermal: lunar surface temperature swings -173°C to +127°C
    T_day_K   = 400   # K daytime
    T_night_K = 100   # K nighttime
    delta_T   = T_day_K - T_night_K

    print(f"\n  Configuration: {N_total} elements "
          f"({N_core} core + {N_out} outriggers)")
    print(f"  Core footprint: {core_footprint:.0f} m × {core_footprint:.0f} m")
    print(f"\n  MASS BUDGET:")
    print(f"    Element antennas:  {mass_elems_kg:8.0f} kg")
    print(f"    Signal cables:     {mass_cable_kg:8.0f} kg")
    print(f"    Electronics:       {mass_elec_kg:8.0f} kg")
    print(f"    TOTAL:             {mass_total_kg:8.0f} kg")
    print(f"\n  POWER BUDGET:")
    print(f"    Receive elements:  {power_rx_W:8.0f} W")
    print(f"    Processing:        {power_proc_W:8.0f} W")
    print(f"    Communications:    {power_comms_W:8.0f} W")
    print(f"    Thermal mgmt:      {power_thermal_W:8.0f} W")
    print(f"    TOTAL:             {power_total_W:8.0f} W")
    print(f"\n  POWER SOURCES:")
    print(f"    Solar panels needed: {panel_area_m2:.1f} m² @ η={solar_efficiency}")
    print(f"    RTG alternative:     {n_rtg} × MMRTG (each ≈110 W EOL)")
    print(f"\n  THERMAL:")
    print(f"    Day/night swing:   {T_day_K - 273:.0f}°C to {T_night_K - 273:.0f}°C "
          f"(ΔT = {delta_T} K)")
    print(f"    Mitigation: multi-layer insulation (MLI) + heaters on electronics")
    print(f"\n  DEPLOYMENT NOTES:")
    print(f"    Core array: 128×128 = {N_core} elements over {core_footprint:.0f}m × "
          f"{core_footprint:.0f}m")
    print(f"    → requires robotic or automated deployment over ~650m footprint")
    print(f"    Outrigger arms: 4 directions, last station at "
          f"~{arm_dists[-1]/1e3:.1f} km from core edge")
    print(f"    → cable/power routed along deployed tether or deployed by rover")
    print(f"\n  FEASIBILITY ASSESSMENT:")
    feasible = mass_total_kg < 5000 and power_total_W < 3000
    print(f"    Mass < 5000 kg: {'YES' if mass_total_kg < 5000 else 'NO'} "
          f"({mass_total_kg:.0f} kg)")
    print(f"    Power < 3 kW:   {'YES' if power_total_W < 3000 else 'NO'} "
          f"({power_total_W:.0f} W)")
    print(f"    Overall feasibility: "
          f"{'TECHNICALLY FEASIBLE' if feasible else 'CHALLENGING — needs phased deployment'}")

    # Save summary
    rows = [
        dict(category="Element antennas", value_kg=mass_elems_kg),
        dict(category="Signal cables",    value_kg=mass_cable_kg),
        dict(category="Electronics",      value_kg=mass_elec_kg),
        dict(category="TOTAL MASS",       value_kg=mass_total_kg),
    ]
    pd.DataFrame(rows).to_csv(os.path.join(ADV_CSV, "adv8_deployment.csv"),
                               index=False)

    # Mass/power pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    mass_labels = ["Elements", "Cables", "Electronics"]
    mass_vals   = [mass_elems_kg, mass_cable_kg, mass_elec_kg]
    ax1.pie(mass_vals, labels=mass_labels, autopct="%1.1f%%",
            colors=["#2196F3", "#4CAF50", "#FF9800"])
    ax1.set_title(f"Mass Budget (Total = {mass_total_kg:.0f} kg)", fontsize=10)

    power_labels = ["Receive elements", "Processing", "Communications", "Thermal"]
    power_vals   = [power_rx_W, power_proc_W, power_comms_W, power_thermal_W]
    ax2.pie(power_vals, labels=power_labels, autopct="%1.1f%%",
            colors=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"])
    ax2.set_title(f"Power Budget (Total = {power_total_W:.0f} W)", fontsize=10)

    fig.suptitle(
        f"ALO Deployment Feasibility — 128×128 Core + Cross 4×4 @ 5 km\n"
        f"({N_total} total elements, {core_footprint:.0f}m core footprint)",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ADV_PLOT, "adv8_deployment.png"), dpi=120)
    plt.close()
    print("  Saved adv8_deployment.png, adv8_deployment.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FULL COMPARISON OF 6 CONFIGURATIONS (A–F)
# Produces separate + combined plots for:
#   1. Sensitivity         2. Beam-pattern quality   3. RMS noise
#   4. UV coverage (100h)  5. Number of detections
# All saved under outputs/.../Comparison/
# ══════════════════════════════════════════════════════════════════════════════

# Colour / style palette — consistent across all comparison plots
CFG_COLORS = {
    "ring_a_c32x32_out2x2_short":   "#1976D2",   # blue
    "ring_b_c32x32_out2x2_long":    "#F57C00",   # orange
    "ring_c_c128x128_out4x4_short": "#388E3C",   # green
    "ring_d_c128x128_out4x4_long":  "#D32F2F",   # red
    "cross_e_c32x32_out2x2_asym":   "#7B1FA2",   # purple
    "cross_f_c128x128_out4x4_asym": "#5D4037",   # brown
}
CFG_SHORT = {
    "ring_a_c32x32_out2x2_short":   "A: Ring 32×32\nshort",
    "ring_b_c32x32_out2x2_long":    "B: Ring 32×32\nlong",
    "ring_c_c128x128_out4x4_short": "C: Ring 128×128\nshort",
    "ring_d_c128x128_out4x4_long":  "D: Ring 128×128\nlong",
    "cross_e_c32x32_out2x2_asym":   "E: Cross 32×32",
    "cross_f_c128x128_out4x4_asym": "F: Cross 128×128",
}
CFG_ORDER = list(CFG_COLORS.keys())

CMP_ROOT = os.path.join(GT_PLOT, "Comparison")
CMP_SEP  = os.path.join(CMP_ROOT, "separate")
CMP_CMB  = os.path.join(CMP_ROOT, "combined")
CMP_CSV  = os.path.join(GT_CSV,  "Comparison")
for _d in [CMP_ROOT, CMP_SEP, CMP_CMB, CMP_CSV]:
    os.makedirs(_d, exist_ok=True)


# ── helper: UV coverage for 100 hours of observation ─────────────────────────

def _uv_100h(cfg, max_hr=100, dt_min=30, freq_MHz=30.0,
             src_ra_deg=344.4, src_dec_deg=20.5):
    """
    Accumulate UV baselines for up to max_hr hours of actual on-source time.
    Uses Moon sidereal rotation; source is 51 Peg b (RA=344.4°, Dec=+20.5°).
    Returns u [kλ], v [kλ], and actual hours accumulated.
    """
    stations   = np.array([(0.0, 0.0)] +
                          [(cx, cy) for cx, cy in cfg["meta"]["out_centres"]])
    src_ra     = np.radians(src_ra_deg)
    src_dec    = np.radians(src_dec_deg)
    lat        = np.radians(PHI_DEG)
    lon        = np.radians(LAM_DEG)
    omega      = 2 * np.pi / (27.3 * 24.0)   # Moon sidereal rotation [rad/hr]
    lam        = C / (freq_MHz * 1e6)
    dt_hr      = dt_min / 60.0
    n_max      = int(35 * 24 / dt_hr)         # search up to 35 days

    u_all, v_all = [], []
    vis_hr = 0.0
    for step in range(n_max):
        if vis_hr >= max_hr:
            break
        t_hr   = step * dt_hr
        psi    = omega * t_hr
        lst    = lon + omega * t_hr
        ha     = lst - src_ra
        sin_el = (np.sin(lat) * np.sin(src_dec) +
                  np.cos(lat) * np.cos(src_dec) * np.cos(ha))
        if sin_el <= 0:
            continue
        vis_hr += dt_hr
        cos_el = np.sqrt(max(1 - sin_el**2, 0.0))
        Rmat   = np.array([[np.cos(psi), -np.sin(psi)],
                            [np.sin(psi),  np.cos(psi)]])
        n_st   = len(stations)
        for i in range(n_st):
            for j in range(i + 1, n_st):
                dxy     = stations[i] - stations[j]
                dxy_rot = Rmat @ dxy
                u_v = dxy_rot[0] / lam * cos_el / 1e3
                v_v = dxy_rot[1] / lam / 1e3
                u_all.extend([u_v, -u_v])
                v_all.extend([v_v, -v_v])
    return np.array(u_all), np.array(v_all), vis_hr


# ── SEPARATE plots — one function per property ────────────────────────────────

def _cmp_sep_dir(cfg_id):
    d = os.path.join(CMP_SEP, cfg_id)
    os.makedirs(d, exist_ok=True)
    return d


def cmp_sep_sensitivity(name, cfg, df_sens_cfg):
    """Sensitivity vs integration time for one config — all 4 bands."""
    sdir   = _cmp_sep_dir(name)
    meta   = cfg["meta"]
    N      = meta["N_total"]
    B_max  = meta["B_max_m"]
    t_arr  = np.logspace(-2, 4, 500)
    colors = {"1-5":"#E53935","5-10":"#FB8C00","10-20":"#1E88E5","20-40":"#8E24AA"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for bl, (f_lo, f_hi), fc in zip(BAND_LABELS, SUBBANDS, BAND_CTR):
        bw   = f_hi - f_lo
        sc   = confusion_limit_Jy(fc, B_max)
        st1  = sigma_thermal_Jy(N, fc, bw, 1.0)
        stot = np.sqrt((st1 / np.sqrt(t_arr)) ** 2 + sc ** 2)
        ax.loglog(t_arr, NSIGMA * stot * 1e3, color=colors[bl], lw=2,
                  label=f"{bl} MHz")
        ax.axhline(NSIGMA * sc * 1e3, color=colors[bl], ls=":", lw=0.9, alpha=0.6)

    ax.axvline(100, color="gray", ls="--", lw=1.2, label="100 h")
    ax.set_xlabel("Integration time [h]", fontsize=10)
    ax.set_ylabel("5σ Total Sensitivity [mJy]", fontsize=10)
    ax.set_title(f"Sensitivity vs Integration Time\n{meta['label']}", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    ax.text(0.97, 0.03, "Dotted = confusion floor",
            transform=ax.transAxes, fontsize=8, ha="right",
            color="gray", style="italic")
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, "sensitivity.png"), dpi=110)
    plt.close()


def cmp_sep_beam(name, cfg, af_data):
    """Beam pattern: 2-D colormap + 1-D cut + contours at 30 MHz."""
    sdir = _cmp_sep_dir(name)
    bl   = "20-40"
    l, m, AF_dB, B_norm = af_data[bl]
    mtr  = beam_metrics(B_norm, l, m, BAND_CTR[BAND_LABELS.index(bl)])
    hpbw = 2 * np.sqrt(mtr["Omega_B"] / np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 2-D beam with contours
    ax = axes[0]
    im = ax.pcolormesh(l, m, AF_dB, vmin=-30, vmax=0, cmap="inferno",
                       shading="auto")
    plt.colorbar(im, ax=ax, label="|AF|² [dB]", fraction=0.046)
    ax.contour(l, m, B_norm, levels=CONTOUR_LVL,
               colors=CONTOUR_COL, linewidths=[0.8, 1.0, 1.3])
    ax.set_aspect("equal")
    ax.set_xlabel("l"); ax.set_ylabel("m")
    ax.set_title(f"2-D Beam  (20–40 MHz ≈ 30 MHz)\n"
                 f"HPBW={np.degrees(hpbw)*60:.1f}′  MSL={mtr['MSL_dB']:.1f} dB",
                 fontsize=9)

    # 1-D cut
    ax2  = axes[1]
    mid  = len(m) // 2
    cut  = 10 * np.log10(B_norm[mid, :] + 1e-20)
    ax2.plot(l, cut, lw=2, color=CFG_COLORS.get(name, "steelblue"))
    ax2.axhline(-3,  color="red",    ls="--", lw=1, label="−3 dB")
    ax2.axhline(-10, color="orange", ls=":",  lw=1, label="−10 dB")
    ax2.axhline(-20, color="purple", ls="-.", lw=1, label="−20 dB")
    ax2.set_ylim(-35, 2)
    ax2.set_xlabel("l (East direction cosine)")
    ax2.set_ylabel("Normalised power [dB]")
    ax2.set_title("1-D beam cut (m = 0)", fontsize=9)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    cfg_lbl = cfg["meta"]["label"]
    fig.suptitle(f"Beam Pattern — {cfg_lbl}", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, "beam_pattern.png"), dpi=110)
    plt.close()


def cmp_sep_rms(name, cfg):
    """RMS noise level vs integration time — thermal, confusion, total."""
    sdir   = _cmp_sep_dir(name)
    meta   = cfg["meta"]
    N      = meta["N_total"]
    B_max  = meta["B_max_m"]
    t_arr  = np.logspace(-2, 4, 500)
    fc     = REF_FREQ      # 30 MHz
    bw_ch  = 1.0           # 1 MHz channel (single-channel RMS)
    bw_ref = REF_BW        # 20 MHz for reference sensitivity

    sc     = confusion_limit_Jy(fc, B_max)
    st1_ch = sigma_thermal_Jy(N, fc, bw_ch, 1.0)   # 1-MHz channel RMS
    st1_bw = sigma_thermal_Jy(N, fc, bw_ref, 1.0)  # 20-MHz band RMS

    st_ch  = st1_ch / np.sqrt(t_arr)
    st_bw  = st1_bw / np.sqrt(t_arr)
    tot_bw = np.sqrt(st_bw**2 + sc**2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(t_arr, st_ch  * 1e3,  lw=2, color="steelblue",
              label=f"Thermal RMS (1 MHz ch)")
    ax.loglog(t_arr, st_bw  * 1e3,  lw=2, color="steelblue", ls="--",
              label=f"Thermal RMS ({REF_BW} MHz band)")
    ax.loglog(t_arr, tot_bw * 1e3,  lw=2, color="tomato",
              label=f"Total noise (thermal + confusion)")
    ax.axhline(sc * 1e3,  color="gray", ls=":", lw=1.5,
               label=f"Confusion floor {sc*1e3:.4f} mJy")
    ax.axvline(100, color="black", ls="--", lw=1, alpha=0.6, label="100 h")

    # Mark RMS at 100h
    idx100 = np.argmin(np.abs(t_arr - 100))
    for val, col, lbl in [
        (st_ch[idx100]  * 1e3, "steelblue", f"σ_th(1MHz,100h)={st_ch[idx100]*1e3:.4f}mJy"),
        (tot_bw[idx100] * 1e3, "tomato",    f"σ_tot(100h)={tot_bw[idx100]*1e3:.4f}mJy"),
    ]:
        ax.scatter([100], [val], s=60, color=col, zorder=5)
        ax.annotate(lbl, (100, val), xytext=(5, 0), textcoords="offset points",
                    fontsize=7.5, color=col)

    ax.set_xlabel("Integration time [h]", fontsize=10)
    ax.set_ylabel("RMS noise [mJy]", fontsize=10)
    ax.set_title(f"RMS Noise Level vs Integration Time  (30 MHz)\n"
                 f"{meta['label']}", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, "rms_noise.png"), dpi=110)
    plt.close()


def cmp_sep_uv(name, cfg, u_kl, v_kl, vis_hr):
    """UV coverage accumulated over 100 hours for one config."""
    sdir = _cmp_sep_dir(name)
    n_st = 1 + len(cfg["meta"]["out_centres"])   # stations
    n_bl = n_st * (n_st - 1) // 2

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(u_kl, v_kl, s=0.4, alpha=0.4, linewidths=0,
               c=CFG_COLORS.get(name, "steelblue"))
    ax.set_xlabel("u  [kλ]  @30 MHz", fontsize=10)
    ax.set_ylabel("v  [kλ]  @30 MHz", fontsize=10)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
    ax.grid(True, alpha=0.2)
    ax.set_title(f"UV Coverage — 100 h observation\n"
                 f"{cfg['meta']['label']}\n"
                 f"{n_st} stations → {n_bl} baselines  |  "
                 f"{vis_hr:.0f} h visibility  |  "
                 f"{len(u_kl)//2:,} UV points",
                 fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, "uv_coverage_100h.png"), dpi=110)
    plt.close()


def cmp_sep_detections(name, cfg, df_sens_cfg):
    """Detection bar chart + scatter for one config."""
    sdir = _cmp_sep_dir(name)
    colors = {"1-5":"#E53935","5-10":"#FB8C00","10-20":"#1E88E5","20-40":"#8E24AA"}

    # Per-band feasible counts
    feas_per_band = {}
    feas_names    = set()
    asp_names     = set()
    for bl in BAND_LABELS:
        sub = df_sens_cfg[df_sens_cfg["frequency_band"] == bl]
        f   = sub[sub["feasibility"] == "feasible"]["target_name"].unique()
        a   = sub[sub["feasibility"] == "aspirational"]["target_name"].unique()
        feas_per_band[bl] = (len(f), len(a))
        feas_names.update(f)
        asp_names.update(a)
    all_det = feas_names | asp_names

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Stacked bar
    feas_vals = [feas_per_band[b][0] for b in BAND_LABELS]
    asp_vals  = [feas_per_band[b][1] for b in BAND_LABELS]
    x_pos     = np.arange(len(BAND_LABELS))
    bars1 = ax1.bar(x_pos, feas_vals, color=[colors[b] for b in BAND_LABELS],
                    edgecolor="white", label="Feasible (<100 h)")
    bars2 = ax1.bar(x_pos, asp_vals, bottom=feas_vals,
                    color=[colors[b] for b in BAND_LABELS], alpha=0.4,
                    edgecolor="white", hatch="//", label="Aspirational (100-1000 h)")
    for i, (f, a) in enumerate(zip(feas_vals, asp_vals)):
        total = f + a
        if total:
            ax1.text(i, total + 0.15, str(total), ha="center",
                     fontsize=10, fontweight="bold")
    ax1.set_xticks(x_pos); ax1.set_xticklabels(BAND_LABELS, fontsize=10)
    ax1.set_ylabel("Number of targets", fontsize=10)
    ax1.set_title(f"Detections per band\n"
                  f"Total unique: {len(feas_names)} feasible / "
                  f"{len(asp_names)} aspirational",
                  fontsize=9)
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # Flux vs required time scatter
    if len(df_sens_cfg):
        best_t = df_sens_cfg.groupby("target_name")["required_t_h"].min().reset_index()
        sub_f  = best_t[best_t["required_t_h"] < 100]
        sub_a  = best_t[(best_t["required_t_h"] >= 100) & (best_t["required_t_h"] < 1000)]
        flux_map = df_sens_cfg.drop_duplicates("target_name").set_index("target_name")["target_flux_mJy"]
        for subset, col, lbl in [(sub_f, "steelblue", "Feasible"),
                                  (sub_a, "orange",    "Aspirational")]:
            if len(subset):
                fluxes = subset["target_name"].map(flux_map)
                ax2.scatter(fluxes, subset["required_t_h"],
                            c=col, alpha=0.75, edgecolors="k", lw=0.4,
                            s=30, label=lbl)
                for _, row in subset.iterrows():
                    if row["required_t_h"] < 100:
                        ax2.annotate(row["target_name"][:12],
                                     (flux_map.get(row["target_name"], np.nan),
                                      row["required_t_h"]),
                                     fontsize=5.5, alpha=0.8,
                                     xytext=(2, 2), textcoords="offset points")
        ax2.axhline(100,  color="red",   ls="--", lw=1.2, label="100 h")
        ax2.axhline(1000, color="orange",ls=":",  lw=1.2, label="1000 h")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("Target flux [mJy]", fontsize=10)
        ax2.set_ylabel("Required integration time [h]", fontsize=10)
        ax2.set_title("Flux vs required time", fontsize=9)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Detection Results — {cfg['meta']['label']}", fontsize=9,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, "detections.png"), dpi=110)
    plt.close()

    # Save target list
    with open(os.path.join(sdir, "detected_targets.txt"), "w") as fh:
        fh.write(f"Config: {name}\n{'='*60}\n")
        fh.write(f"FEASIBLE targets (<100 h):\n")
        for t in sorted(feas_names):
            row = df_sens_cfg[(df_sens_cfg["target_name"] == t) &
                               (df_sens_cfg["feasibility"] == "feasible")]
            best = row["required_t_h"].min() if len(row) else np.inf
            fh.write(f"  {t:<30s}  t_req={best:.1f} h\n")
        fh.write(f"\nASPIRATIONAL targets (100–1000 h):\n")
        for t in sorted(asp_names - feas_names):
            fh.write(f"  {t}\n")


# ── COMBINED plots — all 6 configs on one figure ──────────────────────────────

def cmp_cmb_sensitivity(all_cfgs):
    """Combined sensitivity plot — all 6 configs, at 30 MHz."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    t_arr = np.logspace(-2, 4, 500)
    for ax, core_label, keys in [
        (axes[0], "32×32 core", CFG_ORDER[:2] + [CFG_ORDER[4]]),
        (axes[1], "128×128 core", CFG_ORDER[2:4] + [CFG_ORDER[5]]),
    ]:
        for name in keys:
            cfg   = all_cfgs[name]
            N     = cfg["meta"]["N_total"]
            B_max = cfg["meta"]["B_max_m"]
            bw    = REF_BW
            sc    = confusion_limit_Jy(REF_FREQ, B_max)
            st1   = sigma_thermal_Jy(N, REF_FREQ, bw, 1.0)
            stot  = np.sqrt((st1 / np.sqrt(t_arr)) ** 2 + sc ** 2)
            ax.loglog(t_arr, NSIGMA * stot * 1e3,
                      color=CFG_COLORS[name], lw=2.5,
                      label=CFG_SHORT[name].replace("\n", " "))
        ax.axvline(100, color="gray", ls="--", lw=1.2, label="100 h")
        ax.set_xlabel("Integration time [h]", fontsize=11)
        ax.set_title(f"{core_label} — 5σ Sensitivity (30 MHz, 20 MHz BW)", fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("5σ Total Sensitivity [mJy]", fontsize=11)
    fig.suptitle("Sensitivity Comparison — All 6 Configurations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CMP_CMB, "sensitivity_comparison.png"), dpi=120)
    plt.close()


def cmp_cmb_beam(all_cfgs, af_store_new):
    """Combined beam pattern — 2×3 panel at 30 MHz."""
    bl   = "20-40"
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    metrics_table = []

    for ax, name in zip(axes, CFG_ORDER):
        l, m, AF_dB, B_norm = af_store_new[name][bl]
        fc  = BAND_CTR[BAND_LABELS.index(bl)]
        mtr = beam_metrics(B_norm, l, m, fc)
        hpbw_rad = 2 * np.sqrt(mtr["Omega_B"] / np.pi)
        hpbw_arcmin = np.degrees(hpbw_rad) * 60.0

        im = ax.pcolormesh(l, m, AF_dB, vmin=-30, vmax=0,
                           cmap="inferno", shading="auto")
        ax.contour(l, m, B_norm, levels=CONTOUR_LVL,
                   colors=CONTOUR_COL, linewidths=[0.7, 0.9, 1.2])
        plt.colorbar(im, ax=ax, label="dB", fraction=0.046, pad=0.04)
        ax.set_aspect("equal")
        ax.set_xlabel("l", fontsize=9); ax.set_ylabel("m", fontsize=9)
        ax.set_title(f"{CFG_SHORT[name].replace(chr(10), ' ')}\n"
                     f"HPBW={hpbw_arcmin:.1f}′  MSL={mtr['MSL_dB']:.1f} dB",
                     fontsize=8.5, color=CFG_COLORS[name], fontweight="bold")
        # border colour
        for spine in ax.spines.values():
            spine.set_edgecolor(CFG_COLORS[name]); spine.set_linewidth(2)
        metrics_table.append(dict(config=name, HPBW_arcmin=hpbw_arcmin,
                                  MSL_dB=mtr["MSL_dB"]))

    fig.suptitle("Beam Pattern Comparison — All 6 Configurations  (20–40 MHz ≈ 30 MHz)\n"
                 "Contours: 10% (cyan) / 30% (green) / 50% (white) of peak",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CMP_CMB, "beam_pattern_comparison.png"), dpi=120)
    plt.close()

    # 1-D cuts overlay
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax2, keys, core_l in [(axes2[0], CFG_ORDER[:2]+[CFG_ORDER[4]], "32×32"),
                               (axes2[1], CFG_ORDER[2:4]+[CFG_ORDER[5]], "128×128")]:
        for name in keys:
            l, m, _, B_norm = af_store_new[name][bl]
            mid = len(m) // 2
            cut = 10 * np.log10(B_norm[mid, :] + 1e-20)
            ax2.plot(l, cut, color=CFG_COLORS[name], lw=2,
                     label=CFG_SHORT[name].replace("\n", " "))
        ax2.axhline(-3,  color="red",    ls="--", lw=0.9, label="−3 dB")
        ax2.axhline(-10, color="orange", ls=":",  lw=0.9, label="−10 dB")
        ax2.set_ylim(-35, 2)
        ax2.set_xlabel("l  (East direction cosine)", fontsize=10)
        ax2.set_title(f"{core_l} core — 1-D beam cuts (30 MHz)", fontsize=10)
        ax2.legend(fontsize=8, ncol=2); ax2.grid(True, alpha=0.3)
    axes2[0].set_ylabel("Normalised power [dB]", fontsize=10)
    fig2.suptitle("1-D Beam Cut Comparison — All 6 Configurations  (20–40 MHz, m=0 slice)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CMP_CMB, "beam_1D_comparison.png"), dpi=120)
    plt.close()
    return metrics_table


def cmp_cmb_rms(all_cfgs):
    """Combined RMS noise at 30 MHz — all 6 configs."""
    t_arr = np.logspace(-2, 4, 500)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, keys, core_l in [(axes[0], CFG_ORDER[:2]+[CFG_ORDER[4]], "32×32"),
                              (axes[1], CFG_ORDER[2:4]+[CFG_ORDER[5]], "128×128")]:
        for name in keys:
            cfg   = all_cfgs[name]
            N     = cfg["meta"]["N_total"]
            B_max = cfg["meta"]["B_max_m"]
            sc    = confusion_limit_Jy(REF_FREQ, B_max)
            st1   = sigma_thermal_Jy(N, REF_FREQ, REF_BW, 1.0)
            stot  = np.sqrt((st1 / np.sqrt(t_arr)) ** 2 + sc ** 2)
            ax.loglog(t_arr, stot * 1e3,
                      color=CFG_COLORS[name], lw=2.5,
                      label=CFG_SHORT[name].replace("\n", " "))
            # Confusion floor
            ax.axhline(sc * 1e3, color=CFG_COLORS[name],
                       ls=":", lw=0.8, alpha=0.5)

        # Annotate at 100h
        ax.axvline(100, color="gray", ls="--", lw=1.2, label="100 h")
        ax.set_xlabel("Integration time [h]", fontsize=11)
        ax.set_title(f"{core_l} core — Total RMS Noise (30 MHz, 20 MHz BW)", fontsize=10)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("Total RMS Noise [mJy]  (= √(σ²_th + σ²_c))", fontsize=11)
    fig.suptitle("RMS Noise Level Comparison — All 6 Configurations\n"
                 "Dotted = confusion floor per config",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CMP_CMB, "rms_noise_comparison.png"), dpi=120)
    plt.close()


def cmp_cmb_uv(all_cfgs, uv_data):
    """Combined UV coverage — 2×3 panel."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    for ax, name in zip(axes, CFG_ORDER):
        u_kl, v_kl, vis_hr = uv_data[name]
        n_st = 1 + len(all_cfgs[name]["meta"]["out_centres"])
        n_bl = n_st * (n_st - 1) // 2
        ax.scatter(u_kl, v_kl, s=0.3, alpha=0.4, linewidths=0,
                   c=CFG_COLORS[name])
        ax.set_xlabel("u  [kλ]", fontsize=9)
        ax.set_ylabel("v  [kλ]", fontsize=9)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.4)
        ax.axvline(0, color="gray", lw=0.4)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_edgecolor(CFG_COLORS[name]); spine.set_linewidth(2)
        n_pts = len(u_kl) // 2
        ax.set_title(f"{CFG_SHORT[name].replace(chr(10), ' ')}\n"
                     f"{n_st} stations | {n_bl} baselines | {n_pts:,} UV pts",
                     fontsize=8, color=CFG_COLORS[name], fontweight="bold")
    fig.suptitle("UV Coverage Comparison — 100 Hours Observation  (30 MHz, 51 Peg b)\n"
                 "Moon rotation fills the UV plane over time",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CMP_CMB, "uv_coverage_comparison.png"), dpi=120)
    plt.close()

    # UV filling fraction bar chart
    n_pts_all = {name: len(uv_data[name][0]) // 2 for name in CFG_ORDER}
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars = ax2.bar(range(len(CFG_ORDER)),
                   [n_pts_all[n] for n in CFG_ORDER],
                   color=[CFG_COLORS[n] for n in CFG_ORDER],
                   edgecolor="white", lw=0.5)
    ax2.set_xticks(range(len(CFG_ORDER)))
    ax2.set_xticklabels([CFG_SHORT[n].replace("\n", "\n") for n in CFG_ORDER],
                         fontsize=9)
    ax2.set_ylabel("UV points accumulated in 100 h", fontsize=10)
    ax2.set_title("UV Coverage Richness — 100 h Observation at 30 MHz", fontsize=10)
    for bar, n in zip(bars, CFG_ORDER):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f"{n_pts_all[n]:,}", ha="center", fontsize=8, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CMP_CMB, "uv_richness_bar.png"), dpi=120)
    plt.close()


def cmp_cmb_detections(all_cfgs, df_sens_new):
    """Combined detection yield — grouped bars + best-band breakdown."""
    n_feas, n_asp, feas_names_all = {}, {}, {}
    for name in CFG_ORDER:
        sub   = df_sens_new[df_sens_new["config_name"] == name]
        best  = sub.groupby("target_name")["required_t_h"].min()
        n_feas[name] = int((best < 100).sum())
        n_asp[name]  = int(((best >= 100) & (best < 1000)).sum())
        feas_names_all[name] = sorted(best[best < 100].index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked bar: feasible + aspirational
    x   = np.arange(len(CFG_ORDER))
    ax1 = axes[0]
    fv  = [n_feas[n] for n in CFG_ORDER]
    av  = [n_asp[n]  for n in CFG_ORDER]
    b1  = ax1.bar(x, fv, color=[CFG_COLORS[n] for n in CFG_ORDER],
                  edgecolor="white", label="Feasible (<100 h)")
    b2  = ax1.bar(x, av, bottom=fv,
                  color=[CFG_COLORS[n] for n in CFG_ORDER], alpha=0.4,
                  hatch="//", edgecolor="white", label="Aspirational (100–1000 h)")
    for i, (f, a) in enumerate(zip(fv, av)):
        ax1.text(i, f + a + 0.2, str(f + a), ha="center",
                 fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([CFG_SHORT[n] for n in CFG_ORDER], fontsize=9)
    ax1.set_ylabel("Number of detectable targets", fontsize=10)
    ax1.set_title("Total Detection Yield\n(solid = <100h feasible, hatched = 100–1000h)",
                  fontsize=10)
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # Per-band breakdown (feasible only)
    ax2 = axes[1]
    bw = 0.12
    band_colors = {"1-5":"#E53935","5-10":"#FB8C00","10-20":"#1E88E5","20-40":"#8E24AA"}
    for bi, bl in enumerate(BAND_LABELS):
        per_band = []
        for name in CFG_ORDER:
            sub = df_sens_new[(df_sens_new["config_name"] == name) &
                               (df_sens_new["frequency_band"] == bl) &
                               (df_sens_new["feasibility"] == "feasible")]
            per_band.append(sub["target_name"].nunique())
        offset = (bi - 1.5) * bw
        ax2.bar(x + offset, per_band, bw, color=band_colors[bl],
                label=f"{bl} MHz", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([CFG_SHORT[n] for n in CFG_ORDER], fontsize=9)
    ax2.set_ylabel("Feasible targets per band", fontsize=10)
    ax2.set_title("Feasible Detections (<100 h) by Frequency Band", fontsize=10)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Detection Yield Comparison — All 6 Configurations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CMP_CMB, "detections_comparison.png"), dpi=120)
    plt.close()
    return n_feas, n_asp, feas_names_all


# ── Summary table ─────────────────────────────────────────────────────────────

def cmp_summary_table(all_cfgs, beam_metrics_list, uv_data, n_feas, n_asp,
                      feas_names_all):
    """Generate CSV + PNG summary table."""
    rows = []
    for name in CFG_ORDER:
        cfg  = all_cfgs[name]
        meta = cfg["meta"]
        N    = meta["N_total"]
        B_max = meta["B_max_m"]
        mtr_entry = next((m for m in beam_metrics_list if m["config"] == name), {})
        sc_ref   = confusion_limit_Jy(REF_FREQ, B_max)
        st_ref   = sigma_thermal_Jy(N, REF_FREQ, REF_BW, 100.0)
        stot_ref = np.sqrt(st_ref**2 + sc_ref**2)
        u_kl, v_kl, vis_hr = uv_data[name]
        n_uv = len(u_kl) // 2
        rows.append(dict(
            Config=CFG_SHORT[name].replace("\n", " "),
            N_elements=N,
            B_max_km=round(B_max / 1e3, 2),
            HPBW_arcmin=round(mtr_entry.get("HPBW_arcmin", np.nan), 1),
            MSL_dB=round(mtr_entry.get("MSL_dB", np.nan), 1),
            sigma_th_100h_mJy=round(st_ref * 1e3, 5),
            sigma_confusion_mJy=round(sc_ref * 1e3, 5),
            sigma_total_100h_mJy=round(NSIGMA * stot_ref * 1e3, 5),
            UV_points_100h=n_uv,
            vis_hours=round(vis_hr, 1),
            n_feasible=n_feas[name],
            n_aspirational=n_asp[name],
        ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CMP_CSV, "summary_table.csv"), index=False)

    # PNG table
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.axis("off")
    col_labels = list(df.columns)
    cell_vals  = df.values.tolist()
    tbl = ax.table(
        cellText=cell_vals,
        colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1976D2")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4ff")
        # Highlight best config per column (rows 1-6)
        cell.set_edgecolor("#cccccc")
    plt.tight_layout()
    plt.savefig(os.path.join(CMP_ROOT, "summary_table.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary_table.csv and summary_table.png")
    return df


# ── Interpretation and conclusion text ───────────────────────────────────────

def _write_interpretation(df_sum, n_feas, uv_data, beam_mtr):
    """Write interpretation.txt and final_conclusion.txt."""

    best_sens = df_sum.loc[df_sum["sigma_total_100h_mJy"].idxmin(), "Config"]
    best_beam_hpbw = df_sum.loc[df_sum["HPBW_arcmin"].idxmin(), "Config"]
    best_beam_msl  = df_sum.loc[df_sum["MSL_dB"].idxmin(), "Config"]
    best_rms  = df_sum.loc[df_sum["sigma_th_100h_mJy"].idxmin(), "Config"]
    best_uv   = max(n_feas, key=lambda k: len(uv_data[k][0]))
    best_det  = max(n_feas, key=n_feas.get)
    best_det_n = n_feas[best_det]
    best_det_label = CFG_SHORT[best_det].replace("\n", " ")

    interp = f"""
ALO ARRAY CONFIGURATION COMPARISON — INTERPRETATION
=====================================================

1. SENSITIVITY  (5σ limit at 30 MHz, 20 MHz BW, 100h integration)
   Best: {best_sens}
   • Sensitivity is set by the radiometer equation: σ_th ∝ 1/√(N(N−1)·Δν·t).
     The two 128×128 core configs (C, D, F) vastly outperform the 32×32 configs
     (A, B, E) because N(N−1) is ~256× larger.
   • Within each core size the long-distance configs (B, D) have slightly lower
     confusion floor (smaller HPBW → fewer background sources in the beam).
   • Config D (Ring 128×128, long) and Config F (Cross 128×128) tie for best
     thermal sensitivity; D has marginally lower confusion floor due to its
     larger maximum baseline.

2. BEAM-PATTERN QUALITY  (HPBW and MSL at 30 MHz)
   Narrowest HPBW: {best_beam_hpbw}
   Lowest MSL: {best_beam_msl}
   • HPBW scales as λ/B_max; the long-distance configs (B, D) and the cross
     configs (E, F) all achieve HPBW ≈ 63–75 arcmin at 30 MHz.
   • The short-distance configs (A, C) have much wider beams (HPBW > 200 arcmin)
     due to their shorter maximum baseline; they resolve fewer background sources
     and suffer from higher confusion noise.
   • The cross configs (E, F) have slightly higher sidelobe suppression than the
     symmetric rings because the asymmetric arm lengths (N≤1km, E≤5km, S≤3km,
     W≤3km) break the periodicity that causes coherent aliasing rings.

3. RMS NOISE LEVEL  (thermal RMS at 30 MHz, 20 MHz BW, 100h)
   Best: {best_rms}
   • RMS noise is dominated by thermal noise, which follows σ_th = A/√t where
     A = SEFD_elem / √(N_pol·N(N−1)·Δν).
   • All 128×128 configs achieve σ_th < 0.01 mJy at 100h in the 20–40 MHz band.
   • The 32×32 configs remain above 0.5 mJy even at 100h, making them
     science-limited for all but the brightest targets.

4. UV COVERAGE  (100 hours of observation, 30 MHz, 51 Peg b)
   Most UV points: {CFG_SHORT[best_uv].replace(chr(10), " ")}
   • The number of UV points scales as N_stations × (N_stations−1) × n_time_steps.
   • Configs with more outrigger sub-arrays (all 6 new configs have 17 stations,
     giving 272 unique baselines) fill the UV plane more richly than the old
     single-outrigger designs (5 stations, 20 baselines).
   • Long-distance configs (B, D) spread UV points to larger (u,v) radii,
     providing higher angular resolution and better deconvolution capability.
   • The cross configs (E, F) produce elongated UV tracks due to the asymmetric
     arm lengths, giving anisotropic resolution that may be advantageous for
     specific science targets.

5. NUMBER OF DETECTIONS  (targets detectable in <100h)
   Best: {best_det_label}  ({best_det_n} feasible targets)
   • Detection yield is driven by: (a) collecting area N × A_eff,ele,
     (b) maximum baseline B_max (determines confusion floor), and
     (c) thermal sensitivity σ_th.
   • All 128×128 configs achieve 24–33 feasible detections; 32×32 configs
     achieve only 3 each regardless of outrigger distance.
   • Config D (Ring 128×128, long) and Config F (Cross 128×128) tie at
     33 feasible detections — the maximum achievable with the current
     target catalogue and array sensitivity.
   • The 20–40 MHz band provides the most detections because:
     (i) predicted ECM fluxes are highest here, and
     (ii) the sky temperature is lower than at lower frequencies.
"""

    conclusion = f"""
FINAL CONCLUSION: BEST CONFIGURATION AND TRADE-OFFS
====================================================

OVERALL WINNER: Config D — Ring 128×128, long-distance
  • N_elements = 16640  |  B_max = 10.98 km
  • n_feasible detections = 33  (joint best with Config F)
  • σ_th(100h, 20MHz BW, 30MHz) = {df_sum.loc[df_sum["Config"].str.contains("D:"), "sigma_th_100h_mJy"].values[0]:.5f} mJy
  • σ_confusion(30MHz) = {df_sum.loc[df_sum["Config"].str.contains("D:"), "sigma_confusion_mJy"].values[0]:.5f} mJy
  • HPBW ≈ {df_sum.loc[df_sum["Config"].str.contains("D:"), "HPBW_arcmin"].values[0]:.1f} arcmin  |  MSL ≈ {df_sum.loc[df_sum["Config"].str.contains("D:"), "MSL_dB"].values[0]:.1f} dB

TRADE-OFFS BETWEEN ALL 6 CONFIGURATIONS:

  Config A (Ring 32×32, short): Limited by small collecting area (N=1088).
    Only 3 feasible targets. Beam too wide (HPBW > 500 arcmin) for sub-arcminute
    science. Suitable only for very bright transient sources.

  Config B (Ring 32×32, long): Same collecting area as A but better angular
    resolution (HPBW ≈ 75 arcmin) and lower confusion. Still limited to 3
    feasible targets. Useful as a cross-check or pathfinder array.

  Config C (Ring 128×128, short): Much better sensitivity (N=16640) but
    short baseline limits resolution (HPBW ≈ 250 arcmin) and raises confusion
    noise significantly. 24 feasible targets. Good intermediate option if
    outrigger deployment to 5 km is not feasible.

  Config D (Ring 128×128, long): Best overall configuration.
    Combines maximum collecting area with the longest maximum baseline
    (10.98 km), giving minimum HPBW and minimum confusion.
    33 feasible targets. Symmetric UV coverage aids image reconstruction.

  Config E (Cross 32×32): Same element count as A/B but asymmetric
    baselines. Detection yield identical (3 targets) — limited by collecting
    area, not by geometry. Asymmetric UV coverage provides anisotropic resolution.

  Config F (Cross 128×128): Ties Config D at 33 feasible detections.
    B_max = 8.98 km vs 10.98 km for D, so slightly higher confusion floor
    and wider HPBW. Asymmetric arm lengths suppress coherent aliasing,
    giving better sidelobe characteristics. Preferred if sidelobe suppression
    is a priority (e.g. bright RFI sources near targets of interest).

RECOMMENDATION:
  → Primary: Deploy Config D (Ring 128×128, long) for maximum science yield.
  → If sidelobe contamination is a concern: use Config F (Cross 128×128) instead.
  → Phase 1 pathfinder: Config C (Ring 128×128, short) if outrigger cable
    runs beyond ~1.5 km are not feasible in early deployment stages.
  → Do not deploy 32×32 configurations (A, B, E) as primary science arrays;
    they lack the collecting area to achieve competitive sensitivity.
"""

    with open(os.path.join(CMP_ROOT, "interpretation.txt"), "w") as fh:
        fh.write(interp)
    with open(os.path.join(CMP_ROOT, "final_conclusion.txt"), "w") as fh:
        fh.write(conclusion)
    print(interp)
    print(conclusion)


# ── Master comparison runner ──────────────────────────────────────────────────

def run_full_comparison(targets):
    """
    Generate the full 6-configuration comparison.
    Saves all outputs under outputs/.../Comparison/.
    """
    print("\n" + "=" * 72)
    print("  FULL COMPARISON: 6 CONFIGURATIONS (A–F)")
    print("=" * 72)

    # ── 1. Build / reuse the 6 new configs ──
    all_cfgs = build_new_configs()

    # ── 2. Compute AF for all 6 × 4 bands ──
    print("\n── Computing array factor ──")
    af_store = {}
    for name, cfg in all_cfgs.items():
        af_store[name] = {}
        for bl, fc in zip(BAND_LABELS, BAND_CTR):
            l, m, AF_dB, B_norm = compute_af(cfg["pos_enu"], cfg["meta"], fc, N_GRID)
            af_store[name][bl] = (l, m, AF_dB, B_norm)
    print("  Done.")

    # ── 3. Sensitivity data ──
    print("\n── Computing sensitivity ──")
    df_sens = compute_new_sensitivity(all_cfgs, targets)

    # ── 4. UV coverage for 100h ──
    print("\n── Computing 100-h UV coverage (may take ~1 min) ──")
    uv_data = {}
    for name, cfg in all_cfgs.items():
        u, v, vis = _uv_100h(cfg)
        uv_data[name] = (u, v, vis)
        n_st = 1 + len(cfg["meta"]["out_centres"])
        print(f"  {name}: {vis:.0f} h visibility, {len(u)//2:,} UV points "
              f"({n_st} stations)")

    # ── 5. Separate plots per config ──
    print("\n── Separate plots per configuration ──")
    for name, cfg in all_cfgs.items():
        df_cfg = df_sens[df_sens["config_name"] == name]
        cmp_sep_sensitivity(name, cfg, df_cfg)
        cmp_sep_beam(name, cfg, af_store[name])
        cmp_sep_rms(name, cfg)
        u, v, vis = uv_data[name]
        cmp_sep_uv(name, cfg, u, v, vis)
        cmp_sep_detections(name, cfg, df_cfg)
        # Also generate schematic layout
        plot_schematic_layout(name, cfg, savedir=_cmp_sep_dir(name))
        print(f"  Config {cfg['meta']['cfg_id'].upper()}: "
              f"layout + beam + UV + sensitivity + detections → separate/{name}/")

    # ── 6. Combined comparison plots ──
    print("\n── Combined comparison plots ──")
    cmp_cmb_sensitivity(all_cfgs)
    print("  Saved combined/sensitivity_comparison.png")

    beam_mtr = cmp_cmb_beam(all_cfgs, af_store)
    print("  Saved combined/beam_pattern_comparison.png + beam_1D_comparison.png")

    cmp_cmb_rms(all_cfgs)
    print("  Saved combined/rms_noise_comparison.png")

    cmp_cmb_uv(all_cfgs, uv_data)
    print("  Saved combined/uv_coverage_comparison.png + uv_richness_bar.png")

    n_feas, n_asp, feas_names_all = cmp_cmb_detections(all_cfgs, df_sens)
    print("  Saved combined/detections_comparison.png")

    # ── 7. Summary table ──
    print("\n── Summary table ──")
    df_sum = cmp_summary_table(all_cfgs, beam_mtr, uv_data, n_feas, n_asp,
                               feas_names_all)

    # ── 8. Interpretation + conclusion ──
    print("\n── Interpretation and conclusion ──")
    _write_interpretation(df_sum, n_feas, uv_data, beam_mtr)
    with open(os.path.join(CMP_ROOT, "feasible_targets_all_configs.txt"), "w") as fh:
        fh.write("FEASIBLE TARGETS (<100h) — ALL 6 CONFIGURATIONS\n")
        fh.write("=" * 60 + "\n\n")
        for name in CFG_ORDER:
            fh.write(f"\n{CFG_SHORT[name].replace(chr(10), ' ')}"
                     f"  [{n_feas[name]} targets]\n")
            fh.write("-" * 50 + "\n")
            for t in feas_names_all[name]:
                fh.write(f"  {t}\n")
    print("  Saved feasible_targets_all_configs.txt")

    print(f"\n  All outputs saved to: {CMP_ROOT}")
    print("=" * 72)
    return all_cfgs, af_store, df_sens, df_sum


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("ALO Geometry Trade-off Study — Start")
    print(f"Geometries: {GEOMETRIES}")
    print(f"Core sizes: 32×32, 128×128  |  Distances: {DISTANCES} km")
    print(f"A_eff per element: {A_EFF_ELE} m²  (fixed physical constant)\n")

    # Step 1
    configs = build_all_configs()

    # Step 2
    af_store = compute_all_af(configs)

    # Step 3
    all_metrics = compute_all_metrics(configs, af_store)

    # Load targets (same catalogue as main script)
    print("\n── Loading target catalogue ──")
    targets = load_targets()

    # Step 4
    df_sens = compute_all_sensitivity(configs, targets)

    # Summary table
    df_sum = compile_summary(configs, all_metrics, df_sens)

    # Comparison plots
    comparison_plots(df_sum)

    # Beam contour plots: 10%, 30%, 50% of peak
    beam_contour_plots(configs, af_store)

    # Symmetry diagnostic
    symmetry_diagnostic_plot(configs, af_store)

    # UV coverage with multi-frequency synthesis + cross configs
    cross_cfgs = build_cross_configs()
    plot_uv_coverage(configs, cross_cfgs)

    # Cross configuration analysis
    run_cross_analysis(cross_cfgs, configs, targets)

    # Print summary
    print_summary(df_sum)

    # Asymmetric beam study
    run_asymmetric_beam_study()

    # Extended source analysis
    run_extended_source_analysis()

    # ── NEW CONFIGURATIONS (a–f) ────────────────────────────────────────────
    new_cfgs, af_new, df_sens_new, df_sum_new = run_new_configs_analysis(targets)

    # ── SYMMETRIC CONFIG COMPARISON: old vs corrected ───────────────────────
    run_symmetric_comparison(configs, af_store, new_cfgs, af_new,
                             df_sens, df_sens_new, targets)

    # ── FULL COMPARISON (A–F) ────────────────────────────────────────────────
    run_full_comparison(targets)

    # ── ADVANCED ANALYSES ───────────────────────────────────────────────────
    run_weighted_beamforming(new_cfgs)
    run_element_pattern_study(new_cfgs)
    run_lunar_rotation_uv(new_cfgs)
    run_outrigger_size_study(targets)
    run_calibration_architecture()
    run_plasma_effects()
    run_farside_comparison(targets)
    run_deployment_study()

    print(f"\nAll outputs saved to:\n  {GT_PLOT}\n  {GT_CSV}")
    print("ALO Geometry Trade-off Study — Done")


if __name__ == "__main__":
    main()
