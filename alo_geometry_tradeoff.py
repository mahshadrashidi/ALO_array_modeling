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
    D_SPACE, C, NSIGMA, ETA,
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
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("ALO Geometry Trade-off Study — Start")
    print(f"Geometries: {GEOMETRIES}")
    print(f"Core sizes: 32×32, 128×128  |  Distances: {DISTANCES} km\n")

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

    # Symmetry diagnostic
    symmetry_diagnostic_plot(configs, af_store)

    # Print summary
    print_summary(df_sum)

    # Asymmetric beam study
    run_asymmetric_beam_study()

    print(f"\nAll outputs saved to:\n  {GT_PLOT}\n  {GT_CSV}")
    print("ALO Geometry Trade-off Study — Done")


if __name__ == "__main__":
    main()
