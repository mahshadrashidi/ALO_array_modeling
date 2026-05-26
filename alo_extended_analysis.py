"""
ALO Extended Analysis
=====================
Reads sensitivity_results.csv produced by alo_array_modeling.py and adds:
  A. Solar-system reference emitter detection horizons
  B. Target × config detection matrix heatmap + CSV
  C. Cumulative N_detected vs integration time curves + CSV
  D. Per-band sensitivity curves with target flux scatter overlay
  E. Weighted configuration ranking/scoring + CSV
  F. Aspirational target detailed analysis + CSV
  G. Best config per target CSV
  H. Per-config summary CSV
"""

from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE = "/home/mahshad/Documents/Thesis/ALO_modeling"
CSV_IN = os.path.join(BASE, "outputs/csv/sensitivity_results.csv")
OUT_P = os.path.join(BASE, "outputs/plots")
OUT_C = os.path.join(BASE, "outputs/csv")
os.makedirs(OUT_P, exist_ok=True)
os.makedirs(OUT_C, exist_ok=True)

# ── physical constants (same as alo_array_modeling.py) ───────────────────────
C = 3.0e8
KB = 1.38e-23
JY = 1.0e-26
N_POL = 2
NSIGMA = 5.0
D_SPACE = 5.15          # element spacing [m]
A_EFF_ELE = 6.28         # effective area per element [m²]
ETA = 0.7

BANDS = {
    "1-5":   (1,  5,  3.0,  4.0),   # (f_lo, f_hi, f_centre, bw_eff)
    "5-10":  (5, 10,  7.5,  5.0),
    "10-20": (10, 20, 15.0, 10.0),
    "20-40": (20, 40, 30.0, 20.0),
}


def T_sky(nu_MHz):
    return 180.0 * (nu_MHz / 180.0) ** (-2.6)


def sefd_element(nu_MHz):
    return 2 * KB * T_sky(nu_MHz) / A_EFF_ELE / JY   # Jy


def sigma_thermal(N, nu_MHz, bw_MHz, t_hr):
    """Interferometric radiometer sensitivity [Jy]."""
    n_bl = N * (N - 1)
    t_sec = t_hr * 3600.0
    bw_hz = bw_MHz * 1e6
    sefd = sefd_element(nu_MHz)
    return sefd / np.sqrt(N_POL * n_bl * bw_hz * t_sec)


def confusion_limit(nu_MHz, B_max_m):
    lam = C / (nu_MHz * 1e6)
    theta_r = lam / B_max_m
    theta_am = np.degrees(theta_r) * 60.0
    return 0.2e-3 * (nu_MHz / 74.0) ** 0.8 * (theta_am / 4.0)   # Jy


def max_baseline(core_n, out_dist_km):
    return 2 * out_dist_km * 1e3 + core_n * D_SPACE


def required_t(S_Jy, N, nu_MHz, bw_MHz, B_max_m):
    """Return integration time [hours] for 5σ detection."""
    s_c = confusion_limit(nu_MHz, B_max_m)
    sig_total_floor = np.sqrt(s_c ** 2 + 1e-40)
    if NSIGMA * sig_total_floor >= S_Jy:
        return np.inf
    sefd = sefd_element(nu_MHz)
    n_bl = N * (N - 1)
    bw_hz = bw_MHz * 1e6
    needed_thermal = np.sqrt((S_Jy / NSIGMA) ** 2 - s_c ** 2)
    t_sec = (sefd / (needed_thermal * np.sqrt(N_POL * n_bl * bw_hz))) ** 2
    return t_sec / 3600.0


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading sensitivity_results.csv …")
df = pd.read_csv(CSV_IN)
df["required_t_h"] = pd.to_numeric(
    df["required_t_h"], errors="coerce").fillna(1e12)

configs_ordered = [
    "core32x32_out0.1km",  "core32x32_out0.25km", "core32x32_out0.5km",
    "core32x32_out0.75km", "core32x32_out1.0km",  "core32x32_out5.0km",
    "core128x128_out0.1km", "core128x128_out0.25km", "core128x128_out0.5km",
    "core128x128_out0.75km", "core128x128_out1.0km", "core128x128_out5.0km",
]

# short labels for plots
CLABEL = {c: c.replace("core", "").replace(
    "out", "").replace("km", "km") for c in configs_ordered}

print(f"  {len(df)} rows | {df['target_name'].nunique()} targets | "
      f"{df['config_name'].nunique()} configs | "
      f"feasible={df['feasibility'].eq('feasible').sum()} "
      f"aspirational={df['feasibility'].eq('aspirational').sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# A.  SOLAR-SYSTEM REFERENCE EMITTER DETECTION HORIZONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[A] Solar-system reference emitter analysis …")

# Reference emitter catalogue
#   flux_Jy at reference distance d_ref_AU
#   peak_freq_MHz  (centre of dominant emission band)
#   freq_range_MHz = (f_lo, f_hi) – only observable in this range
SOLAR_EMITTERS = {
    "Jupiter DAM":  dict(flux_Jy=1e6,    d_ref_AU=5.2,  peak_freq_MHz=22.0,  band=(10, 40)),
    "Earth AKR":    dict(flux_Jy=1e7,    d_ref_AU=1.0,  peak_freq_MHz=250.0, band=(100, 700)),
    "Saturn SKR":   dict(flux_Jy=1e5,    d_ref_AU=9.5,  peak_freq_MHz=0.2,   band=(0.02, 1.2)),
    "Uranus UKR":   dict(flux_Jy=1e4,    d_ref_AU=19.2, peak_freq_MHz=0.08,  band=(0.01, 0.9)),
    "Neptune NKR":  dict(flux_Jy=5e3,    d_ref_AU=30.0, peak_freq_MHz=0.03,  band=(0.005, 0.6)),
}

# For each emitter compute detection horizon (max distance [pc] for 5σ in 100h)
# using the best ALO config (core128x128_out5.0km) at the most favourable sub-band

rows_ss = []
d_pc_arr = np.logspace(-2, 4, 1000)  # 0.01 → 10000 pc
AU_PC = 4.848e-6                      # 1 AU in parsec

for name, em in SOLAR_EMITTERS.items():
    # flux scaled as 1/d²; flux at d_ref converted from AU to pc
    d_ref_pc = em["d_ref_AU"] * AU_PC
    flux_at_ref_Jy = em["flux_Jy"]
    flux_arr_Jy = flux_at_ref_Jy * (d_ref_pc / d_pc_arr) ** 2

    # find which ALO sub-band overlaps the emitter band
    best_band = None
    for bk, (f_lo, f_hi, f_c, bw) in BANDS.items():
        if em["peak_freq_MHz"] >= f_lo and em["peak_freq_MHz"] <= f_hi:
            best_band = (bk, f_c, bw)
            break
    if best_band is None:
        best_band = ("1-5", 3.0, 4.0)   # default if out of ALO range

    bk, f_c, bw = best_band
    N_best = 17408        # core128x128_out5.0km
    Bmax = max_baseline(128, 5.0)
    t_arr = np.array([1.0, 10.0, 100.0, 1000.0])

    thresh_arr = []
    for t_h in t_arr:
        sig_th = sigma_thermal(N_best, f_c, bw, t_h)
        sig_c = confusion_limit(f_c, Bmax)
        thresh_arr.append(NSIGMA * np.sqrt(sig_th**2 + sig_c**2))

    # detection horizon at each integration time
    horizons = {}
    for t_h, thresh in zip(t_arr, thresh_arr):
        d_max_pc = d_ref_pc * np.sqrt(flux_at_ref_Jy / thresh)
        horizons[f"horizon_{int(t_h)}h_pc"] = d_max_pc

    rows_ss.append(dict(
        emitter=name,
        flux_at_ref_Jy=flux_at_ref_Jy,
        d_ref_AU=em["d_ref_AU"],
        peak_freq_MHz=em["peak_freq_MHz"],
        alo_band=bk,
        **horizons,
    ))

df_ss = pd.DataFrame(rows_ss)
df_ss.to_csv(os.path.join(
    OUT_C, "solar_system_detection_ranges.csv"), index=False)
print(f"  Saved solar_system_detection_ranges.csv  ({len(df_ss)} rows)")

# ── plot: flux vs distance for each emitter + ALO thresholds ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()
colors_t = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]

for idx, (name, em) in enumerate(SOLAR_EMITTERS.items()):
    ax = axes[idx]
    d_ref_pc = em["d_ref_AU"] * AU_PC
    flux_arr = em["flux_Jy"] * (d_ref_pc / d_pc_arr) ** 2

    ax.loglog(d_pc_arr, flux_arr * 1e3, color="black", lw=2, label=name)

    bk, f_c, bw = best_band  # reuse last; recompute properly
    for bk2, (f_lo, f_hi, f_c2, bw2) in BANDS.items():
        if em["peak_freq_MHz"] >= f_lo and em["peak_freq_MHz"] <= f_hi:
            f_c2_use = f_c2
            bw_use = bw2
            break
    else:
        f_c2_use = 3.0
        bw_use = 4.0

    for t_h, col, ls in zip([1, 10, 100, 1000], colors_t, ["-", "--", "-.", ":"]):
        sig_th = sigma_thermal(N_best, f_c2_use, bw_use, t_h)
        sig_c = confusion_limit(f_c2_use, max_baseline(128, 5.0))
        thresh = NSIGMA * np.sqrt(sig_th**2 + sig_c**2)
        ax.axhline(thresh * 1e3, color=col, ls=ls, lw=1.4, alpha=0.85,
                   label=f"ALO {t_h}h σ×5")

    # mark typical exoplanet distances
    for d_ex, lab in [(1.3, "Prox Cen"), (10, "10 pc"), (100, "100 pc")]:
        ax.axvline(d_ex, color="grey", ls=":", alpha=0.5, lw=0.9)
        ax.text(d_ex*1.05, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1e3,
                lab, fontsize=7, color="grey", va="top")

    ax.set_xlabel("Distance [pc]", fontsize=9)
    ax.set_ylabel("Flux [mJy]",    fontsize=9)
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.01, 1e4)

axes[-1].axis("off")   # 6th panel spare
fig.suptitle("Solar-System Radio Emitters: Flux vs Distance & ALO 5σ Thresholds\n"
             "(best config: core128×128 + 5 km outriggers)", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "solar_system_detection_horizons.png"), dpi=150)
plt.close(fig)
print("  Saved solar_system_detection_horizons.png")


# ══════════════════════════════════════════════════════════════════════════════
# B.  TARGET × CONFIG DETECTION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n[B] Detection matrix (target × config) …")

# For each (target, config) use the best (minimum) t_req across all sub-bands
det = (df.groupby(["target_name", "config_name"])["required_t_h"]
         .min()
         .reset_index()
         .rename(columns={"required_t_h": "best_t_h"}))

# Only keep targets that have at least one feasible or aspirational entry
keep_targets = df.loc[df["feasibility"].isin(["feasible", "aspirational"]),
                      "target_name"].unique()
det_filt = det[det["target_name"].isin(keep_targets)].copy()

# Pivot to matrix
mat = det_filt.pivot(index="target_name",
                     columns="config_name", values="best_t_h")
# reorder columns
mat = mat.reindex(columns=[c for c in configs_ordered if c in mat.columns])
mat = mat.fillna(1e12)
mat_log = np.log10(mat.values.clip(1e-3, 1e12))

det_filt.to_csv(os.path.join(OUT_C, "detection_matrix.csv"), index=False)
print(f"  Saved detection_matrix.csv  ({det_filt['target_name'].nunique()} targets × "
      f"{det_filt['config_name'].nunique()} configs)")

# ── heatmap ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, max(8, len(mat)*0.28)))
cmap = plt.cm.RdYlGn_r
norm = mcolors.Normalize(vmin=-1, vmax=4)

im = ax.imshow(mat_log, aspect="auto", cmap=cmap, norm=norm)

ax.set_xticks(range(len(mat.columns)))
ax.set_xticklabels([CLABEL.get(c, c) for c in mat.columns],
                   rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(mat.index)))
ax.set_yticklabels(mat.index, fontsize=6.5)
ax.set_xlabel("Array Configuration", fontsize=10)
ax.set_ylabel("Target", fontsize=10)
ax.set_title("Detection Matrix — log₁₀(best t_req [h]) per (target, config)\n"
             "Green = fast detection, Red = slow / impossible", fontsize=11)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("log₁₀(t_req [h])", fontsize=9)
cbar.set_ticks([-1, 0, 1, 2, 3, 4])
cbar.set_ticklabels(["0.1h", "1h", "10h", "100h",
                    "1000h", ">10000h"], fontsize=8)

# feasibility boundary lines
ax.axhline(-0.5, color="white", lw=0.4)
for ix in range(len(mat.columns)):
    for iy in range(len(mat.index)):
        v = mat_log[iy, ix]
        if v < np.log10(100):
            ax.add_patch(plt.Rectangle((ix-0.5, iy-0.5), 1, 1,
                                       fill=False, edgecolor="lime", lw=0.6))
        elif v < np.log10(1000):
            ax.add_patch(plt.Rectangle((ix-0.5, iy-0.5), 1, 1,
                                       fill=False, edgecolor="gold", lw=0.4))

plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "detection_matrix_heatmap.png"), dpi=150)
plt.close(fig)
print("  Saved detection_matrix_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# C.  CUMULATIVE N_DETECTED vs INTEGRATION TIME
# ══════════════════════════════════════════════════════════════════════════════
print("\n[C] Cumulative detection curves …")

t_grid = np.logspace(-2, 4, 500)   # 0.01 h → 10000 h
cum_rows = []

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
cmap_c = plt.cm.tab20
color_arr = [cmap_c(i / 12) for i in range(12)]

for ci, cfg in enumerate(configs_ordered):
    sub = df[df["config_name"] == cfg][["target_name", "required_t_h"]].copy()
    # best t per target across all bands
    best_t = sub.groupby("target_name")["required_t_h"].min().sort_values()
    n_cum = np.array([(best_t <= t).sum() for t in t_grid])

    label = CLABEL.get(cfg, cfg)
    core = "128×128" if "128" in cfg else "32×32"
    ax = axes[0] if "32x32" in cfg else axes[1]
    ls = "-"
    ax.semilogx(t_grid, n_cum, color=color_arr[ci % 12], lw=1.8,
                linestyle=ls, label=label)

    for t_h in t_grid:
        cum_rows.append(
            dict(config=cfg, t_h=t_h, n_detected=(best_t <= t_h).sum()))

for ax, title in zip(axes, ["Core 32×32", "Core 128×128"]):
    ax.axvline(100,  color="orange", ls="--", lw=1.2, alpha=0.7, label="100h")
    ax.axvline(1000, color="red",    ls="--", lw=1.2, alpha=0.7, label="1000h")
    ax.set_xlabel("Integration time [h]", fontsize=11)
    ax.set_ylabel("Cumulative N detections", fontsize=11)
    ax.set_title(f"Cumulative Detections – {title}", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.01, 1e4)

fig.suptitle("Cumulative Number of Detected Exoplanets vs Integration Time\n(5σ threshold, best sub-band per target)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "cumulative_detections.png"), dpi=150)
plt.close(fig)

df_cum = pd.DataFrame(cum_rows)
df_cum.to_csv(os.path.join(OUT_C, "cumulative_detection.csv"), index=False)
print(f"  Saved cumulative_detection.csv  ({len(df_cum)} rows)")
print("  Saved cumulative_detections.png")


# ══════════════════════════════════════════════════════════════════════════════
# D.  PER-BAND SENSITIVITY CURVES + TARGET FLUX OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[D] Per-band sensitivity curves with target flux overlay …")

T_INT_CURVES = [0.1, 1, 10, 100, 1000]   # hours
CURVE_COLORS = ["#9C27B0", "#2196F3", "#4CAF50", "#FF9800", "#F44336"]

all_configs_meta = {
    "core32x32_out0.1km":   dict(N=1088,  out_km=0.1,  core_n=32),
    "core32x32_out0.25km":  dict(N=1088,  out_km=0.25, core_n=32),
    "core32x32_out0.5km":   dict(N=1088,  out_km=0.5,  core_n=32),
    "core32x32_out0.75km":  dict(N=1088,  out_km=0.75, core_n=32),
    "core32x32_out1.0km":   dict(N=1088,  out_km=1.0,  core_n=32),
    "core32x32_out5.0km":   dict(N=1088,  out_km=5.0,  core_n=32),
    "core128x128_out0.1km": dict(N=17408, out_km=0.1,  core_n=128),
    "core128x128_out0.25km": dict(N=17408, out_km=0.25, core_n=128),
    "core128x128_out0.5km": dict(N=17408, out_km=0.5,  core_n=128),
    "core128x128_out0.75km": dict(N=17408, out_km=0.75, core_n=128),
    "core128x128_out1.0km": dict(N=17408, out_km=1.0,  core_n=128),
    "core128x128_out5.0km": dict(N=17408, out_km=5.0,  core_n=128),
}

band_list = [("1-5", 1, 5, 3.0, 4.0),
             ("5-10", 5, 10, 7.5, 5.0),
             ("10-20", 10, 20, 15.0, 10.0),
             ("20-40", 20, 40, 30.0, 20.0)]

for bname, f_lo, f_hi, f_c, bw in band_list:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    nu_arr = np.linspace(f_lo, f_hi, 300)

    for col_idx, core_n_sel in enumerate([32, 128]):
        ax = axes[col_idx]
        sel_cfgs = [c for c, m in all_configs_meta.items() if m["core_n"]
                    == core_n_sel]

        for ci_t, (t_h, col) in enumerate(zip(T_INT_CURVES, CURVE_COLORS)):
            # pick representative config (largest outrigger → worst confusion, best thermal)
            cfg_rep = [c for c in sel_cfgs if "5.0km" in c][0]
            meta = all_configs_meta[cfg_rep]
            N = meta["N"]
            Bmax = max_baseline(core_n_sel, meta["out_km"])

            sens_arr = []
            for nu in nu_arr:
                s_th = sigma_thermal(N, nu, bw, t_h)
                s_c = confusion_limit(nu, Bmax)
                sens_arr.append(
                    NSIGMA * np.sqrt(s_th**2 + s_c**2) * 1e3)  # mJy

            ax.semilogy(nu_arr, sens_arr, color=col, lw=1.8,
                        label=f"t={t_h}h")

        # confusion-only limit band across outrigger distances
        out_kms = [0.1, 5.0]
        for ok, lsty in zip(out_kms, [":", "--"]):
            Bmax = max_baseline(core_n_sel, ok)
            s_c_arr = [confusion_limit(nu, Bmax)*1e3 for nu in nu_arr]
            ax.semilogy(nu_arr, s_c_arr, color="black", lw=1.2,
                        ls=lsty, alpha=0.6, label=f"Confusion {ok}km")

        # scatter targets in this band
        tgt_band = df[(df["frequency_band"] == bname)
                      ].drop_duplicates("target_name")
        ax.scatter(tgt_band["target_freq_MHz"], tgt_band["target_flux_mJy"],
                   c="steelblue", s=12, alpha=0.7, zorder=5, label="Targets")

        ax.set_xlabel("Frequency [MHz]", fontsize=10)
        ax.set_ylabel("5σ Sensitivity [mJy]", fontsize=10)
        ax.set_title(
            f"Band {bname} MHz — core {core_n_sel}×{core_n_sel} (+5km outrigger)", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim(f_lo, f_hi)

    fig.suptitle(f"Per-Band Sensitivity Curves with Target Flux Overlay\nBand: {bname} MHz",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(
        OUT_P, f"sensitivity_curves_band{bname.replace('-','_')}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved sensitivity_curves_band{bname.replace('-','_')}.png")


# ══════════════════════════════════════════════════════════════════════════════
# E.  CONFIGURATION RANKING / SCORING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[E] Configuration ranking …")

# Metrics per config (across all sub-bands, best t per target):
#   n_feasible     = targets with t < 100h
#   n_aspirational = targets with 100 <= t < 1000h
#   median_thermal = median thermal sensitivity (Jy) at f_c of each band, 100h
#   min_confusion  = confusion floor (Jy) at 30 MHz (best ALO band)
#   score = w1*n_feas + w2*n_asp + w3*(1/median_thermal_norm) + w4*(1/conf_norm)

score_rows = []
for cfg in configs_ordered:
    sub = df[df["config_name"] == cfg]
    meta = all_configs_meta[cfg]
    N = meta["N"]
    out_km = meta["out_km"]
    cn = meta["core_n"]

    # best t per target
    best_t = sub.groupby("target_name")["required_t_h"].min()
    n_feas = (best_t < 100).sum()
    n_asp = ((best_t >= 100) & (best_t < 1000)).sum()
    n_tot = n_feas + n_asp

    # representative thermal sens at 30 MHz, 100 h integration
    s_th_100 = sigma_thermal(N, 30.0, 20.0, 100.0) * 1e3     # mJy
    # confusion at 30 MHz with this config's max baseline
    Bmax = max_baseline(cn, out_km)
    s_c_30 = confusion_limit(30.0, Bmax) * 1e3               # mJy

    score_rows.append(dict(
        config=cfg,
        core_size=f"{cn}×{cn}",
        outrigger_dist_km=out_km,
        N_elements=N,
        n_feasible=n_feas,
        n_aspirational=n_asp,
        n_detectable_total=n_tot,
        thermal_sens_30MHz_100h_mJy=s_th_100,
        confusion_30MHz_mJy=s_c_30,
    ))

df_score = pd.DataFrame(score_rows)

# normalise metrics for composite score
df_score["_n_norm"] = df_score["n_detectable_total"] / \
    max(df_score["n_detectable_total"].max(), 1)
df_score["_th_norm"] = (1.0 / df_score["thermal_sens_30MHz_100h_mJy"]) / \
    (1.0 / df_score["thermal_sens_30MHz_100h_mJy"]).max()
df_score["_cf_norm"] = (1.0 / df_score["confusion_30MHz_mJy"]) / \
    (1.0 / df_score["confusion_30MHz_mJy"]).max()

w_n, w_th, w_cf = 0.5, 0.3, 0.2
df_score["composite_score"] = (w_n * df_score["_n_norm"] +
                               w_th * df_score["_th_norm"] +
                               w_cf * df_score["_cf_norm"])
df_score = df_score.drop(columns=["_n_norm", "_th_norm", "_cf_norm"])
df_score = df_score.sort_values(
    "composite_score", ascending=False).reset_index(drop=True)
df_score["rank"] = df_score.index + 1

df_score.to_csv(os.path.join(OUT_C, "config_ranking.csv"), index=False)
print(f"  Saved config_ranking.csv  ({len(df_score)} configs)")
print("  Top 3:")
for _, r in df_score.head(3).iterrows():
    print(f"    #{r['rank']}  {r['config']}  score={r['composite_score']:.3f}  "
          f"n_feas={r['n_feasible']}  n_asp={r['n_aspirational']}")

# ── bar chart ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bar_colors = ["#4CAF50" if "128" in r else "#2196F3"
              for r in df_score["config"]]
x_pos = np.arange(len(df_score))

ax = axes[0]
bars = ax.barh(x_pos, df_score["composite_score"],
               color=bar_colors, edgecolor="white", lw=0.5)
ax.set_yticks(x_pos)
ax.set_yticklabels([CLABEL.get(c, c) for c in df_score["config"]], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Composite Score", fontsize=10)
ax.set_title(
    "Configuration Composite Ranking\n(50% detections + 30% thermal + 20% confusion)", fontsize=10)
ax.grid(True, axis="x", alpha=0.3)
legend_el = [Patch(color="#4CAF50", label="Core 128×128"),
             Patch(color="#2196F3", label="Core 32×32")]
ax.legend(handles=legend_el, fontsize=9, loc="lower right")

ax2 = axes[1]
width = 0.4
ax2.barh(x_pos - width/2, df_score["n_feasible"],
         width, color="#4CAF50", label="Feasible (<100h)")
ax2.barh(x_pos + width/2, df_score["n_aspirational"],
         width, color="#FF9800", label="Aspirational (100–1000h)")
ax2.set_yticks(x_pos)
ax2.set_yticklabels([CLABEL.get(c, c) for c in df_score["config"]], fontsize=8)
ax2.invert_yaxis()
ax2.set_xlabel("Number of Targets", fontsize=10)
ax2.set_title("Detectable Targets per Configuration", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, axis="x", alpha=0.3)

fig.suptitle("ALO Configuration Ranking & Detection Yield",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "config_ranking.png"), dpi=150)
plt.close(fig)
print("  Saved config_ranking.png")


# ══════════════════════════════════════════════════════════════════════════════
# F.  ASPIRATIONAL TARGET DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[F] Aspirational target deep-dive …")

# For each aspirational target: best t, best config, best band, flux, freq
asp_best = (df[df["feasibility"] == "aspirational"]
            .sort_values("required_t_h")
            .groupby("target_name")
            .first()
            .reset_index()
            [["target_name", "config_name", "frequency_band", "freq_centre_MHz",
              "target_flux_mJy", "target_freq_MHz", "required_t_h", "total_sens_Jy"]]
            .rename(columns={"required_t_h": "best_t_h", "config_name": "best_config",
                             "frequency_band": "best_band"}))

# Compute potential improvement factor: ratio (t_best_current / t_with_best_config)
best_cfg_name = df_score.iloc[0]["config"]
asp_rows = []
for _, row in asp_best.iterrows():
    tgt = row["target_name"]
    # t with absolute best config
    sub_best = df[(df["target_name"] == tgt) & (
        df["config_name"] == best_cfg_name)]
    t_with_best = sub_best["required_t_h"].min() if len(sub_best) else np.inf
    improvement = row["best_t_h"] / \
        t_with_best if t_with_best > 0 and np.isfinite(t_with_best) else np.nan
    asp_rows.append({**row.to_dict(),
                     "t_with_best_config_h": t_with_best,
                     "improvement_factor": improvement})

df_asp = pd.DataFrame(asp_rows).sort_values("best_t_h")
df_asp.to_csv(os.path.join(OUT_C, "aspirational_targets.csv"), index=False)
print(f"  Saved aspirational_targets.csv  ({len(df_asp)} targets)")

# ── plot: scatter of aspirational targets – t_req vs flux ─────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(df_asp["target_flux_mJy"], df_asp["best_t_h"],
                c=np.log10(df_asp["target_freq_MHz"].clip(1)),
                cmap="plasma", s=40, alpha=0.85, edgecolors="white", lw=0.3)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("log₁₀(freq [MHz])", fontsize=9)

# label the 10 most promising (lowest t)
for _, row in df_asp.head(10).iterrows():
    ax.annotate(row["target_name"], (row["target_flux_mJy"], row["best_t_h"]),
                fontsize=7, alpha=0.85,
                xytext=(4, 2), textcoords="offset points")

ax.axhline(100,  color="orange", ls="--", lw=1.2, label="100 h boundary")
ax.axhline(1000, color="red",    ls="--", lw=1.2, label="1000 h boundary")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Predicted Flux [mJy]", fontsize=11)
ax.set_ylabel("Best Required Integration Time [h]", fontsize=11)
ax.set_title("Aspirational Targets — Flux vs Required Integration Time\n"
             "(colour = log₁₀ emission frequency)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "aspirational_targets_scatter.png"), dpi=150)
plt.close(fig)
print("  Saved aspirational_targets_scatter.png")

# ── bar: top-20 aspirational, best t per target ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
top20 = df_asp.head(20).sort_values("best_t_h")
colors_asp = ["#FF9800" if t < 500 else "#F44336" for t in top20["best_t_h"]]
ax.barh(top20["target_name"], top20["best_t_h"],
        color=colors_asp, edgecolor="white")
ax.axvline(100,  color="limegreen", ls="--", lw=1.2, label="100 h")
ax.axvline(1000, color="red",       ls="--", lw=1.2, label="1000 h")
ax.set_xlabel("Best Required Integration Time [h]", fontsize=11)
ax.set_title("Top-20 Aspirational Targets — Best Integration Time\n(all ALO configurations, all bands)",
             fontsize=11)
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "aspirational_targets_top20.png"), dpi=150)
plt.close(fig)
print("  Saved aspirational_targets_top20.png")


# ══════════════════════════════════════════════════════════════════════════════
# G.  BEST CONFIG PER TARGET
# ══════════════════════════════════════════════════════════════════════════════
print("\n[G] Best config per target …")

best_cfg_per_tgt = (df.sort_values("required_t_h")
                      .groupby("target_name")
                      .first()
                      .reset_index()
                    [["target_name", "config_name", "frequency_band", "freq_centre_MHz",
                        "target_flux_mJy", "target_freq_MHz", "required_t_h", "feasibility",
                        "total_sens_Jy"]]
                    .rename(columns={"config_name": "best_config",
                                     "frequency_band": "best_band",
                                     "required_t_h": "best_t_h"}))
best_cfg_per_tgt.to_csv(os.path.join(
    OUT_C, "best_config_per_target.csv"), index=False)
print(f"  Saved best_config_per_target.csv  ({len(best_cfg_per_tgt)} targets)")

# ── summary stats ─────────────────────────────────────────────────────────────
fc_counts = best_cfg_per_tgt["feasibility"].value_counts()
print("  Feasibility breakdown (best config/band per target):")
for k, v in fc_counts.items():
    print(f"    {k}: {v}")
print("  Most-selected best configs:")
print(best_cfg_per_tgt["best_config"].value_counts().head(
    5).to_string(header=False))


# ══════════════════════════════════════════════════════════════════════════════
# H.  PER-CONFIG SUMMARY CSV
# ══════════════════════════════════════════════════════════════════════════════
print("\n[H] Per-config summary …")

summary_rows = []
for cfg in configs_ordered:
    meta = all_configs_meta[cfg]
    sub = df[df["config_name"] == cfg]
    best_t = sub.groupby("target_name")["required_t_h"].min()
    Bmax = max_baseline(meta["core_n"], meta["out_km"])

    # sensitivity at reference bands
    def sens_ref(nu, bw, t_h=100):
        s_th = sigma_thermal(meta["N"], nu, bw, t_h)
        s_c = confusion_limit(nu, Bmax)
        return NSIGMA * np.sqrt(s_th**2 + s_c**2) * 1e3   # mJy

    summary_rows.append(dict(
        config=cfg,
        core_size=f"{meta['core_n']}×{meta['core_n']}",
        outrigger_dist_km=meta["out_km"],
        N_elements=meta["N"],
        max_baseline_m=round(Bmax, 1),
        n_feasible=(best_t < 100).sum(),
        n_aspirational=((best_t >= 100) & (best_t < 1000)).sum(),
        n_total_detectable=(best_t < 1000).sum(),
        sens_5sigma_3MHz_100h_mJy=round(sens_ref(3.0,  4.0,  100), 6),
        sens_5sigma_7MHz_100h_mJy=round(sens_ref(7.5,  5.0,  100), 6),
        sens_5sigma_15MHz_100h_mJy=round(sens_ref(15.0, 10.0, 100), 6),
        sens_5sigma_30MHz_100h_mJy=round(sens_ref(30.0, 20.0, 100), 6),
        confusion_3MHz_mJy=round(confusion_limit(3.0,  Bmax)*1e3, 6),
        confusion_30MHz_mJy=round(confusion_limit(30.0, Bmax)*1e3, 6),
    ))

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_C, "config_summary.csv"), index=False)
print(f"  Saved config_summary.csv  ({len(df_summary)} configs)")
print(df_summary[["config", "n_feasible", "n_aspirational",
      "sens_5sigma_30MHz_100h_mJy"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── I. Band distribution of feasible and aspirational targets ─────────────────
print("\n[I] Band distribution plot …")
df_feas = df[df["feasibility"].isin(["feasible", "aspirational"])].copy()
df_feas_best = (df_feas.sort_values("required_t_h")
                .groupby(["target_name", "feasibility"])
                .first()
                .reset_index())
band_feas_cnt = df_feas_best.groupby(
    ["frequency_band", "feasibility"]).size().unstack(fill_value=0)
band_order = ["1-5", "5-10", "10-20", "20-40"]
band_feas_cnt = band_feas_cnt.reindex(band_order, fill_value=0)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(band_order))
w = 0.35
for i, (col, color) in enumerate([("feasible", "#4CAF50"), ("aspirational", "#FF9800")]):
    if col in band_feas_cnt.columns:
        ax.bar(x + i*w - w/2, band_feas_cnt[col], w, color=color,
               label=col.capitalize(), edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"{b} MHz" for b in band_order], fontsize=10)
ax.set_xlabel("Frequency Band", fontsize=11)
ax.set_ylabel("Number of Targets", fontsize=11)
ax.set_title(
    "Detectable Targets by Frequency Band\n(best config/band per target)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "band_distribution_detections.png"), dpi=150)
plt.close(fig)
print("  Saved band_distribution_detections.png")

# ── J. Thermal vs confusion noise: radar/parallel plot per config ─────────────
print("\n[J] Thermal vs confusion noise comparison …")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
nu_plot = np.linspace(1, 40, 500)
t_ref = 100.0
bw_ref = 5.0

for col_idx, (core_n_sel, ax) in enumerate(zip([32, 128], axes)):
    sel = [c for c in configs_ordered if all_configs_meta[c]
           ["core_n"] == core_n_sel]
    cmap2 = plt.cm.Blues if core_n_sel == 32 else plt.cm.Greens
    n_s = len(sel)

    for si, cfg in enumerate(sel):
        meta = all_configs_meta[cfg]
        Bmax = max_baseline(meta["core_n"], meta["out_km"])
        color = cmap2(0.35 + 0.55*si/(n_s-1) if n_s > 1 else 0.7)

        s_th = [sigma_thermal(meta["N"], nu, bw_ref, t_ref)
                * 1e3 for nu in nu_plot]
        s_c = [confusion_limit(nu, Bmax)*1e3 for nu in nu_plot]

        lbl = CLABEL.get(cfg, cfg)
        ax.semilogy(nu_plot, s_th, color=color,
                    lw=1.5,   label=f"Thermal {lbl}")
        ax.semilogy(nu_plot, s_c,  color=color, lw=1.5, ls="--")

    ax.set_xlabel("Frequency [MHz]", fontsize=10)
    ax.set_ylabel("Noise [mJy]",     fontsize=10)
    ax.set_title(f"Core {core_n_sel}×{core_n_sel}: Thermal (solid) vs Confusion (dashed)\n"
                 f"(t={t_ref}h, BW={bw_ref} MHz)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

fig.suptitle("Thermal vs Confusion Noise Comparison Across ALO Configurations",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "thermal_vs_confusion_comparison.png"), dpi=150)
plt.close(fig)
print("  Saved thermal_vs_confusion_comparison.png")

# ── K. Feasible targets: t_req distribution histogram ────────────────────────
print("\n[K] Feasible targets t_req histogram …")

feas_best = best_cfg_per_tgt[best_cfg_per_tgt["feasibility"]
                             == "feasible"].copy()
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.logspace(-2, 2, 30)
ax.hist(feas_best["best_t_h"], bins=bins,
        color="#4CAF50", edgecolor="white", alpha=0.85)
ax.set_xscale("log")
ax.set_xlabel("Best Required Integration Time [h]", fontsize=11)
ax.set_ylabel("Number of Targets",                  fontsize=11)
ax.set_title("Distribution of Required Integration Times — Feasible Targets\n"
             "(best config and sub-band per target)", fontsize=11)
ax.axvline(1,  color="navy",   ls="--", lw=1.2, label="1h")
ax.axvline(10, color="orange", ls="--", lw=1.2, label="10h")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "feasible_t_histogram.png"), dpi=150)
plt.close(fig)
print("  Saved feasible_t_histogram.png")

# ── L. Top-20 feasible targets bar chart ─────────────────────────────────────
print("\n[L] Top-20 feasible targets bar chart …")
top20f = feas_best.sort_values("best_t_h").head(20)

fig, ax = plt.subplots(figsize=(11, 7))
col_f = [plt.cm.RdYlGn_r(t/100) for t in top20f["best_t_h"]]
ax.barh(top20f["target_name"], top20f["best_t_h"],
        color=col_f, edgecolor="white")
ax.axvline(1,  color="blue",        ls="--", lw=1, label="1h")
ax.axvline(10, color="darkorange",  ls="--", lw=1, label="10h")
ax.axvline(100, color="red",         ls="--", lw=1, label="100h")
ax.set_xlabel("Best Required Integration Time [h]", fontsize=11)
ax.set_title("Top-20 Easiest Targets for ALO (Feasible)\n"
             "(best config+band, 5σ detection)", fontsize=11)
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_P, "top20_feasible_targets.png"), dpi=150)
plt.close(fig)
print("  Saved top20_feasible_targets.png")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ALO Extended Analysis — COMPLETE")
print("="*70)
print(f"\nCSV outputs ({OUT_C}):")
for f in sorted(os.listdir(OUT_C)):
    fpath = os.path.join(OUT_C, f)
    print(f"  {f:45s}  {os.path.getsize(fpath)//1024} KB")
print(f"\nPlot outputs ({OUT_P}):")
for f in sorted(os.listdir(OUT_P)):
    if f.endswith(".png"):
        fpath = os.path.join(OUT_P, f)
        print(f"  {f:55s}  {os.path.getsize(fpath)//1024} KB")
print()
