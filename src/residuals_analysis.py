"""
02_residuals_analysis.py
=========================
Pre-fit and post-fit residual analysis for the Batch OD solution.

Author : Giuliano Pennacchio
Version: 1.0

Description
-----------
Loads the BLS results from 01_batch_least_squares.py and produces:

  Fig 1 — Pre-fit residuals (range and range-rate, colour-coded by station)
  Fig 2 — Post-fit residuals (range and range-rate, colour-coded by station)
  Fig 3 — Normalised residuals histogram vs unit Gaussian
  Fig 4 — BLS convergence history (|dx|, position error, range RMS per iter)

The ratio  RMS_postfit / sigma_noise  (chi statistic) quantifies whether
the measurement model is consistent with the assumed noise level.  A value
close to 1.0 indicates a well-calibrated model.

Outputs
-------
figures/03_prefit_residuals.png
figures/03_postfit_residuals.png
figures/03_residuals_histogram.png
figures/03_convergence.png

Run
---
    python 02_residuals_analysis.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")    # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from od_utils import SIGMA_RANGE, SIGMA_RANGERATE

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR    = os.path.join(os.path.dirname(__file__), "figures")

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
COLORS = {"Kiruna": "#2563EB", "Svalbard": "#DC2626"}   # blue / red
ALPHA  = 0.65
MS     = 3     # marker size

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
    "figure.dpi":        130,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})


# ===========================================================================
# Helpers
# ===========================================================================

def load_results() -> dict:
    """Load the .npz file saved by 01_batch_least_squares.py."""
    path = os.path.join(DATA_DIR, "bls_result.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BLS result not found: {path}\n"
            "  -> Run 01_batch_least_squares.py first."
        )
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def add_zero_line(ax, lw=0.8):
    """Draw a thin horizontal zero reference line."""
    ax.axhline(0.0, color="black", linewidth=lw, linestyle="--", alpha=0.5)


def add_sigma_bands(ax, sigma, color="gray", n_sigma=(1, 3)):
    """Shade ±n_sigma horizontal bands."""
    for n in n_sigma:
        ax.axhline( n * sigma, color=color, linewidth=0.7,
                    linestyle=":", alpha=0.6, label=f"±{n}σ" if n == 1 else None)
        ax.axhline(-n * sigma, color=color, linewidth=0.7, linestyle=":", alpha=0.6)


def time_axis(ax, obs_times: np.ndarray):
    """Replace raw seconds with hours on the x-axis."""
    ax.set_xlabel("Time from epoch [h]")
    ticks = np.arange(0, obs_times.max() / 3600 + 12, 12)
    ax.set_xticks(ticks)


# ===========================================================================
# Figure 1 & 2: pre-fit and post-fit scatter plots
# ===========================================================================

def plot_residuals(
    obs_times: np.ndarray,
    stations:  np.ndarray,
    res_range: np.ndarray,
    res_rr:    np.ndarray,
    sigma_r:   float,
    sigma_rr:  float,
    title_tag: str,          # "Pre-fit" or "Post-fit"
    filename:  str,
) -> None:
    """
    Two-row scatter plot: range residuals (top), range-rate residuals (bottom).
    Each station has a distinct colour.  Sigma bands are overlaid.
    """
    t_h = obs_times / 3600.0    # convert to hours

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Batch OD — {title_tag} Residuals", fontsize=14, fontweight="bold")

    for gs_name, color in COLORS.items():
        mask = (stations == gs_name)
        if not np.any(mask):
            continue

        # Range residuals [m]
        axes[0].scatter(
            t_h[mask], res_range[mask] * 1000,
            color=color, alpha=ALPHA, s=MS**2, label=gs_name
        )
        # Range-rate residuals [mm/s]
        axes[1].scatter(
            t_h[mask], res_rr[mask] * 1e6,
            color=color, alpha=ALPHA, s=MS**2, label=gs_name
        )

    # Sigma bands and zero line
    for ax, sigma, unit in [(axes[0], sigma_r * 1000, "m"),
                             (axes[1], sigma_rr * 1e6, "mm/s")]:
        add_zero_line(ax)
        add_sigma_bands(ax, sigma)
        ax.legend(loc="upper right", fontsize=9)

    # Labels
    rms_r  = np.std(res_range) * 1000
    rms_rr = np.std(res_rr)    * 1e6
    axes[0].set_ylabel(f"Range residual [m]\n(RMS = {rms_r:.2f} m, σ = {sigma_r*1000:.0f} m)")
    axes[1].set_ylabel(f"Range-rate residual [mm/s]\n"
                       f"(RMS = {rms_rr:.3f} mm/s, σ = {sigma_rr*1e6:.0f} mm/s)")
    axes[1].set_xlabel("Time from epoch [h]")
    axes[1].set_xticks(np.arange(0, t_h.max() + 12, 12))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, filename)
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# Figure 3: normalised residuals histogram
# ===========================================================================

def plot_histogram(
    res_range_pre:   np.ndarray,
    res_rr_pre:      np.ndarray,
    res_range_post:  np.ndarray,
    res_rr_post:     np.ndarray,
    sigma_r:         float,
    sigma_rr:        float,
) -> None:
    """
    Histogram of normalised residuals (y / sigma) compared to N(0,1).
    Four panels: range & range-rate × pre-fit & post-fit.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Normalised Residuals vs Unit Gaussian  N(0,1)",
                 fontsize=14, fontweight="bold")

    z_ref = np.linspace(-4, 4, 200)
    gauss = np.exp(-0.5 * z_ref**2) / np.sqrt(2 * np.pi)

    datasets = [
        (axes[0, 0], res_range_pre   / sigma_r,  "Range — Pre-fit",        "#2563EB"),
        (axes[0, 1], res_range_post  / sigma_r,  "Range — Post-fit",       "#16A34A"),
        (axes[1, 0], res_rr_pre      / sigma_rr, "Range-rate — Pre-fit",   "#DC2626"),
        (axes[1, 1], res_rr_post     / sigma_rr, "Range-rate — Post-fit",  "#9333EA"),
    ]

    for ax, z, label, color in datasets:
        ax.hist(z, bins=30, density=True, color=color, alpha=0.65,
                edgecolor="white", linewidth=0.4, label="Residuals")
        ax.plot(z_ref, gauss, "k-", linewidth=1.5, label="N(0,1)")
        mu  = np.mean(z)
        std = np.std(z)
        ax.set_title(f"{label}\n"
                     f"μ = {mu:+.3f}   σ = {std:.3f}",
                     fontsize=11)
        ax.set_xlabel("Normalised residual  (y / σ)")
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "03_residuals_histogram.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# Figure 4: BLS convergence
# ===========================================================================

def plot_convergence(iter_log: np.ndarray) -> None:
    """
    Three-panel convergence history:
      (top)    |dx| correction norm per iteration [m]
      (middle) true position error per iteration  [m]
      (bottom) pre-fit range RMS per iteration    [m]
    """
    iters    = iter_log[:, 0].astype(int)
    dx_norm  = iter_log[:, 1]       # m
    pos_err  = iter_log[:, 2]       # m
    rms_rng  = iter_log[:, 4]       # m

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle("BLS Convergence History", fontsize=14, fontweight="bold")

    style = dict(marker="o", markersize=6, linewidth=1.8)

    axes[0].semilogy(iters, dx_norm,  color="#2563EB", **style)
    axes[0].set_ylabel("|dx| correction [m]")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(iters, pos_err,  color="#16A34A", **style)
    axes[1].set_ylabel("True position error [m]")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogy(iters, rms_rng,  color="#DC2626", **style)
    axes[2].axhline(SIGMA_RANGE * 1000, color="black", linestyle="--",
                    linewidth=0.9, label=f"σ_range = {SIGMA_RANGE*1000:.0f} m")
    axes[2].set_ylabel("Pre-fit range RMS [m]")
    axes[2].set_xlabel("Iteration")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, which="both", alpha=0.3)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(iters)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "03_convergence.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("Project 2 — Batch OD: Residuals Analysis")
    print("=" * 60)

    d = load_results()

    obs_times = d["obs_times"].astype(float)
    stations  = d["obs_stations"].astype(str)
    sigma_r   = float(d["sigma_range"])
    sigma_rr  = float(d["sigma_rangerate"])
    iter_log  = d["iter_log"]

    res_range_pre  = d["res_range_pre"].astype(float)
    res_rr_pre     = d["res_rr_pre"].astype(float)
    res_range_post = d["res_range_post"].astype(float)
    res_rr_post    = d["res_rr_post"].astype(float)

    # Summary statistics
    print(f"\n  Observations       : {len(obs_times)}")
    print(f"\n  Pre-fit  RMS range     : {np.std(res_range_pre)*1000:.3f} m")
    print(f"  Pre-fit  RMS range-rate: {np.std(res_rr_pre)*1e6:.4f} mm/s")
    print(f"\n  Post-fit RMS range     : {np.std(res_range_post)*1000:.4f} m  "
          f"(σ = {sigma_r*1000:.0f} m,  χ = {np.std(res_range_post)/sigma_r:.3f})")
    print(f"  Post-fit RMS range-rate: {np.std(res_rr_post)*1e6:.6f} mm/s  "
          f"(σ = {sigma_rr*1e6:.0f} mm/s,  χ = {np.std(res_rr_post)/sigma_rr:.3f})")

    # Check: post-fit range chi should be ~1 (noise-consistent estimation)
    chi_r  = np.std(res_range_post) / sigma_r
    chi_rr = np.std(res_rr_post)    / sigma_rr
    if abs(chi_r - 1.0) < 0.2:
        print("\n  [OK] Post-fit range chi ≈ 1  (estimation consistent with noise model)")
    else:
        print(f"\n  [Note] Post-fit range chi = {chi_r:.3f}  "
              "(may indicate residual state error or model mismatch)")

    print("\nGenerating figures...")

    # Figure 1: pre-fit residuals
    plot_residuals(
        obs_times, stations,
        res_range_pre, res_rr_pre,
        sigma_r, sigma_rr,
        title_tag="Pre-fit",
        filename="03_prefit_residuals.png",
    )

    # Figure 2: post-fit residuals
    plot_residuals(
        obs_times, stations,
        res_range_post, res_rr_post,
        sigma_r, sigma_rr,
        title_tag="Post-fit",
        filename="03_postfit_residuals.png",
    )

    # Figure 3: histogram
    plot_histogram(
        res_range_pre, res_rr_pre,
        res_range_post, res_rr_post,
        sigma_r, sigma_rr,
    )

    # Figure 4: convergence
    plot_convergence(iter_log)

    print("\nDone. Next: run 03_covariance_analysis.py\n")


if __name__ == "__main__":
    main()