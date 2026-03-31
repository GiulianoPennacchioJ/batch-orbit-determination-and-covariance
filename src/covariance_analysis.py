"""
03_covariance_analysis.py
==========================
Covariance analysis and uncertainty quantification for the BLS solution.

Author : Giuliano Pennacchio
Version: 1.0

Description
-----------
Loads the covariance matrix P and estimated state from 01_batch_least_squares.py
and produces four analysis outputs:

  Fig 1 — 3-D error ellipsoid (position covariance, 1-sigma and 3-sigma surfaces)
  Fig 2 — Normalised correlation matrix (6x6 heatmap)
  Fig 3 — 1-sigma and 3-sigma uncertainty bars for each state component
  Fig 4 — Estimation error vs covariance bounds (true error vs ±3-sigma)

The covariance matrix P is the formal covariance from the Batch Least Squares:
    P = (H^T W H)^{-1}  =  Lambda^{-1}

where W = diag(1/sigma^2) and H includes the STM mapping.

This is the formal precision of the estimator assuming:
  - The measurement noise model is correct.
  - The force model used in estimation matches the truth (J2-only, consistent).
  - No process noise (deterministic dynamics assumed).

Outputs
-------
figures/04_error_ellipsoid.png
figures/04_correlation_matrix.png
figures/04_sigma_bars.png
figures/04_error_vs_covariance.png

Run
---
    python 03_covariance_analysis.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")

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

STATE_LABELS   = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]
STATE_LABELS_S = ["x", "y", "z", "vx", "vy", "vz"]


# ===========================================================================
# Helpers
# ===========================================================================

def load_results() -> dict:
    path = os.path.join(DATA_DIR, "bls_result.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BLS result not found: {path}\n"
            "  -> Run 01_batch_least_squares.py first."
        )
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def ellipsoid_surface(cov3: np.ndarray, n_sigma: float, n_pts: int = 50):
    """
    Generate the surface of a 3-D error ellipsoid.

    Parameterisation via eigendecomposition:
        surface = mu + sqrt(n_sigma^2) * U @ sqrt(D) @ [sin_u*cos_v, ...]

    Parameters
    ----------
    cov3    : ndarray (3,3)  Position covariance sub-matrix [km^2].
    n_sigma : float          Sigma level (1 or 3).
    n_pts   : int            Grid resolution.

    Returns
    -------
    X, Y, Z : ndarray (n_pts, n_pts)  Surface coordinates [m].
    """
    vals, vecs = np.linalg.eigh(cov3)
    # Safety: ensure positive eigenvalues (numerical noise can cause tiny negatives)
    vals = np.abs(vals)

    # Parametric angles
    u = np.linspace(0, 2 * np.pi, n_pts)
    v = np.linspace(0, np.pi,     n_pts)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale by eigenvalues and rotate by eigenvectors
    # Each point on unit sphere: p = vecs @ diag(sqrt(vals)) @ [x,y,z]
    scale = n_sigma * np.sqrt(vals)
    pts   = (vecs * scale) @ np.array([x.ravel(), y.ravel(), z.ravel()])

    # Convert to metres
    X = pts[0].reshape(n_pts, n_pts) * 1000
    Y = pts[1].reshape(n_pts, n_pts) * 1000
    Z = pts[2].reshape(n_pts, n_pts) * 1000
    return X, Y, Z


# ===========================================================================
# Figure 1: 3-D error ellipsoid
# ===========================================================================

def plot_error_ellipsoid(P: np.ndarray, x_true: np.ndarray, x_est: np.ndarray) -> None:
    """
    Plot 1-sigma and 3-sigma position error ellipsoids centred on
    the estimated position.  The true position is shown as a red dot
    to demonstrate that it falls inside the covariance bound.
    """
    cov3 = P[:3, :3]   # position sub-covariance

    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title("Position Error Ellipsoid\n(1-σ and 3-σ, ECI frame)", pad=12)

    for n_sig, alpha, color, label in [
        (1, 0.25, "#2563EB", "1-σ ellipsoid"),
        (3, 0.10, "#93C5FD", "3-σ ellipsoid"),
    ]:
        X, Y, Z = ellipsoid_surface(cov3, n_sig)
        ax.plot_surface(X, Y, Z, alpha=alpha, color=color, label=label,
                        linewidth=0, antialiased=True)

    # True estimation error (x_est - x_true) in metres — should be inside ellipsoid
    err = (x_est[:3] - x_true[:3]) * 1000   # m
    ax.scatter(*err, color="#DC2626", s=80, zorder=10,
               label=f"True error ({np.linalg.norm(err):.2f} m)")
    ax.scatter(0, 0, 0, color="black", s=40, marker="+",
               label="Estimated position", zorder=10)

    ax.set_xlabel("ΔX [m]", labelpad=8)
    ax.set_ylabel("ΔY [m]", labelpad=8)
    ax.set_zlabel("ΔZ [m]", labelpad=8)

    # Build proxy patches for legend (surfaces don't auto-add to legend)
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#2563EB", alpha=0.5, label="1-σ ellipsoid"),
        Patch(facecolor="#93C5FD", alpha=0.5, label="3-σ ellipsoid"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#DC2626",
                   markersize=8, label=f"True error ({np.linalg.norm(err):.2f} m)"),
        plt.Line2D([0],[0], marker="+", color="black", markersize=8,
                   linestyle="None", label="Estimated position"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "04_error_ellipsoid.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# Figure 2: correlation matrix heatmap
# ===========================================================================

def plot_correlation_matrix(P: np.ndarray) -> None:
    """
    Normalise P to get the correlation matrix C:
        C[i,j] = P[i,j] / sqrt(P[i,i] * P[j,j])
    and display as a colour-coded heatmap.
    """
    sig_diag = np.sqrt(np.diag(P))
    C        = P / np.outer(sig_diag, sig_diag)   # correlation matrix

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Covariance Correlation Matrix", pad=10)

    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation coefficient")

    ax.set_xticks(range(6));  ax.set_xticklabels(STATE_LABELS_S, fontsize=10)
    ax.set_yticks(range(6));  ax.set_yticklabels(STATE_LABELS_S, fontsize=10)

    # Annotate each cell with its value
    for i in range(6):
        for j in range(6):
            txt_color = "white" if abs(C[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{C[i,j]:+.2f}", ha="center", va="center",
                    fontsize=9, color=txt_color, fontweight="bold")

    ax.spines[:].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "04_correlation_matrix.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# Figure 3: sigma uncertainty bars per state component
# ===========================================================================

def plot_sigma_bars(P: np.ndarray) -> None:
    """
    Bar chart of the 1-sigma and 3-sigma formal uncertainties for each
    of the 6 state components (3 position + 3 velocity).
    """
    sigma = np.sqrt(np.diag(P))

    # Convert to human-readable units
    units   = ["m", "m", "m", "mm/s", "mm/s", "mm/s"]
    scales  = [1000, 1000, 1000, 1e6, 1e6, 1e6]
    sig_1   = sigma * scales
    sig_3   = 3 * sigma * scales

    x       = np.arange(6)
    width   = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Formal 1-σ and 3-σ State Uncertainty (from P = Lambda^{-1})", pad=10)

    bars1 = ax.bar(x - width / 2, sig_1, width, label="1-σ", color="#2563EB", alpha=0.85)
    bars3 = ax.bar(x + width / 2, sig_3, width, label="3-σ", color="#93C5FD", alpha=0.85)

    # Value labels above bars
    for bar in list(bars1) + list(bars3):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h * 1.02,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\n[{u}]" for lbl, u in zip(STATE_LABELS_S, units)], fontsize=10
    )
    ax.set_ylabel("Uncertainty [m or mm/s]")
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, which="both", axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "04_sigma_bars.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# Figure 4: estimation error vs covariance bounds
# ===========================================================================

def plot_error_vs_covariance(
    x_true: np.ndarray,
    x_est:  np.ndarray,
    P:      np.ndarray,
) -> None:
    """
    Bar chart comparing the actual estimation error |x_est - x_true|
    to the formal 1-sigma and 3-sigma covariance bounds.

    The estimation errors should be smaller than the 3-sigma bound if the
    covariance is properly calibrated.
    """
    sigma  = np.sqrt(np.diag(P))
    errors = np.abs(x_est - x_true)

    scales = [1000, 1000, 1000, 1e6, 1e6, 1e6]
    units  = ["m", "m", "m", "mm/s", "mm/s", "mm/s"]

    err_sc  = errors * scales
    sig1_sc = sigma  * scales
    sig3_sc = 3 * sigma * scales

    x_pos   = np.arange(6)
    width   = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Estimation Error vs Formal Covariance Bounds", pad=10)

    b_err  = ax.bar(x_pos - width,     err_sc,  width, label="|x_est − x_true|",
                    color="#DC2626", alpha=0.90)
    b_sig1 = ax.bar(x_pos,             sig1_sc, width, label="1-σ bound",
                    color="#2563EB", alpha=0.75)
    b_sig3 = ax.bar(x_pos + width,     sig3_sc, width, label="3-σ bound",
                    color="#93C5FD", alpha=0.75)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [f"{lbl}\n[{u}]" for lbl, u in zip(STATE_LABELS_S, units)], fontsize=10
    )
    ax.set_ylabel("Error / Uncertainty [m or mm/s]")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", axis="y", alpha=0.3)

    # Consistency check annotation
    all_within_3sig = np.all(err_sc < sig3_sc)
    status = "✓ All errors within 3-σ bounds" if all_within_3sig \
             else "✗ Some errors exceed 3-σ bounds"
    color  = "#16A34A" if all_within_3sig else "#DC2626"
    ax.text(0.98, 0.96, status, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, alpha=0.8))

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "04_error_vs_covariance.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("Project 2 — Batch OD: Covariance Analysis")
    print("=" * 60)

    d        = load_results()
    P        = d["P_covariance"].astype(float)
    x_true   = d["x_true"].astype(float)
    x_est    = d["x_estimated"].astype(float)
    x_init   = d["x_initial"].astype(float)
    sigma_r  = float(d["sigma_range"])
    sigma_rr = float(d["sigma_rangerate"])

    sigma = np.sqrt(np.diag(P))

    # -----------------------------------------------------------------------
    # Print covariance table
    # -----------------------------------------------------------------------
    print(f"\n  {'Component':<8}  {'1-σ':>12}  {'3-σ':>12}  {'True error':>12}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")
    scales = [1000]*3 + [1e6]*3
    units  = ["m"]*3 + ["mm/s"]*3
    for k in range(6):
        err = abs(x_est[k] - x_true[k]) * scales[k]
        s1  = sigma[k] * scales[k]
        s3  = 3 * sigma[k] * scales[k]
        flag = "✓" if err < s3 else "✗"
        print(f"  {STATE_LABELS_S[k]:<8}  {s1:>10.6f} {units[k]}  "
              f"{s3:>10.6f} {units[k]}  {err:>10.6f} {units[k]}  {flag}")

    # Condition number of P
    cond_P = np.linalg.cond(P)
    print(f"\n  Condition number of P : {cond_P:.3e}")

    # Correlation highlights
    sig_diag = np.sqrt(np.diag(P))
    C        = P / np.outer(sig_diag, sig_diag)
    max_off_diag = np.max(np.abs(C - np.eye(6)))
    print(f"  Max off-diagonal correlation : {max_off_diag:.4f}")

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------
    print("\nGenerating figures...")

    plot_error_ellipsoid(P, x_true, x_est)
    plot_correlation_matrix(P)
    plot_sigma_bars(P)
    plot_error_vs_covariance(x_true, x_est, P)

    # -----------------------------------------------------------------------
    # Write covariance summary to file
    # -----------------------------------------------------------------------
    summary_path = os.path.join(DATA_DIR, "covariance_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("Project 2 — Covariance Analysis Summary\n")
        fh.write("=" * 55 + "\n\n")
        fh.write("Covariance matrix P = (H^T W H)^{-1}\n")
        fh.write(f"Condition number of P : {cond_P:.3e}\n\n")
        fh.write(f"{'Component':<8}  {'1-sigma':>12}  {'3-sigma':>12}  "
                 f"{'True error':>12}  {'Within 3sig':>12}\n")
        fh.write("─" * 65 + "\n")
        for k in range(6):
            err = abs(x_est[k] - x_true[k]) * scales[k]
            s1  = sigma[k] * scales[k]
            s3  = 3 * sigma[k] * scales[k]
            flag = "YES" if err < s3 else "NO"
            fh.write(f"{STATE_LABELS_S[k]:<8}  {s1:>10.6f} {units[k]}  "
                     f"{s3:>10.6f} {units[k]}  {err:>10.6f} {units[k]}  "
                     f"{flag:>12}\n")
        fh.write("\nCorrelation Matrix\n")
        fh.write("─" * 55 + "\n")
        header = "         " + "".join(f"{lbl:>10}" for lbl in STATE_LABELS_S)
        fh.write(header + "\n")
        for i in range(6):
            row = f"{STATE_LABELS_S[i]:<8} "
            row += "".join(f"{C[i,j]:>+10.4f}" for j in range(6))
            fh.write(row + "\n")

    print(f"\n  Summary saved -> {summary_path}")
    print("\nDone. All analysis complete.\n")
    print("=" * 60)
    print("Project 2 outputs:")
    print(f"  data/ground_truth.csv        — true trajectory")
    print(f"  data/observations.csv        — simulated radar obs")
    print(f"  data/bls_result.npz          — BLS state + covariance")
    print(f"  figures/03_*.png             — residuals + convergence")
    print(f"  figures/04_*.png             — covariance + error analysis")
    print("=" * 60)


if __name__ == "__main__":
    main()