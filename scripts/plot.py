"""Plots an estimated trajectory against ground truth, top view and height.

Usage: python scripts/plot.py <estimate.txt> <sequence dir> <out.png>
           [--before other.txt --before-label "no loop closure"]

The estimate is aligned to ground truth with Umeyama (SE(3), no scale) on
time-associated poses, the same alignment used for ATE.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_tum(path):
    rows = [list(map(float, line.split())) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    a = np.array(rows)
    return a[:, 0], a[:, 1:4]


def associate(t_est, t_gt, max_dt=0.02):
    idx = np.searchsorted(t_gt, t_est).clip(1, len(t_gt) - 1)
    left = np.abs(t_gt[idx - 1] - t_est) < np.abs(t_gt[idx] - t_est)
    idx = idx - left
    ok = np.abs(t_gt[idx] - t_est) < max_dt
    return np.nonzero(ok)[0], idx[ok]


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return R, mu_d - R @ mu_s


def aligned(path, t_gt, p_gt):
    t_est, p_est = load_tum(path)
    i_est, i_gt = associate(t_est, t_gt)
    R, t = umeyama(p_est[i_est], p_gt[i_gt])
    est = p_est[i_est] @ R.T + t
    gt = p_gt[i_gt]
    err = np.linalg.norm(est - gt, axis=1)
    tt = t_est[i_est] - t_est[i_est][0]
    return tt, est, gt, err, np.sqrt(np.mean(err ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("estimate")
    ap.add_argument("sequence")
    ap.add_argument("out")
    ap.add_argument("--label", default="lynx VO")
    ap.add_argument("--before", help="second estimate drawn for comparison")
    ap.add_argument("--before-label", default="before")
    args = ap.parse_args()

    t_gt, p_gt = load_tum(Path(args.sequence) / "groundtruth.txt")
    tt, est, gt, err, rmse = aligned(args.estimate, t_gt, p_gt)

    name = Path(args.sequence).name.replace("rgbd_dataset_", "")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [1, 1.4]})
    ax1.plot(gt[:, 0], gt[:, 1], color="0.35", lw=1.5, label="ground truth")
    title = f"ATE RMSE {100 * rmse:.2f} cm"
    if args.before:
        bt, best, _, berr, brmse = aligned(args.before, t_gt, p_gt)
        ax1.plot(best[:, 0], best[:, 1], color="#4c6ef5", lw=1, ls="--",
                 label=args.before_label)
        ax2.plot(bt, 100 * berr, color="#4c6ef5", lw=1, ls="--",
                 label=f"{args.before_label} ({100 * brmse:.2f} cm)")
        title = "position error after alignment"
    ax1.plot(est[:, 0], est[:, 1], color="#d9480f", lw=1.2, label=args.label)
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title(f"{name}: top view")
    ax1.legend(frameon=False)

    ax2.plot(tt, 100 * err, color="#d9480f", lw=1,
             label=f"{args.label} ({100 * rmse:.2f} cm)")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position error [cm]")
    ax2.set_title(title)
    if args.before:
        ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"{args.out}: ATE RMSE {100 * rmse:.2f} cm on {len(err)} poses")


if __name__ == "__main__":
    main()
