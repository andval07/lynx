"""Runs evo on lynx trajectories and prints a Markdown table.

Usage: python scripts/evaluate.py results/vo_fr1_xyz.txt=data/rgbd_dataset_freiburg1_xyz ...

Each argument is <estimate>=<sequence dir>. ATE is evo_ape on translation after
an SE(3) alignment (no scale), or Sim(3) with --scale for monocular runs.
RPE is evo_rpe on translation over 30-frame
windows, about 1 s at 30 Hz, close to the 1 s delta of the TUM benchmark.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

EVO_BIN = Path(sys.executable).parent


def run_evo(tool, gt, est, extra, scale=False):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "res.zip"
        cmd = [str(EVO_BIN / tool), "tum", str(gt), str(est), "-a",
               *(["-s"] if scale else []), "--save_results", str(out), "--no_warnings", *extra]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with zipfile.ZipFile(out) as z:
            return json.loads(z.read("stats.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="<estimate>=<sequence dir>")
    ap.add_argument("--scale", action="store_true",
                    help="Sim(3) alignment, for monocular runs")
    args = ap.parse_args()

    print("| sequence | frames | ATE RMSE [cm] | ATE max [cm] | RPE trans [cm per 30 frames] |")
    print("|---|---|---|---|---|")
    for pair in args.pairs:
        est, seq = (Path(p) for p in pair.split("=", 1))
        gt = seq / "groundtruth.txt"
        ape = run_evo("evo_ape", gt, est, ["-r", "trans_part"],
                      args.scale)
        rpe = run_evo("evo_rpe", gt, est,
                      ["-r", "trans_part", "-d", "30", "-u", "f"], args.scale)
        frames = sum(1 for line in est.read_text().splitlines()
                     if line and not line.startswith("#"))
        name = seq.name.replace("rgbd_dataset_", "")
        print(f"| {name} | {frames} | {100 * ape['rmse']:.2f} | "
              f"{100 * ape['max']:.2f} | {100 * rpe['rmse']:.2f} |")


if __name__ == "__main__":
    main()
