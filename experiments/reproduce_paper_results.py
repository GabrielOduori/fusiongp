"""Cross-platform reproduction entrypoint for the FusionGP paper experiments.

Examples
--------
Run the main leave-one-station-out evaluation:

    python experiments/reproduce_paper_results.py --device cuda

Only regenerate summary tables from existing runs:

    python experiments/reproduce_paper_results.py --summarize-only

Generate figures for existing runs:

    python experiments/reproduce_paper_results.py --summarize-only --generate-figures
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_STATIONS = ["10_94", "115_50", "121_61", "23_44", "52_41", "61_52", "76_51", "80_26", "88_52"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce FusionGP paper results.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stations", nargs="+", default=DEFAULT_STATIONS)
    parser.add_argument("--run-prefix", default="loso_nobias")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--generate-figures", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], env: dict[str, str] | None = None, log_path: Path | None = None) -> None:
    print(" ".join(cmd), flush=True)
    if log_path is None:
        subprocess.run(cmd, check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        subprocess.run(cmd, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = (project_root / args.outputs_dir).resolve() if not args.outputs_dir.is_absolute() else args.outputs_dir

    if not args.summarize_only:
        for station in args.stations:
            tag = f"{args.run_prefix}_{station}"
            run_dir = outputs_dir / f"demo_run_{tag}"
            final_table = run_dir / "final_results_table.csv"
            if final_table.exists() and not args.rerun_existing:
                print(f"Skipping existing run: {run_dir}")
                continue

            env = os.environ.copy()
            env.update(
                {
                    "MPLCONFIGDIR": env.get("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-debug")),
                    "RUN_TAG": tag,
                    "PRIOR_MEAN_SOURCE": "atmo",
                    "BIAS_CORRECTION_SOURCE": "none",
                    "EPA_HOLDOUT_MODE": "station",
                    "EPA_HOLDOUT_GRID_IDS": station,
                    "EPA_HOLDOUT_FRACTION": "0.15",
                    "FAST_RUN": "1",
                    "PRE_CALIBRATE_TROPOMI": "1",
                    "LEARN_SATELLITE_CALIBRATION": "0",
                    "LEARN_SATELLITE_NOISE": "0",
                    "SATELLITE_NOISE_STD": "20.0",
                    "EPA_NOISE_STD": "2.0",
                    "EPA_TRAIN_FRACTION": "1.0",
                    "BATCH_SIZE": "128",
                }
            )
            run_command(
                [sys.executable, "experiments/run_demo_pipeline.py", "--device", args.device],
                env=env,
                log_path=outputs_dir / f"{tag}.log",
            )

    summarize_cmd = [
        sys.executable,
        "experiments/summarize_loso_results.py",
        "--outputs-dir",
        str(outputs_dir),
        "--run-prefix",
        args.run_prefix,
        "--stations",
        *args.stations,
    ]
    run_command(summarize_cmd)

    if args.generate_figures:
        for station in args.stations:
            run_dir = outputs_dir / f"demo_run_{args.run_prefix}_{station}"
            run_command([sys.executable, "experiments/generate_figures.py", str(run_dir)])


if __name__ == "__main__":
    main()
