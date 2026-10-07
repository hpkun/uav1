"""Render three-seed training curves with individual runs and Student-t CI."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DIRECT_METRICS = (
    "win_rate", "average_return", "average_red_loss",
    "average_red_boundary_exits", "average_red_ground_losses",
    "red_fire_window_episode_rate", "red_kill_episode_rate",
)
PERSISTENT_METRICS = (
    "clear_wave_1_probability", "clear_wave_2_probability",
    "clear_wave_3_probability", "average_waves_cleared", "average_return",
    "average_red_loss", "average_blue_loss", "kill_loss_ratio",
    "average_red_boundary_exits", "average_red_ground_losses", "timeout_rate",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def render(condition: str, run_dirs: list[Path], aggregation_dir: Path,
           output_dir: Path, metrics: tuple[str, ...]) -> list[str]:
    histories=[read_csv(run/"evaluation_history.csv") for run in run_dirs]
    summary=read_csv(aggregation_dir/"training_curve_summary.csv")
    outputs=[]; colors=("#7f7f7f","#a0a0a0","#c0c0c0")
    for metric in metrics:
        agg=[row for row in summary if row["metric"]==metric]
        if not agg or any(metric not in row for history in histories for row in history): continue
        fig,axis=plt.subplots(figsize=(9.5,5.8),constrained_layout=True)
        for seed,history,color in zip((2023,2024,2025),histories,colors):
            axis.plot([float(row["sampled_steps"])/1e6 for row in history],
                      [float(row[metric]) for row in history],color=color,alpha=.55,
                      linewidth=1.15,label=f"seed {seed}")
        x=np.asarray([float(row["sampled_steps"])/1e6 for row in agg]); mean=np.asarray([float(row["mean"]) for row in agg])
        low=np.asarray([float(row["ci95_lower"]) for row in agg]); high=np.asarray([float(row["ci95_upper"]) for row in agg])
        axis.plot(x,mean,color="#2166ac",linewidth=2.3,label="mean")
        axis.fill_between(x,low,high,color="#67a9cf",alpha=.25,label="95% Student-t CI")
        axis.set(xlabel="sampled steps (million)",ylabel=metric,title=f"{condition}: {metric}")
        axis.grid(True,alpha=.25); axis.legend(fontsize=8,ncol=2)
        path=output_dir/f"{condition.lower()}_{metric}.png"; fig.savefig(path,dpi=190); plt.close(fig); outputs.append(str(path))
    return outputs


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    output=args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT/args.output_dir; output.mkdir(parents=True,exist_ok=True)
    rendered=[]
    rendered += render("Direct",[PROJECT_ROOT/"outputs"/f"d999_seed{s}" for s in (2023,2024,2025)],
                       PROJECT_ROOT/"outputs/d999_3seed_training_summary",output,DIRECT_METRICS)
    rendered += render("Persistent",[PROJECT_ROOT/"outputs"/f"pw999_seed{s}" for s in (2023,2024,2025)],
                       PROJECT_ROOT/"outputs/pw999_3seed_training_summary",output,PERSISTENT_METRICS)
    print(json.dumps({"rendered":rendered},indent=2))


if __name__=="__main__": main()
