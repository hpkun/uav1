"""Render MAPPO audit curves and wave-safe trajectory artifacts without Torch."""
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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_curve(rows: list[dict], columns: list[str], output: Path,
               best_step: int, final_step: int, title: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    steps = np.asarray([float(row["sampled_steps"]) for row in rows])
    for column in columns:
        if all(column in row and row[column] not in (None, "") for row in rows):
            axis.plot(steps, [float(row[column]) for row in rows], label=column)
    axis.axvline(best_step, color="green", linestyle="--", label="best_eval")
    axis.axvline(final_step, color="black", linestyle=":", label="final")
    axis.set(xlabel="sampled steps", title=title)
    axis.grid(True, alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles: axis.legend(fontsize=8)
    fig.savefig(output, dpi=190); plt.close(fig)


def trajectory_groups(rows: list[dict]):
    def identity(row):
        return (row["side"], 0 if row["side"] == "red" else int(row["wave_index"]),
                int(row["aircraft"]))
    keys = sorted({identity(row)
                   for row in rows})
    return [(key, [row for row in rows if (row["side"], int(row["wave_index"]),
                                            int(row["aircraft"])) == key
                  or (row["side"] == "red" and key == (
                      "red", 0, int(row["aircraft"])
                  ))]) for key in keys]


def render_trajectory(csv_path: Path, json_path: Path, extra_views: bool) -> list[Path]:
    rows = read_csv(csv_path)
    payload = json.loads(json_path.read_text(encoding="utf-8")); summary = payload["summary"]
    outputs=[]; red_colors=["#7f0000","#b2182b","#d6604d","#f4a582"]
    blue_colors=["#053061","#2166ac","#4393c3","#92c5de"]
    fig=plt.figure(figsize=(12,9),constrained_layout=True); axis=fig.add_subplot(111,projection="3d")
    for (side,wave,aircraft),points in trajectory_groups(rows):
        xyz=np.asarray([[float(p["x_m"]),float(p["y_m"]),float(p["altitude_m"])] for p in points])/1000
        color=(red_colors if side=="red" else blue_colors)[aircraft-1]
        label=f"Red {aircraft}" if side=="red" else f"Blue W{wave}-{aircraft}"
        axis.plot(xyz[:,0],xyz[:,1],xyz[:,2],"-" if side=="red" else "--",
                  color=color,alpha=.9 if side=="red" else .65,
                  linewidth=1.8 if side=="red" else 1.2,label=label)
        axis.scatter(*xyz[0],color=color,marker="D" if points[0]["event"]=="spawn" else "o",s=35)
        axis.scatter(*xyz[-1],color=color,marker="X" if points[-1]["alive"].lower()=="false" else "^",s=55)
    angle=np.linspace(0,2*np.pi,240); axis.plot(5*np.cos(angle),5*np.sin(angle),np.zeros_like(angle),color="#555",label="5 km arena")
    axis.set(xlabel="x (km)",ylabel="y (km)",zlabel="altitude (km)"); axis.view_init(elev=25,azim=-56)
    axis.set_title(f"{payload['checkpoint']} | seed={summary['seed']} | waves={summary['waves_cleared']}/{summary['total_waves']} | return={summary['team_return']:.2f}\nRed loss={summary['red_losses']} | Blue loss={summary['blue_losses']} | {summary['termination_reason']} | steps={summary['episode_length']}")
    axis.legend(fontsize=7,ncol=2); axis.grid(True,alpha=.25)
    png=csv_path.with_suffix(".png"); fig.savefig(png,dpi=220); plt.close(fig); outputs.append(png)
    if not extra_views: return outputs
    transitions=[int(t["step"]) for t in summary["wave_transitions"]]
    for kind in ("topdown","altitude","alive"):
        fig,axis=plt.subplots(figsize=(10,7),constrained_layout=True)
        if kind in ("topdown","altitude"):
            for (side,wave,aircraft),points in trajectory_groups(rows):
                color="#b2182b" if side=="red" else "#2166ac"; label=f"R{aircraft}" if side=="red" else f"B W{wave}-{aircraft}"
                if kind=="topdown": axis.plot([float(p["x_m"])/1000 for p in points],[float(p["y_m"])/1000 for p in points],color=color,alpha=.65,label=label)
                else: axis.plot([int(p["step"]) for p in points],[float(p["altitude_m"])/1000 for p in points],color=color,alpha=.65,label=label)
            if kind=="topdown":
                angle=np.linspace(0,2*np.pi,240); axis.plot(5*np.cos(angle),5*np.sin(angle),color="black"); axis.set_aspect("equal"); axis.set(xlabel="x (km)",ylabel="y (km)")
            else: axis.set(xlabel="step",ylabel="altitude (km)")
        else:
            max_step=int(summary["episode_length"]); steps=np.arange(max_step+1); red_alive=[]; blue_alive=[]; latest={}; ordered=sorted(rows,key=lambda r:int(r["step"])); cursor=0
            for step in steps:
                while cursor<len(ordered) and int(ordered[cursor]["step"])<=step:
                    row=ordered[cursor]; latest[(row["side"],int(row["wave_index"]),int(row["aircraft"]))]=row; cursor+=1
                active=1+sum(step>=t for t in transitions)
                red_alive.append(sum(r["alive"].lower()=="true" for k,r in latest.items() if k[0]=="red"))
                blue_alive.append(sum(r["alive"].lower()=="true" for k,r in latest.items() if k[0]=="blue" and k[1]==active))
            axis.step(steps,red_alive,where="post",color="#b2182b",label="Red alive"); axis.step(steps,blue_alive,where="post",color="#2166ac",label="Active Blue alive"); axis.set(xlabel="step",ylabel="alive aircraft",ylim=(-.1,4.3))
        if kind!="topdown":
            for step in transitions: axis.axvline(step,color="grey",linestyle=":")
        axis.set_title(f"seed={summary['seed']} | {kind}"); axis.legend(fontsize=7,ncol=2); axis.grid(True,alpha=.25)
        path=csv_path.with_name(f"{csv_path.stem}_{kind}.png"); fig.savefig(path,dpi=190); plt.close(fig); outputs.append(path)
    return outputs


def render_analysis(run_dir: Path, analysis_dir: Path) -> list[Path]:
    summary=json.loads((analysis_dir/"training_audit_summary.json").read_text(encoding="utf-8")); best=int(summary["best_training_evaluation"]["sampled_steps"]); final=3_000_000
    evaluations=read_csv(run_dir/"evaluation_history.csv")
    optimization=[json.loads(line) for line in (run_dir/"optimization_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    specs=[(evaluations,["average_return"],"evaluation_return_curve.png","Evaluation return"),
           (evaluations,["clear_wave_1_probability","clear_wave_2_probability","clear_wave_3_probability"],"evaluation_wave_clear_curve.png","Wave clear probability"),
           (evaluations,["average_waves_cleared"],"evaluation_average_waves_curve.png","Average waves cleared"),
           (evaluations,["average_red_loss"],"evaluation_red_loss_curve.png","Red loss"),
           (evaluations,["average_red_survivors_after_wave_1","average_red_survivors_after_wave_2","average_red_survivors_after_wave_3"],"red_survivors_by_wave_curve.png","Red survivors by cleared wave"),
           (evaluations,["average_red_boundary_exits","average_red_ground_losses","average_blue_ground_losses"],"boundary_ground_curve.png","Boundary and ground losses"),
           (optimization,["approx_kl"],"optimization_kl_curve.png","Approximate KL"),
           (optimization,["entropy"],"optimization_entropy_curve.png","Policy differential entropy"),
           (optimization,["value_loss"],"optimization_value_loss_curve.png","Value loss")]
    outputs=[]
    for rows,columns,name,title in specs:
        path=analysis_dir/name; plot_curve(rows,columns,path,best,final,title); outputs.append(path)
    for csv_path in sorted(analysis_dir.glob("trajectory_*.csv")):
        extra=("best_success" in csv_path.stem or "drift_" in csv_path.stem)
        outputs.extend(render_trajectory(csv_path,csv_path.with_suffix(".json"),extra))
    return outputs


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--analysis-dir",type=Path)
    parser.add_argument("--trajectory-dir",type=Path)
    args=parser.parse_args()
    if args.trajectory_dir:
        directory=args.trajectory_dir if args.trajectory_dir.is_absolute() else PROJECT_ROOT/args.trajectory_dir
        outputs=[]
        for csv_path in sorted(directory.glob("*.csv")):
            outputs.extend(render_trajectory(csv_path,csv_path.with_suffix(".json"),True))
    else:
        if not args.run_dir or not args.analysis_dir:
            parser.error("--run-dir and --analysis-dir are required unless --trajectory-dir is used")
        run=args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT/args.run_dir
        analysis=args.analysis_dir if args.analysis_dir.is_absolute() else PROJECT_ROOT/args.analysis_dir
        outputs=render_analysis(run,analysis)
    print(json.dumps({"rendered":[str(p) for p in outputs]},indent=2))


if __name__=="__main__": main()
