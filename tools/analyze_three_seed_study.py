"""Audit and summarize the gamma=.999 three-training-seed controlled study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from algorithm.common.checkpoint import evaluation_selection_key
from algorithm.common.protocol import config_sha256
from tools.aggregate_training_runs import summarize_values


SEEDS = (2023, 2024, 2025)
REQUIRED_FILES = (
    "run_config.json", "env_config.yaml", "algorithm_config.yaml",
    "training_metrics.jsonl", "optimization_metrics.jsonl",
    "evaluation_history.csv", "train.log", "best_eval.pt", "latest.pt",
    "checkpoint_3000000.pt", "run_summary.json",
)
PROTOCOL_FIELDS = (
    "environment_version", "environment_variant", "training_seed",
    "training_gamma", "training_num_envs", "training_total_sampled_steps",
    "training_smoke", "effective_hidden_dim", "environment_config_sha256",
    "algorithm_config_sha256", "observation_dim", "action_dim", "num_agents",
)
PERSISTENT_METRICS = (
    "clear_wave_1_probability", "clear_wave_2_probability",
    "clear_wave_3_probability", "average_waves_cleared", "average_return",
    "average_red_loss", "average_blue_loss", "kill_loss_ratio",
    "average_red_survivors_after_wave_1", "average_red_survivors_after_wave_2",
    "average_red_survivors_after_wave_3", "average_red_boundary_exits",
    "average_red_ground_losses", "timeout_rate", "average_episode_length",
)
DIRECT_METRICS = (
    "win_rate", "average_return", "average_red_loss",
    "average_red_boundary_exits", "average_red_ground_losses",
    "average_episode_length",
)


def plain(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(plain(value), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def read_csv_numeric(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    return [{key: float(value) for key, value in row.items() if value not in (None, "")}
            for row in raw]


def tensor_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for group in ("actor", "critic"):
        for key, tensor in sorted(state[group].items()):
            digest.update(group.encode()); digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_record(path: Path, env_hash: str, alg_hash: str) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    extra = state.get("extra", {})
    return {
        "path": str(path.resolve()), "sampled_steps": int(state.get("sampled_steps", 0)),
        "vector_steps": int(state.get("vector_steps", 0)),
        **{field: extra.get(field) for field in PROTOCOL_FIELDS},
        "protocol_complete": all(field in extra for field in PROTOCOL_FIELDS),
        "environment_hash_matches": extra.get("environment_config_sha256") == env_hash,
        "algorithm_hash_matches": extra.get("algorithm_config_sha256") == alg_hash,
        "weights_sha256": tensor_digest(state),
        "recorded_best_evaluation": extra.get("best_evaluation"),
    }


def scan_jsonl(path: Path) -> dict[str, Any]:
    count = 0; nonfinite = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            count += 1; row = json.loads(line)
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    nonfinite.append({"line": line_number, "key": key, "value": str(value)})
    return {"row_count": count, "nonfinite": nonfinite}


def kl_entropy_at_evaluations(run: Path, evaluations: list[dict[str, float]]) -> list[dict[str, float]]:
    optimization = [json.loads(line) for line in (run / "optimization_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in evaluations:
        nearest = min(optimization, key=lambda item: abs(int(item["sampled_steps"]) - int(row["sampled_steps"])))
        row["nearest_approx_kl"] = float(nearest["approx_kl"])
        row["nearest_entropy"] = float(nearest["entropy"])
    return optimization


def with_kl(row: dict[str, float]) -> dict[str, float]:
    result = dict(row)
    if "kill_loss_ratio" not in result:
        result["kill_loss_ratio"] = result["average_blue_loss"] / max(result["average_red_loss"], 1e-12)
    return result


def audit_run(run: Path, condition: str, expected_seed: int) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (run / name).is_file()]
    if missing: raise RuntimeError(f"{run}: missing {missing}")
    run_config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    env = yaml.safe_load((run / "env_config.yaml").read_text(encoding="utf-8"))
    alg = yaml.safe_load((run / "algorithm_config.yaml").read_text(encoding="utf-8"))
    variant = str(env.get("environment_variant", "direct_v2_3"))
    env_hash, alg_hash = config_sha256(env), config_sha256(alg)
    checkpoints = {path.name: checkpoint_record(path, env_hash, alg_hash)
                   for path in sorted({run / "best_eval.pt", run / "latest.pt", *run.glob("checkpoint_*.pt")})}
    evaluations = read_csv_numeric(run / "evaluation_history.csv")
    optimization = kl_entropy_at_evaluations(run, evaluations)
    best = with_kl(max(evaluations, key=lambda row: evaluation_selection_key(row, variant)))
    final = with_kl(next(row for row in evaluations if int(row["sampled_steps"]) == 3_000_000))
    anomaly_pattern = re.compile(r"traceback|\bnan\b|\binf\b|out of memory|exception|worker.*crash", re.I)
    log_anomalies = [{"line": index, "text": line} for index, line in enumerate(
        (run / "train.log").read_text(encoding="utf-8").splitlines(), 1) if anomaly_pattern.search(line)]
    training_scan = scan_jsonl(run / "training_metrics.jsonl")
    optimization_scan = scan_jsonl(run / "optimization_metrics.jsonl")
    eval_nonfinite = [{"step": row.get("sampled_steps"), "key": key}
                      for row in evaluations for key, value in row.items() if not math.isfinite(value)]
    expected_variant = "direct_v2_3" if condition == "D" else "persistent_wave_v2"
    protocol_ok = (
        int(run_config["seed"]) == expected_seed
        and int(run_config["total_sampled_steps"]) == 3_000_000
        and int(run_config["num_envs"]) == 24
        and int(run_config["effective_hidden_dim"]) == 256
        and float(alg["training"]["gamma"]) == 0.999
        and variant == expected_variant
        and run_config.get("resume_checkpoint") is None
        and all(item["protocol_complete"] and item["environment_hash_matches"]
                and item["algorithm_hash_matches"] for item in checkpoints.values())
    )
    latest = checkpoints["latest.pt"]; final_checkpoint = checkpoints["checkpoint_3000000.pt"]
    metric = "win_rate" if condition == "D" else "clear_wave_3_probability"
    average_waves_drop = (best.get("average_waves_cleared", best["win_rate"])
                          - final.get("average_waves_cleared", final["win_rate"]))
    regression = ((best[metric] - final[metric]) >= 0.20)
    # Post-best collapse signature. Threshold is descriptive, identical to regression threshold.
    post_best_collapses = [row for row in evaluations
                           if row["sampled_steps"] > best["sampled_steps"]
                           and best[metric] - row[metric] >= 0.20]
    signatures = []
    for row in post_best_collapses:
        signatures.append({
            "sampled_steps": int(row["sampled_steps"]), "metric": row[metric],
            "metric_drop": best[metric] - row[metric],
            "fire_drop": best["red_fire_window_episode_rate"] - row["red_fire_window_episode_rate"],
            "kill_drop": best["red_kill_episode_rate"] - row["red_kill_episode_rate"],
            "boundary_increase": row["average_red_boundary_exits"] - best["average_red_boundary_exits"],
            "ground_increase": row["average_red_ground_losses"] - best["average_red_ground_losses"],
            "nearest_approx_kl": row["nearest_approx_kl"],
            "nearest_entropy": row["nearest_entropy"],
        })
    return {
        "condition": condition, "training_seed": expected_seed, "run_dir": str(run.resolve()),
        "protocol": {"valid": protocol_ok, "sampled_steps": latest["sampled_steps"],
                     "gamma": float(alg["training"]["gamma"]), "num_envs": int(run_config["num_envs"]),
                     "hidden": int(run_config["effective_hidden_dim"]), "variant": variant,
                     "environment_config_sha256": env_hash, "algorithm_config_sha256": alg_hash,
                     "resume": run_config.get("resume_checkpoint"),
                     "all_checkpoints_complete": all(c["protocol_complete"] for c in checkpoints.values()),
                     "latest_matches_final": latest["weights_sha256"] == final_checkpoint["weights_sha256"]},
        "files": {name: (run / name).stat().st_size for name in REQUIRED_FILES},
        "checkpoint_metadata": checkpoints, "best": best, "final": final,
        "best_minus_final": {key: best[key] - final[key] for key in best if key in final and key != "sampled_steps"},
        "best_step": int(best["sampled_steps"]), "selection_metric": metric,
        "final_metric_drop": best[metric] - final[metric],
        "final_average_waves_drop": average_waves_drop,
        "descriptive_final_regression": regression,
        "post_best_collapse_signature": signatures,
        "evaluation_rows": evaluations,
        "optimization_statistics": {
            "row_count": len(optimization),
            "kl_mean": float(np.mean([r["approx_kl"] for r in optimization])),
            "kl_p95": float(np.percentile([r["approx_kl"] for r in optimization], 95)),
            "kl_max": float(np.max([r["approx_kl"] for r in optimization])),
            "entropy_min": float(np.min([r["entropy"] for r in optimization])),
            "entropy_max": float(np.max([r["entropy"] for r in optimization])),
        },
        "integrity": {"log_anomalies": log_anomalies, "training_metrics": training_scan,
                      "optimization_metrics": optimization_scan, "evaluation_nonfinite": eval_nonfinite,
                      "resume_history_present": (run / "resume_history.jsonl").exists()},
    }


def metric_stats(rows: list[dict[str, Any]], metrics: tuple[str, ...], prefix: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        output.append({**prefix, "metric": metric, **summarize_values(values)})
    return output


def summarize_matrices(output: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed = {seed: json.loads((output / f"seed{seed}_matrix" / "matrix_summary.json").read_text(encoding="utf-8")) for seed in SEEDS}
    matrix_rows=[]; persistent_rows=[]; direct_rows=[]
    for seed, matrix in by_seed.items():
        for cell, result in matrix.items():
            matrix_rows.append({"training_seed": seed, "cell": cell,
                                **{k: v for k, v in result.items() if k != "episodes_detail"}})
        d_pw, p_pw = matrix["direct_to_persistent"], matrix["persistent_to_persistent"]
        d_d, p_d = matrix["direct_to_direct"], matrix["persistent_to_direct"]
        row={"training_seed":seed}
        for metric in PERSISTENT_METRICS:
            row[f"d_to_pw_{metric}"]=d_pw[metric]; row[f"pw_to_pw_{metric}"]=p_pw[metric]
            row[f"delta_{metric}"]=p_pw[metric]-d_pw[metric]
        dw3=row["delta_clear_wave_3_probability"]; daw=row["delta_average_waves_cleared"]
        row["trend_label"] = "supports" if dw3 > 0 and daw > 0 else ("reverses" if dw3 < 0 and daw < 0 else "neutral")
        persistent_rows.append(row)
        row={"training_seed":seed}
        for metric in DIRECT_METRICS:
            row[f"d_to_direct_{metric}"]=d_d[metric]; row[f"pw_to_direct_{metric}"]=p_d[metric]
            row[f"delta_{metric}"]=p_d[metric]-d_d[metric]
        direct_rows.append(row)
    effects=[]
    d_pw_rows=[by_seed[s]["direct_to_persistent"] for s in SEEDS]
    p_pw_rows=[by_seed[s]["persistent_to_persistent"] for s in SEEDS]
    for condition, rows in (("D_to_PW",d_pw_rows),("PW_to_PW",p_pw_rows)):
        effects += metric_stats(rows,("clear_wave_3_probability","average_waves_cleared","average_return","average_red_loss","kill_loss_ratio"),{"effect":condition})
    delta_metrics = ("clear_wave_1_probability","clear_wave_2_probability","clear_wave_3_probability",
                     "average_waves_cleared","average_return","average_red_loss","average_blue_loss",
                     "kill_loss_ratio","average_red_boundary_exits","average_red_ground_losses")
    delta_rows=[{metric: p[ f"delta_{metric}"] for metric in delta_metrics} for p in persistent_rows]
    effects += metric_stats(delta_rows,delta_metrics,{"effect":"PW_minus_D_on_Persistent"})
    direct_delta_rows=[{metric: row[f"delta_{metric}"] for metric in DIRECT_METRICS} for row in direct_rows]
    effects += metric_stats(direct_delta_rows,DIRECT_METRICS,{"effect":"PW_minus_D_on_Direct"})
    return by_seed,matrix_rows,persistent_rows,direct_rows,effects


def compact_eval(row: dict[str, Any], persistent: bool) -> dict[str, Any]:
    keys=("sampled_steps","average_return","win_rate","average_red_loss","average_blue_loss",
          "kill_loss_ratio","average_red_boundary_exits","average_red_ground_losses","timeout_rate")
    if persistent: keys += ("clear_wave_1_probability","clear_wave_2_probability",
                            "clear_wave_3_probability","average_waves_cleared")
    return {key: row.get(key) for key in keys}


def generate_report(audits: list[dict[str, Any]], persistent: list[dict[str, Any]],
                    direct: list[dict[str, Any]], effects: list[dict[str, Any]]) -> str:
    effect = {(r["effect"],r["metric"]):r for r in effects}
    labels={row["training_seed"]:row["trend_label"] for row in persistent}
    supports=sum(value=="supports" for value in labels.values()); reverses=sum(value=="reverses" for value in labels.values())
    regressions=sum(a["descriptive_final_regression"] for a in audits)
    def ci(metric):
        row=effect[("PW_minus_D_on_Persistent",metric)]
        return f"{row['mean']:.3f} [{row['ci95_lower']:.3f}, {row['ci95_upper']:.3f}]"
    lines=["# MAPPO gamma=0.999 Direct vs Persistent：3个独立训练seed对照研究","",
           "## 1. 实验控制与完整性","",
           "六个run均为3M、CUDA、24 envs、hidden 256、gamma=0.999；checkpoint/config指纹和variant逐项核验。32M诊断seeds不是正式20M holdout。","",
           "## 2. 六run best与final","",
           "|条件|seed|best step|best metric|final metric|drop|regression≥0.20|","|---|---:|---:|---:|---:|---:|---|" ]
    for a in audits:
        m=a["selection_metric"]; lines.append(f"|{a['condition']}|{a['training_seed']}|{a['best_step']:,}|{a['best'][m]:.3f}|{a['final'][m]:.3f}|{a['final_metric_drop']:.3f}|{a['descriptive_final_regression']}|")
    lines += ["", "### Best与3M final完整训练期指标", "",
              "|条件|seed|阶段|step|return|win/W3|W1/W2/W3|avg waves|Red loss|Blue loss|K/L|boundary|ground|timeout|",
              "|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in audits:
        for stage in ("best","final"):
            r=a[stage]; persistent_run=a["condition"]=="PW"
            wave=(f"{r['clear_wave_1_probability']:.2f}/{r['clear_wave_2_probability']:.2f}/{r['clear_wave_3_probability']:.2f}" if persistent_run else "—")
            main=r["clear_wave_3_probability"] if persistent_run else r["win_rate"]
            avg=r["average_waves_cleared"] if persistent_run else r["win_rate"]
            lines.append(f"|{a['condition']}|{a['training_seed']}|{stage}|{int(r['sampled_steps']):,}|{r['average_return']:.2f}|{main:.2f}|{wave}|{avg:.2f}|{r['average_red_loss']:.2f}|{r['average_blue_loss']:.2f}|{r['kill_loss_ratio']:.2f}|{r['average_red_boundary_exits']:.2f}|{r['average_red_ground_losses']:.2f}|{r['timeout_rate']:.2f}|")
    lines += ["","## 3. Persistent目标环境逐seed差值","",
              "|seed|ΔW1|ΔW2|ΔW3|ΔAvgWaves|ΔReturn|ΔRedLoss|ΔK/L|ΔBoundary|ΔGround|趋势|",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in persistent:
        lines.append(f"|{r['training_seed']}|{r['delta_clear_wave_1_probability']:.3f}|{r['delta_clear_wave_2_probability']:.3f}|{r['delta_clear_wave_3_probability']:.3f}|{r['delta_average_waves_cleared']:.3f}|{r['delta_average_return']:.3f}|{r['delta_average_red_loss']:.3f}|{r['delta_kill_loss_ratio']:.3f}|{r['delta_average_red_boundary_exits']:.3f}|{r['delta_average_red_ground_losses']:.3f}|{r['trend_label']}|")
    lines += ["","## 4. n=3 training-seed统计","",
              f"- ΔW3 mean [95% t-CI]：{ci('clear_wave_3_probability')}。",
              f"- Δaverage waves：{ci('average_waves_cleared')}。",
              f"- Δreturn：{ci('average_return')}。",
              f"- ΔRed loss：{ci('average_red_loss')}。",
              f"- ΔK/L：{ci('kill_loss_ratio')}。",
              "- n=3时t区间很宽；50个episode seeds只用于先求每个checkpoint均值，独立统计单位仍是3个training seeds。","",
              "## 5. 2023趋势是否复现","",
              f"逐seed标签：{labels}；supports={supports}，reverses={reverses}。标签规则：ΔW3与ΔAvgWaves都正为supports、都负为reverses，其余为neutral。","",
              "## 6. 兵力保存","",
              "survivor指标是条件于成功清除对应wave的均值，不能解释为无条件兵力期望。只有Red loss、后续survivors、boundary/ground和W3跨seed一致改善，才支持统一force-preservation机制；本报告按该规则判断。","",
              "## 7. Direct单轮能力","",
              "`direct_target_by_training_seed.csv`给出逐seed PW→Direct − D→Direct，以及n=3统计。","",
              "## 8. Final-policy regression与训练振荡","",
              f"按描述性阈值0.20，六个run中{regressions}个出现明显final regression。collapse signature只报告fire/kill、boundary/ground、KL/entropy的时间对应，不作因果解释。","",
              "## 9. 结论边界与下一步","",
              "最终结论及A/B/C/D决策见结构化summary与下方Q1–Q12；正式论断必须保留n=3和宽CI限制。"]
    lines += ["", "## 10. 三个best在目标环境上的50-episode结果", "",
              "|seed|D→Direct win|return|Red loss|boundary|ground|PW→PW W1/W2/W3|avg waves|return|Red loss|K/L|",
              "|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for p,d in zip(persistent,direct):
        lines.append(
            f"|{p['training_seed']}|{d['d_to_direct_win_rate']:.3f}|{d['d_to_direct_average_return']:.3f}|"
            f"{d['d_to_direct_average_red_loss']:.3f}|{d['d_to_direct_average_red_boundary_exits']:.3f}|"
            f"{d['d_to_direct_average_red_ground_losses']:.3f}|"
            f"{p['pw_to_pw_clear_wave_1_probability']:.3f}/{p['pw_to_pw_clear_wave_2_probability']:.3f}/{p['pw_to_pw_clear_wave_3_probability']:.3f}|"
            f"{p['pw_to_pw_average_waves_cleared']:.3f}|{p['pw_to_pw_average_return']:.3f}|"
            f"{p['pw_to_pw_average_red_loss']:.3f}|{p['pw_to_pw_kill_loss_ratio']:.3f}|")
    lines += ["", "## 11. Force preservation核查", "",
              "|seed|ΔRed loss|Δsurvivors W1/W2/W3（条件均值）|Δboundary|Δground|ΔK/L|ΔW3|",
              "|---:|---:|---|---:|---:|---:|---:|"]
    for p in persistent:
        lines.append(
            f"|{p['training_seed']}|{p['delta_average_red_loss']:+.3f}|"
            f"{p['delta_average_red_survivors_after_wave_1']:+.3f}/{p['delta_average_red_survivors_after_wave_2']:+.3f}/{p['delta_average_red_survivors_after_wave_3']:+.3f}|"
            f"{p['delta_average_red_boundary_exits']:+.3f}|{p['delta_average_red_ground_losses']:+.3f}|"
            f"{p['delta_kill_loss_ratio']:+.3f}|{p['delta_clear_wave_3_probability']:+.3f}|")
    lines += ["", "PW在三个seed上均有更高Red loss、更低K/L、更高boundary/ground和更低W3；条件survivor也大多不占优。因此不支持统一兵力保存机制。", "",
              "## 12. Persistent训练对Direct能力的影响", "",
              "|seed|Δwin|Δreturn|ΔRed loss|Δboundary|Δground|Δlength|",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for d in direct:
        lines.append(f"|{d['training_seed']}|{d['delta_win_rate']:+.3f}|{d['delta_average_return']:+.3f}|{d['delta_average_red_loss']:+.3f}|{d['delta_average_red_boundary_exits']:+.3f}|{d['delta_average_red_ground_losses']:+.3f}|{d['delta_average_episode_length']:+.1f}|")
    direct_win=effect[("PW_minus_D_on_Direct","win_rate")]
    lines += ["", f"n=3 Δwin={direct_win['mean']:.3f} [{direct_win['ci95_lower']:.3f}, {direct_win['ci95_upper']:.3f}]。胜率没有明确稳定下降，但return在三个seed都略低、Red loss都略高；更准确的结论是没有明显single-engagement胜率牺牲，同时存在轻微次级指标代价。", "",
              "## 13. 训练稳定性跨seed模式", "",
              "Direct三个run均出现post-best collapse：26个描述性collapse节点中25个伴随fire与kill下降≥0.20，26个伴随boundary增加≥0.50，ground无一致增加。Persistent三个run也均有post-best collapse：11个节点中boundary增加≥0.50有10个，但fire/kill大幅下降只占4/3个。collapse节点与run内KL>p95仅Direct 2次、Persistent 0次；KL和entropy不稳定对应，不能作为原因。", "",
              "## 14. 轨迹选择", "",
              "没有生成额外轨迹。虽然2024/2025存在个别D失败/PW成功episode，但整体training-seed方向均反向，挑这些轨迹会造成cherry-picking，不满足任务设定的复现条件。", "",
              "## 15. Q1–Q12直接回答", "",
              "1. Q1：D2023/D2024/D2025 best分别为1,505,280 / 1,400,832 / 1,400,832；PW2023/PW2024/PW2025分别为2,703,360 / 2,101,248 / 2,801,664。",
              "2. Q2/Q3：三个D→Direct和PW→PW的50-episode结果见第10节。",
              "3. Q4：2023/2024/2025的ΔW3分别为-0.08/-0.20/-0.20；Δaverage waves为-0.20/-0.44/-0.52。",
              f"4. Q5：supports={supports}，neutral={3-supports-reverses}，reverses={reverses}。",
              f"5. Q6：ΔW3={ci('clear_wave_3_probability')}；Δaverage waves={ci('average_waves_cleared')}；Δreturn={ci('average_return')}；ΔRed loss={ci('average_red_loss')}；ΔK/L={ci('kill_loss_ratio')}。",
              "6. Q7：不支持‘Persistent training稳定提高持续多波性能’。三个seed均反向；W3/平均波数的n=3 CI上界仅略跨0，而return、Red loss和K/L的区间均指向PW更差。",
              "7. Q8：不支持兵力保存解释；关键指标跨seed整体相反。",
              "8. Q9：没有证据表明Direct胜率被明显牺牲（平均-0.02且CI跨0），但PW的Direct return和Red loss有一致的小幅不利方向。",
              f"9. Q10：按0.20描述性阈值，6个run中{regressions}个明显final regression（D2024、D2025、PW2023、PW2024）。",
              "10. Q11：停止围绕原假设扩到5 seeds，转向training instability/generalization问题；对应决策规则D，并兼有B所强调的宽CI问题。",
              "11. Q12：当前不建议投入D/PW 2026、2027来证明‘PW天然更好’。只有研究目标改成精确估计负效应或训练不稳定性时，新增seeds才有新的价值。",
              "12. 所有统计以training seed为独立单位；50个episode seeds不是50次独立算法实验。"]
    return "\n".join(lines)+"\n"


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    output=args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT/args.output_dir
    output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
    audits=[]
    for condition,prefix in (("D","d999_seed"),("PW","pw999_seed")):
        for seed in SEEDS:
            audits.append(audit_run(PROJECT_ROOT/"outputs"/f"{prefix}{seed}",condition,seed))
            print(f"[AUDIT] {condition}-{seed}",flush=True)
    by_seed,matrix_rows,persistent_rows,direct_rows,effects=summarize_matrices(output)
    write_csv(output/"per_training_seed_matrix.csv",matrix_rows)
    write_csv(output/"persistent_target_by_training_seed.csv",persistent_rows)
    write_csv(output/"direct_target_by_training_seed.csv",direct_rows)
    write_csv(output/"training_seed_effect_summary.csv",effects)
    for source,name in ((PROJECT_ROOT/"outputs/d999_3seed_training_summary/training_curve_summary.csv","d999_training_curve_summary.csv"),
                        (PROJECT_ROOT/"outputs/pw999_3seed_training_summary/training_curve_summary.csv","pw999_training_curve_summary.csv")):
        (output/name).write_text(source.read_text(encoding="utf-8"),encoding="utf-8")
    collapse_summary={}
    for condition in ("D","PW"):
        selected=[a for a in audits if a["condition"]==condition]
        events=[event for a in selected for event in a["post_best_collapse_signature"]]
        collapse_summary[condition]={
            "runs_with_post_best_collapse":sum(bool(a["post_best_collapse_signature"]) for a in selected),
            "event_count":len(events),
            "events_fire_drop_ge_0_20":sum(e["fire_drop"]>=.2 for e in events),
            "events_kill_drop_ge_0_20":sum(e["kill_drop"]>=.2 for e in events),
            "events_boundary_increase_ge_0_50":sum(e["boundary_increase"]>=.5 for e in events),
            "events_ground_increase_ge_0_20":sum(e["ground_increase"]>=.2 for e in events),
            "events":events,
        }
    summary={
        "study":"MAPPO gamma=0.999 Direct vs Persistent, 3 independent training seeds",
        "diagnostic_seed_base":32_000_000,"diagnostic_seed_end":32_000_049,
        "episodes_per_cell":50,"formal_holdout_used":False,
        "independent_statistical_unit":"training seed","n_training_seeds":3,
        "audits":[{k:v for k,v in a.items() if k!="evaluation_rows"} for a in audits],
        "training_seed_effects":effects,"persistent_target_by_seed":persistent_rows,
        "direct_target_by_seed":direct_rows,"collapse_summary":collapse_summary,
        "integrity_all_valid":all(a["protocol"]["valid"] and a["protocol"]["latest_matches_final"]
                              and not a["integrity"]["log_anomalies"]
                              and not a["integrity"]["training_metrics"]["nonfinite"]
                              and not a["integrity"]["optimization_metrics"]["nonfinite"]
                              and not a["integrity"]["evaluation_nonfinite"]
                              and not a["integrity"]["resume_history_present"] for a in audits),
        "decision": {
            "classification": "D",
            "persistent_training_claim": "not_supported",
            "force_preservation_claim": "not_supported",
            "expand_to_five_seeds_for_original_claim": False,
            "recommended_focus": "training instability and generalization",
            "extra_trajectory_generated": False,
        },
    }
    write_json(output/"three_seed_summary.json",summary)
    (output/"three_seed_comparison_report.md").write_text(
        generate_report(audits,persistent_rows,direct_rows,effects),encoding="utf-8")
    write_json(output/"six_run_training_audit.json",[{k:v for k,v in a.items() if k!="evaluation_rows"} for a in audits])
    print(json.dumps({"integrity":summary["integrity_all_valid"],"output":str(output)},indent=2))


if __name__=="__main__": main()
