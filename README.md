# Multi-UAV Cooperative Air Combat

新增的论文 R4 对照环境为 **v3.46**，以 **v3.44** 为基线，仅替换 R4；MAPPO/IPPO/MAAC 均有独立正式配置。原 **v3.45（45°）已退役**，历史实验记录保留。当前版本说明、验证与训练命令见 [v3.46](docs/combat_environment_v346.md)，连续驻留基线见 [v3.44](docs/combat_environment_v344.md)。下文旧版本参数描述按其各自配置解释。

当前正式 5v5 MAPPO 是 shared-parameter MAPPO with a centralized MLP critic；STEA-MAPPO 保留 entity-attention + GRU actor，并使用完全相同的 MLP critic。Actor/critic 宽度分别读取配置，旧 attention checkpoint 使用各自 run snapshots 继续评估。详见 [critic 协议](docs/centralized_critic_protocol.md)。下文的 4v4 说明保留原 v2.3 协议。

这是一个面向 multi-UAV cooperative air combat 的多智能体强化学习研究代码库。任务为普通单回合 **4v4** 空战：4 架 Red UAV 使用共享的 learned policy，4 架 Blue UAV 使用确定性的 nearest-target pursuit policy。项目提供 **MAPPO** 和 **MADSAC** 两个独立 baseline，以及 Actor 使用实体注意力与 GRU 的独立变体 **STEA-MAPPO**。

环境使用 NED 坐标下的 3DOF point-mass dynamics、RK4 integration 和 `dt=0.1 s`。每个 Red agent 接收 **52D observation**，输出 **3D continuous action**：heading、pitch、speed 的相对指令。物理环境技术版本为 **2.3**。

## Installation

优先在 WSL Ubuntu 中使用 `uav` Conda 环境：

```bash
conda create -n uav python=3.10
conda activate uav
cd /mnt/c/Users/HPK/Desktop/uav1
pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA required'"
```

已有 `uav` 环境时只需激活。训练、checkpoint evaluation 和审计必须使用 CUDA；CUDA 不可用时停止执行。测试和静态检查也在同一 `uav` 环境中运行。WSL 不可用时使用 Windows 的 `uav` Conda 环境。

## Project structure

```text
env/                         # 4v4 combat、控制器、动力学、武器、奖励、观测
algorithm/
    common/                  # 并行环境、指标、checkpoint 与配置指纹
    mappo/                   # 共享 actor、集中式 critic、PPO/GAE
    madsac/                  # 共享 actor、双 attention Q critic、joint replay
    stea_mappo/               # 实体注意力 Actor、GRU、连续序列 recurrent PPO
    train_mappo.py
    evaluate_mappo.py
    train_madsac.py
    evaluate_madsac.py
    train_stea_mappo.py
    evaluate_stea_mappo.py
configs/
    combat_environment.yaml  # 唯一正式环境配置
    mappo.yaml
    madsac.yaml
    stea_mappo.yaml
tools/                       # 环境验证、结果汇总、episode recording/rendering
tests/                       # 环境、算法、训练协议、checkpoint、可视化回归
docs/                        # 环境规范与记录工具说明
outputs/                     # 运行产物，不纳入 Git
```

## Combat environment

`MultiUAVCombatEnv` 是唯一正式环境；`make_combat_environment()` 直接构造该环境。Red 与 Blue 各 4 架，双方使用相同控制器与武器模型。

- 动作裁剪到 `[-1,1]`，对应 heading ±π、pitch ±π/3、speed ±50 m/s 的相对指令。
- 飞行速度范围为 150–300 m/s；pitch 范围为 ±60°。
- 初始化使用随机直径、两队中心半径 4 km 及原有编队偏移和扰动。
- 武器窗口为距离 0–4000 m、真实 3D off-boresight ≤30°。进入合法窗口触发一次尝试，离开窗口后重新武装；命中噪声在攻击机速度坐标系中计算。
- 双方命中同时结算；多个 Red 攻击者共同击杀同一目标时共享击杀奖励。
- 水平半径 5 km 为硬边界；越界与撞地销毁飞机。奖励由原有 R1–R4 组成。
- Blue 全灭为 Red win，Red 全灭为 Blue win，同时全灭为 draw；双方仍存活且到达 `max_steps=1000` 为 timeout/failure。

52D observation 由 7 个 self features、3×7 个 ally features 和 4×6 个 enemy features 组成；死亡实体对应零特征。完整定义见 [环境规范](docs/combat_environment_spec.md)。

## MAPPO smoke test

```bash
python algorithm/train_mappo.py --smoke --device cuda --num-envs 1 --output-dir outputs/mappo_smoke
```

Smoke 使用较小网络和 rollout，总计默认 192 个环境 transitions。MAPPO 默认 `gamma=0.99`。

## MAPPO training

```bash
python algorithm/train_mappo.py --device cuda --seed 1 --output-dir outputs/mappo_seed1
```

默认参数来自 `configs/mappo.yaml`；可用 `--num-envs`、`--total-sampled-steps` 和 `--algorithm-config` 调整训练协议。环境默认使用 `configs/combat_environment.yaml`。

```bash
python algorithm/train_mappo.py --resume outputs/mappo_seed1/latest.pt --device cuda --total-sampled-steps 9000000
```

Resume 校验版本、维度、配置指纹与原始运行参数，恢复网络、优化器和计数器，并从新的 episode 开始采样。它不提供逐 transition 完全相同的轨迹恢复。

## MAPPO evaluation

```bash
python algorithm/evaluate_mappo.py --checkpoint outputs/mappo_smoke/latest.pt --device cuda --seed-base 30000000 --episodes 2 --output outputs/mappo_smoke/holdout.json
```

使用训练运行保存的配置快照评估自定义实验时，加上 `--env-config <run>/env_config.yaml --algorithm-config <run>/algorithm_config.yaml`。Smoke checkpoint 会按保存的有效网络宽度加载。MAPPO 使用 deterministic evaluation。

## STEA-MAPPO training and evaluation

```bash
python algorithm/train_stea_mappo.py --smoke --device cuda --seed 1 --num-envs 2 --total-sampled-steps 96 --output-dir outputs/stea_smoke
python algorithm/train_stea_mappo.py --device cuda --seed 1 --num-envs 16 --total-sampled-steps 2000000 --output-dir outputs/stea_mappo_seed1_2m
python algorithm/evaluate_stea_mappo.py --checkpoint outputs/stea_mappo_seed1_2m/latest.pt --device cuda --seed-base 30000000 --episodes 20 --output outputs/stea_mappo_seed1_2m/holdout.json
```

正式 YAML 默认 24 环境，比较当前 16 环境的 MAPPO 运行时双方应统一为 16。STEA 使用 32 步连续 chunk，512 transitions 的 minibatch 包含 16 个 chunk；最后不足 32 步的 chunk 使用显式 padding mask。Smoke 保留正式 Actor 尺寸，Critic 宽度缩为 64，rollout=32、epochs=2、minibatch=64，评估两回合。STEA checkpoint 与 MAPPO 互不加载；resume 使用 `--resume <run>/latest.pt`，从零 recurrent state 的新回合继续。详细结构、状态生命周期和设计启发见 [STEA-MAPPO 说明](docs/stea_mappo.md)。

## MADSAC training

```bash
python algorithm/train_madsac.py --smoke --device cuda --num-envs 1 --output-dir outputs/madsac_smoke
python algorithm/train_madsac.py --device cuda --seed 1 --output-dir outputs/madsac_seed1
```

默认参数来自 `configs/madsac.yaml`。MADSAC 使用共享的 tanh Gaussian actor、独立的双 centralized attention Q critic、target networks 和 joint-transition replay；保留 delayed actor/target updates。单环境 smoke 默认 32 transitions，包含实际 critic 和 actor 更新。

## MADSAC evaluation

```bash
python algorithm/evaluate_madsac.py --checkpoint outputs/madsac_smoke/latest.pt --device cuda --seed-base 30000000 --episodes 2 --output outputs/madsac_smoke/holdout.json
```

默认使用 stochastic evaluation，可通过 `--mode deterministic` 切换。每个环境 seed 使用独立的 policy-noise RNG stream，评估不会污染训练 RNG。评估校验环境版本、维度、网络架构、配置指纹、实现版本与训练 seed。MADSAC checkpoint 不保存 replay buffer，不能用于 exact resume。

## Tests and validation

```bash
pytest -q
python -m compileall -q env algorithm tools tests
python tools/validate_combat_environment.py --output outputs/combat_environment_validation.json
python tools/check_parallel_env.py --num-envs 2 --steps 10
```

测试覆盖控制器、RK4、动作裁剪、52D observation、初始化、武器与奖励、越界和撞地、各终止结果、并行采样、两个 baseline 的 CUDA training/checkpoint/evaluation 闭环及记录器的语义中立性。

## Experiment outputs

新增独立 [v2.4 8v8 协议](docs/combat_environment_v24.md)，使用 `configs/combat_environment_v24.yaml` 配合 `configs/mappo_8v8.yaml` 或 `configs/stea_mappo_8v8.yaml`；默认环境与旧实验仍是 v2.3 4v4。正式参数的难度审计与 CUDA 小规模闭环由对应 audit/smoke 工具验证，未自动启动 2M 训练。

每次训练使用独立的空目录；入口拒绝覆盖已有非空输出。通常包含：

```text
outputs/<run>/
    env_config.yaml
    algorithm_config.yaml
    run_config.json
    training_metrics.jsonl
    optimization_metrics.jsonl
    evaluation_history.csv
    checkpoint_<sampled_steps>.pt
    best_eval.pt
    latest.pt
    run_summary.json
```

`sampled_steps` 统计 joint environment transitions。评估输出统一包含 `red_win_rate`、`blue_win_rate`、`draw_rate`、`timeout_rate`、`episode_return`、`episode_length`、双方损失与存活、fire attempts、weapon hits、attack kills、boundary exits、ground losses 与 R1–R4 totals。`average_return`、`win_rate` 等 baseline 常用指标同样保留。最优 checkpoint 按 Red win rate、return、较少 Red losses 依次排序。

```bash
python tools/aggregate_training_runs.py outputs/mappo_seed1 outputs/mappo_seed2 --output-dir outputs/training_aggregate
python tools/aggregate_holdout_results.py outputs/mappo_seed1/holdout.json outputs/mappo_seed2/holdout.json --output-dir outputs/holdout_aggregate
```

汇总要求一致协议及不同训练 seed，输出均值、标准差、SEM 和 95% Student-t interval。

## Episode recording and visualization

```bash
python tools/record_combat_episode.py --checkpoint outputs/mappo_smoke/latest.pt --episode-seed 40000000 --device cuda --output-dir outputs/combat_recording
python tools/render_combat_episode.py --trace outputs/combat_recording/episode_trace.npz --metadata outputs/combat_recording/metadata.json --output outputs/combat_recording/episode.png
python tools/render_combat_episode_interactive.py --trace outputs/combat_recording/episode_trace.npz --metadata outputs/combat_recording/metadata.json --output outputs/combat_recording/episode.html
```

记录器支持 MAPPO 与 MADSAC。输出真实射击、命中、击杀、损失事件与奖励数据；交互 HTML 内嵌 Plotly，支持播放、暂停、时间滑块、3D camera 和实体状态查看。静态渲染也支持 GIF 与 MP4；MP4 需要可用的 FFmpeg。详见 [记录工具说明](docs/combat_episode_visualization.md)。

## v3.47 observation ablation

v3.47 继承 v3.46 的全部机制，仅恢复原始 v3.38 的45维观测（动作3维）。新增 MAPPO/IPPO/MAAC 配置及版本、Checkpoint、驻留评估支持；v3.44/v3.46 保持49维。实现与有界验证命令见 [v3.47 文档](docs/combat_environment_v347.md)。不自动启动正式3M训练。

## v3.48 probability weapon ablation

v3.48 保持 v3.47 原始45维观测与 R1–R4/PBRS，复用 v3.42 的三维圆锥概率武器，双方各6次弹药、1秒冷却。新增独立 MAPPO/MAAC 配置、严格 Checkpoint 身份与真实概率攻击指标。实现、公式与有界 CUDA 验证见 [v3.48 文档](docs/combat_environment_v348.md)。不会自动启动3M训练。

## MAAC v3.48 failure audit and opt-in v3.49

新增多种子动作/价值/Reward专项审查工具，以及仅替换R4的独立v3.49候选。原45维观测、概率武器、R1–R3、网络和训练参数保持不变。候选默认不启用，未启动正式3M训练；短程对照不代表性能改善。公式、风险及复现步骤见 [v3.49文档](docs/combat_environment_v349.md)。

## v3.50 paper reward and task-success outcome

新增独立 v3.50，基于 v3.48 仅对齐 Li 等（2023）式(25)的 R1–R4 和“全部 Blue 被消灭”的任务成功条件。移除实际 PBRS 贡献，对同一最近存活 Blue 计算 R3/R4；双方全灭按任务条件判 Red 胜。45维观测、直接增量飞行动力学、概率武器及 MAPPO/MAAC 参数保持不变。这不是完整论文环境或 MAAC 性能复现，不启动3M训练；后续先研究飞行动力学方案。实现约定和验收方式见 [v3.50文档](docs/combat_environment_v350.md)。
