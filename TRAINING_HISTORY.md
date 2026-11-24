# UAV1 训练实验记录

更新时间：2026-11-24（中国标准时间）。此文件记录本地 `runs/` 中每个训练版本的实际结果；以后每次分析训练，继续在本文件追加一节。历史结果只读取，不修改。

## 指标口径

- **全歼率**：评估回合中五架进攻 UAV 全部被捕获的比例。**捕获率**：被捕获的进攻 UAV 数量占总进攻 UAV 数量的比例。两者不能混用。
- **表内峰值评估**：该目录已有周期评估中，按“全歼率、捕获率、团队回报”依次排序的最高一条；它不必等于文件名为 `Best` 的 checkpoint。**末次评估**：最后一条周期评估。表中的场景指评估模式，少数早期运行还评估了训练模式以外的场景。
- 表中百分比来自每次评估的有限回合样本；单次峰值有波动。跨环境版本、场景、奖励或捕获半径的结果不能直接作算法因果比较。`O/S` 是 Actor 观测维度/Critic 状态维度；`d/h` 是距离/航向奖励系数；`r` 是捕获半径（米）；`I` 是采样步更新间隔；`E` 是并行环境数。
- 此清单只覆盖当前本机有 `settings.json` 的训练目录；仅在服务器上而未迁回本地的实验，可在取得结果后补记。早期 `mixed` 目录仅作历史记录，现行实验不使用该模式。

## 训练版本与结果

| 接口 | 算法/评估场景 | 运行目录 | 关键设定 | 进度/更新 | 峰值全歼/捕获 | 末次全歼/捕获 | 末次团队回报 |
|---|---|---|---|---:|---:|---:|---:|
| V1 | MADDPG · 直冲 | [maddpg_direct_rush_2m_seed7](runs/maddpg_direct_rush_2m_seed7/summary.json) | O49/S245 · d0.05 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 30%/65% | 20%/64% | -66.6 |
| V1 | MADDPG · 分散 | [maddpg_direct_rush_2m_seed7](runs/maddpg_direct_rush_2m_seed7/summary.json) | O49/S245 · d0.05 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 5%/15% | 0%/7% | -98.5 |
| V1 | MADDPG · 直冲 | [maddpg_direct_rush_one_penalty_2m_seed7](runs/maddpg_direct_rush_one_penalty_2m_seed7/summary.json) | O49/S245 · d0.05 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/40% | 0%/29% | -69.6 |
| V1 | MADDPG · 直冲 | [maddpg_direct_rush_resolve_all_2m_seed7](runs/maddpg_direct_rush_resolve_all_2m_seed7/summary.json) | O49/S245 · d0.05 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 25%/69% | 0%/41% | -253.8 |
| V1 | MADDPG · mixed | [maddpg_interception_mixed_seed7](runs/maddpg_interception_mixed_seed7/settings.json) | O36/S180 · d0.05 · r5 · γ0.99 · I100 · E4 | 无完整训练结果 | — | — | — |
| V1 | MADDPG · 直冲 | [maddpg_interception_mixed_seed7_retry1](runs/maddpg_interception_mixed_seed7_retry1/summary.json) | O36/S180 · d0.05 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 5%/63% | 0%/17% | -93.8 |
| V1 | MADDPG · 分散 | [maddpg_interception_mixed_seed7_retry1](runs/maddpg_interception_mixed_seed7_retry1/summary.json) | O36/S180 · d0.05 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 25%/58% | 5%/15% | -88.7 |
| V1 | MAPPO · 直冲 | [mappo_direct_rush_2m_seed7](runs/mappo_direct_rush_2m_seed7/summary.json) | O49/S245 · d0.05 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 70%/92% | 5%/38% | -86.7 |
| V1 | MAPPO · 分散 | [mappo_direct_rush_2m_seed7](runs/mappo_direct_rush_2m_seed7/summary.json) | O49/S245 · d0.05 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 25%/51% | 0%/40% | -91.4 |
| V1 | MAPPO · 直冲 | [mappo_direct_rush_one_penalty_2m_seed7](runs/mappo_direct_rush_one_penalty_2m_seed7/summary.json) | O49/S245 · d0.05 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 10%/78% | 0%/56% | -41.3 |
| V1 | MAPPO · 直冲 | [mappo_direct_rush_resolve_all_2m_seed7](runs/mappo_direct_rush_resolve_all_2m_seed7/summary.json) | O49/S245 · d0.05 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 45%/81% | 15%/69% | -84.8 |
| V1 | MAPPO · mixed | [mappo_interception_mixed_seed7](runs/mappo_interception_mixed_seed7/settings.json) | O36/S180 · d0.05 · r5 · γ0.99 · E4 | 无完整训练结果 | — | — | — |
| V1 | MAPPO · 直冲 | [mappo_interception_mixed_seed7_retry1](runs/mappo_interception_mixed_seed7_retry1/summary.json) | O36/S180 · d0.05 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 40%/71% | 5%/40% | -81.8 |
| V1 | MAPPO · 分散 | [mappo_interception_mixed_seed7_retry1](runs/mappo_interception_mixed_seed7_retry1/summary.json) | O36/S180 · d0.05 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 0%/34% | 0%/9% | -92.9 |
| V2 | MADDPG · 直冲 | [maddpg_direct_rush_distance_matching_approach0p5_seed7_2m](runs/maddpg_direct_rush_distance_matching_approach0p5_seed7_2m/summary.json) | O54/S75 · 距离匹配 · d0.5 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/48% | 0%/30% | -47.0 |
| V2 | MADDPG · 直冲 | [maddpg_direct_rush_distance_matching_seed7_2m](runs/maddpg_direct_rush_distance_matching_seed7_2m/summary.json) | O54/S75 · 距离匹配 · d0.05 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/36% | 0%/30% | -68.1 |
| V2 | MADDPG · 直冲 | [maddpg_direct_rush_equal_dynamics_launch2_seed7](runs/maddpg_direct_rush_equal_dynamics_launch2_seed7/summary.json) | O54/S75 · 距离匹配 · d0.5 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/50% | 0%/20% | -69.8 |
| V2 | MADDPG · 直冲 | [maddpg_huang_global_direct_seed7](runs/maddpg_huang_global_direct_seed7/summary.json) | O54/S75 · Huang 全局 · d0.5 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/44% | 0%/22% | -47.4 |
| V2 | MAPPO · 直冲 | [mappo_direct_rush_distance_matching_approach0p5_seed7_2m](runs/mappo_direct_rush_distance_matching_approach0p5_seed7_2m/summary.json) | O54/S75 · 距离匹配 · d0.5 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 0%/46% | 0%/46% | -35.6 |
| V2 | MAPPO · 直冲 | [mappo_direct_rush_distance_matching_seed7_2m](runs/mappo_direct_rush_distance_matching_seed7_2m/summary.json) | O54/S75 · 距离匹配 · d0.05 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 0%/54% | 0%/28% | -70.9 |
| V2 | MAPPO · 直冲 | [mappo_direct_rush_equal_dynamics_launch2_seed7](runs/mappo_direct_rush_equal_dynamics_launch2_seed7/summary.json) | O54/S75 · 距离匹配 · d0.5 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 10%/60% | 0%/40% | -47.9 |
| V3 | MADDPG · 直冲 | [maddpg_huang_target_velocity_direct_per_seed7_2m](runs/maddpg_huang_target_velocity_direct_per_seed7_2m/summary.json) | O36/S75 · Huang 全局 · d0.5 · r5 · γ0.99 · I100 · E4 · PER+均匀 | 2000k/2000k · 19990次 | 0%/46% | 0%/30% | -35.5 |
| V3 | MADDPG · 直冲 | [maddpg_huang_target_velocity_direct_seed7](runs/maddpg_huang_target_velocity_direct_seed7/summary.json) | O36/S75 · Huang 全局 · d0.5 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/50% | 0%/34% | -32.4 |
| V3 | MADDPG · 直冲 | [maddpg_huang_target_velocity_direct_uniform_per_compare_seed7_2m](runs/maddpg_huang_target_velocity_direct_uniform_per_compare_seed7_2m/summary.json) | O36/S75 · Huang 全局 · d0.5 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/42% | 0%/34% | -36.7 |
| V3 | MAPPO · 直冲 | [mappo_huang_target_velocity_direct_seed7](runs/mappo_huang_target_velocity_direct_seed7/summary.json) | O36/S75 · Huang 全局 · d0.5 · r5 · γ0.99 · E4 | 2000k/2000k · 3907次 | 0%/52% | 0%/40% | -22.4 |
| V4 | MADDPG · 直冲 | [maddpg_huang_heading_approach005_direct_seed7_2m](runs/maddpg_huang_heading_approach005_direct_seed7_2m/summary.json) | O36/S75 · Huang 全局 · d0.05 · h0.03 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/46% | 0%/2% | -107.0 |
| V4 | MADDPG · 直冲 | [maddpg_per_heading_approach005_direct_seed7_2m](runs/maddpg_per_heading_approach005_direct_seed7_2m/summary.json) | O36/S75 · Huang 全局 · d0.05 · h0.03 · r5 · γ0.99 · I100 · E4 · PER+均匀 | 2000k/2000k · 19990次 | 0%/32% | 0%/8% | -89.8 |
| V4 | MADDPG · 直冲 | [maddpg_uniform_heading_approach005_direct_seed7_2m](runs/maddpg_uniform_heading_approach005_direct_seed7_2m/summary.json) | O36/S75 · Huang 全局 · d0.05 · h0.03 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/46% | 0%/2% | -107.0 |
| V5 | MADDPG · 直冲 | [maddpg_ideal_global_initial_direct_seed7_2m](runs/maddpg_ideal_global_initial_direct_seed7_2m/summary.json) | O36/S75 · 理想全局 · d0.05 · h0.03 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 10%/42% | 0%/32% | -61.0 |
| V6 | MADDPG · Jia 直线 | [maddpg_ideal_jia_straight_seed7_2m](runs/maddpg_ideal_jia_straight_seed7_2m/summary.json) | O36/S75 · 理想全局 · d0.05 · h0.03 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/46% | 0%/40% | -52.1 |
| V7 | MADDPG · Jia 直线 | [maddpg_ideal_jia_oob_exit_seed7_2m](runs/maddpg_ideal_jia_oob_exit_seed7_2m/summary.json) | O36/S75 · 理想全局 · d0.05 · h0.03 · r5 · γ0.99 · I100 · E4 | 2000k/2000k · 19990次 | 0%/34% | 0%/8% | -87.6 |
| V7 | MADDPG · Jia 直线 | [maddpg_ideal_jia_oob_exit_seed7_2mclear](runs/maddpg_ideal_jia_oob_exit_seed7_2mclear/settings.json) | O36/S75 · 理想全局 · d0.05 · h0.03 · r5 · γ0.99 · I100 · E4 | 无完整训练结果 | — | — | — |
| V8 | MADDPG · Jia 直线 | [maddpg_ideal_jia_circle_heading0p3_gamma099_seed7_300k](runs/maddpg_ideal_jia_circle_heading0p3_gamma099_seed7_300k/summary.json) | O24/S75 · 理想全局 · d0.2 · h0.3 · r5 · γ0.99 · I10 · E1 | 300k/300k · 29898次 | 0%/50% | 0%/46% | +23.7 |
| V8 | MADDPG · Jia 直线 | [maddpg_ideal_jia_circle_targets_gamma099_seed7_300k](runs/maddpg_ideal_jia_circle_targets_gamma099_seed7_300k/summary.json) | O24/S75 · 理想全局 · d0.2 · h0.03 · r5 · γ0.99 · I10 · E1 | 300k/300k · 29898次 | 0%/54% | 0%/28% | -54.6 |
| V8 | MADDPG · Jia 直线 | [maddpg_ideal_jia_obs24_approach02_interval10_critic_lr3e4_single_seed7_300k](runs/maddpg_ideal_jia_obs24_approach02_interval10_critic_lr3e4_single_seed7_300k/summary.json) | O24/S75 · 理想全局 · d0.2 · h0.03 · r5 · γ0.99 · I10 · E1 · Critic LR 0.0003 | 300k/300k · 29898次 | 10%/44% | 0%/40% | -38.8 |
| V8 | MADDPG · Jia 直线 | [maddpg_ideal_jia_obs24_approach02_interval10_single_seed7_300k](runs/maddpg_ideal_jia_obs24_approach02_interval10_single_seed7_300k/summary.json) | O24/S75 · 理想全局 · d0.2 · h0.03 · r5 · γ0.99 · I10 · E1 | 300k/300k · 29898次 | 0%/58% | 0%/40% | -32.9 |
| V8 | MADDPG · Jia 直线 | [maddpg_ideal_jia_obs24_capture3m_gamma0995_single_seed7_300k](runs/maddpg_ideal_jia_obs24_capture3m_gamma0995_single_seed7_300k/summary.json) | O24/S75 · 理想全局 · d0.2 · h0.03 · r3 · γ0.995 · I10 · E1 | 300k/300k · 29898次 | 0%/44% | 0%/14% | -69.3 |
| V8 | MADDPG · Jia 直线 | [maddpg_ideal_jia_obs24_gamma0995_single_seed7_300k](runs/maddpg_ideal_jia_obs24_gamma0995_single_seed7_300k/summary.json) | O24/S75 · 理想全局 · d0.2 · h0.03 · r5 · γ0.995 · I10 · E1 | 300k/300k · 29898次 | 0%/48% | 0%/36% | -42.8 |
| V10 | MADDPG · 1v1 Jia 直线 | [maddpg_one_vs_one_jia_seed7_300k](runs/maddpg_one_vs_one_jia_seed7_300k/settings.json) | O24/S75 · 1 对有效 UAV · d0.2 · h0.03 · r5 · γ0.99 · I10 · E1 | 本机主动停止，进度日志至 80k/300k；无完整汇总 | — | — | — |
| V10 | MADDPG · 1v1 Jia 直线 | [maddpg_one_vs_one_jia_seed7_300k_run2](runs/maddpg_one_vs_one_jia_seed7_300k_run2/summary.json) | O24/S75 · 1 对有效 UAV · d0.2 · h0.03 · r5 · γ0.99 · I10 · E1 | 300k/300k · 29898次 | 70%/70% | 0%/0% | -78.3 |
| V10 | MADDPG · 1v1 Jia 直线 | [maddpg_one_vs_one_jia_no_heading_seed7_300k](runs/maddpg_one_vs_one_jia_no_heading_seed7_300k/summary.json) | O24/S75 · 1 对有效 UAV · d0.2 · h0 · r5 · γ0.99 · I10 · E1 | 300k/300k · 29898次 | 100%/100% | 100%/100% | +53.7 |
| V10 | MADDPG · 1v1 Jia 直线 | [maddpg_1v1_velocity_relative_seed7_500k](runs/maddpg_1v1_velocity_relative_seed7_500k/summary.json) | O24/S75 · 目标相对速度 · d0.2 · h0 · r5 · γ0.99 · I10 · E1 | 500k/500k · 49898次 | 100%/100% | 100%/100% | +53.9 |
| V10 | MADDPG · 1v1 Jia 直线 | [maddpg_1v1_velocity_absolute_seed7_500k](runs/maddpg_1v1_velocity_absolute_seed7_500k/summary.json) | O24/S75 · 目标绝对速度 · d0.2 · h0 · r5 · γ0.99 · I10 · E1 | 500k/500k · 49898次 | 100%/100% | 100%/100% | +53.9 |
| V10 | MADDPG · 1v1 Jia 直线 | [maddpg_1v1_velocity_none_seed7_500k](runs/maddpg_1v1_velocity_none_seed7_500k/summary.json) | O24/S75 · 目标速度两维置零 · d0.2 · h0 · r5 · γ0.99 · I10 · E1 | 500k/500k · 49898次 | 37%/37% | 0%/0% | -78.2 |

### 未完成目录说明

没有 `summary.json` 的目录只表示当前本地尚无完整训练汇总；可能是中断、尚在运行或仅保留配置。表中的“进行中”是生成本文件时的快照，不代表后续完成状态。

## 已分析训练记录

### 2026-11-24 · 理想 Jia 直线场景，24 维 Actor，距离奖励 0.2，单环境更新间隔 10

- [运行结果](runs/maddpg_ideal_jia_obs24_approach02_interval10_single_seed7_300k/summary.json)：300,000 步、920 局、29,898 次更新；`γ=0.99`、捕获半径 5 m、航向系数 0.03。
- 周期评估峰值在 165,000 步：捕获率 58%、全歼率 0%、团队回报约 −12.5。末次 300,000 步：捕获率 40%、全歼率 0%、团队回报约 −32.9。
- 低层捕获能力较先前 2M 步的越界退场版本明显提高，但 165k 后出现退化。Critic 梯度和 TD 误差偏高是稳定性线索，不能单凭这些指标断言价值函数发散。

### 2026-11-24 · 理想 Jia 直线场景，γ=0.995，捕获半径 5 m

- [运行结果](runs/maddpg_ideal_jia_obs24_gamma0995_single_seed7_300k/summary.json)：300,000 步、918 局、29,898 次更新；Actor 24 维，`d=0.2`、`h=0.03`。
- 表内峰值捕获率 48%，末次为 36%；峰值和末次全歼率均为 0%。末次团队回报约 −42.8。
- 相比上条 `γ=0.99` 运行，这组的峰值与末次捕获率都较低；它是单次 seed 和有限评估样本的观察，尚不足以证明折扣系数是唯一原因。

### 2026-11-24 · 理想 Jia 直线场景，γ=0.995，捕获半径 3 m

- [运行结果](runs/maddpg_ideal_jia_obs24_capture3m_gamma0995_single_seed7_300k/summary.json)：300,000 步、914 局、29,898 次更新；其余主要设定与上条相同。
- 表内峰值捕获率 44%，末次降到 14%；峰值和末次全歼率均为 0%。末次团队回报约 −69.3。
- 3 m 捕获判定更严格，观察到的捕获率和后期稳定性弱于 5 m 版本；不宜把不同捕获半径下的绝对捕获率直接当成相同任务成绩。

### 2026-11-24 · 理想 Jia 直线场景，越界退场的旧基线

- [运行结果](runs/maddpg_ideal_jia_oob_exit_seed7_2m/summary.json)：2,000,000 步、6,117 局、19,990 次更新；旧版奖励和观测合同，不能与上面的 24 维版本作单因素比较。
- 表内峰值（1,000,000 步）捕获率 34%，末次为 8%；两次全歼率均为 0%。末次团队回报约 −87.6。
- 该记录主要用于追踪版本演化：后续 24 维、距离奖励 0.2 和更新间隔 10 的联合改动在更少步数内表现出更高的部分捕获率。

### 2026-11-24 · 圆周目标航迹，航向系数 0.03

- [配置](runs/maddpg_ideal_jia_circle_targets_gamma099_seed7_300k/settings.json)、[汇总](runs/maddpg_ideal_jia_circle_targets_gamma099_seed7_300k/summary.json)、[周期评估](runs/maddpg_ideal_jia_circle_targets_gamma099_seed7_300k/evaluations.jsonl)、[checkpoint 索引](runs/maddpg_ideal_jia_circle_targets_gamma099_seed7_300k/checkpoints/index.json)：300,000 步、910 局、29,898 次更新，单环境；Actor 24 维、Critic 状态 75 维、`γ=0.99`、`d=0.2`、`h=0.03`、捕获半径 5 m。
- `Best` 位于 150k：10 局评估的捕获率 54%、全歼率 0%、团队回报 −22.11。`Last` 位于 300k：捕获率 28%、全歼率 0%、团队回报 −54.60；说明峰值后没有保持住捕获表现。
- 20 次评估共 200 局，全歼 0 局；平均捕获率 37.4%、团队回报 −42.43。训练的 910 局中有 1 局全歼，末尾 100 局捕获率 45.4%，与末次固定评估 28% 不一致，提示策略结果对评估样本或训练噪声敏感。
- 末次评估每局平均奖励：捕获 +28.00、突防 −100.00、接近 +15.57、航向 +2.23、越界 −0.40。最后 1000 次更新的平均 TD 绝对误差约 0.720、Critic 梯度范数约 24.53，其中约 86% 的合并梯度范数超过配置裁剪阈值 10；这提示训练压力，但不能单独证明 Q 发散。

### 2026-11-24 · 圆周目标航迹，航向系数 0.3

- [配置](runs/maddpg_ideal_jia_circle_heading0p3_gamma099_seed7_300k/settings.json)、[汇总](runs/maddpg_ideal_jia_circle_heading0p3_gamma099_seed7_300k/summary.json)、[周期评估](runs/maddpg_ideal_jia_circle_heading0p3_gamma099_seed7_300k/evaluations.jsonl)、[checkpoint 索引](runs/maddpg_ideal_jia_circle_heading0p3_gamma099_seed7_300k/checkpoints/index.json)：300,000 步、913 局、29,898 次更新。记录中的配置与上一组逐项核对，除航向奖励系数 `0.03 → 0.3` 外相同；两组的环境实现哈希也一致。
- `Best` 位于 165k：10 局评估捕获率 50%、全歼率 0%、团队回报 +14.40。`Last` 位于 300k：捕获率 46%、全歼率 0%、团队回报 +23.74。20 次评估共 200 局，全歼仍为 0；平均捕获率 41.7%，比低航向系数高 4.3 个百分点，但最佳捕获率反而低 4 个百分点。
- 末次评估每局平均奖励：捕获 +46.00、突防 −100.00、接近 +26.45、航向 +51.32、越界 −0.04。**10 局均突防，平均团队回报仍为正**；回报上升主要来自航向项，不能解读为防御任务已成功。两组 20 次评估的平均回报差约 +43.22，其中航向项差约 +34.56。
- 训练 913 局中有 1 局全歼。最后 1000 次更新的平均 TD 绝对误差约 0.759、Critic 梯度范数约 27.40，约 94% 的合并梯度范数超过阈值 10；高于低航向系数组的 0.720/24.53/86%。Q 均值也更高，可能与奖励尺度增加有关，不能仅据此判定价值估计正确或错误。
- 这组在同一随机种子的多数检查点上部分捕获和越界表现略好，但两组都未在周期评估实现全歼。仅凭这一次 seed、每点评估 10 局，不能确定航向系数提高后的泛化收益。

### 2026-11-24 · 1v1 Jia 场景，航向系数 0.03

- [配置](runs/maddpg_one_vs_one_jia_seed7_300k_run2/settings.json)、[汇总](runs/maddpg_one_vs_one_jia_seed7_300k_run2/summary.json)、[周期评估](runs/maddpg_one_vs_one_jia_seed7_300k_run2/evaluations.jsonl)、[更新日志](runs/maddpg_one_vs_one_jia_seed7_300k_run2/updates.jsonl)：300,000 步、1,134 局、29,898 次更新，seed 7、单环境。只有一对 UAV 有效，其余槽位掩码；1v1 的捕获率即全歼率。
- Best 在 240k，10 局评估拦截 7 局（70%）、团队回报 +21.70；255k 仍为 70%。但 285k 和 Last 300k 均为 0/10，末次团队回报 −78.26（捕获 0、突防 −100、接近 +23.41、航向 −0.28、越界 −1.40）。训练末尾 100 局成功率只有 10%，低于开头 100 局的 24%。
- 全部 20 次周期评估累计 66/200 局成功，均值 33%；训练回合累计 328/1,134 局成功。末尾 1000 次更新的 Critic loss 约 2.55、TD 绝对误差 0.366、裁剪前 Critic 梯度范数 54.85（约 92.5% 超过阈值 10）；Actor 梯度范数约 0.51（约 44.2% 超过阈值 0.5）。这些指标与后期策略退化同时出现，但不足以单独证明 Critic 发散。

### 2026-11-24 · 1v1 Jia 场景，航向奖励关闭

- [配置](runs/maddpg_one_vs_one_jia_no_heading_seed7_300k/settings.json)、[汇总](runs/maddpg_one_vs_one_jia_no_heading_seed7_300k/summary.json)、[周期评估](runs/maddpg_one_vs_one_jia_no_heading_seed7_300k/evaluations.jsonl)、[更新日志](runs/maddpg_one_vs_one_jia_no_heading_seed7_300k/updates.jsonl)：300,000 步、1,175 局、29,898 次更新。与上一组的环境配置仅有 `heading_weight: 0.03 → 0` 差异；保存的实现哈希和奖励描述也因相应代码说明更新而不同，物理规则、网络与训练超参数未变。
- Best 在 270k，10/10 局成功；Last 300k 再次 10/10，团队回报 +53.70（捕获 +20、突防 0、接近 +33.70、越界 0）。但 165k、180k、210k、225k、240k 都是 0/10，285k 为 5/10；不能把末次 100% 视作已稳定收敛。
- 全部周期评估累计 68/200 局成功，均值 34%，与有航向组的 33% 接近；后五次评估均值为 64%，有航向组为 34%。训练末尾 100 局成功率 66%，高于开头 100 局的 20%；训练回合累计 425/1,175 局成功。
- 末尾 1000 次更新的 Critic loss 约 0.84、TD 绝对误差 0.232、裁剪前 Critic 梯度范数 29.59（约 83.9% 超过阈值 10）；Actor 梯度范数约 0.32（约 12.9% 超过阈值 0.5）。相较有航向组，这些末期指标与成功率更好，但两组都存在较高梯度裁剪比例。
- **步数判断**：30 万步已足以出现 1v1 全部拦截，问题不能简单归因于训练步数不够。两组周期评估波动都很大，继续训练可能改善也可能破坏策略。每个检查点只用固定 10 个种子评估；确认泛化前需用独立种子评估 Best/Last，并保留检查点间的波动信息。

### 2026-11-24 · 1v1 目标速度观测消融：相对速度，500k

- [配置](runs/maddpg_1v1_velocity_relative_seed7_500k/settings.json)、[汇总](runs/maddpg_1v1_velocity_relative_seed7_500k/summary.json)、[评估](runs/maddpg_1v1_velocity_relative_seed7_500k/evaluations.jsonl)、[更新](runs/maddpg_1v1_velocity_relative_seed7_500k/updates.jsonl)：500,000 步、2,140 局、49,898 次 Critic/Actor 更新；`γ=0.99`、Actor/Critic LR `0.0003/0.001`、batch 128、warmup 1,024、每 10 步更新一次、replay 容量 200k、单环境、seed 7。三组仅指定目标的速度编码不同，环境实现哈希相同。
- Best 在 300k 首次达到固定 30 局全部成功；350k 又降到 0/30，400k 后恢复，最后五次评估成功率为 `100%, 96.7%, 100%, 100%, 100%`，均值 99.3%。Last 500k 为 30/30、团队回报 +53.87，成功回合平均捕获时间 17.50 s。20 次评估累计 311/600 局成功；训练最后 100 局成功 97 局。
- 最后 1,000 次更新均值：Actor loss −4.51、裁剪前 Actor 梯度范数 0.164（0.8% 超过阈值 0.5）；Critic loss 0.688、TD 绝对误差 0.183、Critic 梯度范数 24.51（71.3% 超过阈值 10）；Q 均值 4.45、target Q 均值 4.47。早期 1,000 次更新的 TD 绝对误差约 0.244，末期较低，但仍有较多 Critic 梯度被裁剪。

### 2026-11-24 · 1v1 目标速度观测消融：绝对速度，500k

- [配置](runs/maddpg_1v1_velocity_absolute_seed7_500k/settings.json)、[汇总](runs/maddpg_1v1_velocity_absolute_seed7_500k/summary.json)、[评估](runs/maddpg_1v1_velocity_absolute_seed7_500k/evaluations.jsonl)、[更新](runs/maddpg_1v1_velocity_absolute_seed7_500k/updates.jsonl)：500,000 步、1,918 局、49,898 次 Critic/Actor 更新；训练参数与相对速度组一致。
- Best 在 475k 首次达到固定 30 局全部成功，Last 500k 仍为 30/30、团队回报 +53.85；但 450k 为 0/30，后五次评估均值仅 60.0%，标准差约 38.9 个百分点。20 次评估累计 220/600 局成功；训练最后 100 局成功 90 局。Last 成功回合平均捕获时间 19.05 s，比相对速度组的 17.50 s 慢约 1.55 s。
- 最后 1,000 次更新均值：Actor loss +6.11、Actor 梯度范数 0.272（6.9% 超过阈值）；Critic loss 2.902、TD 绝对误差 0.250、Critic 梯度范数 24.84（65.0% 超过阈值）；Q/target Q 均值约 −6.19/−6.16。Actor loss 与 Q 均值的正负受价值基线和采样分布影响，不能直接与另一组比较为策略优劣。

### 2026-11-24 · 1v1 目标速度观测消融：无目标速度，500k

- [配置](runs/maddpg_1v1_velocity_none_seed7_500k/settings.json)、[汇总](runs/maddpg_1v1_velocity_none_seed7_500k/summary.json)、[评估](runs/maddpg_1v1_velocity_none_seed7_500k/evaluations.jsonl)、[更新](runs/maddpg_1v1_velocity_none_seed7_500k/updates.jsonl)：500,000 步、1,803 局、49,898 次 Critic/Actor 更新。Actor 的指定目标速度两维恒为零，保留自身速度；Critic 仍看完整全局速度状态。
- Best 在 75k 达到 11/30（36.7%），300k 也有 11/30；其余多数检查点为 0/30，最后五次全部 0/30。Last 的捕获奖励 0、突防 −100、接近 +21.84、团队回报 −78.16。20 次评估累计 32/600 局成功；训练最后 100 局却成功 54 局，表明带探索噪声的采样表现和确定性评估严重背离，不能把训练窗口的 54% 当作部署成功率。
- 最后 1,000 次更新均值：Actor loss +8.43、Actor 梯度范数 0.205（3.0% 超过阈值）；Critic loss 2.800、TD 绝对误差 0.261、Critic 梯度范数 32.28（75.7% 超过阈值）；Q/target Q 均值约 −8.47/−8.45。Critic 能拟合当前采样分布，不意味着 Actor 已学会稳定拦截。
- 这一次 seed 的结果支持“目标速度输入对当前无历史帧的加速度 Actor 有帮助”，但不能证明没有目标速度在所有设置中必定失败。三组均使用相同 30 个固定周期评估种子；末次 30/30 仍需要独立种子测试。相对与绝对速度在保留自身速度时信息等价，差别主要是表示形式及优化路径；当前单 seed 的相对速度组学习更早、末期更稳定。

### 2026-11-24 · 1v1 相对速度，距离系数 0.5，600k 从头训练

- [配置](runs/maddpg_1v1_velocity_relative_approach05_seed7_600k/settings.json)、[汇总](runs/maddpg_1v1_velocity_relative_approach05_seed7_600k/summary.json)、[周期评估](runs/maddpg_1v1_velocity_relative_approach05_seed7_600k/evaluations.jsonl)、[更新](runs/maddpg_1v1_velocity_relative_approach05_seed7_600k/updates.jsonl)：600,000 步、2,754 局、59,898 次更新；单环境、seed 7、`γ=0.99`、Actor/Critic LR 为 `0.0003/0.001`、batch 128、更新间隔 10、uniform replay 200k；1v1 Jia 直线、相对目标速度、航向奖励 0，距离系数 0.5。
- 首次 30/30 是 180k，亦是 Best；但 210k、240k、270k 连续三次 0/30，说明出现严重中期策略退化。300k 恢复 30/30；300k 以后 10 次评估平均成功率 95.3%，最低 540k 的 24/30。Last 600k 为 30/30、团队回报 +104.61、捕获时间均值 18.12 s。20 次固定种子周期评估累计 429/600 成功；训练最后 100 局 97/100，最后 500 局 468/500。
- Last 每局平均奖励：捕获 +20、距离 +84.61、突防与越界均为 0；距离占这两项正回报约 80.9%。在 210k～270k 的 0/30 评估期间，距离项仍有约 +64～+69，表明缩距本身不足以保证最终捕获。训练末尾 1,000 次更新：Actor loss −28.71、Actor 梯度范数 0.254（7.1% 超过 0.5）；Critic loss 1.283、TD 绝对误差 0.238、Critic 梯度范数 35.03（83.6% 超过 10），Q/target Q 均值约 28.64/28.67。Critic 更新压力较高，单凭这些指标不能证明 Q 发散。

### 2026-11-24 · 1v1 相对速度，距离系数 0.2，700k 从头训练

- [配置](runs/maddpg_1v1_velocity_relative_seed7_700k_from_scratch/settings.json)、[汇总](runs/maddpg_1v1_velocity_relative_seed7_700k_from_scratch/summary.json)、[周期评估](runs/maddpg_1v1_velocity_relative_seed7_700k_from_scratch/evaluations.jsonl)、[更新](runs/maddpg_1v1_velocity_relative_seed7_700k_from_scratch/updates.jsonl)：700,000 步、3,220 局、69,898 次更新；核心学习参数及场景与上条相同，距离系数为 0.2。两组还存在训练预算 600k/700k 与评估间隔 30k/25k 的差异；对比时应以共同训练步数和捕获指标为主。
- 首次 30/30 在 300k，亦是 Best；350k 再次跌至 0/30，400k 恢复，之后至 700k 连续 13 次评估均达到 29/30 或 30/30，其中 450k～700k 的 11 次均为 30/30。Last 700k 为 30/30、团队回报 +53.85、捕获时间均值 17.59 s。28 次固定种子周期评估累计 551/840 成功；训练最后 100 局 96/100，最后 500 局 481/500。
- Last 每局平均奖励：捕获 +20、距离 +33.85、突防与越界均为 0；距离占这两项正回报约 62.9%。训练末尾 1,000 次更新：Actor loss −16.95、Actor 梯度范数 0.210（3.6% 超过 0.5）；Critic loss 0.467、TD 绝对误差 0.151、Critic 梯度范数 20.39（54.6% 超过 10），Q/target Q 均值约 16.91/16.92。两组回报与 Q 尺度不同，不能直接依据 Actor loss 或 Q 大小判断策略优劣；这一次 seed 的 0.2 组在后期固定评估上更平稳，0.5 组更早出现 30/30，但中期退化和末期波动均存在。

## 后续追加格式

每次用户要求分析一次训练，就在“已分析训练记录”末尾新增日期与运行目录的小节，记录：实际配置与完成状态、评估峰值和末次的全歼/捕获/团队回报、主要奖励分量、Actor/Critic 学习指标、训练趋势、可比性限制、`settings.json`/`summary.json`/`evaluations.jsonl` 证据链接。未完成训练明确标注快照，不写成最终结论。

