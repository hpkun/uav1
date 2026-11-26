# UAV1 训练实验记录

更新时间：2026-11-26（中国标准时间）。此文件记录本地 `runs/` 中每个训练版本的实际结果；以后每次分析训练，继续在本文件追加一节。历史结果只读取，不修改。

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

### 2026-11-25 · 5v5 理想全局分配，距离 0.2、无航向、捕获半径 5 m，2M

- [配置](runs/maddpg_5v5_ideal_jia_d02_h0_seed7_2m/settings.json)、[汇总](runs/maddpg_5v5_ideal_jia_d02_h0_seed7_2m/summary.json)、[周期评估](runs/maddpg_5v5_ideal_jia_d02_h0_seed7_2m/evaluations.jsonl)、[更新](runs/maddpg_5v5_ideal_jia_d02_h0_seed7_2m/updates.jsonl)：2,000,000 步、6,081 局、199,898 次更新；单环境、seed 7、5v5、Actor 24 维、Critic 状态 75 维、理想全局一对一分配、Jia 直线来袭；`γ=0.99`、Actor/Critic LR `0.0003/0.001`、batch 128、每 10 步更新、uniform replay 200k。
- 20 次固定 30 种子的周期评估均为 0/30 全歼。最高个体捕获率 52.7% 出现在 1.4M，Best 因此保存于该步；当时团队回报 −21.68，平均捕获奖励 +52.67、突防 −100、距离 +25.72、越界 −0.07。Last 2M 个体捕获率 33.3%、团队回报 −46.45，奖励分量为捕获 +33.33、突防 −100、距离 +20.72、越界 −0.51。全程 600 个固定周期评估回合中没有全歼；训练 6,081 局仅 9 局全歼，最后 100 局 1 局全歼、个体捕获率 42.2%。1.4M 峰值后没有稳定保持，1.9M 降至 22.0%。
- 最后 1,000 次更新均值：Actor loss −5.97、Actor 裁剪前梯度范数 0.299；Critic loss 9.207、TD 绝对误差 0.763、旧版五 Critic 联合裁剪前梯度范数 23.63、Q/target Q 均值 5.75/5.76。联合范数超 10 的更新约 91.5%，但这不能与新版每 Critic 独立裁剪比例直接比较。TD 误差比最初 1,000 次更新的约 0.347 高，且策略峰值后退化；这提示训练稳定性问题，不足以单独认定 Q 发散。

### 2026-11-25 · 5v5 理想全局分配，距离 0.2、无航向、捕获半径 3 m，2M

- [配置](runs/maddpg_5v5_ideal_jia_capture3m_d02_h0_seed7_2m/settings.json)、[汇总](runs/maddpg_5v5_ideal_jia_capture3m_d02_h0_seed7_2m/summary.json)、[周期评估](runs/maddpg_5v5_ideal_jia_capture3m_d02_h0_seed7_2m/evaluations.jsonl)、[更新](runs/maddpg_5v5_ideal_jia_capture3m_d02_h0_seed7_2m/updates.jsonl)：2,000,000 步、6,046 局、199,898 次更新；记录的环境与算法参数除捕获半径 3 m 外与上条相同。训练代码版本不同：本组使用按有效 Critic 分别裁剪及新统计口径，上组使用旧版联合裁剪，因此不能把性能差异全部归因于半径。
- 20 次周期评估均为 0/30 全歼。最高个体捕获率 42.0% 同样在 1.4M，Best 保存于该步；当时团队回报 −33.18，平均捕获 +42、突防 −100、距离 +24.97、越界 −0.15。Last 2M 个体捕获率 26.0%、团队回报 −54.04，奖励分量为捕获 +26、突防 −100、距离 +20.16、越界 −0.20。全程 600 个固定周期评估回合及 6,046 个训练回合均无全歼；最后 100 训练局个体捕获率 20.4%。1.4M 后也未保持峰值。
- 最后 1,000 次更新均值：Actor loss +0.088、Actor 裁剪前梯度范数 0.245、约 23.4% 更新发生至少一次 Actor 裁剪；Critic loss 8.133、TD 绝对误差 0.718、有效 Critic 平均裁剪前梯度范数 9.332、约 58.4% 更新发生至少一次 Critic 裁剪，平均有效 Critic 数量 5；Q/target Q 均值 −0.263/−0.254。TD 误差高于最初 1,000 次更新的约 0.364；Q 与 target Q 接近不证明估值准确。与 5 m 组相比，同一 1.4M 评估点少捕获约 0.53 架/局、末次少约 0.37 架/局，距离奖励在两组末次相近；更严格的几何判定与训练代码差异共同影响比较。

### 2026-11-25 · 中心型径向 5v5，局部奖励求和，2M

- [配置](runs/maddpg_5v5_radial_capture5m_local_sum_seed7_2m/settings.json)、[汇总](runs/maddpg_5v5_radial_capture5m_local_sum_seed7_2m/summary.json)、[周期评估](runs/maddpg_5v5_radial_capture5m_local_sum_seed7_2m/evaluations.jsonl)、[更新](runs/maddpg_5v5_radial_capture5m_local_sum_seed7_2m/updates.jsonl)：2M 步、6,729 局、199,898 次更新；单环境、seed 7、5 个有效 Critic、24/75 维、捕获半径 5 m、距离系数 0.2、航向 0、每目标突防 −50；团队 Return 为全局事件奖励加五架局部奖励之和。
- Best 在 300k：固定 30 局全歼 2/30（6.7%）、个体捕获率 66.0%、团队回报 +62.66（捕获 +66.00、突防 −85.00、距离 +81.66）。Last 2M：全歼 0/30、捕获率 26.7%、回报 −71.74（+26.67、−183.33、距离 +84.93）。20 次周期评估仅 2/600 局全歼；训练末尾 100 局全歼 0、捕获率 17.6%、距离 +69.01、回报 −119.39。300k 峰值没有保持，后期存在大幅波动。
- 末尾 1,000 次更新均值：Actor loss +44.18、裁剪前梯度范数 0.100、裁剪比例 3.1%；Critic loss 3.841、TD 绝对误差 0.491、梯度范数 11.69、裁剪比例 77.7%；Q 均值 −44.33、标准差 63.47。首 1,000 次 TD 误差 1.283、Critic loss 33.02；数值下降并未对应稳定捕获。团队 Return 的新统计口径不改变 MADDPG 每个 Critic 的逐智能体训练奖励。

### 2026-11-25 · 中心型径向 1v1，2M

- [配置](runs/maddpg_radial_1v1_capture5m_seed7_2m/settings.json)、[汇总](runs/maddpg_radial_1v1_capture5m_seed7_2m/summary.json)、[周期评估](runs/maddpg_radial_1v1_capture5m_seed7_2m/evaluations.jsonl)、[更新](runs/maddpg_radial_1v1_capture5m_seed7_2m/updates.jsonl)：2M 步、10,879 局、199,898 次更新；一个有效 Actor/Critic，其余四槽掩码；物理、奖励系数和 MADDPG 超参数与中心型 5v5 相同，每目标捕获 +20、突防 −50。
- Best 在 500k：30/30 全歼、回报 +43.39（捕获 +20、距离 +23.39），平均捕获时间 13.17 s。Last 2M 仍 30/30、回报 +43.33、距离 +23.33，但平均捕获时间增至 23.67 s。20 次评估的平均成功率只有 59.7%；1M、1.1M、1.2M、1.4M、1.8M 的评估均为 0/30，说明不是稳定收敛。训练末尾 100 局 100/100 成功、距离 +22.86。
- 末尾 1,000 次更新均值：Actor loss −14.87、梯度范数 0.531、裁剪比例 45.3%；Critic loss 0.370、TD 误差 0.192、梯度范数 25.22、裁剪比例 88.2%；Q 均值 +14.75。末尾较强梯度裁剪与中途策略反复同时存在，不能仅据此判定发散。

### 2026-11-25 · 中心型径向 2v2，2M

- [配置](runs/maddpg_radial_2v2_capture5m_seed7_2m/settings.json)、[汇总](runs/maddpg_radial_2v2_capture5m_seed7_2m/summary.json)、[周期评估](runs/maddpg_radial_2v2_capture5m_seed7_2m/evaluations.jsonl)、[更新](runs/maddpg_radial_2v2_capture5m_seed7_2m/updates.jsonl)：2M 步、8,354 局、199,898 次更新；两个有效 Actor/Critic，其余三槽掩码；两条进攻航线约相隔 180°，其余设置同上。
- Best 在 1M：30/30 全歼、目标捕获率 100%、回报 +86.23（捕获 +40、距离 +46.23），平均回合 13.14 s；1.1M～1.3M 仍为 30/30。Last 2M 跌到 6/30 全歼、捕获率 60%、回报 +20.34（捕获 +24、突防 −40、距离 +36.34）。20 次评估平均全歼率 32.2%、捕获率 53.2%；训练末尾 100 局全歼率 37%、捕获率 65%。后期退化明显。
- 末尾 1,000 次更新均值：Actor loss −5.15、梯度范数 0.209、裁剪比例 7.1%；Critic loss 1.335、TD 误差 0.290、梯度范数 16.65、裁剪比例 79.4%；Q 均值 +4.63。三组更新频率同为约 0.1 次/采样步，Actor loss 和 Q 均值受有效智能体数量与奖励尺度影响，不能直接横向排序。

### 中心型距离奖励口径核查

- 末次固定评估的距离奖励总额依次为 1v1 +23.33、2v2 +36.34、5v5 +84.93；按有效防御机平均约 +23.33、+18.17、+16.99。5v5 总额并非与另两组相等；每架平均贡献较低，但不代表距离项主要被负值抵消。
- 5v5 训练末尾 100 局中，33 局零捕获的距离奖励仍平均 +59.41；2v2 零捕获的 7 局仍平均 +33.84。另用三个种子的无动作对局检查，1v1、2v2、5v5 全部失败时距离项均值约 +19.04、+36.92、+91.11。蓝方主动飞近防御环即可提供正缩距奖励；这些演示种子与固定评估样本不同，不能用数值差直接推断训练策略优劣。现有结果只保存回合累计分量，无法量化失败局中逐步负距离奖励的出现频率。

### 2026-11-26 · 中心型径向 5v5 高速、捕获半径 8 m，2M

- [配置](runs/maddpg_radial_5v5_fast_capture8m_seed7_2m/settings.json)、[汇总](runs/maddpg_radial_5v5_fast_capture8m_seed7_2m/summary.json)、[周期评估](runs/maddpg_radial_5v5_fast_capture8m_seed7_2m/evaluations.jsonl)、[更新](runs/maddpg_radial_5v5_fast_capture8m_seed7_2m/updates.jsonl)、[训练回合](runs/maddpg_radial_5v5_fast_capture8m_seed7_2m/episodes.jsonl)：2M 步、9,562 局、199,898 次更新；单环境、seed 7、5 个有效 Actor/Critic、24/75 维。双方最大速度 7 m/s、最大加速度 3 m/s²、蓝方初速度 2.8 m/s、捕获半径 8 m；其余关键学习参数仍为 `γ=0.99`、Actor/Critic LR `0.0003/0.001`、batch 128、uniform replay 200k、每 10 步更新。
- Best 在 1.4M：固定 30 局全歼 0/30、目标捕获率 74.0%、团队回报 +103.81（捕获 +74、突防 −65、距离 +94.81）。Last 2M：全歼 0/30、捕获率 20.0%、回报 −101.71（捕获 +20、突防 −200、距离 +78.29）。20 次固定 30 局评估均无全歼。训练末尾 100 局全歼 4、捕获率 55%、平均回报 +34.82；与末次确定性评估有明显差异，不能把近期训练回合表现代替固定评估。
- 后 1M～2M 的 11 次评估，距离分量范围 +68.13～+97.14、标准差 8.55；团队回报范围 −109.54～+103.81、标准差 66.02。由于每局五个目标均结算，事件奖励等于 `−250 + 70 × 捕获数`。1.4M 到 2M 的平均捕获数由 3.7 降到 1.0，导致事件奖励下降 189；距离项仅下降 16.52，合计 Return 下降 205.52。正距离奖励与完成拦截不是同一个指标。
- 最后 1,000 次更新均值：Actor loss +6.149、裁剪前梯度范数 0.178、裁剪比例 14.7%；Critic loss 8.428、TD 绝对误差 0.866、梯度范数 31.20、裁剪比例 99.7%、Q 均值 −7.43、Q 标准差 57.24。首 1,000 次 Critic loss/TD 误差约 41.59/1.566；误差下降但评估捕获率未稳定。末期 Critic 长期高频裁剪与策略波动并存，不能仅凭裁剪证明其为根因或 Q 发散。
- 相比旧版 5 m 中心型，最佳目标捕获率从 66% 到 74%，但旧版固定评估曾有 2/30 全歼，新版 20 次评估均无全歼；Last 捕获率由 26.7% 降至 20%。本组同时改变速度、加速度、初速度和捕获半径，不能把差异归于单一参数；两组都未表现出稳定 5v5 全歼能力。

### 2026-11-26 · Jia 速度—航向径向 5v5：d0.2 / h0.1 / 每步时间 −0.05

- [配置](runs/maddpg_radial_5v5_jia_heading_d02_h01_time005_seed7_2m/settings.json)、[汇总](runs/maddpg_radial_5v5_jia_heading_d02_h01_time005_seed7_2m/summary.json)、[周期评估](runs/maddpg_radial_5v5_jia_heading_d02_h01_time005_seed7_2m/evaluations.jsonl)、[更新汇总](runs/maddpg_radial_5v5_jia_heading_d02_h01_time005_seed7_2m/updates_summary.jsonl)、[训练回合](runs/maddpg_radial_5v5_jia_heading_d02_h01_time005_seed7_2m/episodes.jsonl)：2M 采样步、8,263 局、199,898 次更新、单环境、seed 7；26/85 维、捕获半径 5 m、理想一对一分配。三组共用同一环境实现哈希与 MADDPG 配置，仅距离、航向、时间系数不同；20 次评估均对同一组 30 个种子执行。
- Best 在 1.7M：固定评估全歼 0/30、目标捕获率 61.3%、团队回报 +87.51。Last 2M：全歼 0/30、捕获率 42.0%、回报 +20.14；每局平均分量为捕获 +42.00、突防 −145.00、距离 +89.58、航向 +46.39、时间 −12.03、越界 −0.80。1.3M 捕获率仅 4%，到 1.7M 升至 61.3%，1.8M 又降至 26%，后期没有稳定保持峰值。
- 全部周期评估的 600 局没有全歼；20 次评估的平均目标捕获率 30.1%，最后五次均值 36.1%。探索采样的 8,263 个训练回合有 41 局全歼；末尾 100 局全歼 0、捕获率 33.4%、回报 −44.63。训练和确定性评估口径不同，不能混用。
- 最后约 1,000 次更新：Actor loss +33.37、裁剪前梯度范数 0.265、至少一个有效 Actor 被裁剪的更新比例 32.6%；Critic loss 9.24、TD 绝对误差 0.838、有效 Critic 平均裁剪前梯度范数 19.29、至少一个有效 Critic 被裁剪比例 98.1%，Q/target Q 均值约 −34.40/−34.35。首约 1,000 次 TD 误差 1.424；误差下降没有对应稳定的全歼表现。

### 2026-11-26 · Jia 速度—航向径向 5v5：d0.3 / h0.02 / 每步时间 −0.03

- [配置](runs/maddpg_radial_5v5_jia_heading_d03_h002_time003_seed7_2m/settings.json)、[汇总](runs/maddpg_radial_5v5_jia_heading_d03_h002_time003_seed7_2m/summary.json)、[周期评估](runs/maddpg_radial_5v5_jia_heading_d03_h002_time003_seed7_2m/evaluations.jsonl)、[更新汇总](runs/maddpg_radial_5v5_jia_heading_d03_h002_time003_seed7_2m/updates_summary.jsonl)、[训练回合](runs/maddpg_radial_5v5_jia_heading_d03_h002_time003_seed7_2m/episodes.jsonl)：2M 步、8,241 局、199,898 次更新；与前组的环境参数和算法超参数一致，只改 `d/h/time` 三项。
- Best 在 800k：全歼 0/30、捕获率 54.0%、回报 +87.69。Last 2M：全歼 0/30、捕获率 38.0%、回报 +25.55；分量为捕获 +38.00、突防 −155.00、距离 +141.08、航向 +8.69、时间 −7.22、越界 0。1.5M 捕获率跌至 4.7%，末五次评估捕获率为 41.3%、34.0%、40.0%、45.3%、38.0%，较前组末五次波动小，但未出现全歼。
- 全部固定周期评估 0/600 全歼，平均捕获率 28.5%；训练回合全歼 11/8,241，末尾 100 局全歼 0、捕获率 33.2%、回报 +6.53。末尾 100 局有 14 局零捕获，其距离奖励仍平均 +118.02，说明距离奖励不能替代捕获指标。
- 最后约 1,000 次更新：Actor loss +39.87、梯度范数 0.272、裁剪比例 32.4%；Critic loss 10.09、TD 误差 0.813、有效 Critic 梯度范数 15.64、裁剪比例 92.0%，Q/target Q 均值约 −40.32/−40.27。首约 1,000 次 TD 误差 1.452；末期数值误差下降，策略仍有明显阶段性退化。

### 2026-11-26 · Jia 速度—航向径向 5v5：d0.8 / h0.03 / 每步时间 −0.015

- [配置](runs/maddpg_radial_5v5_jia_heading_d08_h003_time0015_seed7_2m/settings.json)、[汇总](runs/maddpg_radial_5v5_jia_heading_d08_h003_time0015_seed7_2m/summary.json)、[周期评估](runs/maddpg_radial_5v5_jia_heading_d08_h003_time0015_seed7_2m/evaluations.jsonl)、[更新汇总](runs/maddpg_radial_5v5_jia_heading_d08_h003_time0015_seed7_2m/updates_summary.jsonl)、[训练回合](runs/maddpg_radial_5v5_jia_heading_d08_h003_time0015_seed7_2m/episodes.jsonl)：2M 步、8,296 局、199,898 次更新；其他环境和学习参数与前两组相同。
- Best 在 1.8M：固定 30 局全歼 3 局（10.0%）、目标捕获率 60.7%、回报 +403.32；因 Best 首先按全歼率选取，所以它不是最高捕获率检查点，1.7M 捕获率 63.3% 但全歼 0。Last 2M：全歼 0/30、捕获率 62.7%、回报 +401.23；分量为捕获 +62.67、突防 −93.33、距离 +425.31、航向 +10.14、时间 −3.55、越界 0。末五次评估捕获率均值 57.3%，但 1.3M 曾跌至 10%，不能视为稳定收敛。
- 全部 20 次固定评估只在 1.8M 的一次中出现 3 局全歼；重复使用相同 30 个评估种子，不能将 3/600 当成独立泛化试验。探索采样训练全歼 83/8,296 局；末尾 100 局全歼 1、捕获率 41.0%、回报 +315.93。训练所有 1,119 个零捕获回合的距离奖励均值仍为 +256.04、团队回报均值 +9.02，约 56.9% 的零捕获回合回报为正。高回报与完成拦截明显不等价。
- 最后约 1,000 次更新：Actor loss +10.75、梯度范数 0.358、裁剪比例 62.2%；Critic loss 15.32、TD 误差 1.142、有效 Critic 梯度范数 24.99、裁剪比例 100%，Q/target Q 均值约 −11.40/−11.35。首约 1,000 次 TD 误差 1.538。此组的回报和 Critic target 尺度更大，loss/TD 误差不能不经尺度校正就解释为网络一定更差；但近乎持续的梯度裁剪与 1.3M 性能低谷都值得记录。
- 三组末次回报差主要受距离奖励系数影响：本组距离项 +425.31，前两组 +89.58/+141.08。将距离项除以各自系数后，末次每局累计的未加权距离变化约 531.6/447.9/470.3；此量包含蓝方运动与目标切换，不是防御机主动接近的净位移。三组同种子、同环境实现哈希与训练参数，但同时改变三项奖励，且首组在 Ubuntu 服务器、另两组在本地 WSL/CUDA 运行；单次结果不能把差异归因于某一个系数。三组所有训练回合的奖励分量均与团队回报对账，时间分量均等于 `−系数×步数`。

### 2026-11-26 · MATD3，Jia 速度—航向径向 5v5：d0.2 / h0.3 / 每步时间 −0.1，2M

- [配置](runs/matd3_radial_5v5_jia_heading_time01_seed7_2m/settings.json)、[汇总](runs/matd3_radial_5v5_jia_heading_time01_seed7_2m/summary.json)、[周期评估](runs/matd3_radial_5v5_jia_heading_time01_seed7_2m/evaluations.jsonl)、[更新汇总](runs/matd3_radial_5v5_jia_heading_time01_seed7_2m/updates_summary.jsonl)、[训练回合](runs/matd3_radial_5v5_jia_heading_time01_seed7_2m/episodes.jsonl)：2M 采样步、8,240 局、199,898 次 Critic 更新；单环境、seed 7，26/85 维、理想一对一分配、捕获半径 5 m、uniform replay 200k，batch 128、`γ=0.99`、`τ=0.005`、Actor/Critic LR `0.0003/0.001`、每 10 步更新。MATD3 有每智能体双 Critic、目标动作平滑及 `policy_delay=2`，故实际 Actor 和 target 更新各 99,949 次，而非 199,898 次。
- Best 在 1.9M：固定 30 局全歼 12 局（40.0%；该样本 Wilson 95% 区间约 24.6%～57.7%）、目标捕获率 76.7%、团队回报 +180.29。Last 2M：全歼 2/30（6.7%）、捕获率 64.7%、回报 +121.33。1.4M/1.6M/1.8M 分别全歼 6/10/7 局，1.7M 只有 1 局；1.9M 后仅 100k 步就从 12/30 降至 2/30，未稳定收敛。20 次评估累计 45 次全歼、平均目标捕获率 50.8%，但每次重复相同 30 个评估种子，不能当成 600 个独立测试回合。
- Best 每局平均分量：捕获 +76.67、突防 −58.33、距离 +111.68、航向 +73.11、时间 −22.83、越界 0。Last：捕获 +64.67、突防 −88.33、距离 +109.86、航向 +59.12、时间 −23.98、越界 0。Best→Last 的团队回报减少约 58.96，其中捕获与突防事件净额减少 42.00，距离分量只减少 1.82；主要退化是完成捕获的目标数下降。Last 仍有 28/30 局失败，正 Return 不是全歼成功率。
- 探索采样的训练回合仅 3/8,240 全歼，最后 100 局全歼 0、捕获率 38.2%、回报 +44.81；最后 500 局全歼 1、捕获率 40.6%。这与 Best 的确定性固定种子评估差距很大，可能同时涉及动作噪声与种子分布，现有文件不能区分原因。训练回合奖励账本与 Return 对账，时间分量等于 `−0.1×步数`。
- 最后约 1,000 次 Critic 更新：Actor loss +38.36（仅延迟 Actor 更新时记录）、Actor 裁剪前梯度范数 0.327、Actor 更新裁剪比例 73.8%；两组 Critic loss 分别为 14.57/14.63，总和 29.21，双 Critic 平均 TD 绝对误差 0.881，双 Critic 有效网络平均梯度范数 19.79，99.2% 的 Critic 更新至少有一个网络超出裁剪阈值 10；Q1/Q2 平均绝对差 0.448，Q/target Q 均值约 −44.64/−44.57。首约 1,000 次 TD 误差 1.392。近乎持续裁剪提示更新压力，但 1.9M→2M 间 TD 误差略降仍伴随全歼率下滑，不能单凭 TD 或裁剪比例确定策略退化的根因。
- 与[同环境、同奖励的 MADDPG 对照](runs/maddpg_radial_5v5_jia_heading_time01_seed7_2m/evaluations.jsonl)核对：环境配置与实现哈希、共同训练超参数、seed 和评估种子设置一致。MADDPG Best 为 2/30 全歼、74.0% 目标捕获；Last 为 0/30、19.3%。MATD3 在这一次固定种子评估的 Best/Last 均更好，但它同时改变双 Critic、目标平滑和 Actor 延迟更新，不能归因于单个机制；也需要新的独立评估种子和重复训练种子验证泛化与稳定性。MADDPG 单 Critic loss 与 MATD3 双 Critic loss 总和不能直接比较。

## 后续追加格式

每次用户要求分析一次训练，就在“已分析训练记录”末尾新增日期与运行目录的小节，记录：实际配置与完成状态、评估峰值和末次的全歼/捕获/团队回报、主要奖励分量、Actor/Critic 学习指标、训练趋势、可比性限制、`settings.json`/`summary.json`/`evaluations.jsonl` 证据链接。未完成训练明确标注快照，不写成最终结论。

