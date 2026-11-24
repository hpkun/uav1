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

## 后续追加格式

每次用户要求分析一次训练，就在“已分析训练记录”末尾新增日期与运行目录的小节，记录：实际配置与完成状态、评估峰值和末次的全歼/捕获/团队回报、主要奖励分量、Actor/Critic 学习指标、训练趋势、可比性限制、`settings.json`/`summary.json`/`evaluations.jsonl` 证据链接。未完成训练明确标注快照，不写成最终结论。

