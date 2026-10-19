# MATD3 v3.41 实际交付与验证报告

实现已完成，真实 v3.41 CUDA 训练、保存、重载、评估与 canonical recorder 测试闭环通过。未启动正式 300k 或 3M 训练，也未使用历史模型初始化。完整文件清单、数学位置、配置、Replay/mask、噪声与 checkpoint 设计见 [实现说明](matd3_v341.md)。

## 运行环境与隔离范围

- PowerShell，`D:\conda_envs\envs_dirs\uav\python.exe`，PyTorch `2.11.0+cu128`。
- CUDA：NVIDIA GeForce RTX 5060 Laptop GPU；CUDA 不可用或指定 CPU 时显式报错。
- 既有 Git HEAD：`7323533bd777ce10a0c81647966bc2a3f5843c3d`。初始工作区干净。
- 既有文件只修改 `algorithm/maddpg/protocol.py` 的 3 处 v3.41 身份登记和 `tools/record_combat_episode.py` 的 MATD3 分发分支。
- 环境代码、环境 YAML、历史算法数学、历史算法配置和入口默认值未改；新增 MADDPG v3.41 配置与 v3.39 配置逐字段比较，仅环境版本不同。
- 新增 8 个 MATD3 包文件、2 个 CLI、2 个配置、1 个专项测试文件、2 个说明文档。原有训练产物未覆盖。

## 实际测试结果

| 检查 | 结果 |
| --- | --- |
| 修改前相关历史基线 | 365 passed，35.81s |
| 最终专项与相关历史回归 | 410 passed，62.20s |
| 其中 MATD3 / 新增 v3.41 专项 | 45 passed |
| 全仓库测试 | 4,422 passed / 44 failed / 4 skipped，共 4,470 项，482.51s |
| 未修改 HEAD 复现全库失败集合 | 44/44 同样失败 |
| 独立 CLI CUDA smoke + 重载评估 + 三 checkpoint 审计 | 全部通过 |
| 确定性动作、8 个 Q 输出、smoothing RNG 往返 | 完全一致 |
| 同开发集种子的重载评估 | 与训练结束评估完全一致 |

全库运行后补充了一个 MADDPG v3.41 checkpoint 往返测试，并增强 recorder 的逐步环境 RNG 比较；最终 410 项专项回归包含这些最终版本检查。全库结果保留原始运行数量，未用后续专项结果冒充一次新的全库运行。

新增测试覆盖独立网络/参数存储/优化器、45/3/192 形状、同智能体 clipped double Q、Gaussian/噪声裁剪/动作裁剪/死亡置零顺序、独立 RNG、无 Target 梯度、Q1-only Actor 与 replay 其他动作、Critic 阶段 Actor 不变、delay=2、跨事件延迟计数和重新抽样、Polyak 系数方向、终止与截断、死亡当步奖励、全死 batch、真实 reset 前 next observation、3M 理论计数、optimizer.step 前裁剪、固定 batch 双 Critic 下降、冻结 Critic 的 Actor 梯度方向、保存加载身份/哈希/结构/真实步数，以及录像对动作、奖励、观测和环境 RNG 的不变性。均使用正式 256×256 网络宽度。

## 最终 CLI smoke 数值

正式请求配置保存在 `smoke_final/requested_algorithm_config.yaml`；生效配置单独保存在 `smoke_final/algorithm_config.yaml`。正式预算/采样参数为 16 环境、3M transitions、batch1024、warmup10k、replay1M、20 局评估；本次仅使用 2 环境、512 transitions、batch64、warmup128、replay4096、1 局评估。其余核心网络/优化公式保持正式配置。

| 项目 | 实际值 |
| --- | --- |
| 环境版本 / 观测 / 动作 / Red | 3.41 / 45 / 3 / 4 |
| 在线 Actor / 在线 Q / 非训练 Target 网 | 4 / 8 / 12 |
| Actor 单网 / Q 单网参数 | 78,339 / 115,457 |
| 在线总参数 / Target 总参数 | 1,237,012 / 1,237,012 |
| sampled_steps / vector_steps / Replay size | 512 / 256 / 512 |
| Critic 更新轮 / Actor 更新轮 | 3 / 1 |
| 每个 Q1 optimizer 实际 step | 3, 3, 3, 3 |
| 每个 Q2 optimizer 实际 step | 3, 3, 3, 3 |
| 每个 Actor optimizer 实际 step | 1, 1, 1, 1 |
| 剩余 transition credit | 84 |
| 实际 Critic round UTD / Actor round UTD | 3/512 / 1/512 |

Loss 是各智能体 current-alive masked loss 的算术平均；无有效样本的智能体单独跳过优化器。下表中未更新轮的 Actor loss 为 null，不复用上一轮数值。

| sampled_steps | Critic 轮 | Actor 轮 | Q1 loss | Q2 loss | Actor loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 228 | 1 | 0 | 0.00751103186 | 0.000554356664 | 本轮不更新 |
| 328 | 2 | 1 | 0.00524596507 | 0.000316170495 | 0.016644208 |
| 428 | 3 | 1 | 0.00306629356 | 0.000176715155 | 本轮不更新 |

所有非空数值优化诊断均有限。在线网络相对初始参数的 L2 漂移：

| 网络组 | R1–R4 L2 |
| --- | --- |
| actor_parameter_drift_l2 | 0.0216380116, 0.0241628867, 0.0236279629, 0.024497347 |
| critic_parameter_drift_l2 | 0.04718161, 0.0876026824, 0.0868449882, 0.0829484239 |
| critic2_parameter_drift_l2 | 0.0768838525, 0.0558472872, 0.0465047024, 0.0435129553 |

Target 相对初始化拷贝的 L2 漂移，证明三组 Target 均发生了实际变化；其非延迟轮保持不变和延迟轮 Polyak 精确关系另由专项测试验证：

| Target 网络组 | R1–R4 L2 |
| --- | --- |
| target_actors | 2.16421795e-05, 2.41683538e-05, 2.36341511e-05, 2.45073261e-05 |
| target_critics | 3.6560923e-05, 6.02922992e-05, 5.9749691e-05, 5.8715701e-05 |
| target_critics2 | 5.7445508e-05, 4.43423269e-05, 3.61283819e-05, 3.38692516e-05 |

评估种子 10,000,000，episode length=298，Return=-39.583943025452，Red win rate=0，Red losses=4，Blue losses=0。该评估仅用于检验闭环；未将 smoke 胜率作为训练效果结论。

环境配置 SHA256：`05e6e9464fd57a284e6c388fd275800761ec71e79e4f78a86593d1e785fa8d89`。

生效算法 SHA256：`9e73593854cf2b8dce2a2e11aa83dac59a181501fe785b61b9df1badbc14b852`。

best/latest/final 均通过独立协议检查。checkpoint 保存 OU、Replay sampling、Target smoothing 及其他全局 RNG 状态、全部权重/优化器/真实计数/评估选择信息。Replay 内容和完整环境状态未保存，`formal_exact_resume_supported=false`，不提供伪精确恢复。

## 既有失败的定位与处理

全部 44 项在 `git archive HEAD` 解包的独立目录、同一 Windows uav 环境复现；未删除测试、放宽断言、修改历史期望值，未修改相关历史算法来消除这些失败。

| 类别 | 数量 | 定位 |
| --- | ---: | --- |
| 历史完整轨迹 SHA256 参考不一致 | 14 | 旧 v2/v3 轨迹 reference 与本机运行输出不同；未进一步断言其最初形成原因 |
| MAPPO 系列日志测试夹具缺少 env_config | 10 | `_accumulate_rollout_metrics` 读取字段时 AttributeError |
| 旧版 MADSAC 版本号/错误文本及 MAAC/MADSAC 范围文本期望 | 14 | 当前 implementation v5 或注册范围已变化，旧测试仍期望旧版本/文本 |
| 历史 Actor 参数逐字节比较 | 4 | `torch.equal` 与保存参考不相等；未放宽为近似比较 |
| MADSAC v3.3 checkpoint attention_head_dim | 1 | 旧测试 checkpoint 元数据不满足现协议 |
| 写死全部 YAML 文件名的名单 | 1 | 原 HEAD 已比旧名单多 17 个配置；本次新增 2 个后多 19 个，同位置仍失败 |

原始全库输出：`outputs/matd3_v341_validation_20261019/all_tests.log` 和 `.xml`；未修改 HEAD 复现：`baseline_failures.log`、`baseline_failures.xml`、`baseline_reproduction.json`。这些既有问题未影响 MATD3 专项闭环，但不能把全仓库描述为全部通过。

跳过项保留原标记：

- `tests.test_combat_replay_pipeline.test_mp4_realtime_duration_and_png`：FFmpeg not available
- `tests.test_combat_v24_videos.test_mp4_probe_and_stride_fire_event_visible`：FFmpeg/ffprobe unavailable; PNG and event visibility checked
- `tests.test_stea_representation_diagnostic.test_real_checkpoint_reproducibility_hash_and_formal_eval`：local completed experiment not available
- `tests.test_stea_representation_diagnostic.test_batched_checkpoint_diagnostics_reproducible_and_episode_independent`：local completed experiment unavailable

## 可复现命令

```powershell
Set-Location C:\Users\HPK\Desktop\uav1
& 'C:\Users\HPK\anaconda3\shell\condabin\conda-hook.ps1'
conda activate 'D:\conda_envs\envs_dirs\uav'
python -m compileall -q algorithm/matd3 algorithm/train_matd3.py algorithm/evaluate_matd3.py tests/test_matd3.py
python -m pytest tests/test_matd3.py tests/test_maddpg.py tests/test_madsac_v341.py tests/test_v341_training_configs.py tests/test_combat_v341.py -q
python -m pytest tests -q
python outputs/matd3_v341_validation_20261019/reproduce_baseline.py
python -u algorithm/train_matd3.py --env-config configs/combat_environment_v341.yaml --algorithm-config configs/matd3_4v4_v341.yaml --output-dir outputs/matd3_v341_validation_20261019/smoke_final --device cuda --seed 1 --num-envs 2 --total-sampled-steps 512 --smoke
python -u algorithm/evaluate_matd3.py --checkpoint outputs/matd3_v341_validation_20261019/smoke_final/final.pt --device cuda --episodes 1 --output outputs/matd3_v341_validation_20261019/smoke_final/evaluation_reload.json
python outputs/matd3_v341_validation_20261019/audit_smoke.py
```

以上记录本次实际参数；重新执行训练或 baseline 解包应换一个新的输出目录。正式 3M 训练命令仅交付，**本次未执行**：

```powershell
python -u algorithm/train_matd3.py --env-config configs/combat_environment_v341.yaml --algorithm-config configs/matd3_4v4_v341.yaml --output-dir outputs/matd3_v341_seed1_3m --device cuda --seed 1 --num-envs 16 --total-sampled-steps 3000000
```

3M 理论计数测试验证 Critic=29,900 轮，Actor=14,950 轮。没有将更新次数误记为 4 倍 agent transitions。

## 失败复现完整名单

- `tests.test_combat_protocol.test_factory_and_public_api_are_combat_only`：未修改 HEAD 同样失败。
- `tests.test_combat_v25.test_old_protocol_full_episode_replay[case4]`：未修改 HEAD 同样失败。
- `tests.test_combat_v26.test_madsac_still_rejects_v26`：未修改 HEAD 同样失败。
- `tests.test_combat_v31.test_legacy_bit_level_trajectories`：未修改 HEAD 同样失败。
- `tests.test_combat_v310.test_complete_pre_extension_v39_reference`：未修改 HEAD 同样失败。
- `tests.test_combat_v311.test_previous_v310_full_trajectory_reference`：未修改 HEAD 同样失败。
- `tests.test_combat_v312.test_v311_full_reference_and_legacy_entity_actor_exact`：未修改 HEAD 同样失败。
- `tests.test_combat_v313.test_v312_full_trajectory_reference_unchanged`：未修改 HEAD 同样失败。
- `tests.test_combat_v32.test_legacy_complete_terminal_trajectories_unchanged`：未修改 HEAD 同样失败。
- `tests.test_combat_v32.test_true_truncated_episode_in_training_metrics_and_summary[mappo]`：未修改 HEAD 同样失败。
- `tests.test_combat_v32.test_true_truncated_episode_in_training_metrics_and_summary[rmappo]`：未修改 HEAD 同样失败。
- `tests.test_combat_v32.test_true_truncated_episode_in_training_metrics_and_summary[ea-mappo]`：未修改 HEAD 同样失败。
- `tests.test_combat_v32.test_true_truncated_episode_in_training_metrics_and_summary[stea-mappo]`：未修改 HEAD 同样失败。
- `tests.test_combat_v327.test_minimal_inheritance_config_algorithm_and_state_identity`：未修改 HEAD 同样失败。
- `tests.test_combat_v33.test_historical_v30_v31_v32_full_info_and_reward_bytes`：未修改 HEAD 同样失败。
- `tests.test_combat_v33.test_offpolicy_v33_cuda_dimensions_checkpoint_and_eval[madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v330.test_old_checkpoint_version_rejected[madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v332.test_old_action_semantics_checkpoint_rejected[3.30-madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v332.test_old_action_semantics_checkpoint_rejected[3.31-madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v333.test_historical_checkpoint_rejected[3.30-madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v333.test_historical_checkpoint_rejected[3.31-madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v333.test_historical_checkpoint_rejected[3.32-madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v334.test_checkpoint_version_isolation[3.30-madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v334.test_checkpoint_version_isolation[3.31-madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v334.test_checkpoint_version_isolation[3.32-madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v334.test_checkpoint_version_isolation[3.33-madsac]`：未修改 HEAD 同样失败。
- `tests.test_combat_v34.test_v33_pre_change_full_reference`：未修改 HEAD 同样失败。
- `tests.test_combat_v35.test_v34_full_prechange_reference`：未修改 HEAD 同样失败。
- `tests.test_combat_v36.test_complete_v35_pre_hook_reference`：未修改 HEAD 同样失败。
- `tests.test_combat_v37.test_configuration_isolation_and_original_reference`：未修改 HEAD 同样失败。
- `tests.test_combat_v38.test_inherited_parameters_methods_algorithm_bytes_and_old_reference`：未修改 HEAD 同样失败。
- `tests.test_combat_v39.test_complete_pre_extension_v38_trajectory_reference`：未修改 HEAD 同样失败。
- `tests.test_compact_training_metrics.test_one_weighted_summary_after_whole_rollout[MAPPOTrainingRunner]`：未修改 HEAD 同样失败。
- `tests.test_compact_training_metrics.test_one_weighted_summary_after_whole_rollout[RMAPPOTrainingRunner]`：未修改 HEAD 同样失败。
- `tests.test_compact_training_metrics.test_one_weighted_summary_after_whole_rollout[STEAMAPPOTrainingRunner]`：未修改 HEAD 同样失败。
- `tests.test_compact_training_metrics.test_one_weighted_summary_after_whole_rollout[EAMAPPOTrainingRunner]`：未修改 HEAD 同样失败。
- `tests.test_compact_training_metrics.test_no_episodes_null_and_theoretical_count`：未修改 HEAD 同样失败。
- `tests.test_compact_training_metrics.test_accumulation_and_serialization_do_not_consume_rng`：未修改 HEAD 同样失败。
- `tests.test_formal_8v8.test_5v5_actor_weights_forward_and_deterministic_action_unchanged[mappo]`：未修改 HEAD 同样失败。
- `tests.test_formal_8v8.test_5v5_actor_weights_forward_and_deterministic_action_unchanged[rmappo]`：未修改 HEAD 同样失败。
- `tests.test_formal_8v8.test_5v5_actor_weights_forward_and_deterministic_action_unchanged[ea_mappo]`：未修改 HEAD 同样失败。
- `tests.test_formal_8v8.test_5v5_actor_weights_forward_and_deterministic_action_unchanged[stea_mappo]`：未修改 HEAD 同样失败。
- `tests.test_mappo_environment_identity.test_madsac_diagnostic_ranges_cover_v338`：未修改 HEAD 同样失败。
- `tests.test_mappo_environment_identity.test_maac_diagnostic_reports_registered_range`：未修改 HEAD 同样失败。
