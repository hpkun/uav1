# PS-IPPO v3.41

本实现是参数共享 IPPO：4 架 Red 共用一个 Actor 和一个 LocalValueCritic，每架分别使用自身 45 维观测。不是四套独立参数的 IPPO。

## 结构和信息边界

| 项目 | MAPPO 正式 v3.41 | PS-IPPO |
|---|---|---|
| Actor | SharedMAPPOActor，45→256→256→3，ReLU | 同一个现有类 |
| Gaussian | state_independent，log_std=-0.5，范围[-5,0.5] | 相同 |
| Critic | CentralizedMLPCritic，225→256→256→1 | LocalValueCritic，45→256→256→1 |
| Critic 信息 | 4架联合观测与目标机观测 | 仅对应飞机的45维观测 |
| 参数 | Actor 78,342；Critic 123,905 | Actor 78,342；Critic 77,825；在线总计156,167 |

`algorithm/ippo/networks.py:LocalValueCritic.forward` 直接在最后一维执行共享 MLP，支持 `[B,4,45]` 和 `[T,E,4,45]`。不拼接智能体，不使用 agent index，不读取额外环境状态。输出可乘 alive mask；该 mask 不作为网络输入。自身观测中原有相对队友、敌机信息保留。

`IPPOTrainer` 独立构造本地网络、优化器和 checkpoint 协议；复用 MAPPO 的动作采样、确定性推理、GAE 和稳定性统计。PPO 更新内核保留现有 MAPPO 方程，仅增加每个 minibatch 的梯度裁剪频率统计。专项测试将原版 MAPPO 更新函数连接到相同局部网络，使用相同数据、随机数状态，核对所有公共指标和更新后的每一个参数完全相同。原有 MAPPO 源码、配置和环境源码不改动。

## 数学和采样协议

- 采样保存真实执行动作、Gaussian raw action 和当时 old_log_prob；动作与环境执行值不一致时直接报错，不静默重算旧概率。
- Tanh Jacobian 直接复用 `SharedMAPPOActor._squashed_log_prob` 的稳定 latent-action 实现；评估为 `tanh(mean)`，不采样、不加噪声。
- `compute_gae` 从 `algorithm.mappo.trainer` 直接导入，逐智能体计算奖励、GAE 和 return。continuation=(1-done)*next_alive；terminated/truncated 都不 bootstrap。
- 当前活着、当步死亡的奖励保留；之后死亡样本不参与 loss 或 advantage normalization。全死 rollout 安全跳过且所有指标有限。
- 使用 `ParallelVectorEnv.transition_next_observations` 和 `next_alive_masks`，不是 reset 后下一局观测；当前局终止不会向下一局递推 GAE。
- Advantage normalization 对整个 rollout 有效 agent-step 联合统计；minibatch 按联合环境 transition 排列。
- 正式 rollout `[256,16,4,45]` 展平为 `[4096,4,45]`；512 条联合 transition/minibatch，一次共享 Actor step 和一次共享 Critic step。每 epoch 8 次；最多10 epoch/80次更新。
- PPO clip=0.2，value clipping=0.2，value coefficient=0.5，entropy coefficient=0.001；Actor/Critic 独立 Adam，各自 global L2 grad clip=0.5。
- KL 为 `(ratio-1)-log_ratio` 的存活 masked mean；按完整 epoch 的 minibatch KL 平均，超过0.015才停止后续 epoch。与 MAPPO 相同，没有更改其规则。
- `sampled_steps` 单位为环境 transition，不乘4。3M预算：732个完整 rollout +108 vector-step 尾部，共733次 rollout update；无提前停止、无无效 minibatch 时上限58,600次 Actor/Critic optimizer step。
- 为保持预算精确，CLI/Runner 要求总预算可被环境数整除。smoke 默认4096，最多16384，仍使用正式网络、rollout、minibatch、epoch 配置。

## 配置、Checkpoint 和种子

配置 `configs/ippo_4v4_v341.yaml` 沿用现有嵌套结构。`network.parameter_sharing=true`、`critic_scope=local_observation`、`critic_type=local` 为实际验证条件。训练、implementation、runtime_logging 字段与正式 MAPPO v3.41 完全相同。工厂不允许静默覆盖网络宽度。

Checkpoint `algorithm='ippo'`、`ippo_impl_version=1`。包括 Actor/Critic 和独立优化器、完整架构及参数数目、Gaussian/PPO/GAE参数、环境和算法配置指纹、算法完整配置、采样/vector/PPO/optimizer实际计数、训练种子、评估种子来源、best选择信息与评估历史。正式评估和官方录像均通过同一个 IPPO factory、严格验证器和 trainer.load，拒绝配置指纹不一致和其他算法 checkpoint。

明确声明 `resume_capability=evaluation_reload_only`、`bit_exact_training_resume=false`。可以完整保存和确定性评估重载；没有环境与全部随机状态快照，不提供训练恢复入口。

- 训练 seed 范围小于10,000,000；Runner 保守检查预算范围不会碰到评估。
- development 范围 `[10,000,000,20,000,000)`，默认10,000,000起20局；训练 best 只使用该范围及公共 `evaluation_selection_key`。
- 录像/诊断范围 `[20,000,000,30,000,000)`；每次训练结束使用20,000,000+training_seed完整记录一次独立诊断飞行，不参与 best 选择。
- holdout 范围 `>=70,000,000`，必须显式 `--split holdout`；只用于训练结束后的评估。

## 日志和诊断

标准产物：env_config.yaml、algorithm_config.yaml、run_config.json、training_metrics.jsonl、optimization_metrics.jsonl、evaluation_history.csv、best_eval.pt、latest.pt、final.pt、run_summary.json、train.log。已有非空训练目录拒绝覆盖。终端 stdout/stderr 写入 train.log。

优化指标包含 actor_loss、value_loss、explained_variance、approximate_kl/approx_kl、clip_fraction、configured/effective_ppo_epochs、kl_early_stop、policy_entropy、policy_sigma_heading/pitch/speed、actor/critic_gradient_norm_preclip、gradient_clipped_ratio、actual_actor/critic_optimizer_steps、valid_alive_agent_samples、ratio分位数及每epoch统计。

`actual_*_optimizer_steps` 为本次 rollout 更新实际次数，checkpoint 的 actor_updates/critic_updates 为累计次数；valid_alive_agent_samples 是该 rollout 唯一存活 agent-step 数，不乘 epoch；gradient_clipped_ratio 是本次所有 Actor/Critic 梯度裁剪调用中 preclip norm>0.5 的比例。KL/损失沿用 MAPPO 对 minibatch 的平均单位。

`action_diagnostics.jsonl`：实际执行动作三维mean/std、|a|>0.99比例、局内相邻存活控制步变化量、由 v3.41 自身观测反解的实际航向/俯仰变化和速度。只统计有效连续对，注明单位、样本覆盖和within_rollout边界；死亡后的零观测不用于推断死亡当步角度。

`diagnostic_flight/` 及官方 replay 的 `ippo_flight_diagnostics.json` 使用完整 canonical 状态和事件，保留死亡当步物理状态：动作统计和相邻变化、航向/俯仰变化、速度、持续同向转弯比例、360度累计转向事件、首次进入武器距离/开火/击毁时间、首次开火瞬间ATA/HA、双方武器击毁/边界/触地/触顶计数、各架Red损失时间与原因。角度为弧度、速度m/s、时间秒；持续转弯定义为同号实际航向变化连续至少10步，360事件为同一连续转向段内累计完整2π。均为描述性指标，不直接判断训练失败或收敛。

`diagnostic_observations.npz` 从首次训练 rollout 前16个vector-step保存固定观测及真实训练seed、环境SHA和来源。`algorithm/diagnose_ippo.py` 对多个checkpoint输出确定性动作、各维均值/方差、相邻checkpoint差异及持续饱和比例；可额外输入多个同场景seed录像，比较重叠存活帧的位置变化及完整飞行诊断。拒绝holdout/未知来源观测集合，不根据holdout反馈修改训练。

## PowerShell 命令

所有命令在仓库目录，通过Windows的uav环境运行。训练、评估、checkpoint审计必须CUDA。

```powershell
& 'C:\Users\HPK\anaconda3\shell\condabin\conda-hook.ps1'
conda activate uav
python -m pytest tests/test_ippo.py -q --tb=short
python -u algorithm/train_ippo.py --smoke --output-dir outputs/ippo_smoke_new
```

正式3M命令（本任务未运行）：

```powershell
python -u algorithm/train_ippo.py --env-config configs/combat_environment_v341.yaml --algorithm-config configs/ippo_4v4_v341.yaml --device cuda --seed 0 --num-envs 16 --total-sampled-steps 3000000 --output-dir outputs/ippo_3m_seed0_new
```

确定性评估：

```powershell
python -u algorithm/evaluate_ippo.py --checkpoint outputs/ippo_3m_seed0_new/best_eval.pt --seed-base 10001000 --episodes 20 --split development --output outputs/ippo_development_new.json
python -u algorithm/evaluate_ippo.py --checkpoint outputs/ippo_3m_seed0_new/best_eval.pt --seed-base 70000000 --episodes 20 --split holdout --output outputs/ippo_holdout_new.json
```

官方录像（mp4需要FFmpeg；无FFmpeg可指定png html csv report）：

```powershell
python -u tools/replay_combat.py --checkpoint outputs/ippo_3m_seed0_new/best_eval.pt --env-config configs/combat_environment_v341.yaml --episode-seed 20000071 --output-dir outputs/ippo_replay_new --formats mp4 png html csv report --ffmpeg C:/path/to/ffmpeg.exe
python algorithm/diagnose_ippo.py --checkpoints outputs/ippo_3m_seed0_new/best_eval.pt outputs/ippo_3m_seed0_new/final.pt --observation-bank outputs/ippo_3m_seed0_new/diagnostic_observations.npz --output outputs/ippo_policy_comparison_new.json
```

本次实测结果和完整命令见 `docs/ippo_v341_acceptance.md`。不以 smoke 胜率作为验收标准。
