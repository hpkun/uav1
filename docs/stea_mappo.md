# STEA-MAPPO v1

## 当前正式 5v5 比较

正式 5v5 MAPPO/STEA-MAPPO 均使用同一个 `CentralizedMLPCritic`（390→256→256→1，166,145 参数）。MAPPO 保留 shared MLP actor；STEA 保留 entity-attention + GRU actor（167,110 参数）和当前 state-independent Gaussian。主要结构差异是 actor representation。见 [centralized_critic_protocol.md](centralized_critic_protocol.md)。下文的 52D attention critic、state-dependent Gaussian 和计数描述原 v2.3 legacy 协议，继续支持旧 run snapshots。

STEA-MAPPO（Spatial-Temporal Entity-Attention MAPPO）是独立算法变体，checkpoint 标识为 `STEA-MAPPO`，`STEA_MAPPO_IMPL_VERSION=1`。扁平观测缺少对不同友机、敌机相关性的显式建模，单帧输入也难以完整表达动态态势。本版本以实体注意力和 GRU 提供空间聚合与时间上下文；这些是结构动机，性能提升需要正式多 seed 对照实验验证。

## Actor 与 CTDE

每个 agent 仅使用自己的本地 52D observation 和自己的历史 hidden state。4 个 Red agent 共享 Actor 参数，不共享 hidden state。三个类型内共享 encoder 保持同类型实体的交换结构，无 slot positional embedding。

| 输入 | 精确切片 | Encoder | 输出 |
|---|---|---|---|
| self | `0:7` | Linear(7,64)+ReLU | 64 |
| allies | `7:28` → `[3,7]` | 共享 Linear(7,64)+ReLU | `[3,64]` |
| enemies | `28:52` → `[4,6]` | 共享 Linear(6,64)+ReLU | `[4,64]` |

Self embedding 分别作为两组独立 attention 的 query，ally/enemy embedding 分别产生 key、value。每组使用 2 heads、每 head 32D，`softmax(QKᵀ/√32)`。两个 64D context 与 64D self embedding 拼成 192D，经过 Linear(192,128)+ReLU、单层单向 GRU(128,128)、Linear(128,128)+ReLU，然后独立 mean/log_std Linear(128,3) heads。log_std clamp 为 [-5,2]，动作是原有 heading/pitch/speed 三维 tanh Gaussian；复用 MAPPO 的稳定 tanh Jacobian 公式，保存 raw latent action 以避免饱和反变换误差。正式 Actor 参数 167,494。

Critic 直接复用原 `CentralizedValueCritic` 类：每个 agent 的 52D 输入编码为 256D，2-head centralized attention，输出原有 per-agent value，并使用原 alive mask。Critic 无 GRU、无实体 encoder，参数 474,113；正式总参数 641,607。初始化时按相同 seed 构造原 MAPPO 网络，保留同一 Critic 初始化，只替换 Actor 和 Actor optimizer。训练是 CTDE，执行 Actor 不访问其他 agent 的 hidden state 或 Critic 信息。

## 掩码与 recurrent 生命周期

实体末维 `alive > 0.5` 构成显式 validity mask，并与当前 focal agent 的 alive mask 相交。softmax 前将无效 score 填为 dtype 的最小有限值，softmax 后再次乘 mask 并对剩余权重重新归一化（分母下界 1e-12）。死亡实体权重严格为零；空实体集合所有权重及 context 严格为零，避免 `softmax(all -inf)` 的 NaN。

Runner 显式持有 `actor_hidden_states[E,4,128]`、`episode_start_masks[E]`，初始分别为零和一。Rollout 记录的是当前动作**之前**的 hidden state `[T,E,4,128]` 与 starts `[T,E]`；采样后立即乘 transition 的 next_alive，死亡 agent 清零；terminated/truncated 后整个 environment 的 hidden 清零。自动 reset 后使用新观测、starts=1。每步 GRU 前再次应用 `hidden *= alive * (1-episode_start)`，防止 chunk 中途跨回合泄漏。死亡 agent 的输出 hidden、action、log_prob 保持零。

Evaluation 每回合创建零 hidden，连续维护历史，用 deterministic `tanh(mean)`，在死亡和回合结束时清零。评估的 hidden 是局部变量，不触碰训练 Runner 的 hidden。Resume 与 baseline 一样恢复网络、优化器及计数器，切换到新 scenario episode 并清零 hidden；不承诺逐 transition 精确恢复旧轨迹。

## Recurrent PPO

Formal rollout=256，sequence length L=32。按 environment 单独切连续时间 chunk，保留 `[B,L,4,...]`。一个 rollout 每环境 8 chunks；只随机打乱 chunk 顺序，chunk 内保持时间顺序。Minibatch=512 **joint environment transitions**，因此 B=512/32=16，不按 4 个 agents 再除一次。配置必须满足 minibatch 和 rollout 均可被 L 整除，非法配置明确报错。

最后一个短 rollout 的末尾 chunk 不丢弃：补齐到 L，`valid_steps` 显式记录有效时间，padding 的 alive mask 为零、episode_start 为一。GAE 先在全部真实 `[T,E,4]` transitions 上计算，再 pack；Critic 可以对 sequence minibatch 的 B×L 做普通 feedforward，Actor 必须逐时间展开。每个 chunk 从 rollout 保存的 before-action hidden 开始并 detach，chunk 内执行 truncated BPTT，每一步应用 episode/death reset。

沿用 MAPPO 的 GAE、advantage normalization、PPO clipped objective、bounded-policy entropy Monte Carlo estimate、clipped value loss、gradient clipping 和 Adam 参数。新版本没有辅助损失。每次 update 在任何 optimizer step 前重新 forward 全部 chunks，对所有 live transitions 审计 ratio：最大 `|ratio-1|` 必须 ≤1e-4，否则立即报错。这是实现一致性审计；之后 PPO epochs 中 ratio 偏离 1 是正常更新行为。

## 协议、日志与运行

`configs/stea_mappo.yaml` 从 MAPPO 复制训练、Critic、implementation 超参数，仅把正式 target 设为 2M，并增加 Actor/recurrent 字段。默认 24 environments；当前 MAPPO 2M 运行使用 16，公平比较时命令指定 `--num-envs 16`。采样步数统计 joint environment transitions，按完整 vector batch 收集；24 环境的 2M target 最终可能到 2,000,016，16 环境可精确到 2,000,000。

```bash
conda activate uav
cd /mnt/c/Users/HPK/Desktop/uav1
python algorithm/train_stea_mappo.py --device cuda --seed 1 --num-envs 16 --total-sampled-steps 2000000 --output-dir outputs/stea_mappo_seed1_2m
python algorithm/evaluate_stea_mappo.py --checkpoint outputs/stea_mappo_seed1_2m/latest.pt --device cuda --seed-base 30000000 --episodes 20 --output outputs/stea_mappo_seed1_2m/holdout.json
```

在 WSL Ubuntu 的 `uav` 环境运行；WSL 不可用时用 Windows `uav`。训练、checkpoint audit/evaluation 强制 CUDA，不自动回退 CPU。入口拒绝覆盖非空新目录，保存环境/算法 YAML 快照与 run_config，支持严格 snapshot 验证、stale checkpoint 拒绝与日志 rollback 的 resume。不要运行两个进程写入同一 run 目录。

Checkpoint 保存独立 algorithm/version、52/3/4 dimensions、环境 2.3、训练 seed/gamma/num_envs/target/smoke、两份配置 SHA256、完整 architecture（actor_type/entity_dim/entity_heads/spatial_hidden/gru_hidden/gru_layers/L/critic_type/critic_hidden/critic_heads）和参数计数。Resume/evaluation 严格验证。MAPPO/STEA 双向拒绝加载，旧 MAPPO validator 未改。holdout 使用连续且与训练分离的 seed 范围；聚合器同时检查 algorithm、STEA version 和 architecture，只允许不同训练 seeds 的同协议结果聚合。Smoke 与 formal 有明确标记，不能混合。

Optimization JSONL 保留原 PPO diagnostics、ratio 分位数、first minibatch/epoch metrics、log_std 和动作饱和率，增加 `gru_hidden_norm_mean`、两类 `attention_entropy`、`attention_top1_mean`、padding 数量和 pre-update ratio 审计。Evaluation 同样输出聚合 attention/hidden diagnostics。通过 `return_attention=True` 可读取 `[E,4,heads,slots]` 权重；不逐步写入 attention 矩阵，诊断不改变 forward 数值路径。

## 文献启发与范围

以下是本次设计选择的概念启发，不是对原文算法的直接复现，也不声称其实验结论适用于当前 4v4 环境。

- **MTVO**：[Short-range air combat maneuver decision of UAV swarm based on multi-agent Transformer introducing virtual objects](https://doi.org/10.1016/j.engappai.2023.106358)。借鉴实体组织和 attention/masking 的表示思路；本版本没有 virtual object 或 CLS token。
- **E-MATD3 / ATT-MATD3**：[An evolutionary multi-agent reinforcement learning algorithm for multi-UAV air combat](https://www.sciencedirect.com/science/article/abs/pii/S0950705124006348)。借鉴友机/敌机分开聚合及 death masking；未引入 evolutionary population、self-play 或 TD3 的训练目标。
- **CAAFF**：[A Context-Aware Feature Fusion Method for Multi-UAV Cooperative Air Combat](https://doi.org/10.1109/TITS.2025.3530463)。作为时间/上下文态势表示的动机；本版本只在 spatial fusion 后使用 GRU，不复现其分层特征融合或图网络。
- **SIIS-MARL**：[Towards Efficient Multi-UAV Air Combat: An Intention Inference and Sparse Transmission Based Multiagent Reinforcement Learning Algorithm](https://doi.org/10.1109/TAI.2025.3567431)。按本次设计需求，将 recurrent encoding 支持上下文表示作为启发；本版本的 GRU 是独立设计选择，未声称复现原文编码器，未使用原文的 ToM、意图推断损失或 sparse communication。

本版本不新增 GAT、显式通信、稀疏通信、ToM、意图辅助任务、自博弈、目标选择动作或武器控制动作。环境 2.3、Blue 策略、动力学、52D observation、3D action、R1–R4、武器和终止条件均保持不变。

## 实现文件与本次验收

| 新增文件 | 用途 |
|---|---|
| `algorithm/stea_mappo/__init__.py` | 独立公共导出 |
| `algorithm/stea_mappo/networks.py` | Entity encoders、masked attention、显式状态 GRU Actor |
| `algorithm/stea_mappo/trainer.py` | RecurrentRolloutBatch、chunk batching、PPO、ratio 审计 |
| `algorithm/stea_mappo/factory.py` | Config 验证与 trainer 构造 |
| `algorithm/stea_mappo/protocol.py` | CUDA 约束、checkpoint/version/architecture 验证 |
| `algorithm/stea_mappo/runner.py` | 并行 rollout 与 hidden 生命周期、独立 checkpoint/resume |
| `algorithm/stea_mappo/evaluation.py` | Deterministic recurrent evaluation 与 holdout metadata |
| `algorithm/train_stea_mappo.py` | 训练 CLI、快照、日志和 resume |
| `algorithm/evaluate_stea_mappo.py` | Holdout CLI |
| `configs/stea_mappo.yaml` | 独立正式 2M 配置 |
| `tests/test_stea_mappo.py` | 36 项新测试，覆盖 A–N 及补齐、BPTT 边界和协议拒绝 |
| `docs/stea_mappo.md` | 本说明 |

修改仅有 README（说明与入口）、`tests/test_combat_protocol.py`（新增 YAML 白名单）、`tools/aggregate_holdout_results.py`（增加 STEA version/architecture 协议字段）。原环境、MAPPO、MADSAC、公共训练代码、旧入口与旧配置共 42 个文件的 SHA256 与实施前一致，检查清单保存在 `outputs/stea_mappo_validation/baseline_hashes_before.json`。

WSL Ubuntu 的 `uav`、RTX 5060 Laptop GPU 上，真实 CUDA smoke 完成 96 transitions、2 次 rollout update、4 次 Actor/Critic optimizer update、两回合评估及 checkpoint save/load。两个 rollout 更新前 ratio mean/min/max 均为 1.0、最大误差 0.0；末尾短 rollout 有 32 个显式屏蔽的 padding transitions。Reload 后确定性评估与保存前完全一致。

额外使用正式 Actor/Critic 尺寸与 rollout=256、epochs=10、minibatch=512、L=32 完成 512 transitions 的 CUDA 审计，并重新加载评估。完整测试 `372 passed in 36.92s`；compileall 和 git diff --check 通过。已执行检查中没有 warning、NaN 或 recurrent-state leakage；这些是短程实现验收，未运行完整 2M，不能据此推断算法胜率优于 MAPPO。机器可读证据见 `outputs/stea_mappo_validation/audit_report.json`。
