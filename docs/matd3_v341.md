# MATD3 v3.41 实现与验证

MATD3 使用独立 Trainer、网络、配置协议、训练与评估入口。仅复用 MADDPG 的 JointReplayBuffer、OU、transition schedule、日志格式，以及公共并行环境、战斗指标、checkpoint 排序和 canonical recorder。环境源文件与 v3.41 环境 YAML 均未修改；MADDPG 只增加 v3.41 身份映射及独立 YAML。

算法依据：[Ackermann et al. 2019](https://arxiv.org/abs/1910.01465)、[Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)。项目参数、OU 行为探索、其他智能体使用 replay 动作及 terminated/truncated 处理遵守本仓库约定，不声称是论文实验的原始超参数。

## 文件与调用关系

| 文件 | 职责 |
| --- | --- |
| `algorithm/matd3/__init__.py` | 独立算法包 |
| `algorithm/matd3/networks.py` | 本地确定性 Actor、独立集中式标量 Q MLP |
| `algorithm/matd3/trainer.py` | 双 Q target、两个独立优化器、延迟 Actor、Polyak、计数、checkpoint |
| `algorithm/matd3/factory.py` | 根据正式配置构建完整宽度网络 |
| `algorithm/matd3/protocol.py` | v3.41、算法身份、结构、完整配置指纹、更新次数与优化器状态校验 |
| `algorithm/matd3/runner.py` | 并行 rollout、OU、Replay、transition credit、训练和评估产物 |
| `algorithm/matd3/evaluation.py` | 严格加载、开发集种子、无噪声评估 |
| `algorithm/matd3/diagnostics.py` | 梯度 norm、masked 分布、动作统计、参数距离 |
| `algorithm/train_matd3.py`、`algorithm/evaluate_matd3.py` | 独立 CLI，默认 v3.41 |
| `configs/matd3_4v4_v341.yaml` | MATD3 正式 3M 配置 |
| `configs/maddpg_4v4_v341.yaml` | v3.39 MADDPG 正式配置仅修改环境版本 |
| `algorithm/maddpg/protocol.py` | 增量登记 v3.41、强制匹配环境身份，不修改数学或历史入口默认值 |
| `tools/record_combat_episode.py` | MATD3 验证、构造、加载分发；复用原轨迹记录与渲染 |
| `tests/test_matd3.py` | 数学、正式宽度 CUDA 更新、边界、调度、身份、smoke、录像回归 |

实施前实际检查了 MADDPG networks/trainer/replay/noise/schedule/runner/evaluation/protocol/factory，以及公共 vector_env/evaluator/checkpoint/protocol 的源码与调用关系。

## 网络与数学

四架 Red 的局部观测均为 45 维，动作 3 维。每个 Actor 为 `45→256→256→3`，ReLU/ReLU/Tanh；每个 Q 为 `192→256→256→1`，ReLU/ReLU/线性。Critic 输入是先展平全部观测，再拼接展平的全部动作。Actor 不读取其他智能体信息。

| 网络 | 数量 | 单网参数 | 合计 |
| --- | ---: | ---: | ---: |
| 在线 Actor | 4 | 78,339 | 313,356 |
| 在线 Q1、Q2 | 8 | 115,457 | 923,656 |
| 在线总数 | 12 | — | 1,237,012 |
| 非训练 Target | 12 | 同对应在线网络 | 1,237,012 |

Q1/Q2 分别构造、分别初始化、分别使用 Adam。Target 复制对应在线网，存储不共享，`requires_grad=False`。

`trainer.smoothed_target_actions` 在 `no_grad` 中计算所有 Target Actor 的下一动作；独立 `torch.Generator` 生成 `N(0,0.2²)`，先裁剪噪声至 ±0.5，再把加噪动作裁剪至 ±1，最后将 next-dead 槽归零。

`trainer.targets` 对同一智能体自己的 Target Q1/Q2 取最小值，`critic_target` 计算：

```text
y_i = reward_i + 0.99 × (1-done) × next_alive_i × min(Q1_target_i, Q2_target_i)
done = terminated OR truncated
```

`update_critic` 分别最小化 `masked_mean((Qki-y_i)², current_alive_i)`。当前存活、本步死亡的样本保留即时奖励；下一步死亡不 bootstrap。Critic 的观测和动作均按 alive mask 用 `where` 归零，防止死亡槽 NaN 传播。没有 current-alive 样本则跳过 optimizer.step，空 masked mean 返回零。

每完整双 Q 阶段增加一轮 `critic_update_count`，即使个别智能体因无有效样本跳过也保留调度轮次；实际 Q1/Q2 optimizer 次数另行逐智能体记录。`critic_update_count % policy_delay == 0` 才执行 Actor 与所有 Target 更新。计数器跨更新事件连续累积。

`update_actor(i)` 以当前 Actor_i 替换 replay 联合动作中的自身动作，其他动作 detach，冻结该智能体的两个 Critic，只通过 Q1 优化 `-masked_mean(Q1_i, alive_i)`。没有 smoothing、min、熵项或额外正则。三个其余 Actor 不调用、不更新。

所有梯度均在 optimizer.step 前逐智能体、逐网络按 global L2 norm 裁剪至 0.5。`polyak_update` 仅在 Actor 阶段对全部三个 Target 网络组执行 `target = 0.999*target + 0.001*online`。

## Replay、随机数、预算

Joint Replay 的形状为 `[B,4,45]` observations/next_observations、`[B,4,3]` actions、`[B,4]` rewards/alive/next_alive、`[B]` done。存储 `executed_red_actions`，并核对它等于已裁剪行为动作。next observation 使用自动 reset 前的 `transition_next_observations`；下一个 rollout 使用 reset 后 observation。

OU 与 MADDPG 相同：theta=.15、sigma=.20、mu=0，scale 在 3M joint environment transitions 从 1 线性降至 .1。OU 使用独立 NumPy Generator，种子为 `seed ^ 0x0A17`；Replay 采样用另一 NumPy Generator，种子为 `seed ^ 0x31A5`；Target Gaussian 用 CUDA Generator，种子为 `seed ^ 0x7D3001`。环境命中使用环境自己的 NumPy RNG；评估建立独立环境并使用 10,000,000 起的开发集种子。

每个 vector step 只增加 `num_envs` 条 sampled_steps，不乘以四。warmup 后每新增 100 条触发一次事件，每事件重新 sample 一个 batch，进行一轮双 Q 更新。正式 3M 对应 `(3,000,000-10,000)/100=29,900` Critic 轮、14,950 Actor 轮；16 环境对应 187,500 vector steps。

正式 batch=1024、replay=1M、warmup=10k、16 环境、lr=1e-4、gamma=.99、tau=.001、policy_delay=2；每 100k transitions 以 20 局 deterministic evaluation 评估。`--smoke` 使用 512 transitions、warmup128、batch64、replay4096、1 局评估，保留正式网络和数学。CLI 的 num-envs/seed/steps/device 覆盖会写入生效算法快照；原请求 YAML 独立保存，并分别计算哈希。

## Checkpoint 与诊断

`algorithm=matd3`、`implementation_version=1`。加载时验证环境身份、完整环境与算法 SHA256、嵌入配置、网络数量及每个网络所有张量形状、参数数量、更新轮次、credit 余数、实际 per-agent 优化器步数及 Adam 的 step。与 MADDPG/MADSAC 双向拒绝错误身份。

保存全部在线/Target 权重、12 个 optimizer 状态、轮次和逐网络计数、训练种子、sampled/vector steps、OU 状态及 RNG、Replay sampling RNG、Target Gaussian RNG、Python/NumPy/Torch/CUDA RNG、评估历史及 best 选择结果。未保存 Replay 内容和完整环境状态；明确记录 `formal_exact_resume_supported=false`、`replay_buffer_included=false`，`resume()` 显式报错。`load()` 用于严格加载权重、优化器和计数，并恢复 smoothing RNG；不提供精确训练恢复承诺。

产物：生效环境/算法 YAML、请求算法 YAML、run_config.json、initial_diagnostics.json、training_metrics.jsonl、optimization_metrics.jsonl、evaluation_history.csv、best_eval.pt、latest.pt、final.pt、按采样数命名的 checkpoint、run_summary.json。非空目录拒绝写入。Best 排序保持 Red win rate、return、较少 Red loss。

优化日志包含两个 Q 的 pre-step masked 均值、总体标准差、min/max、TD error 和 loss，target 均值/方差，signed/absolute Q gap，梯度裁剪前后 norm、动作逐维分布和饱和比例、next-alive 坐标的平滑噪声 RMS/clip 比例、alive 数量和比例、轮次/真实 optimizer 步数及各自 UTD、在线/Target 距离和相对初始参数的漂移。参数距离在前两轮及每 diagnostics_interval_updates 个 Critic 轮记录；最终 summary 再完整计算。run_config 中明确统计口径。Twin Q 分歧不代表真实过估计偏差。

统一战斗指标继续复用公共 evaluator：胜率、return、存活/损失、fire attempts/hits/kills、episode length、boundary/ground/ceiling、R1–R4 等。正式评估和录像仅调用在线 Actor，不消耗 OU 或 smoothing 随机数。

## PowerShell 命令

先进入仓库、加载 Conda hook 并激活 Windows uav；直接调用 conda.bat 的 activate 可能不改变当前 PowerShell 的 Python。

```powershell
Set-Location C:\Users\HPK\Desktop\uav1
& 'C:\Users\HPK\anaconda3\shell\condabin\conda-hook.ps1'
conda activate 'D:\conda_envs\envs_dirs\uav'
python -c "import sys,torch; print(sys.executable); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python -m pytest tests/test_matd3.py -q
python -m pytest tests -q
python -m compileall -q algorithm/matd3 algorithm/train_matd3.py algorithm/evaluate_matd3.py

python -u algorithm/train_matd3.py --env-config configs/combat_environment_v341.yaml --algorithm-config configs/matd3_4v4_v341.yaml --output-dir outputs/matd3_v341_smoke_new --device cuda --seed 1 --num-envs 2 --total-sampled-steps 512 --smoke
python -u algorithm/evaluate_matd3.py --checkpoint outputs/matd3_v341_smoke_new/final.pt --device cuda --episodes 1 --output outputs/matd3_v341_smoke_new/evaluation_reload.json
```

评估不传 YAML 时加载 checkpoint 同目录生效快照；传入其他配置将严格检查完整指纹。不要用正式 YAML 去评估 smoke checkpoint。

正式训练启动命令（本次未执行）：

```powershell
python -u algorithm/train_matd3.py --env-config configs/combat_environment_v341.yaml --algorithm-config configs/matd3_4v4_v341.yaml --output-dir outputs/matd3_v341_seed1_3m --device cuda --seed 1 --num-envs 16 --total-sampled-steps 3000000
```

本次实际测试数量、CUDA smoke 数值和审计结果见同目录的 `matd3_v341_validation.md`，原始验证产物在 `outputs/matd3_v341_validation_20261019/`。
