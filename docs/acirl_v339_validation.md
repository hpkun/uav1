# ACIRL v3.39 中文实施与验证报告

ACIRL 的独立网络、异步意图状态机、双时间尺度 PPO、回报监督 Gate Q、七种消融、严格 checkpoint、状态化评估和 canonical 对局录制已经实现。已在真实 v3.39 环境完成 CUDA 小规模训练和保存/重载/评估/录像闭环。**没有执行正式 3M 训练，也没有验证科研效果或相对基线优势。**

文件及完整网络结构、参数量、真实数据流和数学公式的实现位置见 [详细实现说明](acirl_v339.md)。本报告记录实际执行证据，原始产物目录为 `outputs/acirl_v339_validation_20261019/`。

## 1. 变更和环境隔离

- 新增 `algorithm/acirl/` 的 12 个模块、2 个独立 CLI、7 个独立配置、`tests/test_acirl.py` 和 2 个中文文档。
- 唯一修改的既有源码是 `tools/record_combat_episode.py`，增加 13 行可选的 ACIRL 加载/会话/sidecar hooks；既有算法不触发这些 hooks。
- 无环境代码或环境 YAML 变更。未改 MAPPO、MAAC、MADSAC、MADDPG、MATD3 等的算法、checkpoint 格式、正式参数或历史训练入口默认值。历史测试及期望值未修改。
- 开发前工作区干净，基准 HEAD 为 `01371a20ab4c680ca1b65a165f55cd712378323f`。
- 运行环境：PowerShell；`D:\conda_envs\envs_dirs\uav\python.exe`；PyTorch `2.11.0+cu128`；NVIDIA GeForce RTX 5060 Laptop GPU。

## 2. 完整网络及消融结构审计

Full：Policy 642,835 参数，两个独立 Critic 共 550,915 参数，总计 **1,193,750**。Policy 包含实体 Encoder 167,552、GRU 100,224、Continuous Actor 116,742、Intention Actor 108,550、Gate Actor 115,713、队友推断 33,670、两套 intention embeddings 共 384。Team Value 251,649，Gate Q 299,266。

默认 Encoder 输出128、4 attention heads、GRU hidden128、K6、latent embedding32、策略/价值 MLP256×2。Full 动作头输入192，意图头160，Gate头193，集中式 Team输入724，Gate Q输入909。所有 smoke 使用正式网络宽度。

下表来自真实构造的七种网络，未仅将不适用的输出置零：

| 模式 | Policy 参数 | Critic 参数 | Action 输入 | Team 输入 | Gate Q 输入 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 642,835 | 550,915 | 192 | 724 | 909 |
| no_intent | 368,134 | 244,481 | 128 | 696 | 无 |
| fixed_interval | 527,122 | 251,649 | 192 | 724 | 无 |
| no_gate_q | 642,835 | 251,649 | 192 | 724 | 无 |
| no_teammate_inference | 584,397 | 550,915 | 160 | 724 | 909 |
| no_temporal_critic | 642,835 | 550,915 | 192 | 724 | 909 |
| no_switching_cost | 642,835 | 550,915 | 192 | 724 | 909 |

七种模式全部完成非空数据上的真实 CUDA 参数更新单元测试。Full 另外完成真实并行环境训练和完整评估/录像。未宣称每个消融已完成长训练实验。

## 3. 训练数学与信息流检查

- `networks.py` 编码当前真实 7/6/5 实体布局，零槽局部 mask，空注意力返回零；不将 AA/ATA/HA 改作坐标或攻击动作。
- `state.py` 每架独立执行 min3/max30 门控，初始/max/fixed forced 与 learned gate 分开；只在 renew 采样意图，每步重新采样物理动作。
- `advantages.py:team_gae` 的 gamma=.99、lambda=.95；环境 terminated OR truncated 停止 bootstrap，人为 rollout cut 保留 next value。
- `advantages.py:macro_advantages` 实现真实 tau 的 discounted macro reward、gamma^tau bootstrap 和 gamma^tau×lambda_high 递推，死亡/episode 停止连接。
- 跨 rollout 保留运行意图与 memory，旧行为策略的宏样本只在当前片段 bootstrap，不用新策略后续 reward 回填，不制造新选择事件。
- `trainer.py:_fit_critics` 用固定 rollout 的 team λ-return 监督两个 Critic；Gate Q 只监督实际采样分支。Gate Actor 的优势来自拟合后 Q 的 detached 预测和行为 Bernoulli 概率，不把当前 Q 自身当监督标签。
- `trainer.py:evaluate_sequences/update` 做真实连续 GRU unroll、完整可用 prefix burn-in、padding masks、三个独立 PPO ratio/event views。动作使用保存的 pre-tanh action 和稳定 Jacobian 修正。
- 高层和门控各自以有效事件数量归一化。共享 Policy 单一优化器，Team/Gate Critic 各自独立优化器；空事件不推进对应策略头 Adam 动量。
- 全局联合信息只进入 Critic。真实队友意图只在推断 CE 的 label 路径使用，Actor 使用本机历史、局部观测、自身意图及推断分布。改变其他 UAV 的真实意图/hidden/局部行不会改变固定本机输入的动作与门控概率，此项有专项测试。

## 4. 实际测试结果

| 执行阶段 | 结果 |
| --- | --- |
| 修改前 v3.39/MAPPO/MATD3 相关基线 | 107 passed，90.31s |
| 修正首轮问题后的专项 | 51 passed，35.78s |
| 加入 warm restart / 空 gate 动量检查后的相关回归 | 375 passed，1 skipped，125.63s |
| 最终版本 ACIRL 专项 | **53 passed，49.80s** |
| 最后生命周期日志调整的直接回归 | 2 passed，51 deselected，2.86s |
| 两个已知历史失败：当前树与未修改 HEAD | 两边均复现 2/2；未改旧测试 |
| 最终独立 CLI smoke、两种评估、重载录像、CUDA 审计 | 通过 |

相关回归覆盖 `test_acirl`、`test_mappo`、`test_maddpg`、`test_maac`、`test_madsac_v341`、`test_matd3`、`test_combat_v339`、`test_combat_replay_pipeline`、`test_gaussian_protocol`。保留的原跳过项：`tests.test_combat_replay_pipeline.test_mp4_realtime_duration_and_png: FFmpeg not available`。

专项包括网络/参数归属、零实体/死亡 NaN 槽、局部信息隔离、tanh logp 饱和稳定性、门控初始/min/max/异步/死亡/reset、强制门控零梯度、手算 3/5/10 步宏回报与 gamma^tau、宏递推、死亡 reward 和 terminal/cut、跨 rollout 不拼新策略、auto-reset 真实 next observation、实际 Gate 分支梯度、Critic 与 Actor 梯度隔离、PPO clip、invalid/padding mask、旧 logp 重算、真实各头参数变化、全部消融、warm restart、配置/模式/版本/Adam 计数损坏拒绝，以及完整 deterministic/stochastic/holdout episode 和录像/渲染。

首轮 8 项失败的处理：1 项测试直接把需梯度 Tensor 交给 pytest.approx，修正为 detach 后比较；6 项暴露 padding 位置的 Gate/Intention 审计 mask 遗漏，修正打包 mask，原 2e-4 阈值未放宽；1 项暴露 recorder 公共指纹元数据缺失，补齐并严格校验。没有修改历史测试来绕过失败。

## 5. 最终独立 CUDA smoke 数值

正式请求 YAML 单独保存。实际仅用 2 个 spawn 并行 v3.39 环境、128 joint environment transitions、64 vector steps、2 次联合更新。每 rollout32 步、PPO2 epochs、agent-time minibatch128、Critic minibatch64；网络宽度/意图数学/持续时间边界不缩小。最终 Policy Adam 的所有已建立参数状态 step=8，Team/Gate Critic 的状态 step=4。

| sampled transitions | 低层有效 agent-time | 新意图事件 | 可学习 gate 事件 | Action loss | Intention loss | Gate loss | Team V loss | Gate Q loss | Inference CE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 256 | 66 | 118 | -0.006231226 | -0.011350684 | -0.036184588 | 0.00027971367 | 0.0003972847 | 1.8050978 |
| 128 | 256 | 60 | 135 | 0.0005545821 | -0.0021756226 | -0.039296038 | 7.1333277e-05 | 0.0008021009 | 1.7832862 |

这些 loss 是实际优化分量的 minibatch 均值，不是伪造或单独随机修改参数的结果。Policy 总 loss 两轮分别为 0.108469181、0.119313102；全部非空数值指标有限。更新前 logp 重算最大误差≤2.38418579e-07，三个概率分量均通过同参数 ratio≈1 审计。Policy、Team、Gate 的梯度均在 optimizer.step 前裁剪至≤.5。

各模块相对初始化的真实 L2 参数漂移：

| 模块 | L2 漂移 |
| --- | ---: |
| encoder | 0.523216665 |
| gru | 0.500461221 |
| action_actor | 0.303660154 |
| team_critic | 0.165632069 |
| gate_actor | 0.611232042 |
| intention_actor | 0.332997352 |
| inference | 0.178597361 |
| gate_critic | 0.275123984 |

128 transitions 的原始环境 reward 总和=0.512000024319，训练尺度团队 reward 总和=0.12800000608，累计 switching cost=0.0294999992475，内部训练 reward 总和=0.0985000059009。数值尺度和成本分开记录。此次训练采样未结束完整训练 episode，不将部分 episode 当成完整回报；完整对局由独立评估提供。

## 6. 保存、加载与完整评估

best/latest/final 都通过配置、架构、参数量、身份、模式和真实 Adam step 审计。重载后 deterministic/stochastic 会话的动作、hidden、意图、age、gate 输出精确一致；专项测试还验证两个重载 checkpoint 的完整 episode 评估复现及评估不消耗训练/全局 RNG。holdout=70,000,000 的完整 episode 已在专项中执行。

| 固定种子10,000,000 | 原始团队 Return（sum） | Episode steps | Red/Blue losses | 意图选择计数 | 主动 / 强制 renew |
| --- | ---: | ---: | --- | --- | --- |
| deterministic | -29.465180908518 | 168 | 4/1 | [23, 0, 0, 0, 0, 0] | 0/23 |
| stochastic | -29.462914430270 | 164 | 4/1 | [30, 23, 19, 20, 21, 21] | 130/4 |

确定性评估明确记录 `['SINGLE_INTENTION_CLASS_IN_EVALUATION', 'NO_LEARNED_RENEW_IN_EVALUATION']`。随机评估仍使用六类意图；不能将短训练的 argmax 集中直接等同于随机策略概率坍缩，也不能把此次 0% 胜率当作机制失效或有效的统计证据。未宣称研究效果已验证。

训练行为 RNG seed domain=`seed ^ 0xB3A001`；entropy=`seed ^ 0xE17001`；permutation=`seed ^ 0xA017`；每个评估 episode 的私有 RNG=`episode_seed ^ 0xAC17E`。环境命中仍使用原环境 RNG，互不复用。

环境 SHA256：`c6c0eb20a17ee359bda53c6c36f6c0a545d67a33929d5b2ea943dc78f615d915`。

生效 smoke 算法 SHA256：`8bac26d227e2fd40bb7ce8a931122ba90555dfb8a3b3f84ecb0229cedfca0fa8`。

GRU/意图/age/previous action 的运行时摘要写入 checkpoint 供审计，全部模型/优化器/算法 RNG/计数可加载。完整并行环境状态没有保存，明确 `formal_exact_resume_supported=false`。warm-start 测试从真实中间 checkpoint64 恢复权重/优化器/计数/RNG，在新回合继续至原预算128，记录新 seed base67；没有声称逐 transition exact resume。

## 7. 录像与可视化

最终重载模型以同一开发集种子执行了 168 步，生成：

- `replay_validated/episode_trace.npz`、`metadata.json`：兼容原 schema3，实际连续动作和环境状态对齐。
- `acirl_decisions.npz`、`acirl_intention_events.json`：每步意图、门控、age、概率、强制/可学习 mask 和真实生命周期。
- `trajectory.png`、`interactive.html`、`all_uav_states.csv`、`fire_events.csv`、`detailed_report.md`：复用原渲染器。已查看 PNG，轨迹、事件和双方状态正常显示。

专项逐步比较普通环境和 RecordingCombatEnv 的观测、reward、执行动作和环境 RNG state，确认记录过程不增加环境随机数抽样或改变动作。独立审计验证 forced/learned 不重叠、有效 gate 只在合法 age 出现、keep 时意图不变、age≤30，以及录制 Return 与 deterministic 评估在浮点累加精度内一致。未验证 MP4 编码器。

## 8. 历史失败和仍有影响的限制

本次未重新运行整个四千余项的全库套件；报告中的 375/1 是上述相关回归集合。额外选择两个已知历史失败，用 `git archive HEAD` 在独立目录、同一 uav 环境复现：

1. `test_factory_and_public_api_are_combat_only` 的全部 YAML 文件名清单陈旧：原 HEAD 已多19个配置，本次新增7个 ACIRL 配置后多26个；没有删除合法配置或修改旧期望清单。
2. `test_no_episodes_null_and_theoretical_count` 的 bare MAPPO 测试夹具缺少 `env_config`，原 HEAD 同样 AttributeError；生产 MAPPO 数学未改。

完整比较在 `known_failure_comparison.json`；没有把历史失败隐藏成全部通过。ACIRL 当前未解决/未验证的边界是：正式多种子学习效果尚未测量；Gate Q 未执行分支是模型估计而非环境反事实真值；全零可观测实体与缺失槽不可区分；不支持 exact environment resume；没有验证 MP4。它们不阻碍已验证的训练/评估/记录闭环，且均没有被伪装成已解决的科研结论。

## 9. 实际执行命令和正式入口

均在 PowerShell 中加载 Conda hook 并激活 Windows uav：

```powershell
Set-Location C:\Users\HPK\Desktop\uav1
& 'C:\Users\HPK\anaconda3\shell\condabin\conda-hook.ps1'
conda activate 'D:\conda_envs\envs_dirs\uav'
python -m compileall -q algorithm/acirl algorithm/train_acirl.py algorithm/evaluate_acirl.py tests/test_acirl.py
python -m pytest tests/test_acirl.py -q
python -m pytest tests/test_acirl.py tests/test_mappo.py tests/test_maddpg.py tests/test_maac.py tests/test_madsac_v341.py tests/test_matd3.py tests/test_combat_v339.py tests/test_combat_replay_pipeline.py tests/test_gaussian_protocol.py -q
python outputs/acirl_v339_validation_20261019/reproduce_known_failures.py
python -u algorithm/train_acirl.py --smoke --output-dir outputs/acirl_v339_validation_20261019/smoke_validated --device cuda --seed 1 --num-envs 2 --total-sampled-steps 128
python -u algorithm/evaluate_acirl.py --checkpoint outputs/acirl_v339_validation_20261019/smoke_validated/final.pt --episodes 1 --mode deterministic --output outputs/acirl_v339_validation_20261019/deterministic_validated.json
python -u algorithm/evaluate_acirl.py --checkpoint outputs/acirl_v339_validation_20261019/smoke_validated/final.pt --episodes 1 --mode stochastic --output outputs/acirl_v339_validation_20261019/stochastic_validated.json
python -u tools/replay_combat.py --checkpoint outputs/acirl_v339_validation_20261019/smoke_validated/final.pt --env-config outputs/acirl_v339_validation_20261019/smoke_validated/env_config.yaml --episode-seed 10000000 --output-dir outputs/acirl_v339_validation_20261019/replay_validated --device cuda --formats trace png html csv report
python outputs/acirl_v339_validation_20261019/audit_release.py
```

上述路径记录本次真实产物；重新训练/录像/解包 baseline 时必须换新的输出目录。正式训练和各消融/评估/holdout/录像完整命令另见 [实现说明](acirl_v339.md)。正式训练核心命令仅提供，**未自动执行**：

```powershell
python -u algorithm/train_acirl.py --env-config configs/combat_environment_v339.yaml --algorithm-config configs/acirl_4v4_v339.yaml --output-dir outputs/acirl_v339_seed1_3m --device cuda --seed 1 --num-envs 16 --total-sampled-steps 3000000
```

已实现：独立算法与实验入口。已测试：上述数学/协议/消融/真实 CUDA 闭环。已验证有效：仅工程可运行性与审计不变量；尚未验证科研性能提升。
