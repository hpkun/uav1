# EAA-MAPPO v1：固定 v3.38 的实体/智能体注意力消融

研究问题：显式实体关系建模和集中式智能体协同关系建模是否能够改善固定 4v4 空战中的多无人机协同决策。

Whether explicit entity-level relevance modeling in the decentralized policy and agent-level coordination modeling in the centralized value function improve cooperative decision making in fixed-scale 4v4 UAV air combat.

本次完成结构、协议、诊断与 CUDA 闭环，没有进行性能对比训练，因此不声称 attention 已提高胜率或回报。没有训练 3M、多 seed，也没有创建新环境版本。

## 参考思想与本项目适配

[Tan et al. 2026, BRMA-MAPPO](https://doi.org/10.1093/jcde/qwag014) 的直接参考思想是实体特征映射、multi-head scaled dot-product attention、自身实体输出聚合战场表示、死亡实体屏蔽。该文还包含随机掩码与时序模块；本次只采用空间实体表示思想，不是完整复现 BRMA-MAPPO。

项目适配包括 v3.38 的 self7/ally6/enemy5 三类异维编码器、固定 8 tokens、embedding64/4 actor heads、无位置编码、复用现有集中式 agent-attention critic，以及冻结当前 MAPPO PPO 协议。不实现 biased random mask、mask generator、Gumbel-Softmax、信息瓶颈 mask loss、Top-M、OOD dropout、规模泛化、GRU/LSTM、时序注意力或 Transformer block 堆栈。

[Liu et al. 2022](https://doi.org/10.3390/aerospace9100563) 仅作为 multi-UCAV MAPPO 与 CTDE 的基础参考，不将 attention 思想归于该文。本文引用范围不包含对原文实验环境、数值超参数或性能结论的复现。

## 审计与四种结构

修改前检查了 MAPPO Actor、两个 Critic、Trainer/GAE/PPO、runner、训练入口、checkpoint/evaluator/common 协议和 v338 observation 源码。当前正式 baseline 为 45D observation、3D action、4 个学习 Red，`SharedMAPPOActor + CentralizedMLPCritic`，实现版本 2。现有 attention critic 的按活机 key 屏蔽、排除自身、无其他活机时零 context 的语义可直接复用；没有确认需要改动 baseline 的核心算法错误。

| 方法 | Actor | Critic | Actor 参数 | Critic 参数 | 总参数 |
|---|---|---|---:|---:|---:|
| MAPPO | SharedMAPPOActor | CentralizedMLPCritic | 78,342 | 123,905 | 202,247 |
| MAPPO-CA | SharedMAPPOActor | CentralizedValueCritic | 78,342 | 472,321 | 550,663 |
| EA-MAPPO | EntityAttentionMAPPOActor | CentralizedMLPCritic | 113,414 | 123,905 | 237,319 |
| EAA-MAPPO | EntityAttentionMAPPOActor | CentralizedValueCritic | 113,414 | 472,321 | 585,735 |

表中均为一份共享网络的 trainable 参数，不乘 4。Actor 由四架 Red 共享；Critic 也为现有共享 per-focal value 网络。保留 baseline hidden256，没有为了等参数强行缩小网络。性能分析必须同时报告容量差异，不能把可能收益全部归因于 attention。

历史 `algorithm/ea_mappo/` 的 5v5/8v8、双头实体结构与此次不同，保持原样。本次 EA 4v4 使用独立 `algorithm/eaa_mappo/` 和 checkpoint family，旧 EA 权重不能静默迁移到新结构。

## 实体解析、attention 与策略

`split_v338_entities(Tensor)` 支持任意 leading dimensions，唯一解析入口为：

| 输入区域 | 输出 |
|---|---|
| self [0:7] | `[...,7]` |
| allies [7:25] | `[...,3,6]`，各 slot 长度 6 |
| enemies [25:45] | `[...,4,5]`，各 slot 长度 5 |

enemy 原有 distance、velocity、AA、ATA、HA 的顺序与归一化完全不变。解析只拆 Tensor，不改 observation。

三类映射分别为 Linear(7,64)+ReLU、Linear(6,64)+ReLU、Linear(5,64)+ReLU。同类所有 slots 共用一个 encoder，没有独立 slot 参数或 positional embedding。拼接 `[self,ally1..3,enemy1..4]` 得到 `[...,8,64]`。

Q/K/V 各由 Linear(64,64) 产生，各 head 为 `[...,4,8,16]`：

`score_h = Q_h @ K_h^T / sqrt(16)`

`A_h = softmax(score_h.masked_fill(~valid_key, -inf))`

`C = concat_h(A_h @ V_h)`，形状 `[...,8,64]`。没有 output projection、dropout、residual 或 LayerNorm。

ally/enemy 的 validity 为原始 slot 中 `any(abs(feature)>1e-8)`；self 恒 valid，死 agent 的最终动作沿用 Trainer 的 alive mask 清零。无效 key 在 softmax 前屏蔽，不能通过 Linear bias 重新产生非零权重。只剩 self 时其权重严格为 1，无 all-masked NaN。无效 query 的输出不供策略使用，策略只读取 self query。

取 `context_self=C[...,0,:]`，与原始 `self_embedding=X[...,0,:]` 拼接成 128D。随后为 `128→256 ReLU→256 ReLU`，接 baseline 相同的 Gaussian mean head 和 state-independent log_std。`sample`、`evaluate_actions`、`_squashed_log_prob`、entropy Monte Carlo、latent raw action、tanh、log_std 初始化与 clamp 都直接继承 `SharedMAPPOActor`，没有复制/改写分布公式。

`return_attention=True` 可用于 Actor.forward/distribution/representation，返回 self-query weights `[...,4,8]`。`entity_attention(tokens,valid)` 还可直接返回完整 `[...,4,8,8]` 权重和全部 context，便于数学测试。

同类实体交换使 token/attention 对应置换，而 self representation、policy mean/action 保持等价。测试同时覆盖两个存活 slot 交换，以及存活/死亡 slot 交换，不只测试全活的理想输入。

## Critic 与 CTDE

CA/EAA 原样复用 `CentralizedValueCritic(45,256,2)`。输入 `[B,4,45]`，每个 Red observation 经共享 `45→256→256` embedding；Q/K/V 为 256D、2 heads、每 head128D。每个 focal query 只关注其他活 Red，focal embedding 与 context 拼成512D，经 `512→256→256→1` 输出每 agent 的 V。

死 Red 不作为 key/value，其 value 和 query 权重按现有 alive contract 清零。若仅剩一架 Red，因排除 self 没有其他 key，现有实现显式归零并安全重归一化，attention 全零、context 全零、value 有限；此时不要求空行权重和为 1。至少有一个其他活 Red 的有效 query，其权重和才为 1。

Actor 只读自身 45D local observation；Critic 读取四个 Red 的 observation。Blue 信息已在 local observation 中，不额外添加 Blue agent tokens。集中式 Critic 不参与执行动作生成。

## 冻结的 PPO 与 baseline 保护

正式三份配置的 training、implementation、runtime_logging、environment 与当前 `mappo_4v4_v338.yaml` 逐项相等。仅 algorithm 标签和 network 结构不同，validate_config 拒绝不符合消融设计的配置。

冻结项包括 gamma=.99、lambda=.95、clip=.2、Actor/Critic lr=3e-4、value coefficient=.5、entropy=.001、rollout256、epochs10、minibatch512、envs16、3M 配置预算、target KL=.015、grad clip=.5、state-independent std、log_std_init=-.5、mean gain=.01、log_std clamp[-5,.5]。当前 baseline 默认 seed=0，未机械改成其他值；CLI 可显式指定实验 seed。

GAE、死亡 continuation、advantage normalization、clipped surrogate、value clipping、bounded entropy、KL early stop、Adam、梯度裁剪全部由原 `MAPPOTrainer.update` 执行。rollout 收集、terminal next observation、done/truncated、评估种子与 best selection 复用原 runner/common evaluator。

原 `networks.py`、`trainer.py`、`factory.py`、`evaluation.py`、common 模块、正式 baseline YAML 与 implementation version=2 未变。现有 MAPPO 代码仅有两个小接口/日志改动：runner 增加默认关闭的构造器注入，并在 START 显示参数量；原训练入口只对显式 `algorithm: MAPPO-CA` 路由新协议。`algorithm: MAPPO` 仍走原构造器、参数和 checkpoint 路径，不能声称整个 runner 文件字节零修改。

与修改前源码快照的 CUDA 比较确认 baseline 初始参数、固定 RNG 的 actions/raw actions/log_prob、完整 PPO update 指标和 update 后 Actor/Critic 参数逐位相同。已有 `outputs/mappo_v338_seed1_3m/best_eval.pt` 继续在 CUDA 上加载，文件 hash 未变。

## 诊断与 trace

训练更新后用 no-grad、不采样的诊断，不参与 loss 或 optimizer，不改变 Torch RNG。

Actor 机器字段：`actor_attention_self_mass`、`actor_attention_ally_mass`、`actor_attention_enemy_mass`、`actor_attention_entropy`，以及 `actor_attention_head_{0..3}_{self|ally|enemy}_mass` 和 `actor_attention_head_{0..3}_entropy`。mass 按存活 focal agent 与 heads 聚合，三类总和为1；entropy 使用自然对数。控制台只追加 `Attn S/A/E=.xx/.xx/.xx`。

Critic 字段：`critic_attention_entropy`、`critic_attention_max_weight`、`critic_attention_valid_queries`。只统计至少还有一名活队友的 live query；没有有效 query 时为0，不把空 context 的零行当作概率分布。CA 没有 Actor entity 字段，EA 没有 Critic attention 字段。诊断写入 optimization 与 training JSONL，不打印完整矩阵。

`tools/analyze_eaa_attention.py` 运行一个 deterministic episode，输出 `attention_trace.csv`、`attention_trace.json`。CSV 每行是 action 执行前的 step/time/alive Red agent、head-average self/ally/enemy mass、head-average 最强 entity slot/weight。JSON 另保存每 head 的8槽权重，直接作为 heatmap 数据。它只用于 EA/EAA，CA 的 MLP Actor 没有 entity attention。

v3.38 ally slots 按 Red 数组索引排除 focal 自身后排列；enemy slots 按 opponent 数组索引排列，死亡不压缩 slots。它们不是“永久战术身份”。工具保留 observation slot 语义，不宣称“网络一直关注 Blue-1”。权重也不自动构成因果解释。

实际工具检查：EAA CUDA smoke checkpoint，seed=10,000,000，一局172步、655条 alive-Red 记录，CSV/JSON/各head和三类mass均 finite 且归一化。该短 smoke 模型未经过性能训练，这局结果不能代表 EAA 效果。

## Checkpoint

三种新消融使用 `checkpoint_family=eaa_mappo_v338`、独立 implementation_version=1 和准确的 algorithm/algorithm_variant。记录环境版本/hash、算法配置/hash、完整45D observation layout、actor/critic type、embedding64/4heads/head16、critic heads、hidden layers、PPO 配置与 Gaussian 协议、参数量、优化器、更新/采样计数。runner extra 另记录实际 seed/env count/训练目标及 smoke 状态；配置中的正式预算不等于本次实际运行预算。

load 在改动网络前验证根与 extra 的身份/配置/结构一致性，再 strict load_state_dict。错误 variant、baseline、历史 EA、错误配置/hash/version 均拒绝；反向用 baseline loader 载入新消融也拒绝。模型评估/加载与 roundtrip 已支持；v1 未实现 training resume，不暗示精确环境/RNG恢复。

## 使用入口

后续运行入口示例（本次没有执行正式预算）：

```powershell
python -u algorithm/train_mappo.py --env-config configs/combat_environment_v338.yaml --algorithm-config configs/mappo_ca_4v4_v338.yaml --device cuda --seed 1 --output-dir outputs/mappo_ca_v338_seed1_3m
python -u algorithm/train_eaa_mappo.py --env-config configs/combat_environment_v338.yaml --algorithm-config configs/ea_mappo_4v4_v338.yaml --device cuda --seed 1 --output-dir outputs/ea_mappo_v338_seed1_3m
python -u algorithm/train_eaa_mappo.py --env-config configs/combat_environment_v338.yaml --algorithm-config configs/eaa_mappo_4v4_v338.yaml --device cuda --seed 1 --output-dir outputs/eaa_mappo_v338_seed1_3m
```

已实际运行的短验证：

```powershell
python -m pytest tests/test_eaa_mappo.py tests/test_mappo.py -q
python -u tools/validate_eaa_mappo_smoke.py
python -u tools/analyze_eaa_attention.py --checkpoint outputs/eaa_mappo_validation/eaa_mappo/latest.pt --env-config configs/combat_environment_v338.yaml --episode-seed 10000000 --output-dir outputs/eaa_mappo_validation/attention_trace --device cuda
```

smoke 的正式网络、256-step rollout、epochs10、minibatch512 等 PPO 参数保持不变；通过 CLI 把实际并行环境降为2、实际总预算降为512，新方法只评估1个固定 seed，并将报告/存盘间隔设到结束点。baseline 用原正式入口的512步短预算；因其原日程不会在512步评估，验证工具单独执行 deterministic evaluation。原 MAPPO `--smoke` 的历史缩小网络语义没有被修改。

## 文件范围

新增 `algorithm/eaa_mappo/{__init__,networks,trainer,factory,protocol,runner,evaluation,entry}.py`、`algorithm/train_eaa_mappo.py`，三份 `*_4v4_v338.yaml` 消融配置，`tests/test_eaa_mappo.py`，`tools/analyze_eaa_attention.py`、`tools/validate_eaa_mappo_smoke.py` 与本文档。

修改 `algorithm/mappo/runner.py` 的构造接口和参数量日志，以及 `algorithm/train_mappo.py` 的 CA 路由；`tests/test_combat_protocol.py` 的严格配置白名单显式加入三份新 YAML，没有放宽清单断言。没有改动 env/、v338 环境 YAML、原正式 MAPPO YAML、MAPPO 核心网络/优化器数学、历史 EA/其他算法或 common 协议。

## 36项验收结果

| 项 | 结果 |
|---|---|
| 1 文件修改/新增 | 见上节，独立 eaa package；baseline 仅构造/日志/显式CA路由 |
| 2 环境零修改 | 对照修改前 SHA256 保留性快照，最终检查另见 validation report |
| 3 baseline 训练语义 | CUDA 修改前/后 bitwise 参数、采样和 PPO update 比较通过 |
| 4 MAPPO-CA | 原 MLP Actor + 原 CentralizedValueCritic |
| 5 EA-MAPPO | 新 EntityAttentionMAPPOActor + 原 MLP Critic |
| 6 EAA-MAPPO | 新 Entity Actor + 原 Agent Attention Critic |
| 7 45D拆分 | 7 + 3×6 + 4×5，单一显式 parser |
| 8 encoder | 三类 Linear→64→ReLU |
| 9 同类共享 | 不依赖 slot 编号，支持同类置换等价性 |
| 10 positional embedding | 无 |
| 11 Q/K/V | 每个 token 64D；每 head16D |
| 12 heads | Actor 4；Critic 2，未随 Actor 改动 |
| 13 head dim | Actor16；Critic128 |
| 14 scaled dot product | QKᵀ/√16，独立手算测试通过 |
| 15 dead mask | raw slot validity，softmax前 -inf key mask；bias 非零测试通过 |
| 16 all-masked | Actor self保底权重1；Critic无队友时零行/零context，finite |
| 17 self 聚合 | 仅 C[...,0,:] 进入 policy |
| 18 fusion | self64 + context64 = 128 |
| 19 Gaussian | 所有采样/log-prob/entropy函数直接继承 baseline；latent roundtrip通过 |
| 20 PPO冻结 | 原 Trainer.update/GAE，共同config逐项比较通过 |
| 21 Critic复用 | 同一个现有 class，没有另写 critic |
| 22 构造测试 | baseline/CA/EA/EAA 四种 exact class 组合通过 |
| 23 ally permutation | 两活slot、活/死交换，representation不变/weights对应置换通过 |
| 24 enemy permutation | 两活slot、活/死交换同样通过 |
| 25 dead attention | dead ally/enemy权重严格0；self-only=1；finite通过 |
| 26 Actor diagnostics | S/A/E mass、entropy及per-head，日志验证通过 |
| 27 Critic diagnostics | entropy/max_weight/valid_queries，日志验证通过 |
| 28 参数量 | 四种方法见结构表；START与metadata均记录 |
| 29 checkpoint | 独立family/v1、variant/type/layout/PPO/hash；双向错误加载拒绝 |
| 30 trace工具 | 已运行172步/655行；CSV/JSON与heatmap数据有效 |
| 31 CPU专项 | 见下方最终测试记录 |
| 32 CUDA smoke | 四种512步/完整256网络/PPO，各10个Actor与Critic更新；保存/加载/评估通过 |
| 33 全量回归 | 见下方最终测试记录，不将历史失败说成全绿 |
| 34 GRU | 未加入 |
| 35 biased random mask | 未加入 |
| 36 正式3M | 未启动；四结构总计仅2048条训练transition |

## 最终测试记录

最终结果：

- 新增 attention CPU 专项 **44 passed**；CUDA 专项 **3 passed**，覆盖三种消融 checkpoint roundtrip、反向 baseline 拒绝加载，以及 EA/EAA self/ally/enemy encoders、WQ/WK/WV、policy backbone 的 finite 且非零梯度。
- attention 与既有 MAPPO 联合专项 **60 passed**。
- 最终全仓回归 **3576 passed / 41 failed / 4 skipped**。41 个失败项与已有基线完全一致；用本次修改前的 MAPPO 源码快照再次单独重跑，仍为 **41 failed**，没有新增失败。失败主要涉及历史 trajectory 快照和旧模型/协议预期；本次没有修改这些环境/其他算法来消除它们。
- 四种 CUDA smoke 全部通过：每种512 transitions、完整256-width网络和正式PPO参数、Actor/Critic各10次更新、GAE/采样/latent log-prob/PPO/确定性评估/保存加载均覆盖。没有正式3M或多seed实验。
- baseline 修改前后 CUDA 参数、采样和 PPO update **bitwise equal**；原正式 best checkpoint 可继续加载且 SHA256 未变。
- 208 项文件快照中，仅两处预期的 baseline 构造/日志/路由接口文件变化，其余206项保持原样。`env/`、v338环境YAML、baseline正式YAML、核心网络/PPO/factory/evaluator、common及其他算法源码均保持原样。AST与 `git diff --check` 通过。
- trace 实际运行172步，655条 alive-Red 记录，所有 head 权重、三类 mass 归一化检查通过。

完整证据：[最终审计 JSON](../outputs/eaa_mappo_validation/final_audit_report.json)、[CUDA smoke与逐位兼容报告](../outputs/eaa_mappo_validation/cuda_smoke_report.json)、[最终全仓日志](../outputs/eaa_mappo_validation/regression_final.log)、[attention CSV](../outputs/eaa_mappo_validation/attention_trace/attention_trace.csv)、[JSON/heatmap 数据](../outputs/eaa_mappo_validation/attention_trace/attention_trace.json)。

运行环境为 PowerShell `conda activate uav`、PyTorch 2.11.0+cu128、RTX5060 Laptop CUDA。WSL 枚举返回 E_ACCESSDENIED，依据仓库规则使用 Windows uav 回退；没有 CPU 训练或 checkpoint/evaluation 审计回退。CPU 仅用于专项单元测试。
