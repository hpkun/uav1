# ACIRL v3.39 实现、数学约定与运行方式

ACIRL 是独立的分层 recurrent on-policy 实现，没有继承 MAPPO Trainer，也没有修改既有算法的数学。环境继续使用原 v3.39 4v4、45D 观测、3D 连续物理动作、0.1s 仿真步长、Blue 固定策略、武器、奖励和终止规则。本实现经过有界 CUDA 验证；研究效果尚未验证，不能据此宣称优于任何基线。实际结果见 [验证报告](acirl_v339_validation.md)。

## 实际检查与集成

实施时检查了 v3.39→v3.38 的观测继承、v3.36 ATA/HA 武器、v3.39 Blue 策略、环境配置冻结规则、MAPPO/RMAPPO 的 GAE、tanh-Gaussian、sequence unroll，以及 vector_env、evaluator、checkpoint 和 canonical recorder 的实际源码。

特别注意：v3.39 的敌方 5D 来自 `env/v338_observation.py`，是距离、速度、AA、ATA、HA；不是更早 v3.30 的机体系方位/俯仰误差。网络只编码这些既有特征，不把角度当全局坐标或新物理动作。

| 文件 | 功能 |
| --- | --- |
| `algorithm/acirl/networks.py` | 实体编码/掩码注意力、GRU、推断头、三个策略头、独立 Team/Gate Critics |
| `algorithm/acirl/state.py` | 每环境每 UAV 的 hidden、previous action、意图、age、reset；约束门控与独立随机采样 |
| `algorithm/acirl/advantages.py` | 团队 GAE、异步 SMDP macro 优势、事件 masked mean、独立 PPO 分量 |
| `algorithm/acirl/rollout.py` | 同步对齐的联合 transition、真实 reset 前下一观测、跨 rollout 状态、宏片段及自然生命周期事件 |
| `algorithm/acirl/trainer.py` | recurrent burn-in/unroll、三个策略 ratio、Critic 回报监督、梯度、checkpoint、状态化执行 |
| `algorithm/acirl/diagnostics.py` | 作战以外的动作/意图/Gate Q/梯度/损失/KL/类别与持续时间统计和告警 |
| `algorithm/acirl/evaluation.py` | 独立会话的 deterministic/stochastic/holdout 评估和严格加载 |
| `algorithm/acirl/protocol.py` | 维度、模式、配置、架构、种子和 checkpoint 契约 |
| `algorithm/acirl/recording.py` | canonical trace 以外的意图/门控 sidecar 和生命周期记录 |
| `algorithm/acirl/factory.py`、`__init__.py` | 独立构造入口和算法包 |
| `algorithm/train_acirl.py`、`evaluate_acirl.py` | 训练和评估 CLI |
| `configs/acirl_4v4_v339*.yaml` | Full 与六种完整消融配置 |
| `tests/test_acirl.py` | 网络、门控、手算优势、信息隔离、序列训练、七种消融、CUDA、warm restart、评估及录像 |
| `tools/record_combat_episode.py` | 唯一修改的既有源文件；增加 ACIRL 加载和可选 sidecar hooks |

复用的是公共 ParallelVectorEnv、统一战斗指标、checkpoint 排序、轨迹/渲染工具。未复用不兼容的旧 PPO 训练过程。新算法当前注册 v3.39 和同布局的 v3.41；其他布局必须显式增加 adapter，模型模块本身通过配置处理维度和数量。

## 完整网络

所有 Actor 参数在 UAV 间共享，各 UAV 的运行状态独立。令实体 embedding=128、GRU=128、意图 K=6、意图 embedding=32。

| 模块 | 默认结构 / 输入输出 | 参数 |
| --- | --- | ---: |
| 自身 encoder | 7→128→128，ReLU | 17,536 |
| 队友 encoder | 每槽 6→128→128，3 槽共享 | 17,408 |
| 敌方 encoder | 每槽 5→128→128，4 槽共享 | 17,280 |
| 注意力 Q/K/V | 128→128，4 heads×32；分别池化队友和敌方 | 49,536 |
| Fusion | [self, ally pool, enemy pool] 384→128→128 | 65,792 |
| Entity encoder 总计 | 45→128，带实体掩码 | 167,552 |
| GRUCell | [embedding128, previous action3] 131→hidden128 | 100,224 |
| 队友意图推断 | [own hidden128, ally embedding128] 256→128→6，每本机每队友分布 | 33,670 |
| 自身意图 / 团队预测 embedding | 各 6×32 | 384 合计 |
| Intention Actor | [hidden128, inferred context32] 160→256→256→6 | 108,550 |
| Gate Actor | [hidden128, context32, own intent32, normalized age1] 193→256→256→1 | 115,713 |
| Continuous Actor | [hidden128, context32, selected own intent32] 192→256→256→6 | 116,742 |
| Policy 总计 | 含全部共享 encoder/GRU/heads/embeddings | 642,835 |
| Team Value | 4×[obs45, hidden128, alive1, previous intent6, age1]=724→256→256→1 | 251,649 |
| Gate Q | [joint724, focal181, agent id4]=909→256→256→2 | 299,266 |
| Critics 总计 | 与 Policy 无参数共享；两个 Critic 也不共享参数 | 550,915 |

总可训练参数 1,193,750。动作头最后 6 维分为 mean3、log_std3；初始 log_std=-1.5，范围 [-5,.5]，mean head gain=.01。所有上述宽度、头数、K、边界和重要训练权重均在 YAML 配置。smoke 不缩小网络。

队友槽是“按 UAV 编号升序移除自身”，敌方槽保持固定编号。零实体槽被判为局部无效；完全没有有效实体时注意力权重和 context 都为零，不对全 mask 的 softmax 产生 NaN。观测没有独立实体 alive bit，因此若一个存活实体的全部可观测特征恰好为零，它与缺失槽不可区分；执行端不引入训练用全局 mask 去消除这个环境信息限制。

推断模块只读取本机 history 和本机观测中的 ally embedding。推断分布乘可学习 embedding，再按本机有效队友 mask 平均得到 team_context。真实队友意图只在 Trainer 的交叉熵标签构造处出现。Actor 的联合 batch 只是并行计算轴，任意本机输出不读取其他 UAV 行的观测、hidden 或真实意图；集中式信息仅供 Critic。

## 一个时间步的真实数据流

1. 从当前本机观测和上一次实际连续动作更新本机 GRU；按 reset/death 清理历史。
2. 推断可观测队友意图分布，聚合本机 team_context。
3. Team Value 和 Gate Q 的状态是当前观测、当前观测编码后的联合 GRU 摘要、门控之前保持的联合意图和 age。
4. 初始意图缺失时强制初始化；age<3 不能普通更新；age≥30 强制更新；其余位置独立采样 Bernoulli gate。
5. 只有 renew 才从 Categorical 重新采样意图。重新采到相同类别也算一个新的决策，age 重置，主动 renew 成本照计。
6. 根据当前 hidden、推断 context、自身当前意图重新产生物理动作。训练使用 tanh-Gaussian 随机动作；deterministic 使用 tanh(mean)。意图 keep 不会重复上一步动作。
7. 并行环境执行三维动作；核对 `executed_red_actions` 与送入的 bounded action 一致。保存 pre-tanh action 和三个旧 log probability。
8. 保存真实 `transition_next_observations`、before/next alive、terminated/truncated、reset、hidden before/after、意图与 age、强制/可学习门控 mask、价值、episode id 和 rollout 边界。随后才使用自动 reset 后的 observation 开始下一个交互。

Rollout 的 `[T,E,A,D]` observations 本身就是对齐的联合观测，Actor 只取各自局部行。sampled_steps 每 vector step 增加 E，不乘 A。自然意图终止事件与为了 PPO 构造的 rollout 截断宏片段分别记录在 `intention_events.jsonl`、`macro_decisions.jsonl`，不会把 censored fragment duration 冒充真实完整意图持续时间。

## 数学约定与优化顺序

原环境奖励不改。默认 `r_team=mean_i(r_i)`，可配置 sum；`r_train=r_team-kappa*mean_i(gate_i*gate_valid_i)`，kappa=.001。初始化、max-duration、固定周期强制更新均不收费。raw 环境 return、训练尺度团队 return、内部训练 return 和成本分别记录；checkpoint 排序始终使用原始环境 return。

低层：`team_gae` 计算 `delta=r_train+gamma*(1-done)*V_next-V`、`A=delta+gamma*lambda*(1-done)*A_next`。团队任务仍继续时单架死亡不终止整个团队价值；该 UAV 的动作 loss 自下一步起由 alive mask 排除。done=terminated OR truncated，环境 timeout 不 bootstrap，与项目约定一致。人为 rollout 截断保留真实 next value bootstrap。

高层：`macro_advantages` 按每个环境/每架 UAV 的真实选择边界扫描，累加 `R_macro=sum gamma^k*r_train`，计算 `delta_macro=R_macro+gamma^tau*V_end-V_start`；对相连的下一真实意图决策递推 `A_high=delta_macro+gamma^tau*lambda_high*A_high_next`。死亡/回合结束时 bootstrap 和递推均停止，但死亡当步 reward 保留。λ_high 本身不取 tau 次方，遵守任务给定公式。

跨 rollout：运行时 hidden、意图、age、previous action 保留。若本 rollout 中的新意图尚未结束，对该旧行为策略样本只累计当前 rollout 内片段并用旧行为价值 bootstrap；下一 rollout 的继续状态不是新选择事件，不再训练这条旧 intention logp，也不会用新策略的数据回填旧 macro。自然生命周期继续独立记录，直到真正 renew/death/end。

每次更新先对固定行为 rollout 构造低层 GAE、团队 λ-return 和高层宏优势，然后：

1. Team Critic 用 λ-return 监督，支持以旧 V 为中心的 value clipping。
2. Gate Critic 只在真实可学习 gate 事件上，用实际被采样分支的团队 λ-return 监督；未执行分支不作为已知标签。两个分支都保留。其估计是函数近似产生的反事实预测，不是环境干预实验的结果。
3. Critic 拟合后冻结其输出，按行为概率计算 `A_gate=Q_chosen-[(1-p_old)*Q_keep+p_old*Q_renew]`。Gate Q 不接入策略反向图。
4. 以连续动作、真实 intention event、可学习 gate event 三个独立索引视图计算三个独立 PPO ratio/clipped objective，分别按有效 action 或事件数量归一化，不把高层优势复制到每个物理步。强制/不可选择门控完全排除。
5. 加入 masked teammate inference CE 和三种策略各自可配置 entropy bonus。动作 entropy 用新重参数化样本的 squashed `-logp` 估计；Categorical/Bernoulli 使用解析 entropy。
6. 共享 Encoder/GRU/三个策略头统一累加梯度、统一 policy optimizer step；Team/Gate Critic 分别使用独立 Adam。所有优化器在 step 前执行 global L2 norm≤.5 裁剪。空 gate/intention event 集也不会推进对应头的 Adam 动量。

动作概率从保存的 raw action 计算，使用稳定修正 `log|tanh'(u)|=2*(log(2)-u-softplus(-2u))`，不通过 inverse tanh 重建采样值。advantage normalization 在三个各自有效集合上分别进行。每个 ratio 都记录 approximate KL 和 clip fraction；任一有效策略分量的 epoch KL 超阈值时提前结束剩余 PPO epochs。

## Recurrent PPO

默认 sequence length32、连续动作 minibatch512 按 agent-time 样本计数；4v4 时每个 joint sequence 有 32×4=128 个 agent-time 位置，每 minibatch 取 4 条 joint sequences。只随机排列完整连续序列，不打乱时间步。

训练每条 chunk 前，从 rollout 起点的已记录 hidden 状态出发，在当前参数下 no-grad 重放该环境从 rollout 起点到 chunk 起点的完整 prefix，作为 burn-in；chunk 内按真实时间顺序做可微 GRU unroll。rollout 起点是明确的 TBPTT 边界，之前的运行记忆保留为条件状态，不宣称对无限过去精确反向传播。padding 同时屏蔽 alive、gate event、intention event。更新前重算三个 logp 审计，最大误差超 2e-4 即失败；过期 behavior_version 的 rollout 拒绝再次训练。

## 七种消融的完整定义

| 模式 / YAML 后缀 | 策略与训练变化 |
| --- | --- |
| full / 无后缀 | 全部机制 |
| no_intent | 移除意图 embedding/head、gate、推断与 Gate Q；保留 Encoder/GRU 连续动作 PPO 和 Team Value；不存在高层/Gate/推断损失 |
| fixed_interval | 移除 Gate Actor 和 Gate Q，默认每 10 步强制重选；保留真实周期边界的 SMDP intention PPO；无可学习 gate ratio、无更新成本 |
| no_gate_q | 移除 Gate Q 及优化器；Gate PPO 使用普通 step GAE，仍只在可选择事件上更新 |
| no_teammate_inference | 移除推断 head、team embedding/context 及 CE；Actor 输入维度相应缩减，不输入真实队友意图 |
| no_temporal_critic | 按题目定义，用选择时刻的普通 step GAE 替换 SMDP high advantage；保留事件 mask，其余 Full 机制不变 |
| no_switching_cost | kappa=0，保留全部网络和其余目标 |

每种配置和模式都写入架构元数据与完整配置 SHA256，跨模式错误加载会拒绝；没有复制七份训练器。

## 评估、checkpoint、恢复和录像

评估使用每回合独立 PolicySession，保存每架 UAV 的 GRU/意图/age，且每步重新计算动作。确定性评估使用 gate sigmoid≥.5、意图 argmax、tanh(mean)；随机评估使用独立 seeded generator。训练行为、训练 entropy、minibatch permutation、评估、环境命中随机数流分离。训练选择固定配置的 evaluation_mode 和 10M 起开发集；70M 起 holdout 必须显式 `--holdout`，不会混入训练 best 排序。

checkpoint 包含全部网络/embedding/优化器、sampled/vector/update/optimizer counters、训练种子、环境及算法快照与指纹、架构及模式/持续协议、行为/entropy/permutation/Torch/CUDA/NumPy/Python RNG、当前运行时 GRU/意图/age 摘要、评估历史与 best 选择。加载同时检查顶层及 recorder 公共指纹、所有模型张量形状/有限性、参数量、算法和模式身份及 Adam step 范围。

没有保存完整并行环境状态；`formal_exact_resume_supported=false`，`resume()` 显式拒绝。`load()` 恢复网络、优化器、计数和算法自有 RNG。`--warm-start` 要求严格匹配配置，并在新输出目录中以新环境回合继续剩余原预算，清空旧回合 hidden/意图和待续生命周期，记录来源和新的环境 seed base。它不承诺原训练逐 transition 一致，不能用它静默更换消融或配置。

canonical recorder 增加可选 ACIRL hooks，原有算法不走这些 hooks。基础 `episode_trace.npz` 和既有 trace schema/渲染器不变；另存 `acirl_decisions.npz`、`acirl_intention_events.json`，记录每步实际内部意图、门控、强制/可学习 mask、持续时间、选择概率和结束原因。统一 trace 保存环境实际执行的三维连续动作。PNG、HTML、CSV、report 复用已有工具；本次没有验证 MP4 编码器。

## 诊断与解释边界

训练日志含统一战斗指标、原始/内部 reward 与成本、每个 episode 的更新次数；优化日志含逐维动作 mean/std/abs/near-zero/saturation/相邻变化、entropy、KL/clip fraction、raw/weighted losses、各策略模块和 Critic 的梯度、宏/低层/Gate 优势分布、Q_keep/Q_renew/Delta、监督误差与分支样本数、推断准确率/预测与标签类别计数、时间加权意图使用率/决策计数、自然持续时间与 censored 宏片段时长、异步更新时间相关矩阵和真实参数漂移。

NaN/Inf loss/gradient 显式失败；坍缩、全部 learned keep/renew、活跃头零梯度产生可见 alerts。评估也单独标记单类别意图和无主动 renew 等观察结果；deterministic argmax 的单类别使用本身不证明 stochastic 分布已坍缩。smoke 不用于检验研究效果。

## 运行命令

```powershell
Set-Location C:\Users\HPK\Desktop\uav1
& 'C:\Users\HPK\anaconda3\shell\condabin\conda-hook.ps1'
conda activate 'D:\conda_envs\envs_dirs\uav'
python -c "import sys,torch; print(sys.executable); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pytest tests/test_acirl.py -q

# 有界验证：128 joint transitions，2 环境，完整网络。
python -u algorithm/train_acirl.py --smoke --output-dir outputs/acirl_v339_smoke_new --device cuda --seed 1 --num-envs 2 --total-sampled-steps 128

# 正式 3M：仅提供命令，本次未启动。
python -u algorithm/train_acirl.py --env-config configs/combat_environment_v339.yaml --algorithm-config configs/acirl_4v4_v339.yaml --output-dir outputs/acirl_v339_seed1_3m --device cuda --seed 1 --num-envs 16 --total-sampled-steps 3000000

# 使用训练目录快照评估；不能给 smoke 模型传入正式 YAML。
python -u algorithm/evaluate_acirl.py --checkpoint outputs/acirl_v339_seed1_3m/best_eval.pt --mode deterministic --episodes 20 --output outputs/acirl_v339_seed1_3m/eval_deterministic.json
python -u algorithm/evaluate_acirl.py --checkpoint outputs/acirl_v339_seed1_3m/best_eval.pt --mode stochastic --episodes 20 --output outputs/acirl_v339_seed1_3m/eval_stochastic.json
python -u algorithm/evaluate_acirl.py --checkpoint outputs/acirl_v339_seed1_3m/best_eval.pt --mode deterministic --holdout --seed-base 70000000 --episodes 20 --output outputs/acirl_v339_seed1_3m/eval_holdout.json

# 一次 rollout；复用 canonical trace、PNG、HTML、CSV、报告导出。
python -u tools/replay_combat.py --checkpoint outputs/acirl_v339_seed1_3m/best_eval.pt --env-config outputs/acirl_v339_seed1_3m/env_config.yaml --episode-seed 10000000 --output-dir outputs/acirl_v339_replay_new --device cuda --formats trace png html csv report

# 消融示例：其余模式按表中后缀替换算法配置，分别使用新的输出目录。
python -u algorithm/train_acirl.py --algorithm-config configs/acirl_4v4_v339_no_gate_q.yaml --env-config configs/combat_environment_v339.yaml --output-dir outputs/acirl_no_gate_q_seed1_3m --device cuda --seed 1
```
