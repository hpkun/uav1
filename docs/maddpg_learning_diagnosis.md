# MADDPG-v3：参考对齐、学习诊断与修正

本次只修改 MADDPG、对应配置/测试/文档。环境固定为 Combat Environment v3.38。短诊断固定 seed=1、16 个并行环境、CUDA、每候选 300,000 environment transitions。没有启动正式 3M/7M 训练。

## 1. 完整参考对照矩阵

见 [30 项源码对照矩阵](maddpg_reference_alignment_audit.md)。其中 Current 是本次修改前的 v2，而 Decision 对应本次 v3。矩阵在修改训练源码前完成，区分 PAPER、OFFICIAL CODE、PROJECT ADAPTATION。

主要来源：[Lowe et al. 2017 正文与附录](https://arxiv.org/html/1706.02275v4)、[OpenAI trainer](https://github.com/openai/maddpg/blob/master/maddpg/trainer/maddpg.py)、[裁剪工具](https://github.com/openai/maddpg/blob/master/maddpg/common/tf_util.py)、[分布参数化](https://github.com/openai/maddpg/blob/master/maddpg/common/distributions.py)。论文附录确实报告了 .01 学习率、.95 gamma、.01 tau 等实验设置，不能将这些都写成“论文没有规定”；它们也不是必须照搬到空战环境的算法定义。

## 2. 真正的实现问题

确认 v2 仅计算梯度 norm，没有执行梯度裁剪。v3 对 Actor/Critic 都补上真实裁剪，这是训练数学变化，版本已从 2 升至 3。

另一个工程问题是 Actor step 结束时无条件恢复 Critic 的 `requires_grad=True`，会覆盖外部有意冻结的 Critic。现在逐参数恢复进入该 step 前的标志，固定 Critic 的学习测试验证了这一点。常规训练中 Critic 原本均可训练，因此不能把它宣称为原训练失败的原因。

没有发现需要改写的 own-action policy gradient、target 网络来源、Polyak 方向或死亡 reward/mask 错误。诊断日志缺口与训练强度不可显式配置的问题也已补齐。

## 3. 超参数差异

项目 lr=1e-4、gamma=.99、tau=.001、256/256 网络、10k warmup、OU 探索、45/3/4 维度与官方实验设置不同。这些归为 PROJECT-ADAPTATION，不能仅凭数值差异叫 bug。A/B 均保持这些值不变，正式默认仍每事件 1 次梯度更新。

## 4. 原来是否缺失梯度裁剪

是。原 v2 可用日志中 Actor 的四个 norm 最大值均低于 .085，未超过 .5；Critic 各 agent 的 norm 超过 .5 的比例约 67.08%、43.45%、51.51%、65.66%。因此缺失裁剪对 Critic 的实际相关性明显更高，但这不是“已证明裁剪解释全部学习失效”。原始统计保存在 `outputs/maddpg_alignment_diagnosis/original_v2_diagnosis.json`。

原 `outputs/maddpg_v2_v338_seed1_3m` 在读取时只有约 1.954M 步日志、最后评估 1.9M，缺少最终 summary/latest/final，因此没有被当作完整 3M 结果；也没有修改或续跑该目录。可用评估胜率 0–10%，best checkpoint 为 800k。

## 5. 最终裁剪方式与实际更新验证

每个 agent 的 Actor/Critic 独立执行：先算全参数 global L2 norm，再 `clip_grad_norm_(..., .5, error_if_nonfinite=True)`，再记录 post-clip norm，然后 optimizer.step。每行保留四个 agent 的 pre/post、是否 clipped，以及有有效样本 agent 中的 clipped fraction。

这是用户指定的 **每 agent 网络 global norm**；OpenAI helper 是 **每个参数张量单独 norm**。阈值相同，裁剪范围不同，文档没有将二者写成完全等价。

超大梯度测试验证真实 SGD 参数差等于 `-lr * clipped_gradient`，更新 L2≈.05（lr=.1、clip=.5）；另用正式 Adam 验证一阶矩输入和实际首步参数差。Adam 的参数步长不等于简单 `lr * norm` 上界，因此不能仅看 Adam 步长来判断是否裁剪。

## 6. Actor regularization

**OFFICIAL-ONLY / NOT PORTED。** 官方 `p_reg` 惩罚 policy distribution 的 flat parameters；Box 分布包含 mean/logstd。当前 Actor 是 deterministic tanh 映射，没有严格等价的 action² 或 weight decay 映射。仍使用 `loss_actor_i=-E[Q_i]`，没有引入默认开启或关闭的额外正则项。

## 7. Actor 数学

保持 `a_i=mu_i(o_i)`，其他动作取 detached replay `a_-i`，符合论文 sampled own-action gradient 与官方 placeholder 替换结构。其他当前 Actor 不参与这条图；Critic 参数冻结且不会在 Actor step 被 optimizer 更新。独立 Actor、独立 centralized scalar Q 的经典结构保留。

## 8. Critic target 与 alive mask

`y_i=r_i+gamma*(1-done)*next_alive_i*Q'_i(o',mu'_1(o'_1),...,mu'_N(o'_N))`。

所有 next actions 来自 target Actors；Q 来自本 agent 的 target Critic；整个 target 无梯度。终止与当前有限回合协议的 truncation 不 bootstrap；本机死亡不 bootstrap。alive-before=1 的死亡 transition 即时 reward 参与学习；死亡后的 slot 不参与该 agent loss。死机观测/动作先归零，terminal next observation 使用 reset 之前的数据。

## 9. Target 更新

`target=(1-tau)*target+tau*online`，方向正确，先 Critic、再 Actor、再 target。tau=.001 不是实现错误，其半衰期约 693 个梯度更新。B 每 100 条 eligible transitions 更新 target 4 次，等效事件权重 .003994；A 为 .001。A/B 提高更新强度时同时提高了 target 在环境时间上的跟随速度，这一混杂必须保留在解释中。

## 10. UTD 与理论次数

UTD 分母是 sampled environment transitions，不是 agent transitions；16 个环境不把 UTD 乘 16。Actor/Critic 的组更新数相同，另外记录每 agent 的真实 optimizer 次数。

| 配置 | g | 名义 Actor/Critic UTD | 300k 理论每类更新组 | 3M 理论每类更新组 | 3M 含 warmup 实际比率 |
|---|---:|---:|---:|---:|---:|
| A/reference | 1 | .01 | 2,900 | 29,900 | .0099667 |
| B/moderate | 4 | .04 | 11,600 | 119,600 | .0398667 |

两者都是 10k warmup、每新增 100 条 eligible transitions 触发一次，保留余数；每次梯度更新都重新采样。3M 数字仅计算，没有启动训练。可用 v2 日志实际为 19,441 / 1,954,112 = .00994876；已完成 MAAC 日志的 Actor/Critic 比率约 .0370253，区别于其名义 .04（12 env runner 的 cadence overshoot）。不将 MAAC 的 target 次数当成 Actor 更新次数。MADDPG 一组包含四个独立 Actor/Critic，MAAC 的共享结构与优化目标不同；这里比较的是更新频率口径，不能据此宣称等计算量或等优化效果。

## 11. 新增诊断与协议

- 每 agent 的有效 alive 数/比例，以及 active-only Q、target 的 mean/std/min/max 和 TD signed mean/abs mean/abs max。
- Actor deterministic heading/pitch/speed 各自 mean/std、|a|≥.95/.99 饱和率，汇总与每 agent 均有。
- 活机动作坐标的 OU RMS/absolute mean、缩放后的 added-noise RMS、actor+noise 被裁到 ±1 的比例。
- 初始化时保存不求导的独立参数 clone；第一及每 100 次梯度更新记录 initial-reference L2 drift、online-target L2 lag，最终 summary 再即时计算一次。
- pre/post clip norms、clip flags/fractions、trigger event、event 内梯度序号、真实 Actor/Critic UTD。

旧 global `target_mean`/`target_abs_max` 仍包含所有 slots；科研比较用新增 active-only per-agent 字段。漂移只能证明参数在变，不能单独证明策略变好。动作分布来自训练访问状态，也受状态分布变化影响。

v3 checkpoint 明确记录 clipping scope/norm、g、interval、nominal/observed UTD、事件/梯度/每 agent 计数、初始参考、配置 hash。拒绝 v1/v2，验证余数与更新算术，不承诺 exact resume（不含 replay/环境完整状态）。

## 12. Critic fixed-batch overfit

CUDA，固定 batch，冻结 Actors，terminal Bellman targets，仅更新 Critic 0；使用生产 Critic optimizer/loss/clip 路径，200 步 MSE **.0806698054 → 5.72329371e-11**。证明优化链可拟合固定监督目标，不等于证明不断变化的真实 Bellman 学习已解决。

## 13. Actor frozen-critic improvement

CUDA，固定 observations/replay a_-i 和已知冻结 Q，仅训练 Actor 1，100 步平均 Q **.082749173 → .994375706**。其他三个 Actor 参数逐项完全不变，Critic 参数/冻结状态不变。证明梯度方向与 own-action 计算图正确，不等于真实环境策略收益保证。

## 14. 单元与回归检查

针对 MADDPG 的测试 **58 passed**，含两个学习测试、真实裁剪参数差、Adam、梯度隔离、death/mask、schedule/remainder/fresh minibatch 和 checkpoint rejection。全仓回归 **3529 passed / 41 failed / 4 skipped**。41 项失败均属于已有基线；切回本次修改前的 HEAD MADDPG-v2 模块单独重跑这 41 项，仍为 **41 failed**。没有新增失败项，不能将整仓说成全部通过。失败涉及历史轨迹快照、非 MADDPG 协议与旧 checkpoint 兼容测试，本次未修改对应环境/算法来消除它们。原始日志与失败项清单保存在 `outputs/maddpg_alignment_diagnosis/`。

## 15. CUDA smoke

**PASS**：RTX 5060 Laptop GPU，8,192 transitions / 512 vector steps，Actor/Critic 各 80 组更新，每 agent 均 80，remainder=64，14 个完成回合。所有网络在 CUDA、所有日志有限、online/target checkpoint roundtrip 精确一致，reload 后确定性评估一致。见 `outputs/maddpg_alignment_diagnosis/smoke_validation.json`。

## 16. 300k A/B 实测结果

两组均完成 300,000 transitions / 18,750 vector steps / 2,900 trigger events，余数为 0。A 每个 Actor/Critic 实际更新 2,900 次，B 均为 11,600 次，没有全死 minibatch 导致的实际 optimizer skip。实际 Actor/Critic UTD 分别为 .00966667 和 .03866667。

配置逐项复核：两组只在 g 和不参与训练的实验标签上不同。CUDA checkpoint 审计确认初始四个 Actor、四个 Critic 参数逐项完全相同。两组均在 0/50k/100k/150k/200k/250k/300k 上用相同 20 个 development seeds 评估，训练源码 SHA256 全程一致；最终 v3 checkpoint contract 检查通过。

| 项目 | A：g=1 | B：g=4 |
|---|---:|---:|
| 初始确定性回报 | -38.1177 | -38.1177 |
| 最佳回报 | **-32.0769（200k）** | -37.8494（250k） |
| 最终回报 | **-36.5125** | -38.3748 |
| 最终减初始 | +1.6052 | -0.2571 |
| 最终评估胜率 | 0/20 | 0/20 |
| 最终平均 Red 损失 | 4.00 | 4.00 |
| 最终平均 Blue 损失 | .40 | .20 |
| 最终 fire / kill episode rate | 75% / 30% | 40% / 15% |

A 有阶段性改善，随后回落；B 的四倍更新没有带来更好的任务回报。不能用 B 更低的 Critic loss 将其选为更优配置。A 运行时并行执行过全仓测试，运行耗时不是严格受控的吞吐率对照。

条件 C **未启动**：A 已出现阶段性收益/任务改善，不符合“A、B 都完全不学习”后才运行 C 的条件。没有同时提高 lr/tau，也没有启动正式 3M/7M。

## 17. Q、TD、梯度、饱和、drift/lag 对照

早期窗口为 (10k,60k]，末段为 (250k,300k]，按相同 transitions 比较。下表中的多 agent 指标先按每次更新统计，再在窗口/agent 间平均；不是独立同分布样本的显著性检验。

| 指标 | A 早期 | A 末段 | B 早期 | B 末段 |
|---|---:|---:|---:|---:|
| Critic MSE | .40429 | .38059 | .37729 | **.28712** |
| mean absolute TD | .09104 | **.10243** | .10807 | .13439 |
| Q mean | -.04951 | -.15268 | -.09431 | -.20905 |
| mean within-batch Q std | .07581 | .19656 | .15587 | .72898 |
| Actor grad pre | .00812 | .01005 | .01118 | .06847 |
| Critic grad pre | .13494 | .32406 | .30789 | 1.63630 |
| Critic clipped fraction | .10% | 9.10% | 13.88% | **99.58%** |
| heading sat99 | 7.03% | 33.19% | **45.64%** | 26.94% |
| pitch sat99 | 18.22% | 45.15% | 42.56% | 33.24% |
| speed sat99 | 17.59% | 32.34% | 38.75% | 30.91% |

MSE 与 absolute TD 不是同一统计量；两组目标与访问状态分布也在变化。B 的 MSE 下降、Q 分布变宽，表明 Critic 并非完全不更新，但 absolute TD 没有下降，策略回报也没有改善，不能宣称 Bellman 学习已稳定。Q 分布变宽本身不能证明校准正确或过估计。

两组 Actor 梯度的窗口均值均非零，全程未触发 .5 裁剪；各 agent 全程最大 norm，A 为 .018–.040，B 为 .123–.164。B 的 Critic 末段 pre norm 各 agent 约 1.28–2.17，post 均约 .499–.500；实际所有 post norm 均不超过 .500002。裁剪真实生效，B 的高梯度/裁剪比例仍显示较大优化压力。

B 更早出现动作饱和，末段有所回落，不能把饱和描述成持续单调恶化；A 的 pitch 末段仍有约 45% 输出 |a|≥.99。没有为此添加 action² penalty 或 weight decay。

| 最终参数诊断（四 agent 均值） | A | B |
|---|---:|---:|
| Actor initial-reference L2 drift | 10.2952 | 17.1285 |
| Critic initial-reference L2 drift | 4.6118 | 10.6201 |
| Actor online-target L2 lag | 3.1936 | 1.4113 |
| Critic online-target L2 lag | 1.2417 | .7004 |

B 更新更多、参数漂移更大，target 更接近 online，仍没有带来收益提升。tau=.001 下 target 确实在跟随，没有发现 Polyak 方向错误；这些距离不能单独证明 tau 已最优，或证明 target lag 是学习失败的根因。

有效 batch 充足：A 末段各 agent 平均有效样本约 600/751/743/814；B 为 670/602/814/747，均相对 batch=1024 报告。全程每 agent 最小值，A 为 548/696/630/669，B 为 550/541/616/688，没有样本耗尽或大量跳过训练的证据。

末段 OU RMS 为 A .3782、B .3773；有效 added-noise RMS 为 .3470/.3461，行为动作被裁到 ±1 的坐标比例为 31.42%/28.13%。探索机制有实际作用且存在明显边界裁剪，未发现噪声没有加入；两组没有更改 OU/decay。该比例不是 deterministic actor saturation。

完整每 agent 的 Q/target mean/std/min/max、TD signed/absolute mean/max、pre/post gradients、动作 mean/std/sat95/sat99，见 [量化附录](../outputs/maddpg_alignment_diagnosis/quantitative_appendix.md)、[机器可读统计](../outputs/maddpg_alignment_diagnosis/short_diagnostics_analysis.json) 与 [per-agent 明细](../outputs/maddpg_alignment_diagnosis/per_agent_diagnostics.csv)。其中附录 min/max 是 minibatch 极值的窗口平均，原始逐更新日志保留真实各 batch 极值。

![同预算学习曲线](../outputs/maddpg_alignment_diagnosis/diagnostic_learning_curves.png)

## 18. 哪套配置显示出学习迹象

**A 有阶段性但不稳定的实际策略改善，B 没有可信的持续回报改善。** A 初始到 200k 回报提升约 6.04，Blue 损失 .15→.85，kill episode rate 15%→60%；300k 后回落为 -36.51，Red 仍全损。B 的主要证据是价值网络/参数变化，不等同于更好的策略。现有结果支持继续保留 g=1 的 reference 默认，不能把 g=4 宣称为已校准的更优 baseline。

原 v2 也并非严格“完全不学习”：其 100k/200k/400k 回报约 -37.64/-35.05/-31.99，随后又回落。它是历史背景，没有作为第三个同预算受控组。因此没有证明缺失裁剪是全部根因，也没有证明补上裁剪已解决任务性能。

只运行了一个训练 seed，20 个重复使用的 development seeds 用于诊断与选择，不是独立最终测试；不作统计显著性或跨 seed 稳健性结论。

## 19. 剩余实现层疑点与限制

在本次源码对照、梯度隔离、mask/terminal、target、真实参数更新、学习测试与 CUDA 短训练覆盖内，没有确认新的核心 MADDPG 计算图错误。不能将测试通过当成算法在该环境必然成功的证明。

仍待解释的是：移动 Bellman target 下 absolute TD 没有下降、Actor 输出较早饱和、更多优化未改善确定性回报，以及阶段性收益的回落。这些属于后续优化/超参数校准问题；本次没有通过隔离实验确定 lr、tau、gamma 或探索衰减中哪个是根因，也没有据此修改它们。没有把问题归咎于环境不适合 MADDPG 或算法本身。

环境和 MAPPO/MAAC/MADSAC 的 161 个受保护文件 SHA256 保持一致，整个 `git diff` 没有这些算法或环境的改动。原 v2 输出保持只读。

## 20. 是否建议直接正式训练

**目前不建议直接启动正式 3M 或 7M，尤其不建议把 B 直接放大到多百万步。** A 的阶段性信号不足以证明稳定改进，B 的四倍更新在本次对照中未显示收益，两个最终评估仍 Red 全损、胜率 0/20。保留 v3/g=1 作为可审计的 reference；后续应先通过单因素短校准与重复性检查获得更稳定的回报/TD/动作证据，再决定正式训练配置。本次不额外启动这些实验。

这次已完成的是参考审计、两项实现修正、日志/协议补齐及完整同预算 A/B；任务性能不稳定这一现象尚未被证明解决。

## 复查入口

- [正式 v3 使用与协议说明](maddpg_v338.md)
- [逐项参考矩阵](maddpg_reference_alignment_audit.md)
- [两组全部评估](../outputs/maddpg_alignment_diagnosis/diagnostic_evaluations.csv)
- [保留性与回归复核](../outputs/maddpg_alignment_diagnosis/preservation_and_regression.json)
- [CUDA smoke](../outputs/maddpg_alignment_diagnosis/smoke_validation.json)
- 短诊断配置、原始 training/optimization/evaluation 日志及 final/best checkpoint：`outputs/maddpg_alignment_diagnosis/A_reference/`、`B_moderate/`；对应 A/B YAML 位于其父目录。

运行环境：WSL 枚举返回 E_ACCESSDENIED，按仓库运行规则回退至 PowerShell `conda activate uav`；训练与 checkpoint 审计均使用 CUDA，没有 CPU 回退。
