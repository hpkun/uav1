# v3.44 / v3.45 连续驻留对照实验

实施日期：2026-10-22。结果目录：`outputs/v344_v345_validation_20261022/`。本轮仅为离线抽象点质点仿真实现与验证，没有启动正式 3M、7M 或 500K 训练，没有覆盖历史模型、录像或实验输出。

## 论文依据与本轮边界

直接读取项目目录 Xu 等（2026）作者接受稿 PDF，共15页，DOI 10.1109/TAES.2026.3734384，并检查第3、4、8页图像。

- 第3页 II-B / Figure 2：区域为沿瞬时速度方向的前向球面封顶锥体；区域内持续驻留规定时间后目标失效。
- 第4页式(4)：目标姿态条件是攻击优势指标的一部分。本轮没有把该指标的目标姿态约束附加为失效条件。
- 第8页 IV-A1：2000 m、45°、0.5 s；原文 dt=.25 s，描述两步。当前实验沿用 dt=.1 s，因此默认五次有效采样。

本轮只借用区域/驻留机制及上述区域参数，不是 Xu 全系统复现。不接入 APH/Hungarian、15动作 Blue、Xu Reward、Beta 网络或 Xu 三自由度微分方程。旧场景、4v4、NED 点质点增量运动、固定 Blue、PPO、终止规则都沿用 v3.43。第4页的相关目标分配文字不改变本轮既有目标选择器。

## 精确版本差异

| 版本 | 事件区域 | 抽象事件规则 | Reward | Actor观测 |
|---|---|---|---|---|
| v3.43 | 0–2000 m，三维半角30° | 原共享高斯噪声概率命中、6次资源、1秒冷却 | 原 R1/R2/PBRS R3/连续R4 | 49D |
| v3.44 | 同一0–2000 m、三维半角30° | 连续0.5秒采样达到阈值，确定性失效；无资源/冷却限制 | 完全继承 v3.43 | 同一49D |
| v3.45 | 仅半角改成45° | 完全继承 v3.44 | 同上 | 同上 |

v3.44 相对 v3.43 改的是**整个事件机制**，包含按要求去除旧资源/冷却制约；不是在仍受6次额度/1秒冷却限制下仅替换随机函数。v3.44/v3.45 的唯一机制参数差异是半角；版本身份也不同。

两版保持 dt=.1、max_steps=1000、原初始分布、原 Blue 及动作空间、原 arena、R1/R2/R3/R4 所有系数、gamma=.99、GAE lambda=.95；MAPPO/IPPO YAML 相对 v3.43 仅环境身份变化，网络维度仍49/3/4。

`DwellEnvelope` 的配置只包含 `range_min=0`、`range_max=2000`、`off_boresight_max`。不含旧概率尺度或6次额度。父构造器所需的兼容资源分配为零、cooldown_seconds=0，并以 `legacy_ammo_cooldown: disabled` 明示不适用。资格函数不读取 ammo，事件钩子不读取冷却数组；即使测试把资源置零、冷却置999也不阻断驻留。

新事件不调用任何概率命中函数，不消耗命中随机数。`attempt_hit_for_pair` 在新版本直接报错，用于阻止意外进入旧路径。reset 场景随机性和 Blue 的既有正常随机语义不变。

## 几何与离散驻留时序

直接复用 `engagement_geometry` 的 `distance/off_boresight`，包含航向和俯仰。保留 NED forward=`[cos(theta)cos(psi), cos(theta)sin(psi), -sin(theta)]`，距离0沿用 alpha=0。边界沿用 `angle_within` 的1e-12角度舍入容差。没有添加100 m最小射程、独立ATA/HA门限或目标姿态门限；非有限飞行状态明确拒绝，不将 NaN/Inf 返回训练张量。

时间来源是 `fire_control.dwell_seconds`。所需样本数为 `ceil(dwell_seconds/dt)`，以十进制字符串做比例运算以避免 .3/.1 的浮点进位伪差；例如 .5/.1=5、.5/.25=2、.5/.2=3、.51/.1=6。这个离散化、inclusive边界和采样解释是本实验明确的工程选择。

每一步仍按原顺序执行：

1. step开始为 Blue 确定最近追击目标；读取当前势函数。
2. 双方执行运动，处理非战斗损失。
3. 在统一的运动后/交战前状态计算 R4 和两方几何。
4. Red 对当前最近合法几何目标累计；Blue 只对 step开始确定的追击目标累计，运动后不重新挑选另一个目标。
5. 分别收集双方本步完成驻留的 `(attacker,target,True)` 内部结算凭据。
6. 双方事件都收集完后，才调用已有统一失效结算器，随后计算 R1/R2、终止状态与事件后的 R3 势函数。

首次有效采样记1，连续采样为1、2、3、4、5，第五次首次产生事件。**不是第一次进区就产生失效，也不是等待第六次。** 首个采样按一个 dt 计入，表示采样区间持续满足条件的离散假设，不做区间内插值。例如首次采样11.8秒、第五次12.2秒，尽管采样时间戳差为.4秒，计入的是五个.1秒采样区间。不能据此宣称知道真实连续边界进入时刻。

离开区域、攻击者死亡、目标死亡，均清零进度。原目标切到新的合法目标，旧段中断，新段当步从1开始。当前连续目标、计数和资格在 `DwellState` 中独立存储；双方各一份，一架实体每步最多累计一个目标。

完成后该驻留段结束，进度清零；`sampled_targets/sampled_counts`保存本步交战前身份与第五次采样，`completed_this_step`保存完成掩码。settlement后对死亡实体再次失效化。回合reset重建所有状态和统计；terminated/truncated结束时也关闭残余段。终止不把尚未完成的驻留段转成击毁。

## 同步失效、多人归属与奖励

两方完成事件基于同一份交战前有效集收集，因此 Red 先进入 resolver 不会取消已成立的 Blue 事件。已有 `_resolve_combat` 按目标去重、排序攻击者并记录唯一目标损失；已有 `_event_reward_components` 把一个目标的+10在完成的攻击者之间均分。多人完成数可以大于唯一失效数，二者不能混用。沿用原float32事件分项分摊的舍入语义，不改历史奖励实现。

双方同时全灭仍为 `draw_mutual_destruction`。本轮冻结场景测试中，四对实体第5步同时完成，双方各失效4个、平局；每个Red的+10与-10抵消。四个Red同时完成一个Blue时，完成数4、实际目标损失1、每个Red R1=2.5。

未复制 `step_core`、死亡处理或奖励公式。R3当前势函数仍在运动前，下一实际势函数仍在事件后；死亡/无目标势函数为零，terminated/truncated将用于shaping的下一势函数归零。R4仍来自交战前状态，死亡所在transition可以保留该步R4。每步真实奖励直接等于 r1+r2+r3+r4，逐项累计核对，完整回合折扣R3仍满足 `sum(.99**t*R3[t])=-Phi(initial)`。

R4的候选系数、可能的持续态势收益和既有设计限制原样保留。没有因驻留预检结果改奖励、Blue、场景、终局项或PPO超参数。

## 观测、部分可观测性与诊断

前45个字段与追加的四个alpha字段位置完全保留，没有向Actor加入驻留计数、剩余时间或目标身份。独立DwellState是额外环境内部状态，因此相同49D观测可能对应不同的驻留历史；**额外部分可观测性是本轮实验风险**。本轮没有通过扩大Actor输入消除它。

每方 info 提供：`dwell_targets/counts/eligible`（结算后的有效进度），`dwell_sampled_targets/sampled_counts/completed_this_step`（本步事件前采样），以及以下累计：

- dwell_eligible_pair_steps：双方仍有效且在几何窗口内的配对步总数，不受单目标选择限制。
- dwell_active_agent_steps：实际对一个目标累计的攻击者步数。
- dwell_started、dwell_interrupted、dwell_target_switches：新连续段、未完成关闭段、活动驻留目标之间的切换。Blue在窗口外的普通追击目标变化仍由原Blue诊断另行记录，不混入这个窗口内驻留切换指标。
- dwell_completed：完成阈值的攻击者段数；dwell_neutralizations：经原resolver去重后的目标失效数。
- first_dwell_geometry_step、first_dwell_start_step、first_dwell_complete_step；max_dwell_duration_steps。
- mean_dwell_duration_steps：**已结束段**（完成或中断）的平均样本数；尚开放段不进入均值，最大值包含当前段。回合结束会关闭开放段。

全局 `event_semantics=continuous_dwell_neutralization`，`legacy_fire_metrics_applicable=False`、`ammo_cooldown_applicable=False`。旧 fire_window/fire_attempts/weapon_hits/ammo_used保持零、相应首次时间为None，作为不适用的接口兼容占位。attack_kills及首次kill时间仍表示真实唯一目标失效，不改含义。

训练JSONL和评估CSV包含事件身份及驻留统计；完成回合后按回合归约驻留累计。普通发射/命中率不应与v3.43直接比较。录像事件为 `dwell_complete`，唯一失效仍为 `attack_kill`，cause标注continuous_dwell；不伪造fire_attempt。新增 `dwell_events.csv`，trace包含逐step两方计数、身份和完成采样。资源trace用-1表示不可用，元数据initial_ammo=None、cooldown_available=False。PNG/HTML/MP4绘图识别驻留完成事件。

## 实现文件与历史隔离

新增：

- `env/combat_v344.py`、`env/combat_v345.py`、`env/v344_dwell.py`、`env/v344_config.py`、`env/v345_config.py`。
- `configs/combat_environment_v344.yaml`、`configs/combat_environment_v345.yaml`；两个版本各自的 `mappo_4v4_v*.yaml` 和 `ippo_4v4_v*.yaml`，共六份。
- `tests/test_combat_v344_v345.py`。
- `tools/preflight_v344_v345.py`、`tools/verify_v344_v345_readiness.py`、本文档。

修改的共享接入：`env/config.py`、`env/combat_env.py`、`env/v343_config.py`（仅扩展训练gamma一致性检查）；`algorithm/common/{checkpoint,evaluator}.py`、`algorithm/mappo/{runner,evaluation}.py`、`algorithm/ippo/{protocol,runner,diagnostics}.py`；`tools/{combat_trace,combat_visualization,record_combat_episode,combat_replay_report,render_combat_episode,render_combat_episode_interactive}.py`。

新版本正常工厂、49/3/4维度、MAPPO/IPPO、评估、严格身份/指纹、固定观测诊断和规范回放均已接入。即使维度相同也拒绝跨3.43/3.44/3.45静默加载。IPPO保持旧有只支持确定性评估重载、不保证精确续训的能力边界。修改不使其他算法族进入历史兼容路径。用户原始Xu PDF是输入文献，不属于本轮生成代码。

## 验证结果

### 测试与历史保护

全部在PowerShell的 uav 环境执行，PyTorch/CUDA可用，GPU为RTX 5060 Laptop GPU。

1. `python -m pytest tests/test_combat_v344_v345.py -q`：**95 passed**，12.59秒。包括几何、时序、切换、死亡/reset、多人共享、同时全灭、禁概率路径、无额度/冷却限制、PBRS/观测与配置冻结、完整回合可重复性。
2. 下方Ubuntu测试命令所列六个测试文件：**496 passed, 1 skipped**，270.25秒；1项为旧视频测试因ffmpeg/ffprobe不在PATH跳过（`regression_tests.log`）。该批包括95个新测试与v3.41/3.42/3.43、旧录像管道测试，不重复相加。
3. 将已存在的FFmpeg目录加入本次测试进程PATH后，`python -m pytest tests/test_combat_v344_v345.py tests/test_combat_replay_pipeline.py::test_mp4_realtime_duration_and_png -q`：**96 passed**，17.73秒（`final_targeted_tests.log`），包含全部95个新测试及原先跳过的视频项。没有放宽阈值。
4. `python -m compileall -q env algorithm tools tests`、`git diff --check`通过；Git仅提示部分文件的LF/CRLF策略。不是全仓库所有测试通过的声明。

改动前提交 `cbb9ff20f612f0e7d6d9ccf83349cd2df7dc0b60`。379 个已有环境/配置和MAPPO网络/训练器文件逐文件比较不变（共享注册器及新增版本gamma检查单独列为改动）。独立HEAD源代码归档对照：旧8版本32个完整回合的状态、观测、奖励、资源及RNG哈希逐位相同；又专门对v3.41/v3.42/v3.43比较12个完整回合，全部精确相同。三个旧版本的历史MAPPO诊断checkpoint均在CUDA正常重载并完成确定性推理，且都被拒绝用于v3.44/v3.45。见 `source_protection.json`、`history_equivalence.json`、`recent_history_equivalence.json`、`historical_checkpoint_validation.json`。

### 240局固定策略预检

三版本×四策略×20个reset seeds（0–19）=240局，正常step、正常失效与终止。控制器复用未修改的既有审计函数：Zero完全零动作；Pursuit最近目标追击/225m/s；Vertical在8–12秒指令+30°、16–20秒-30°；Lateral在12–20秒增加30°航向偏差。没有训练模型参与。

| 版本 | 策略 | Red胜/负/平 | 时长秒 | 损失R/B | 完成驻留攻击者R/B | 实际唯一击毁R/B |
|---|---|---|---|---|---|---|
| 3.43 | zero | 30%/70%/0% | 14.70 | 3.20/2.35 | N/A | 2.35/3.20 |
| 3.43 | pursuit | 35%/65%/0% | 14.75 | 3.15/2.60 | N/A | 2.60/3.15 |
| 3.43 | lateral | 0%/100%/0% | 14.97 | 4.00/1.15 | N/A | 1.15/4.00 |
| 3.43 | vertical | 25%/75%/0% | 15.01 | 3.60/1.95 | N/A | 1.95/3.60 |
| 3.44 | zero | 0%/0%/100% | 14.40 | 4.00/4.00 | 4.0/4.0 | 4.00/4.00 |
| 3.44 | pursuit | 0%/0%/100% | 14.40 | 4.00/4.00 | 4.0/4.0 | 4.00/4.00 |
| 3.44 | lateral | 0%/100%/0% | 14.30 | 4.00/1.00 | 1.0/5.0 | 1.00/4.00 |
| 3.44 | vertical | 0%/100%/0% | 14.70 | 4.00/3.00 | 4.0/4.0 | 3.00/4.00 |
| 3.45 | zero | 0%/0%/100% | 14.40 | 4.00/4.00 | 4.0/4.0 | 4.00/4.00 |
| 3.45 | pursuit | 0%/0%/100% | 14.40 | 4.00/4.00 | 4.0/4.0 | 4.00/4.00 |
| 3.45 | lateral | 0%/100%/0% | 14.30 | 4.00/3.00 | 3.0/4.0 | 3.00/4.00 |
| 3.45 | vertical | 0%/0%/100% | 14.70 | 4.00/4.00 | 4.0/4.0 | 4.00/4.00 |

驻留版本详细曝光与时序（均值；曝光分母为该方所有存活配对步，而非资源可用配对）：

| 版本 | 策略 | 几何曝光R/B | 开始R/B | 中断R/B | 首次几何秒R/B | 首次完成秒R/B |
|---|---|---|---|---|---|---|
| 3.44 | zero | 1.671%/1.671% | 5.0/5.0 | 1.0/1.0 | 11.80/11.80 | 12.20/12.20 |
| 3.44 | pursuit | 1.815%/1.815% | 5.0/5.0 | 1.0/1.0 | 11.80/11.80 | 12.20/12.20 |
| 3.44 | lateral | 0.426%/2.132% | 2.0/7.0 | 1.0/2.0 | 11.80/11.80 | 12.20/12.20 |
| 3.44 | vertical | 1.632%/1.818% | 7.0/7.0 | 3.0/3.0 | 12.40/12.00 | 13.30/12.40 |
| 3.45 | zero | 1.671%/1.671% | 5.0/5.0 | 1.0/1.0 | 11.80/11.80 | 12.20/12.20 |
| 3.45 | pursuit | 1.815%/1.815% | 5.0/5.0 | 1.0/1.0 | 11.80/11.80 | 12.20/12.20 |
| 3.45 | lateral | 1.899%/1.899% | 6.0/6.0 | 3.0/2.0 | 11.80/11.80 | 12.20/12.20 |
| 3.45 | vertical | 1.546%/1.546% | 5.0/5.0 | 1.0/1.0 | 12.00/12.00 | 12.40/12.40 |

所有组20局均观察到双方几何曝光；两个新版本所有组均有双方驻留完成。没有Blue完全无法交战、无驻留完成或大量超时；240局超时均为0。多攻击者完成与唯一击毁不同，例如v3.44 Lateral的Blue均值5个完成、4个失效，不是重复扣了5个目标生命。目标切换次数、均值/最大持续时长以及逐局/逐事件位置与alpha均在CSV中，不能用占位fire_attempts来解释。

Reward（每局Red四机总和均值）及非零比例：

| 版本 | 策略 | R1 | R2 | R3 | R4 | R1/R2/R3/R4非零比例 |
|---|---|---|---|---|---|---|
| 3.43 | zero | -8.500000 | 0.000000 | -0.596078 | -0.685345 | 0.95%/0.00%/100.00%/42.76% |
| 3.43 | pursuit | -5.500000 | 0.000000 | -0.598470 | -0.682760 | 1.01%/0.00%/100.00%/42.71% |
| 3.43 | lateral | -28.500000 | 0.000000 | -0.595750 | -0.841300 | 0.86%/0.00%/100.00%/42.92% |
| 3.43 | vertical | -16.500000 | 0.000000 | -0.590940 | -1.015257 | 0.95%/0.00%/100.00%/43.85% |
| 3.44 | zero | 0.000000 | 0.000000 | -0.585255 | -0.636366 | 0.00%/0.00%/100.00%/41.48% |
| 3.44 | pursuit | 0.000000 | 0.000000 | -0.587872 | -0.632141 | 0.00%/0.00%/100.00%/41.48% |
| 3.44 | lateral | -30.000000 | 0.000000 | -0.584846 | -0.744618 | 0.55%/0.00%/100.00%/41.59% |
| 3.44 | vertical | -10.000000 | 0.000000 | -0.574853 | -0.916542 | 0.92%/0.00%/100.00%/41.91% |
| 3.45 | zero | 0.000000 | 0.000000 | -0.585255 | -0.636366 | 0.00%/0.00%/100.00%/41.48% |
| 3.45 | pursuit | 0.000000 | 0.000000 | -0.587872 | -0.632141 | 0.00%/0.00%/100.00%/41.48% |
| 3.45 | lateral | -10.000000 | 0.000000 | -0.584809 | -0.743315 | 0.18%/0.00%/100.00%/41.59% |
| 3.45 | vertical | 0.000000 | 0.000000 | -0.580294 | -0.941910 | 0.00%/0.00%/100.00%/42.65% |

非零比例按step开始存活的Red agent-step归约，死亡所在transition也计入。所有逐步返回与R1–R4相加逐位一致、逐项episode累计一致，跨求和顺序最大总误差 `7.11e-15`。每个回合均验证折扣R3望远镜恒等式。没有修改R4来抵消预检中的负值或改变系数。

v3.43与v3.44比较事件机制；v3.44与v3.45比较半角。所有同seed reset状态相同，事件改变后轨迹自然分叉，不声称全程严格配对。扩大半角在**同一状态**不会减少几何资格，但不同版本有不同生存轨迹和曝光分母，因此动态整局曝光比例不必单调。

固定外生几何诊断另提供18组 `fixed_trajectory.csv`，不推进死亡/奖励、首次事件锁存，避免后续失效分叉：对准4样本时新版本不完成；对准5样本时均第5次完成；35°/45°持续8样本只有v3.45完成（第5次）；46°均无曝光；第3样本离区后重入，新版本直到第8样本才完成。v3.43按固定seed的原概率/资源/冷却计算首次事件；本次对准时第1次命中。这是该seed的实际结果，不是概率模型必然首发命中。新规则在所有这些诊断中命中正态抽样次数为0。

复跑每版本/策略seed=0的12局，轨迹哈希、总奖励精确复现，见 `repeatability.json`。

### CUDA Smoke与回放

| 版本 | 算法 | 环境采样 | PPO更新 | Actor/Critic优化步 | KL | Entropy | EV |
|---|---|---|---|---|---|---|---|
| 3.44 | mappo | 4096 | 1 | 80/80 | 0.002823 | 1.797949 | 0.002015 |
| 3.44 | ippo | 4096 | 1 | 80/80 | 0.002827 | 1.797940 | 0.001224 |
| 3.45 | mappo | 4096 | 1 | 80/80 | 0.002829 | 1.797352 | 0.002950 |
| 3.45 | ippo | 4096 | 1 | 80/80 | 0.002834 | 1.797284 | 0.003154 |

四组总采样16384。每组保持16 env×256 rollout、两层256、10 PPO epochs、minibatch512，以及全部原LR/KL阈值/entropy/gamma/GAE参数；Actor/Critic梯度、参数与Adam状态均有限且有非零更新。各自诊断checkpoint保存和重载后参数/确定性动作精确相同。拒绝旧版本及另一个驻留半角版本checkpoint，拒绝缺失指纹。各做2局CUDA固定种子评估、1局规范诊断记录，核对dwell_complete数量与去重失效数，验证49D固定观测库；四组worker均确认关闭。

Smoke复用已有通用验证器，显式4096预算且不调用正式run循环。training_smoke=False用于保留MAPPO正式网络与PPO参数；diagnostic文件名、预算和summary标识其不是正式实验模型。seed使用既有通用验证器的342/记录20000342，属于独立诊断种子，不是版本号或新超参数。

另外通过正式 `tools/replay_combat.py` CLI，以新v3.44 MAPPO诊断模型和seed20000344生成MP4/PNG/HTML/CSV/报告；事件标签为Dwell completed，`dwell_events.csv`与逐step驻留trace齐全，未生成随机fire_attempt事件。MP4已全帧解码核对，PNG已目视检查。此单局Red存活1、Blue存活0，历时14.7秒；**一次Smoke模型胜局不能解释为训练成功**。产物在 `replay_cli_v344/`。

### 异常、限制与500K建议

**目前不建议把两版无条件视为已通过500K效果对比的科研验收。** 技术实现、可复现性、Reward账务与CUDA链路通过，但固定策略揭示以下限制，需要在启动学习诊断前明确研究解释：

- v3.44与v3.45的Zero/Pursuit均为100%双方全灭平局；v3.45 Vertical也为100%双方全灭。不是Zero胜率异常高，而是确定性对称互毁的退化现象，风险筛查已列出5个版本/策略组。
- 这些组虽然交战充分，正负R1却逐机抵消，R1非零比例为0。不能把完成次数多等同于击毁学习信号丰富；R3/R4仍有信号，但两者设计本轮原样保留。
- v3.44 Lateral/Vertical和v3.45 Lateral均为Blue全胜，本轮没有证据表明扩大角度或改成驻留一定改善策略可学习性。
- 现有初始分布的随机量仅是整体径向旋转，编队、速度和高度固定。确定性机制去掉随机命中后，20个seed产生相同类别事件/结局并不代表20种战术场景的独立多样性；没有擅自改变初始分布来改善此结果。
- 49D输入隐藏驻留进度，额外部分可观测性仍在。没有把进度加入Actor或引入别的任务解决它。
- R4不是势差奖励，其长期保持态势的收益问题仍需研究；本轮不以调参、额外终局奖励、改Blue或扩大边界消除风险。

若研究人员随后选择500K，应把它定义为检验上述对称互毁、R1抵消与隐藏计时状态的**有风险学习诊断**，不能提前承诺胜率改善或正式收敛。下面提供原参数独立目录的命令供使用；本轮未启动500K、3M或7M。


## Ubuntu复现与训练命令

本轮实际命令按用户既有要求在PowerShell激活Windows `uav`执行；CPU只用于数学和固定控制器预检，训练与checkpoint评估均使用CUDA。以下供Ubuntu终端复制，必须使用新输出目录。

```bash
conda activate uav
export OMP_NUM_THREADS=1
python -c 'import torch; assert torch.cuda.is_available(), "CUDA mandatory"; print(torch.__version__, torch.cuda.get_device_name(0))'
OUT="outputs/v344_v345_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
python -m pytest tests/test_combat_v344_v345.py tests/test_combat_v342.py \
  tests/test_combat_v343.py tests/test_combat_v341.py \
  tests/test_combat_visualization.py tests/test_combat_replay_pipeline.py -q
python -u tools/preflight_v344_v345.py --seeds 20 --workers 4 --output "$OUT/preflight"
python -u tools/verify_v344_v345_readiness.py --output "$OUT/cuda_smoke"
python -m compileall -q env algorithm tools tests
git diff --check
```

以下是**供研究人员按风险结论决定是否运行**的500K诊断命令，本轮未执行。独立输出/日志，不改PPO参数；VERSION可改为345，ALG可改为ippo，环境、算法配置必须一起改。500K不是正式3M/7M预算。

```bash
conda activate uav
export OMP_NUM_THREADS=1
VERSION=344
ALG=mappo
RUN_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p outputs/training_logs
set -o pipefail
python -u "algorithm/train_${ALG}.py" \
  --env-config "configs/combat_environment_v${VERSION}.yaml" \
  --algorithm-config "configs/${ALG}_4v4_v${VERSION}.yaml" \
  --device cuda --seed 1 --num-envs 16 --total-sampled-steps 500000 \
  --output-dir "outputs/${ALG}_v${VERSION}_seed1_500k_${RUN_TS}" \
  2>&1 | tee "outputs/training_logs/${ALG}_v${VERSION}_seed1_500k_${RUN_TS}.log"
```

已有诊断模型可通过正常入口评估/回放：

```bash
python -m algorithm.evaluate_mappo \
  --checkpoint "$OUT/cuda_smoke/mappo_v344/diagnostic_4096.pt" \
  --env-config configs/combat_environment_v344.yaml \
  --algorithm-config configs/mappo_4v4_v344.yaml \
  --device cuda --seed-base 10000000 --episodes 2 --output "$OUT/mappo_eval.json"
python -m algorithm.evaluate_ippo \
  --checkpoint "$OUT/cuda_smoke/ippo_v345/diagnostic_4096.pt" \
  --env-config configs/combat_environment_v345.yaml \
  --algorithm-config configs/ippo_4v4_v345.yaml \
  --device cuda --seed-base 10000000 --episodes 2 --split development \
  --output "$OUT/ippo_eval.json"
python -u tools/replay_combat.py \
  --checkpoint "$OUT/cuda_smoke/mappo_v344/diagnostic_4096.pt" \
  --env-config configs/combat_environment_v344.yaml \
  --algorithm-config configs/mappo_4v4_v344.yaml \
  --episode-seed 20000344 --device cuda --formats png html csv report \
  --output-dir "$OUT/replay_v344"
# 需要MP4时加入 --formats mp4 png html csv report，并确保FFmpeg可用。
```
