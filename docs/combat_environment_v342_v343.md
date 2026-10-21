# v3.42 / v3.43 三维几何与连续奖励研究实验

本轮实现与验证日期：2026-10-21。结果目录：`outputs/v342_v343_validation_20261021/`。

两版均为离线、抽象 4v4 点质点研究仿真；不包含真实平台、硬件或外部系统功能。本轮没有启动正式 3M 训练，没有读取、覆盖既有训练模型或历史实验结果。下面的“继承公式”指仓库已有、标记为论文简化公式的实现；本轮不声称核验或复现了原论文的完整系统。

## 代码核查与版本边界

实际代码与任务描述一致。`env/geometry.py` 定义 `ATA=wrap(LOS_heading-psi)`、`AA=wrap(target.psi-LOS_heading)`、`HA=atan2(-dz,horizontal)`，并已提供考虑 psi/theta 的 `off_boresight` 和机体系有符号方位/俯仰误差。NED 下正 theta 向上、正 dz 表示目标更低。重合位置沿用已有 alpha=0、两个机体系误差=0 的约定。

v3.41 继承链为 v341→v340→v339→v338→v337→v336→…→v330。实际资格采用 `PaperATAHAWeapon` 的独立 ATA/HA 门限；概率也输入 ATA/HA。运动采用 `v311_kinematics.advance_aircraft`，不是父类保留的旧动力学控制器。Blue 为 `DistanceTierNearestTargetBluePolicy`。v330 构造器和 step 核心绕过了若干历史 step 包装，不能只根据类继承顺序推断奖励重复次数。

`step_core` 的顺序保持为：当前势函数→双方动作/运动→非战斗损失→R4 稠密几何快照→双方资格与事件尝试→同时结算事件→R1/R2→终止判断→事件后势函数→优势奖励→别名和累计→观测。Red 选择当前合法目标中最近者、同距离按槽号；Blue 的事件目标仍绑定 step 开始的最近追击目标。没有新增跟踪或锁定状态机。

## v3.42 相对 v3.41 的精确差异

新增独立 `CombatEnvironmentV342`。新 `ConeWeapon` 直接使用 `engagement_geometry(...).off_boresight <= pi/6`，替换独立 ATA/HA 矩形门限。保留原 `angle_within` 的 1e-12 数值容差。

概率输入替换为 `boresight_azimuth_error` / `boresight_elevation_error`；复用旧 `attempt_hit` 和 `expected_hit_probability` 方法，保留 `threshold=pi*exp(-d/effective_hit_distance)`、两个误差共享同一个 epsilon、每次尝试恰好一次 `normal(0,1)`。未重新标定发生率。正对目标时与旧概率一致；水平以外的状态可能因坐标系或资格改变而分叉。

配置用 `weapon.off_boresight_max` 替换 `ata_max/ha_max`。校验在深拷贝上映射到 v3.41 校验，入参不变，其他冻结约束不放松。双方使用同一对象和同一角度输入路径。直接转发 v3.26 的事件钩子，仅绕过旧矩形几何诊断包装；目标、资源、冷却、抽样和同时结算实现不重写。

保持 2000 m 距离上限、6 次资源、1 秒冷却、0.1 秒步长、4v4、45 维观测、旧 R1–R4、Blue 策略、终止规则以及 MAPPO/IPPO 全部优化参数。诊断中的 `range_az/geometry` 在新版明确标注为三维锥体语义，历史日志不变。

**v3.42 故意保留旧 R3/R4：几何事件与奖励几何没有统一，这是单变量实验设计。** 本轮固定控制器中其 R4 仍为零，不能据此宣称已解决战术奖励曝光问题。

## v3.43 相对 v3.42 的精确差异

新增独立 `CombatEnvironmentV343`，完全继承 v3.42 运动和事件机制。

R1/R2 沿用原结算：有效击毁 +10，多攻击者沿用原分享规则；战斗损失和触地沿用原 -10 死亡口径；出界 -10。没有团队终局奖励或终止规则变化。

新 `research_reward` 是明确标注的科研候选参数，不是原论文已实现或已最优的规则。旧 `reward` 中 R3/R4 参数保留以冻结父类 schema，但 v3.43 的 R3/R4 实际只读取 `research_reward`。

每个存活 Red 以三维欧氏距离选择最近存活 Blue，同距按目标槽号。目标不受资源或冷却筛选；与现有全局敌机观测保持一致，不另引入传感器跟踪规则。选择器完全由当前状态确定，不缓存目标，因此目标死亡、切换和同 step 击毁都进入同一个状态势函数：

```text
fd = 1-clip(d/8000,0,1)
fa = 1-alpha/pi
Phi = .1*(.5*fd+.5*fa)
R3 = .99*Phi(next)-Phi(current)
```

死亡 Red 或无存活目标的势函数为零。当前状态在运动前读取，下一状态在交战结算后读取；terminated 和 truncated 都把用于 shaping 的下一势函数归零，实际下一势函数仍单独保留在 info。截断在本实验中作为有限回合终点，与现有 PPO done 处理一致。训练 gamma 必须等于 shaping gamma=.99，runner、IPPO 配置及 checkpoint 验证均检查。

目标切换可能造成单步势差跳变，但没有目标切换奖金或重置基准。任意完整长度 T 的回合满足 `sum(gamma**t * R3[t]) = -Phi(initial)`。这项测试检验记账，不代表 R3 没有学习引导作用。静态或闭合状态循环不会把势函数变成持续正奖励。

```text
w = clip((4000-d)/2000,0,1)
g(alpha) = clip(1-alpha/(pi/3),0,1)
tail = (1+cos(AA_red))/2
R4_off_pair = w*g(alpha_red_to_blue)*(.005+.015*tail)
R4_def_pair = -.01*w*g(alpha_blue_to_red)
R4 = max(valid offense pairs) + min(valid defense pairs)
```

死亡和无效配对取零；进攻/防御独立归约，目标重排不变。R4 在角度和距离分段点连续，有限且每机每步范围为 [-.01,.02]。沿用原时序，在交战前存活的状态上计算，所以本 step 随后被击毁的飞机仍可有 R4；R3 的下一势函数则反映同 step 的死亡。

精确重合点的 LOS 没有方向，沿用 alpha=0 的有限数值约定；不能对该点宣称任意方向的位置极限都连续。

R4 不是势差：理论上 1000 步、每步最大进攻收益仍可累计 20/机，即两次击毁等价。这是候选规则的已知上界，不是“任何策略都无法刷分”的证明。本轮测试了静态远距无效循环和固定控制器，未改系数、加时间惩罚或裁剪收益；学习策略是否存在利用方式仍需小规模训练观察。

## 奖励记账与日志

不复制或修改共享 `step_core`。v3.43 通过 `potential`、`shaping_next_potential`、`advantage_reward`、`reward_aliases` 四个钩子扩展：

1. dense 阶段只返回 R4，guide 初始为零。
2. advantage 阶段计算一次 R3，返回 R3+R4。
3. aliases 阶段将同一个 R3 写入 r3 和 guide 诊断，把 adv 写入 tactical 总诊断；这些诊断不会再次加入真实奖励。
4. 原 transition 各累计器更新一次，父 wrapper 从 r1–r4 返回 float64 奖励。

因此 `actual_reward=R1+R2+R3+R4`，R4 等于进攻加防御。info 分项与逐局累计逐一验证；总和的不同浮点加法顺序允许机器精度误差，不能声称不同顺序的浮点总和逐位相同。

v3.43 用 `research_rewards` / `episode_research_reward_total`，移除 `paper_reward` 和 `madsac_r*` 标签；训练汇总和 evaluator 相应使用 research 名称。历史版本继续使用原日志名称。`guide` 在 v3.43 是 R3 别名，`tactical_total_rewards` 是 R3+R4，`tactical_rewards` 是 R4；这些均不额外发放奖励。

## 49 维观测、网络与身份

v3.43 的前 45 个字段逐位保留，索引不变。尾部 `[45:49]` 按四个敌机槽位追加 `alpha/pi`，而不是在旧敌机五元组中间插入，以满足原始索引完全兼容的要求。死亡本机整行零；死亡敌机的旧槽位和追加字段均零。没有增加资源或冷却观测。

MAPPO 从配置读取 45/49；IPPO 局部 Critic、trainer、架构与参数数量校验、固定观测库均改为显式支持对应维度。v3.43 IPPO actor/critic 参数数量分别为 79366/78849；v3.41/v3.42 仍为 78342/77825。

版本注册、维度查询、runner 身份、评估指标、checkpoint resume/evaluation 指纹路径均加入两个新版本。即使 v3.41/v3.42 同为 45 维，也必须精确版本匹配；缺失环境或算法指纹会被拒绝。IPPO 继续沿用原有“仅确定性评估重载、不支持逐位续训”的能力边界，没有宣称新增精确 resume。

固定观测库记录环境身份与指纹；新版本比较时不允许缺省身份进入旧兼容路径。录像按实际类分发，记录 alpha、两个机体系误差和 cone_pass；新版本 CSV 同步输出锥体判定。trace 保存物理状态，没有 45 维硬编码。其他算法族的显式版本白名单未扩展，因为本次只授权 MAPPO/IPPO，不能静默接入旧兼容路径。

## 数学验证

`forward=[cos(theta)cos(psi),cos(theta)sin(psi),-sin(theta)]`，alpha 直接复用既有点积反余弦实现。distance=1500、ATA=0、HA=-20°、theta=+50° 时 alpha=70°：v3.41 通过旧角度资格、v3.42 拒绝。theta=-20° 时恢复对准。另验证 HA=50° 且 theta=50° 时新规则通过、旧规则拒绝，证明是替换门限。

覆盖向上/向下、水平与垂直错位、30°上下及边界、±pi 航向跨界、不同象限/高度差、极近和重合位置。通过构造机体系 LOS 独立验证有符号角度和实际概率钩子输入。1000 个固定随机三维样本检验概率有限以及传感器包含关系。每次事件一次 normal，双方 4 架同时合资格为 8 次；冷却期间和不合资格状态不消耗随机数。固定种子事件轨迹可复现。

传感器使用相同 forward/LOS 三维几何，窗口为 d<=10000 且 alpha<=60°。事件窗口 d<=2000 且 alpha<=30° 严格包含于该几何窗口；资源、冷却、目标选择只会缩小可发生事件的集合。这个集合包含结论没有新增传感器锁定机制，也不表示观测改成了传感器掩码。

## 实际验证结果与复现

下方统计由本轮结构化结果补充；所有 Python、pytest、训练/评估命令均在 PowerShell 激活 Windows `uav` 环境执行。用户本轮明确指定 PowerShell，优先于仓库默认 WSL 规则。CUDA 为 PyTorch 2.11.0+cu128，NVIDIA GeForce RTX 5060 Laptop GPU。几何/奖励/固定控制器预检按本轮要求使用 CPU，训练与模型评估只用 CUDA。

### 测试与历史保护

- 最终专项：`python -m pytest tests/test_combat_v342.py tests/test_combat_v343.py -q`，**170 passed**（23.90 秒，`delivery_tests.log`）。v3.42 首批 62 项先通过，之后才实现 v3.43。
- 近代版本/入口回归命令见下节：**1020 passed, 5 failed, 1 skipped**（316.70 秒，`tests.log`）。当时包含最初 149 项新专项；与最终 170 项有重叠，不重复相加。
- 更早版本及汇总回归：**442 passed, 14 failed**（78.57 秒，`legacy_reference_tests.log`）。
- 19 个旧失败在独立原始 HEAD 归档中逐个重现：`baseline_known_failures.log` 5 failed，`baseline_legacy_failures.log` 14 failed。没有修改或绕过这些测试；未宣称全仓库绿灯。
- 原始提交 `f201624a10e9b0e5d4e06f35f92bdda546dbca89`；365 个已有环境/配置文件的 LF 规范化 SHA256 保持一致。8 个历史版本共 32 个完整回合，与原始 HEAD 的状态/观测/奖励/资源/RNG 哈希完全相同，见 `history_trajectory_equivalence.json`。
- `python -m compileall -q env algorithm tools tests` 与 `git diff --check` 通过；仅 Git 提示个别 LF 将按已有设置转换 CRLF。

已确认的既有失败分组：

1. 3 项 MADSAC checkpoint 测试期望 environment_version 错误，但当前旧实现先拒绝 implementation-v3/v4。
2. 2 项 MADSAC/MAAC 版本范围错误信息仍匹配旧 v3.38 文案。
3. 4 项历史固定轨迹/观测 JSON 快照与当前运行结果不匹配；原始 HEAD 也失败。没有把未经证实的差异归因于本轮改动或特定平台。
4. 10 项旧 runner 测试桩未提供 env_config，原始实现的 controller 诊断分支触发 AttributeError。

完整测试 ID：

```text
tests/test_combat_v330.py::test_old_checkpoint_version_rejected[madsac]
tests/test_combat_v332.py::test_old_action_semantics_checkpoint_rejected[3.30-madsac]
tests/test_combat_v332.py::test_old_action_semantics_checkpoint_rejected[3.31-madsac]
tests/test_mappo_environment_identity.py::test_madsac_diagnostic_ranges_cover_v338
tests/test_mappo_environment_identity.py::test_maac_diagnostic_reports_registered_range
tests/test_combat_v31.py::test_legacy_bit_level_trajectories
tests/test_combat_v32.py::test_legacy_complete_terminal_trajectories_unchanged
tests/test_combat_v32.py::test_true_truncated_episode_in_training_metrics_and_summary[mappo]
tests/test_combat_v32.py::test_true_truncated_episode_in_training_metrics_and_summary[rmappo]
tests/test_combat_v32.py::test_true_truncated_episode_in_training_metrics_and_summary[ea-mappo]
tests/test_combat_v32.py::test_true_truncated_episode_in_training_metrics_and_summary[stea-mappo]
tests/test_combat_v37.py::test_configuration_isolation_and_original_reference
tests/test_combat_v39.py::test_complete_pre_extension_v38_trajectory_reference
tests/test_compact_training_metrics.py::test_one_weighted_summary_after_whole_rollout[MAPPOTrainingRunner]
tests/test_compact_training_metrics.py::test_one_weighted_summary_after_whole_rollout[RMAPPOTrainingRunner]
tests/test_compact_training_metrics.py::test_one_weighted_summary_after_whole_rollout[STEAMAPPOTrainingRunner]
tests/test_compact_training_metrics.py::test_one_weighted_summary_after_whole_rollout[EAMAPPOTrainingRunner]
tests/test_compact_training_metrics.py::test_no_episodes_null_and_theoretical_count
tests/test_compact_training_metrics.py::test_accumulation_and_serialization_do_not_consume_rng
```

本轮开发中还修正了两个新建验证代码的问题：静态无交战测试原先把目标放在场外；审计把 float32 alive mask 当作索引。修复后专项及 240 局重跑通过。保留初次审计失败于 `preflight.log`。一次补充测试调用拼错测试节点名，没有执行任何测试，见 `final_targeted_tests.log`；最终正确命令结果为上述 170 passed。这些不属于残留环境失败。

### 240 局固定控制器预检

每版 4 策略 × seeds 0–19，共 80 局。复用未修改的 `tools.audit_v339_attack_geometry.red_actions`：Zero 完全零控制；Pursuit 追最近存活目标并维持 225 m/s；Lateral 在 12–20 秒增加 30°航向偏差；Vertical 在 8–12 秒指令 +30°、16–20 秒 -30°，其余为水平。Blue 仍用原固定策略。所有回合使用正常致命 step。

| 版本 | 策略 | Red胜/负/平 | 平均秒 | Red/Blue损失 | Red/Blue尝试数 | 首次几何秒 Red/Blue |
|---|---|---|---|---|---|---|
| 3.41 | zero | 6/14/0 | 14.695 | 3.20/2.35 | 6.60/7.35 | 11.80/11.80 |
| 3.41 | pursuit | 7/13/0 | 14.750 | 3.15/2.60 | 6.70/7.20 | 11.80/11.80 |
| 3.41 | lateral | 0/20/0 | 14.970 | 4.00/1.15 | 4.30/8.90 | 11.80/11.80 |
| 3.41 | vertical | 8/12/0 | 15.270 | 2.95/2.60 | 7.40/8.45 | 12.00/12.00 |
| 3.42 | zero | 6/14/0 | 14.695 | 3.20/2.35 | 6.60/7.35 | 11.80/11.80 |
| 3.42 | pursuit | 7/13/0 | 14.750 | 3.15/2.60 | 6.70/7.20 | 11.80/11.80 |
| 3.42 | lateral | 0/20/0 | 14.970 | 4.00/1.15 | 4.30/8.90 | 11.80/11.80 |
| 3.42 | vertical | 5/15/0 | 15.010 | 3.60/1.95 | 6.05/7.95 | 12.55/12.00 |
| 3.43 | zero | 6/14/0 | 14.695 | 3.20/2.35 | 6.60/7.35 | 11.80/11.80 |
| 3.43 | pursuit | 7/13/0 | 14.750 | 3.15/2.60 | 6.70/7.20 | 11.80/11.80 |
| 3.43 | lateral | 0/20/0 | 14.970 | 4.00/1.15 | 4.30/8.90 | 11.80/11.80 |
| 3.43 | vertical | 5/15/0 | 15.010 | 3.60/1.95 | 6.05/7.95 | 12.55/12.00 |

首次几何在运动/非战斗损失后、同时事件前采样，只计存活配对及距离/角度，不含资源和冷却。全部组的 20 局均观察到双方几何机会。资格差异比例以双方存活且在 2 km 内的配对步为分母；Zero/Pursuit/Lateral 为 0，Vertical 的 v3.41 轨迹为 1.5766%，v3.42/v3.43 轨迹为 2.1453%。因后续轨迹不同，两比例不是同一轨迹的严格配对估计。

所有实际事件的三维 alpha（合并四种策略；角度单位度，分布是实际尝试条件下的分布）：

| 版本 | 方 | 尝试 | 命中 | alpha均值 | 中位数 | P90 | 最大 | 平均距离m |
|---|---|---|---|---|---|---|---|---|
| 3.41 | red | 500 | 198 | 9.246 | 7.257 | 27.699 | 39.879 | 1835.9 |
| 3.41 | blue | 638 | 284 | 0.210 | 0.118 | 0.411 | 5.318 | 1796.8 |
| 3.42 | red | 473 | 181 | 8.465 | 5.499 | 26.934 | 29.353 | 1846.5 |
| 3.42 | blue | 628 | 297 | 0.207 | 0.118 | 0.416 | 5.318 | 1810.2 |
| 3.43 | red | 473 | 181 | 8.465 | 5.499 | 26.934 | 29.353 | 1846.5 |
| 3.43 | blue | 628 | 297 | 0.207 | 0.118 | 0.416 | 5.318 | 1810.2 |

全部存活配对和射程内配对的 alpha 分位数见 `episodes.csv`；全配对 10° 分箱分布见 `summary.json`。`events.csv` 含每次实际尝试的 ATA/HA、alpha、两个机体系误差、理论概率、hit 和时间；不把多次命中数混同于唯一击毁数。

奖励累计均值（每局 Red 四机合计）：

| 版本 | 策略 | R1 | R2 | R3 | R4 | R3非零 | R4非零 |
|---|---|---|---|---|---|---|---|
| 3.41 | zero | -8.50000 | 0.00000 | 0.44693 | 0.00000 | 96.16% | 0.00% |
| 3.41 | pursuit | -5.50000 | 0.00000 | 0.44673 | 0.00000 | 96.25% | 0.00% |
| 3.41 | lateral | -28.50000 | 0.00000 | 0.44534 | 0.00000 | 94.06% | 0.00% |
| 3.41 | vertical | -3.50000 | 0.00000 | 0.45507 | 0.00000 | 95.15% | 0.00% |
| 3.42 | zero | -8.50000 | 0.00000 | 0.44693 | 0.00000 | 96.16% | 0.00% |
| 3.42 | pursuit | -5.50000 | 0.00000 | 0.44673 | 0.00000 | 96.25% | 0.00% |
| 3.42 | lateral | -28.50000 | 0.00000 | 0.44534 | 0.00000 | 94.06% | 0.00% |
| 3.42 | vertical | -16.50000 | 0.00000 | 0.45494 | 0.00000 | 96.57% | 0.00% |
| 3.43 | zero | -8.50000 | 0.00000 | -0.59608 | -0.68535 | 100.00% | 42.76% |
| 3.43 | pursuit | -5.50000 | 0.00000 | -0.59847 | -0.68276 | 100.00% | 42.71% |
| 3.43 | lateral | -28.50000 | 0.00000 | -0.59575 | -0.84130 | 100.00% | 42.92% |
| 3.43 | vertical | -16.50000 | 0.00000 | -0.59094 | -1.01526 | 100.00% | 43.85% |

非零分母为 step 开始存活的 Red agent-step，包含死亡所在 transition；不把死亡后槽位充入分母。R4 合计为零可能来自无曝光或正负抵消，因此独立保存进攻/防御项：

| v3.43策略 | 进攻累计 | 防御累计 | 进攻绝对占比 | 防御绝对占比 | 正R3累计 | 负R3累计 |
|---|---|---|---|---|---|---|
| zero | 0.69233 | -1.37767 | 33.45% | 66.55% | 0.000052 | -0.596130 |
| pursuit | 0.72383 | -1.40659 | 33.98% | 66.02% | 0.003144 | -0.601614 |
| lateral | 0.58838 | -1.42968 | 29.16% | 70.84% | 0.001297 | -0.597047 |
| vertical | 0.44969 | -1.46495 | 23.49% | 76.51% | 0.008710 | -0.599650 |

240 局逐步实际奖励与四项相加逐位相同，逐项 episode 累计逐位相同；不同求和顺序下总奖励最大误差 `7.11e-15`。v3.43 每局折扣 R3 均检验等于 -Phi(initial)。Zero 的正 R3 累计均值约 0.000052，含终止扣回后的净 R3 为 -0.596078，没有发现被动接近赚取大量 R3。审计筛查每机累计稠密奖励大于 20、或团队稠密净收益超过 max(10,事件绝对累计) 的两倍：异常 0 局。筛查是明确阈值，不是策略不可利用的数学证明。

三版同 seed 初始状态 SHA256 全相同。v3.42/v3.43 全 80 局固定控制器物理轨迹 hash 相同，证明本轮奖励与观测修改未改变其运动/事件。v3.41/v3.42 的前三种策略同轨迹；Vertical 因俯仰相关事件窗口及概率输入改变，胜局由 8/20 降为 5/20，不能解释为新版本更优。没有宣称三版全程严格配对。另复跑 12 局（每版本/策略 seed=0），轨迹 hash、奖励、时长及事件总数精确复现，见 `repeatability.json`。

### CUDA 有界 Smoke 与正常入口

| 版本 | 算法 | 采样 | PPO更新 | Actor/Critic步数 | rollout形状 | 参数变化L2 Actor/Critic |
|---|---|---|---|---|---|---|
| 3.42 | mappo | 4096 | 1 | 80/80 | [256, 16, 4, 45] | 1.619351/1.828760 |
| 3.42 | ippo | 4096 | 1 | 80/80 | [256, 16, 4, 45] | 1.620230/1.440631 |
| 3.43 | mappo | 4096 | 1 | 80/80 | [256, 16, 4, 49] | 1.671114/1.681396 |
| 3.43 | ippo | 4096 | 1 | 80/80 | [256, 16, 4, 49] | 1.670363/1.435573 |

每组保持 16 env × 256 rollout、两层 256、原 learning rate/10 epochs/minibatch512/KL/entropy/gamma；每组只做一个 PPO update，总采样 16384。所有梯度、参数、优化器状态和指标有限且有非零更新；各自 checkpoint 保存/加载后参数与确定性动作精确相同。通过精确版本拒绝所有跨版本组合，并拒绝缺失指纹。每组从自己的诊断 checkpoint 做 2 局 CUDA 评估及 1 局规范录像；IPPO 额外用固定观测库验证重载。

四组是验证工具直接调用正常 runner/更新/保存/加载/评估接口；未进入正式 run 循环。checkpoint 的 training_smoke=False 用于保留 MAPPO 正式网络和 PPO 参数，4096 预算、diagnostic 文件名、输出目录与 summary 明确标识诊断用途，不应误当成正式实验模型。初版审计 L2 读数触发一次 PyTorch requires_grad→float 提示，已在工具中用 detach 修正，不影响保存数据或更新。

另以命令行 `python -m algorithm.evaluate_mappo` / `evaluate_ippo` 对 v3.43 各做 2 局 CUDA 评估，均成功，结果为 `mappo_cli_evaluation.json` / `ippo_cli_evaluation.json`。还通过 `tools/replay_combat.py --formats csv report` 的 v3.43 IPPO 命令行回放，生成 149 步 trace、CSV 和报告，见 `replay_cli_ippo_v343/`。完整命令参数与下方 Linux 复现示例一致。

### 是否进入训练

建议 v3.43 进入**小规模、独立输出的学习诊断**：新几何/奖励/账务专项、固定控制器和 CUDA 链路没有残留关键失败，事件机会充足，R3/R4 有曝光，未发现本轮固定策略奖励异常。监控 R4 进攻/防御贡献、累计收益/事件收益、目标切换和策略盘旋；当前证据不证明更易收敛或性能更优。

v3.42 可作为几何单变量对照运行，但其旧 R4 在全部 80 局为零、且奖励几何刻意不统一。如果研究目标要求有战术 R4 学习信号，这仍是明确限制，不能宣布达到该目标的正式训练条件。两版都不建议直接以本次 Smoke 代替训练研究验收，不自动启动 3M。19 项既有回归失败已给出基线证据，仍作为仓库测试债报告。


## 可复制的 Linux 复现命令

在仓库根目录，先激活已有的同依赖 `uav` 环境。以下输出目录必须全新；命令仅执行测试、固定控制器审计及有界 CUDA 验证，不启动正式 3M。

```bash
conda activate uav
export OMP_NUM_THREADS=1
python -c 'import torch; assert torch.cuda.is_available(), "CUDA mandatory"; print(torch.__version__, torch.cuda.get_device_name(0))'
OUT="outputs/v342_v343_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
python -m pytest tests/test_combat_v342.py tests/test_combat_v343.py -q
python -u tools/preflight_v342_v343.py --seeds 20 --workers 4 --output "$OUT/preflight"
python -u tools/verify_v342_v343_readiness.py --output "$OUT/cuda_smoke"
python -m compileall -q env algorithm tools tests
git diff --check

python -m algorithm.evaluate_mappo \
  --checkpoint "$OUT/cuda_smoke/mappo_v343/diagnostic_4096.pt" \
  --env-config configs/combat_environment_v343.yaml \
  --algorithm-config configs/mappo_4v4_v343.yaml \
  --device cuda --seed-base 10000000 --episodes 2 --output "$OUT/mappo_eval.json"
python -m algorithm.evaluate_ippo \
  --checkpoint "$OUT/cuda_smoke/ippo_v343/diagnostic_4096.pt" \
  --env-config configs/combat_environment_v343.yaml \
  --algorithm-config configs/ippo_4v4_v343.yaml \
  --device cuda --seed-base 10000000 --episodes 2 --split development \
  --output "$OUT/ippo_eval.json"
python -u tools/replay_combat.py \
  --checkpoint "$OUT/cuda_smoke/ippo_v343/diagnostic_4096.pt" \
  --env-config configs/combat_environment_v343.yaml \
  --algorithm-config configs/ippo_4v4_v343.yaml \
  --episode-seed 20000342 --device cuda --formats csv report \
  --output-dir "$OUT/replay_ippo_v343"
```

要复现 v3.42 单独评估，把上述 v343 环境/算法/诊断 checkpoint 路径同时替换为 v342，不能混用版本。两个版本都有独立 MAPPO/IPPO YAML；正式训练入口可正常读取，但本轮未调用正式预算训练。

历史回归命令（保留已知失败，不改旧断言）：

```bash
python -m pytest tests/test_combat_v341.py tests/test_combat_v340.py \
  tests/test_combat_v339.py tests/test_combat_v338.py tests/test_combat_v337.py \
  tests/test_combat_v336.py tests/test_combat_v330.py tests/test_combat_v332.py \
  tests/test_combat_v342.py tests/test_combat_v343.py \
  tests/test_mappo_environment_identity.py tests/test_ippo.py \
  tests/test_v341_training_configs.py tests/test_combat_replay_pipeline.py -q
python -m pytest tests/test_combat_environment.py tests/test_combat_v31.py \
  tests/test_combat_v32.py tests/test_combat_v37.py tests/test_combat_v39.py \
  tests/test_compact_training_metrics.py -q
```

历史完整轨迹比较要求提供**改动前**提交的独立代码归档。不要用新代码本身冒充 baseline：

```bash
# BASELINE_COMMIT 应设为 source_protection.json 中记录的改动前提交
BASELINE_COMMIT=f201624a10e9b0e5d4e06f35f92bdda546dbca89
mkdir -p "$OUT/baseline_source"
git archive "$BASELINE_COMMIT" env algorithm tests tools configs pyproject.toml \
  | tar -x -C "$OUT/baseline_source"
python -u tools/audit_v342_v343_history.py \
  --baseline-root "$OUT/baseline_source" --output "$OUT/history_equivalence.json"
```

## 实际新增与修改文件

新增环境与数学模块：`env/combat_v342.py`、`env/combat_v343.py`、`env/v342_weapon.py`、`env/v342_config.py`、`env/v343_config.py`、`env/v343_reward.py`、`env/v343_observation.py`。

新增配置：`configs/combat_environment_v342.yaml`、`configs/combat_environment_v343.yaml`、`configs/mappo_4v4_v342.yaml`、`configs/mappo_4v4_v343.yaml`、`configs/ippo_4v4_v342.yaml`、`configs/ippo_4v4_v343.yaml`。

新增测试/工具/文档：`tests/test_combat_v342.py`、`tests/test_combat_v343.py`、`tools/preflight_v342_v343.py`、`tools/verify_v342_v343_readiness.py`、`tools/audit_v342_v343_history.py`、本文档。

修改共享接入：`env/config.py`、`env/combat_env.py`、`algorithm/common/checkpoint.py`、`algorithm/common/evaluator.py`、`algorithm/mappo/runner.py`、`algorithm/mappo/evaluation.py`、`algorithm/ippo/networks.py`、`algorithm/ippo/trainer.py`、`algorithm/ippo/protocol.py`、`algorithm/ippo/runner.py`、`algorithm/ippo/diagnostics.py`、`tools/combat_visualization.py`、`tools/combat_replay_report.py`。没有修改任何历史环境专属实现或配置。

实验数据均在新结果目录：`preflight/episodes.csv`、`events.csv`、`comparison.csv`、`paired_initialization.csv`、`summary.json`；`cuda_smoke/` 中每个算法/版本的诊断 checkpoint、参数/梯度/更新摘要、评估 JSON、规范录像 trace 和元数据；以及各测试日志、baseline 对照日志、历史源文件指纹、重复性报告。旧模型、视频和统计结果不覆盖。
