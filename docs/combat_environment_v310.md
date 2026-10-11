# Combat Environment v3.10

本版只将 v3.9 的 Blue tiered policy 替换为全局 nearest-target pursuit。没有对照实验、参数扫描或正式训练。v3.0–v3.9 的策略、配置和轨迹不回写。

## 论文直接描述：Section 2.5 Fixed Strategy

- 每个 simulation step 获取所有 Red UAV 的位置信息。
- 选择最近 Red 作为 pursuit target；最近目标改变则立即切换。
- attack phase 保持相同 pursuit strategy。
- 开火资格与命中由 attack model 自动决定。

## 当前实现约定

每个存活 Blue 每步重新选择所有存活 Red 中 `(distance_squared, Red_index)` 最小者：

```
distance_squared = (red.x-blue.x)^2
                 + (red.y-blue.y)^2
                 + (red.z-blue.z)^2
```

完全相同距离取较小 Red index。不存在目标锁定、保持时间、迟滞、概率切换。旧目标死亡后当前 step 立即重选。Blue 已死亡或所有 Red 死亡时返回 `[0,0,0]`。

`PaperNearestTargetBluePolicy` 继承 `NearestTargetPursuitPolicy`。`nearest_target_index`、`action`、`action_toward` 为原方法本身；wrapper 的 `team_actions` 直接调用原方法，只额外计算诊断，不复制 heading/elevation/control 数学。

heading 为目标当前 XY LOS；elevation 沿用现有 NED `atan2(own.z-target.z, horizontal_distance)`。使用原连续 action controller，desired_speed=250 m/s。250 是当前场景固定策略参数，**不是论文给出的唯一 Blue 巡航速度**；论文允许速度范围为 150–300 m/s。

Blue 目标选择**不使用 radar visible mask**。即使目标在 Blue FOV 外、无目标可见、人数劣势或 Red 很分散，仍追当前最近存活 Red。Red 的 66D observation 编码及雷达权限保持原样，没有额外全局 Blue 信息。

## 删除的正式 Blue 决策

唯一决策为 `GLOBAL_NEAREST_PURSUIT`。正式 v3.10 不再有 escape、guerrilla、centripetal、雷达门控目标选择、close_distance、dispersion_threshold、人数比较、质心/RMS spread 或随机目标。

`reset(seed=None)` 仅清零诊断，seed 不参与行为；新 policy 不创建 RNG、无 rng 成员、不调用随机采样。父类构造的临时旧 policy 在 `__init__` 结束前替换，未参与任何 simulation step；运行时父类 reset 调用新的兼容 reset。初始化与 Gaussian hit 仍使用原环境 RNG，未接入其它流。

## 诊断

- `blue_paper_nearest_agent_steps`：累计存活且存在目标的 Blue agent decisions。
- `blue_target_switches`：上次与本次目标均非空，且 index 不同的累计次数。首次获取目标、死亡变空不算 switch；旧目标死亡后改选另一 index 算 switch。
- `blue_last_target_indices`：最近一次自动 Blue 决策的每架目标 index；无目标或 Blue 死亡为 None。

诊断对应每步 **dynamics 前** 的真实状态；该步随后可能发生命中/死亡，所以不能拿 dynamics 后状态反推所选目标是否合法。诊断不参与 action、reward 或 target selection。

正式 evaluator/训练 rollout summary/运行 summary 提供平均 `blue_paper_nearest_agent_steps` 和 `blue_target_switches`。不输出不存在的 escape/guerrilla/centripetal 行为统计。历史版本保留原诊断。

外部显式 Blue action override 的语义仍保留：不调用 fixed policy，不更新 pursuit/target counters；info 不伪造自动决策。

## 冻结的环境与算法协议

`CombatEnvironmentV310` 直接继承 `CombatEnvironmentV39`，只实现 `__init__` 替换 fixed_policy，不 override reset/step：

- dt=0.1、max_steps=1000；arena radius=5000 m、altitude=0–6000 m。
- v3.8 chord 10 km diameter initialization、lateral/altitude/speed/heading 分布和 RNG 顺序不变。
- sensor range=10000 m、FOV=π/2；observation=66D、action=3D。
- weapon range=2000 m、ATA/HA≤30°、ammo=6、Gaussian Dhit/c4/c5、strict pair-entry、simultaneous resolution 不变。
- R1/R2/R3/R4、guide_distance_min=4000 m、tactical_distance_max=4000 m、safety、paper/extended 分解全部不变。
- Blue boundary return 仍由原 `_resolve_noncombat_losses` 强制位置/高度修正；这是既有环境边界规则，**不是新增追中心或防御决策**。Red 越界损失规则不变。

新 Blue 配置只有 `desired_speed: 250.0`，validator 拒绝其它字段及速度改动。配置校验适配器只为调用历史 validator 恢复旧 schema，不用于运行时决策。

四份 v310 算法 YAML 与 v39 字节相同，网络/GRU/attention、学习率、gamma/lambda/entropy、PPO epochs/rollout/minibatch、evaluation/checkpoint cadence 不变。环境版本支持加入 config/factory/dimension dispatcher/evaluator 和正式算法 allowlists，checkpoint contract 仍检查新版本与 SHA。

## 验证范围

专项覆盖全局 3D 最近、FOV外追击、即时切换/死亡重选、tie、人数劣势、分散、无目标、NED同高度/上方/下方、100个随机合法场景与原策略逐值一致、无RNG、相同seed重复轨迹、显式Blue控制下Gaussian RNG一致、边界返回和奖励分解、历史 v3.9 完整 reference。

仅在 v3.10 做 ZERO/PURSUIT × seeds 1、31、10000000 的 sanity，逐 agent-step 验证最近目标，并记录 selected indices、switches、首次 fire-window/attempt/kill 和回合长度。没有 v3.9/v3.10 对照实验，也不按结果调参。

还运行完整 pytest、compileall、真实 spawn vector reset，以及四算法各512 transitions/一次真实update/save-reload/2回合deterministic evaluation CUDA smoke。完成后停止，不自动启动1M/3M训练。
