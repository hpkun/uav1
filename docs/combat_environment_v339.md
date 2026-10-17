# Combat Environment v3.39

v3.39 directly inherits v3.38. 唯一行为变化是 **Blue speed command**。

`CombatEnvironmentV339(CombatEnvironmentV338)` 仅在初始化后替换 `fixed_policy`。
`DistanceTierNearestTargetBluePolicy(PaperNearestTargetBluePolicy)` 仅覆写
`action()`：复用 `nearest_target_index()`、`action_toward()`，完整继承
`team_actions()` 和 target-switch diagnostics。没有新增控制状态或 RNG。

对每架存活 Blue，使用当前最近存活 Red 的三维距离：

| 当前目标距离 d | desired speed |
|---|---:|
| d > 3000 m | 240 m/s |
| 1000 < d <= 3000 m | 220 m/s |
| d <= 1000 m | 200 m/s |

因此恰好 3000 m 使用 220，恰好 1000 m 使用 200。全部初始速度仍为
225 m/s。速度动作继续为 `clip((desired_speed - own.v) / 2.5, -1, 1)`，
由原运动学每仿真步最多改变 2.5 m/s（dt=0.1 s），禁止直接赋值状态速度。
实际速度范围仍为 150–300 m/s。

Blue 仍然每步选择 nearest living Red，立即 retarget，以原 LOS heading /
elevation 做 pure pursuit；没有目标或自身死亡时返回零动作。新增速度选择
仅使用目标距离；速度动作还使用自身速度。没有 hysteresis、目标速度匹配、
能量管理、协同、规避或 fire-aware 控制。

这是 **PROJECT DESIGN / CALIBRATION**，不是 MADSAC 论文公开参数。
设计目的是去掉全程 225 m/s 的机械恒速，以简单规则提供远距离温和追赶、
中距离正常接敌和近距离较低 closure / overshoot；不复制 Red 长期贴
150 m/s 的简化运动学 exploit。这些是设计意图；现有 frozen Red A/B
实测结果见[训练 readiness 文档](combat_v339_training_readiness.md)。

配置相对 v3.38 仅 `environment_version`、`blue_policy` 不同。
五个 Blue policy 字段全部显式提供；严格拒绝缺失、额外字段（含旧
`desired_speed`）、非有限数、非法阈值和超出 150–300 的目标速度。
其余配置反向映射至 v3.38 validator 检查。正式配置使用上表数值。

Red 动作、Reward、1 km PaperATAHAWeapon、命中公式及标定、6 发弹药、
1 s cooldown、编队、初始化、45D observation、传感器、运动学和 outcome
均继承 v3.38。显式提供相同 Red/Blue actions 时，两版完整 transition、
Reward、非版本身份 info、ammo、cooldown、weapon outcomes 和 RNG 一致。

原有 `blue_paper_nearest_agent_steps`、`blue_target_switches`、
`blue_last_target_indices` 继续有效并由继承的 reset 清零。本版不新增
可选速度诊断状态；现有 canonical recorder 已记录实际 Blue actions、
逐帧速度与 pursuit target indices，不额外调用策略。

v3.39 已完成正式训练接入。MAPPO、MAPPO-CA、EA-MAPPO、EAA-MAPPO、
MAAC、MADDPG、MADSAC 均有独立的 v3.39 正式配置。
v3.38/v3.39 checkpoint 按环境版本、配置 hash 及适用的 checkpoint family
严格隔离，不能互相冒充或用于跨版本正式 resume/evaluation。
详细配置、训练 readiness、CUDA smoke 和 frozen Red A/B 结果统一见
[训练 readiness 文档](combat_v339_training_readiness.md)。当前未启动 3M 正式训练。

验证入口：`tests/test_combat_v339.py`。覆盖精确边界、3D 距离、MRO 调用、
目标切换、渐进收敛、自动动作、reset 等同、完整对局同动作等同、
canonical observer 单次调用及 RNG 等同、实际 trace 读写、元数据隔离和配置拒绝。
历史测试不修改 expected results。

环境专项验证已通过。当前训练 readiness 的最终验证结果（包括测试通过、
历史失败和跳过项的说明）以[训练 readiness 文档](combat_v339_training_readiness.md)
为准，本环境文档不重复维护测试计数或完整 A/B 统计表。
