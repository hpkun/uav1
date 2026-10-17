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
150 m/s 的简化运动学 exploit。这些是设计意图，尚未经训练对照实验证实。

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

算法及 checkpoint validator 均未修改，没有新增算法配置、加载历史模型做
跨环境实验或启动训练。v3.38 checkpoint 的身份仍是 3.38，不能冒充 3.39；
配置 SHA 也不同。当前任务只注册并验证环境，不扩展算法各自的版本支持范围。

验证入口：`tests/test_combat_v339.py`。覆盖精确边界、3D 距离、MRO 调用、
目标切换、渐进收敛、自动动作、reset 等同、完整对局同动作等同、
canonical observer 单次调用及 RNG 等同、实际 trace 读写、元数据隔离和配置拒绝。
历史测试不修改 expected results。

2026-10-17 验证结果（WSL 访问被拒，按仓库规则使用 Windows `uav`）：

- v3.39 专项：49 passed。
- 所选历史回归：782 passed、6 failed、1 skipped。包括完整 v3.38、v3.37、
  v3.36、v3.35、v3.34、v3.32，以及 combat environment、canonical replay、
  policy modes、MAPPO environment identity 测试。
- 6 个失败位于 v3.34（4 个）和 v3.32（2 个）的历史 MADSAC checkpoint
  测试：期望 `environment_version` 异常，但当前 MADSAC v5 提前拒绝 v3/v4
  implementation。已用隔离临时包中的 HEAD 原版 `env/config.py` 和
  `env/combat_env.py` 复跑这 6 项，全部复现相同失败；未修改算法或旧测试。
- 1 个跳过项因 FFmpeg/ffprobe 不可用。v3.39 正式 trace 读写、完整对局
  observer 等同及单次策略调用均通过；未声称视频编码测试通过。
- 合计所选测试：831 passed、6 existing failures、1 skipped；新增失败 0。
  未执行全仓库 pytest。CUDA checkpoint/runner 相关回归使用 RTX 5060 Laptop GPU。
- `git diff --check` 通过；未启动训练。
