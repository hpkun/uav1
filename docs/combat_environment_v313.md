# Combat Environment v3.13

v3.13 在 v3.12 基础上修正 Height Line-of-Sight Angle（HA）。按本任务给定的 MADSAC Equation (6) / Figure 2 约定，HA 描述目标相对于攻击者水平面的垂直视线角：

```
horizontal = hypot(dx, dy)
los_elevation = atan2(altitude_target - altitude_attacker, horizontal)
HA = abs(los_elevation)                       # [0, pi/2]
```

NED 内部等价于 `atan2(-(z_target-z_attacker), horizontal)`。公式不读取攻击者或目标的 pitch。仅改变 pitch、保持位置及 heading 时，distance、ATA、AA、HA 均保持不变；实际飞行中 pitch 仍会通过后续运动改变位置关系。

历史 v3.12 及此前版本保留 `angular_difference(los_elevation, own_pitch)`，表示垂直瞄准误差，仅用于历史兼容。v3.13 不修改 `v36_geometry.py`。ATA/AA 仍是 LOS 水平 heading 与自身/目标 heading 的最小圆周角差，真实三维距离定义也不变。

垂直正上/下方且不重合时 HA=π/2。完全重合时 ATA=HA=0，AA 继续使用攻击者 heading 与目标 heading 的稳定角差；带符号 los_elevation 显式为 0。历史 helper 在重合时使用 own_pitch 作为 elevation 占位值，此特殊值仅在新模块中改为无方向的 0；非重合情形 signed LOS 与旧版逐值相同。

`CombatEnvironmentV313` 继承 `CombatEnvironmentV312`，只覆盖 version、build_weapon、attempt_hit_for_pair、dense_combat_reward、_observations。它不复制 step、reset、动力学或初始化。weapon eligibility、hit sampling 的 HA 输入、R3、R4 offense、reverse R4 defense 和 enemy observation 全部使用 `combat_geometry_v313`。

开火仍要求双方存活、ammo>0、0≤d≤4000、ATA≤30°、HA≤30°；AA 和 radar 不参与。Equation (8) 原采样实现直接继承，每次发射只采样一次 epsilon，并共享给 ATA/HA。D_hit=4000/ln(6)，c4=c5=0.9984711359155588 不变；这些是项目透明标定，并非论文公布值。

R3 仍是 d≥4000、ATA/HA≤30° 的任一活目标给予 +0.001。R4 保留 30°/15°/5° 分级、offense 最大值和 reverse defense 最小值；R1/R2/safety、PBRS off、terminal=0、casualty=0 全部不变。

Observation 仍为 Self7 + 4×Ally6 + 5×Enemy5=56D，固定 identity，死亡 slot 全零。Self/Ally 归一化、flight-path frame、phi=0 完全保留。所有活 Blue 均可观测，不加入 ammo/alive/radar flag，不使用 radar gate。Enemy 为：

```
[d/10000, (v-225)/75, AA/pi, ATA/pi, HA/(pi/2)]
```

HA 0°/30°/45°/90° 分别映射至 0/1⁄3/0.5/1。Self theta 和 Ally body-frame 仍可能随自身 pitch 改变，不能误要求整个 observation 对 pitch 不变。

环境 YAML 与 v3.12 仅版本号不同；validator 委托冻结的 v3.12 参数合同。四算法 YAML 是 v3.12 的字节副本，网络、PPO、centralized critic 协议不变。虽然两个版本均为 56D，v3.12 checkpoint 必须因 environment_version 不匹配而拒绝用于 v3.13。

验证产物位于 `outputs/v313_validation`：修改前源码哈希、六条完整 v3.12 轨迹 reference、专项/full pytest、compileall、真实 spawn vector reset、四个旧 checkpoint 隔离、同状态差异审计，以及每算法 512 transitions / 1 update / save-reload / 2 deterministic CUDA evaluation。详见该目录的 `report.md`。没有正式训练或参数调整。
