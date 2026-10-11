# Combat Environment v3.9

v3.9 直接继承 v3.8 的 10 km 直径、随机直径端点附近的 chord 初始化、5v5、66D observation 和 3D action。只修改雷达距离、R3/R4 距离分段及对应奖励诊断。历史 v3.0–v3.8 不回写。

## 实际雷达与武器配置

雷达使用原有 true 3D 距离和前向视场，range_max=10000 m，off_boresight_angle_max=π/2；角度模型没有改变。MADSAC 论文描述机头前方扇区扫描，未给出可以直接采用的数值探测距离。10 km 是本项目为适配 10 km combat airspace、减少完全无目标信息阶段的设置，**不是论文雷达参数**。

武器仍为 range_max=2000 m、ATA/HA≤30°、每架 6 发弹药，Gaussian Equation (8) 的 Dhit/c4/c5 标定、strict pair-entry、同时结算全部保持 v3.8。论文 Table 1 的 D_firemax=4 km 与项目的 2 km 武器存在差异；本次不改变该差异。2–4 km 内可以有战术奖励，但不能开火。

## R3 / R4

R3 每个存活 Red 只要存在一个满足条件的目标，就获得 0.001，不随目标数量相加：

```
visible && d >= reward.guide_distance_min (4000 m)
        && |ATA| <= 30° && |HA| <= 30°
```

R41 使用 Red→Blue 几何，visible、d≤reward.tactical_distance_max (4000 m)、|AA_r|≤30°时，按 ATA/HA 同时满足 30° / 15° / 5° 的最强级别给 +0.01 / +0.02 / +0.10。

R42 使用 Blue→Red reverse geometry，双方存活、d≤4000 m、|AA_b|≤30°时，按 Blue ATA/HA 的最强级别给 −0.015 / −0.025 / −0.15。防御项保留原实现：不要求威胁目标处于 Red 的 radar FOV，不额外加入 Blue-visible gate。

每个 Red 的 offense 取 strongest positive level，defense 取 strongest negative level；R4=offense+defense，不对多个目标求和。死机不贡献奖励。几何奖励仍在 dynamics 后、同时命中结算前计算，避免击毁目标导致当步几何奖励丢失。

**d=4000 m 同时满足 R3 的 ≥ 和 R4 的 ≤，不排除重叠。** 两个 reward 距离参数独立于 weapon range。

与 v3.8 比较：R3 从 visible 且 2000<d≤4000 改为 visible 且 d≥4000（受 10000 m 雷达约束）；R4 从 d≤2000 改为 d≤4000。所有角度、级别系数、聚合方式不变。

## Paper-defined reward vs 5v5 implementation

| 条目 | Equation (25) 的依据 | 本项目实现约定 |
|---|---|---|
| R1 | 击毁 +10，被毁 −10 | 同时多攻击者共享 +10；ground loss 映射为 destroyed/loss |
| R2 | 出空战区 −10 | boundary 只进入 R2，不重复进入 R1 |
| R3 | ATA/HA≤30°、d≥4 km、+0.001 | visible gate；满足任一目标只奖励一次 |
| R4 | AA≤30°、d≤4 km、30/15/5°三级系数 | strongest-target 聚合、offense 与 defense 相加 |
| Sensor | 机头前方扇区；未提供数值距离 | range=10 km、原前向 FOV |
| Safety | 不属于 Equation (25) | 保留 5v5 collision/proximity penalty，safe_distance=200 m、scale=0.2 |

R4 以正式 Equation (25) 的 AA≤30° 为权威定义，不用前文 AA<90° 的定性优势描述替代。

`paper_reward = R1+R2+R3+R4` 指这四项的实现分解，包含上述 5v5 实现约定，并不宣称仿真与论文逐字完全等价。

`extended_reward = paper_reward+safety` 是实际返回给算法的训练奖励。team casualty/outcome 为 0，PBRS off；没有额外爬升、生存、边界距离 shaping。

v3.9 用 float64 直接返回 extended_reward，避免先经过父类 float32 中间结果后产生累计分解误差；算法仍使用原有 tensor conversion，网络、更新流程、超参数均未改。`local_rewards` 与实际返回值同步。

每步 info 提供 `paper_rewards`、`extended_rewards`、`safety_rewards`（前两者也有单数别名）；累计提供 episode_r1/r2/r3/r4_total、episode_paper_reward_total、episode_safety_total、episode_extended_reward_total 以及逐 agent 累计向量。reset 时全部归零。

四算法共享 runner 的训练 JSONL 记录 mean_paper/safety/extended_reward 和完成回合累计均值；正式 evaluator 输出 average_episode_paper_reward/safety/extended_reward_total。v3.9 专属分支不会为历史版本增加字段。rollout 中没有完成回合时 episode 累计均值为 null，不伪造为已完成 episode。

## 隔离与验证

新类只 override dense_combat_reward、reset/step 的最小诊断包装和计数 reset；不复制 step 主流程。v3.8 初始化与 RNG、weapon、controller、dynamics、Blue policy 代码、边界语义完全继承。Blue 使用共享 sensor 配置，所以会按同一 10 km 雷达感知，但没有增加新的策略规则。

四份 v39 算法配置与 v38 对应配置字节相同；运行时另行指定环境配置。checkpoint 仍严格检查环境版本与 SHA。v3.8 的 3M 最终 checkpoint 不可跨版本用于 v3.9 审计，不绕过 metadata 检查。

验证包括：精确距离/角度/级别、4 km 闭区间重叠、2–4 km 奖励与开火分离、100 个初始化 seeds、step/episode 分解一致性、真实 spawn-worker reset、历史完整轨迹与哈希、完整 pytest/compileall、四算法 512 transitions/1 update/save-reload/2 episode CUDA smoke。

小规模 ZERO/PURSUIT 对照用 seeds 1、31、10000000。first detection 包含 reset 的 step 0；first R3/R4 统计实际 step 返回的非零奖励，从 step 1 开始；first fire-window/attempt/kill 取双方最早事件。缺失事件为 null。只报告，不按结果调参，不启动正式训练。
