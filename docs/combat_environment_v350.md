# v3.50：Li 等（2023）Reward 与任务胜负判定对齐

本版本仅对齐论文第 3.3.1 节的任务成功条件和第 3.3.4 节式 (25) 的奖励。
**不能称为完整论文环境复现，更不能将功能测试通过解释为论文 MAAC 性能复现。**
后续应先研究并确定飞行动力学方案；本轮不启动 3M 训练。

论文：[Li et al., Aerospace 2023, 10(7), 574](https://doi.org/10.3390/aerospace10070574)。
同时核对仓库已有论文抽取文本 `outputs/madsac_v341_paper_audit/paper_text.txt` 的第 3.3.1、3.3.4 节。

## 独立版本和冻结范围

- 新环境 `env/combat_v350.py` 直接继承 `CombatEnvironmentV348`。
- 新配置 `configs/combat_environment_v350.yaml`、`configs/mappo_4v4_v350.yaml`、`configs/maac_4v4_v350.yaml`。
- 环境配置相对 v3.48 只改变版本、移除 `research_reward`，并将 `reward.guide_distance_min` 从 3000 改为论文的 4000 米。
- 算法配置仅改变 `environment.environment_version`，其余网络、优化器、损失、超参数和默认采样预算逐字段相同。
- 保留 4v4、原始 v3.38 的 45 维观测和字段顺序、3 维动作、直接增量运动学、Blue 规则策略和步首绑定目标、初始场景、种子协议、0.1 秒步长、1000 步上限及原空域。
- 保留 v3.48 的 2000 米射程、30°三维攻击圆锥、距离/机体系高斯概率、6 发弹药、1 秒冷却、环境 RNG 与同步毁伤结算。
- 历史 v3.48/v3.49 环境、Reward 和配置文件不改；公共入口只新增 v3.50 分支及能力注册。

## 论文公式与实现对应

总奖励始终为逐 Red 的 `R1 + R2 + R3 + R4`，团队 Return 继续采用既有求和方式。

| 分量 | v3.50 规则 | 实现 |
|---|---|---|
| R1 | 击毁 Blue +10，Red 损失 −10 | 复用 `_event_reward_components` 和同步唯一毁伤结算 |
| R2 | Red 离开现有合法空域 −10 | 复用原有边界损失分类，不重新定义空域 |
| R3 | `abs(ATA_r)<=30° && abs(HA_r)<=30° && d_r>=4000` 时 +0.001，否则 0 | `paper_pair_rewards`；不使用三维离轴角 |
| R4 进攻 | `abs(AA_r)<=30° && d_r<=4000`；ATA/HA 同时满足 5°/15°/30°时依次 +0.1/+0.02/+0.01，否则 0 | 复用已验证的 `v346_reward.paper_branch` |
| R4 防御 | 进攻外层条件不满足时，使用 Blue 视角 `abs(AA_b)<=30° && d_b<=4000`；5°/15°/30°依次 −0.15/−0.025/−0.015，否则 0 | 同一 `paper_branch`，交换攻击方/目标 |

每个档位都要求 ATA、HA **同时** 满足，按 5°、15°、30°从严到宽只选一个，不能累加。
按式 (25) 外层列示顺序使用 `if R41_gate / else R42_gate`；进攻外层条件成立而内层奖励为零时不转入防御分支。
两外层条件在正常非重合几何下不能同时成立；重合等退化情况同样采用先列进攻分支的确定性约定。

角度使用现有、已核对的 `engagement_geometry`：ATA 是 LOS 水平角与本机航向之差，AA 是目标航向与 LOS 水平角之差，HA 是高度 LOS 角 `atan2(-dz,horizontal)`，不减本机俯仰角。
Blue 视角将两机角色互换后重新计算。攻击资格继续使用三维离轴角，但该角度不替代 Reward 的 ATA/HA。
角度保留项目 `angle_within` 的 1e-12 弧度浮点舍入容差，距离使用字面 `>=4000` / `<=4000` 比较。
4000 米处 R3 和 R4 可同时有效，它们属于两个独立分量，不能误当成档位重复。

## 4v4 必要实现补充：不声称来自论文

论文没有完整规定多目标奖励聚合和多人击毁分配，本版本明确采用以下补充：

1. 每架存活 Red 选择距离最近的存活 Blue，距离相同时按 Blue 槽位索引从小到大。
2. 同一个目标同时用于该 Red 的 R3 和 R4，选择过程与奖励值、弹药及 2000 米攻击资格无关；不跨目标拼接正负奖励。
3. 无存活目标或本机死亡时 R3/R4 为零，奖励目标索引为 −1。
4. 保留 v3.48 奖励采样时机：运动及非战斗损失处理后、同步战斗毁伤结算前。因此本步被击毁的飞机仍可能获得本步战术项，不改变原有 transition 顺序。
5. 多个 Red 同步命中同一 Blue 时，仅结算一架毁伤，将该事件总计 +10 在命中的不同 Red 间均分；不向每个命中者重复发 +10。
6. Red 撞地沿用 R1 损失 −10；水平越界/超高沿用 R2 −10。地面原因优先，同一飞机不会同时收到 R1 和 R2 的死亡处罚，既有死亡后不重复处罚。超高是边界损失子类，统计和奖励不再额外加一次。

## 相对 v3.48 删除或替换的 Reward

- R3 从 PBRS 改为论文的 +0.001 远距条件项，阈值固定 4000 米。
- 实际 Reward 不再调用距离/三维角度势函数、gamma 势差、终止势补偿。
- R4 保留论文六个常数和数学分段，删除跨 Blue 的最大进攻/最小防御聚合，改为选定的同一个最近存活目标。
- 不调用 v3.49 的候选连续 R4；不加入距离连续权重、尾后权重、连续对准项或额外攻击引导。
- 无安全、团队伤亡、终局胜负奖励贡献。配置中继承的相应系数严格锁为 0，仅保留兼容字段。
- `phi_*` 兼容日志为零，不是残留 PBRS 奖励；`pbrs_enabled=false`，不使用 research 命名报告论文分量。

## 胜负判定的真实差异

| 场景 | 当前 v3.48/v3.49 | v3.50 |
|---|---|---|
| Blue 全灭、Red 尚存 | Red 胜 | Red 胜：`red_win_elimination` |
| Red 全灭、Blue 尚存 | Blue 胜 | Blue 胜：`blue_win_elimination` |
| 双方同步全灭 | 平局：`draw_mutual_destruction` | **Red 胜：`red_win_mutual_destruction`** |
| 超时，双方存活数任意且 Blue 尚存 | 已经是 Blue 胜 | Blue 胜：`blue_win_timeout` |
| 正常进行中 | ongoing，无胜者 | ongoing，无胜者 |

代码核对发现，当前 v3.48 本来就没有按超时存活数决定胜负，因此不虚构这一项为新增修复。
双方全灭按用户指定的“全部 Blue 被消灭”字面条件判 Red 胜，论文并未专门讨论互毁。
继续使用原有存活状态/损失解析结果判断全灭，不另改 Blue 非战斗损失机制。
不附加任何终局奖励或任务失败惩罚。

## 训练、评估、Checkpoint 与回放

- 环境注册、45/3/4 维度、MAPPO/MAAC 配置身份和 CUDA 门禁支持 v3.50；旧配置默认行为不变。
- 新 Checkpoint 同时保存并校验版本、环境/算法配置 SHA256、Reward 语义和任务结果语义。v3.48/v3.49/v3.50 不允许错用模型。
- 统一从环境 `_outcome` 产生训练终止信息、评估胜率和回放终止原因。
- 分别保留 R1–R4 逐步数组、逐回合每机累计与团队总计；实际返回 Reward 与分量逐步一致。
- 保留真实攻击/成功命中/唯一击毁、弹药消耗、双方损失、Return、胜率。驻留指标明确不适用。
- `mutual_annihilation_draw_rate` 在 v3.50 为 0；额外记录 `mutual_destruction_red_win_rate`，不把互毁成功误记为平局。
- 可继续使用 `tools/replay_combat.py`。v3.50 回放报告明确注明只对齐 Reward/任务条件。

## 验证方法

所有测试及检查在 `uav` Conda 环境中执行。WSL 访问拒绝时使用 PowerShell；训练更新/模型审查必须 CUDA。

```powershell
python -m pytest tests/test_combat_v350.py tests/test_combat_v348.py tests/test_combat_v349.py -q
python tools/verify_v350_readiness.py --stage history --output outputs/v350_validation_20261024
python tools/verify_v350_readiness.py --stage mappo --output outputs/v350_validation_20261024
python tools/verify_v350_readiness.py --stage maac --output outputs/v350_validation_20261024
python tools/verify_v350_readiness.py --stage checkpoints --output outputs/v350_validation_20261024
```

上述输出目录是本次审查专用目录，包含修改前的 `baseline_source`。历史比对依赖该快照；重复执行应另建目录并先保存待比较版本，不能覆盖本次产物。
CUDA 功能检查显式限制 MAPPO 为 4096 个样本/1 次 PPO 更新、MAAC 为 1200 个样本/1 个原有更新周期（4 critic + 4 actor + 1 target）。使用完整配置网络，不进入正式训练循环。
另检查固定策略动作、固定种子随机环境、Checkpoint 重载、跨版本拒绝和回放 Reward 分解。

最终实测结果见 `outputs/v350_validation_20261024/验收报告.md` 与同目录 XML/JSON/日志。

## 尚未对齐论文的设计

本轮保持当前直接增量飞行动力学和限幅，不恢复论文飞行动力学/控制跟踪过程；武器仍为 v3.48 的 2000 米三维圆锥、概率命中、6 发和 1 秒冷却，不改为论文攻击机制。
原有观测、策略、初始场景等均按本轮冻结要求保留；不宣称已逐项复现论文的所有环境设定。
**本轮只完成 Reward 和胜负判定对齐，功能测试不证明原论文 MAAC 学习性能已经复现。后续先研究并确定飞行动力学方案，不启动 3M 训练。**
