# Multi-UAV Combat Environment v2.4

v2.4 是独立的 8v8 普通单回合协议；默认入口仍使用 `combat_environment.yaml` 的 v2.3 4v4。两者共用 3DOF 点质量动力学、RK4、动作控制器、概率命中与 R1–R4 奖励，使用明确的版本分支隔离武器资格和入窗状态。没有 missile、ammo、HP、radar、communication、learned Blue 或复杂脚本 Blue。MADSAC 明确仅支持 v2.3。

## 正式参数

| 项目 | v2.4 |
|---|---|
| 每方飞机 / 每机动作 / 每机观测 | 8 / 3 / 104 |
| dt / max steps / 场地半径 | 0.1 s / 1000 / 5000 m |
| 编队中心间距 / 高度 | 8000 m / 3000 ± 100 m |
| 初速 / 航向扰动 | 225 ± 10 m/s / ±5° |
| 编队横向 offsets | −1050, −750, −450, −150, 150, 450, 750, 1050 m |
| 飞机速度 / pitch 限制 | 150–300 m/s / ±60° |
| Blue | 原 NearestTargetPursuitPolicy，desired_speed=250 m/s |
| 攻击距离 / 真正 3D off-boresight | 0–4000 m / ≤30° |
| target aspect / 速度 | ≤45° / attacker.v ≥ target.v |

四项攻击资格必须同时满足。`target_aspect = abs(aa)`，其中 `aa = wrap(target.psi − LOS_horizontal)`；攻击者位于目标尾后且同向时为 0，迎头时为 π。这是现有水平目标姿态角定义，不能误称为 3D target-aspect；自身攻击锥仍使用机体前向向量与 3D LOS 的夹角。速度严格比较，允许相等，不加 margin。

资格成立后执行一次原概率模型：`threshold = π exp(−distance / 2232.442506204989)`，两次独立标准正态噪声，azimuth/elevation noise scale 都为 1；命中即击毁。资格成立不保证命中。双方以同一 pre-hit 状态判断攻击，再同时结算击杀，允许互相击毁及多人命中同一目标的共享奖励。

## 入窗状态

v2.3 保留每个 attacker 一个 `FireState.armed` 的旧逻辑。v2.4 双方分别维护 8×8 的 `PairFireState.armed`：

1. reset 后所有 pair armed=True。
2. pair 未满足完整资格，或任一实体死亡：该 pair armed=True。
3. 在 qualified AND armed 的目标中选择距离最近者，距离相同按 target index。
4. 每个 attacker 每步最多尝试一次，仅被选中的 pair 变为 False，未选中的 pair 保持 True。
5. 同一 pair 持续合格不能再次尝试；离开完整资格区后重新进入可再次尝试。

因此对 A 尝试过不会阻塞另一个合格且 armed 的 B。几何预筛选只提前排除不满足速度、距离、aspect 的 pair，不改变门限、目标排序或随机数消耗。

## 观测与算法

`observation_dim_for_team_size(N) = 7 + 7(N−1) + 6N = 13N`。v2.4：自身 7D、按原索引顺序排列的 7 个 ally×7D、8 个 enemy×6D。死亡槽位清零，死亡自己的整行观测清零；归一化与每槽特征不变。旧 `OBSERVATION_DIM=52`、类默认维度 4/52/3 保留，只在实例上读取实际协议。

向量环境先接收 worker 维度元数据，再分配 `[E,N]` alive mask 和 `[E,N,13N]` observation；使用长期存活的 spawn 进程并行。MAPPO runner、集中 critic、rollout、评估和 checkpoint 兼容检查读取实际 N/obs。STEA 按末维除以 13 推导实体数，保持全部参数名和 v2.3 state_dict 结构；8v8 输入为 `[E,8,104]`，序列为 `[B,L,8,104]`，hidden 为 `[E,8,128]`。死亡、episode reset 均清除对应 GRU memory。

8v8 两份 YAML 使用 16 env、2M sampled steps、rollout 256、PPO epochs 10、minibatch 512、eval interval 100k，其余优化器和基线超参数不变。STEA 保持 entity_dim=64、heads=2、spatial=128、GRU=128、L=32、critic=256。实现版本仍为 MAPPO=2、STEA-MAPPO=1。checkpoint 显式记录实际环境版本、obs/action/agents 和配置指纹，跨环境版本拒绝加载到训练/评估协议。

## 难度与设计参照

难度来自 4v4→8v8 的多实体交互、目标尾后姿态限制、攻击者速度优势、以及逐 pair 的完整资格入窗语义。所有精确门限由本项目需求规定，不宣称完整复现任何论文。

本需求给出的同行环境设计对照为：

- [CAAFF](https://doi.org/10.1109/TITS.2025.3530463)：简化 3DOF，并联合攻击角、目标后方/逃逸角、速度、距离定义优势攻击条件。
- [MTVO](https://doi.org/10.1016/j.engappai.2023.106358) 及相关简化 multi-UAV 环境：简化 3DOF 短距空战与脚本最近目标追逐策略。
- [E-MATD3](https://doi.org/10.1016/j.knosys.2024.112000)：简化 3DOF、多 UAV、概率攻击及实体数量变化。
- [SIIS-MARL](https://doi.org/10.1109/TAI.2025.3567431)：4v4、8v8、16v16 等规模用于研究协同复杂度随规模增加的问题。

文献核验边界：本次可获取的 MTVO 出版社文本确认短距空战、多实体 attention 和不同机群规模；E-MATD3 出版社文本确认 3D 多机环境、attention/death masking 与数量变化。CAAFF、SIIS-MARL 全文当前无法获取，上述具体环境/规模对照沿用任务提供的描述，未独立核验正文；不能据此把 45°、4000 m、速度相等允许攻击或本项目入窗规则归为论文原始参数。E-MATD3 提到传感器也不表示本项目实现了传感器模型。

## 验证与复现

在 WSL Ubuntu 的 `conda activate uav` 后运行，训练与 checkpoint audit 必须有 CUDA：

```bash
python -m pytest -q
python -m compileall -q env algorithm tools tests
python tools/audit_combat_v24.py --episodes 1000 --workers 2 --trace-count 10
python tools/review_combat_v24_traces.py
python tools/smoke_combat_v24.py
```

审计 Red、Blue 都使用原 250 m/s 最近目标脚本，种子固定为 40000000–40000999，两版本各 1000 局。物理仿真使用 NumPy，审计 coordinator 检查 CUDA，worker 不导入 PyTorch。输出逐局事件、几何分布、首次事件的条件均值与样本数、双方差异和至少十条完整逐步轨迹。trace review 核验尾后投影、完整资格与重复 pair 之间的失效区间，并绘制三张真实 3D 轨迹。

审计诊断阈值是本工具的 review flag：攻击资格任何违规；双方胜率相差 >20 个百分点；双方尝试率均 <10%；超时 >90%；半数以上对局在 50 步内结束；非战斗损失超过一半。这些不是论文定义，也不代表阈值内必然适合训练；高超时仍应结合 fire/kill 和代表性轨迹解释。发现严重异常时保留正式参数、停止 CUDA update smoke 并报告。smoke 工具重新读取逐局记录检查诊断，并要求匹配配置指纹的 1000 局审计通过后，才各执行 512 sampled steps 的正式架构 rollout/update、checkpoint 保存/加载与确定性评估。

旧版验证包含修改前捕获的 6 组 reset+256 transitions 完整观测/奖励/info SHA256 指纹，以及已有两份 2M checkpoint 在修改前与修改后源码下的同种子 CUDA 评估。实际本次结果详见 `outputs/v24_validation/acceptance_report.md` 与 `outputs/v24_combat_audit/`。
