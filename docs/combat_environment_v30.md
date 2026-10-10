# Combat Environment v3.0

v3.0 的定位是“基于同行公开空战参数构建的、适配现有三维点质量动力学的简化多UAV非完全信息空战环境”。它不是对某篇论文环境的逐项复现。3 km sensor、1 km weapon、90° sensor/weapon cone、6 枚弹药、70% 毁伤概率采用本次任务指定的同行参数思想；任务未提供具体论文出处，本文不补造来源。

## 配置与隔离

使用 `configs/combat_environment_v30.yaml` 显式选择新环境族；默认仍为 v2.3。v2.3–v2.9 的配置、动力学、控制器、积分器、Blue 脚本、武器与奖励实现保持原有路径。

| 参数 | v3.0 |
| --- | --- |
| 规模 / observation / action | 5v5 / 66 / 3 |
| 步长 / 最大步数 | 0.1 s / 1000 |
| 速度 / pitch | 150–300 m/s / ±π/3 |
| 三项控制时间常数 / normal load max | 2 s / 8 |
| 圆形战区 / 高度 | 半径 5000 m / 0–6000 m |
| 初始两队中心距离 | 5000 m，随机 radial 朝向 |
| 队形横向偏移 | −600、−300、0、300、600 m |
| 初始高度 / 速度 / heading 扰动 | 3000±100 m / 225±10 m/s / ±5° |
| sensor / weapon | 3000 m / 1000 m，真实三维 off-boresight ≤90° |
| ammo / 命中率 | 每机 6 枚 / 固定 Bernoulli(0.7) |
| Blue guard | radius ≥4500，altitude ≤500 或 ≥5500 m |
| Blue desired speed / guard-search altitude | 250 m/s / 3000 m |

上述参数未进行 sweep 或依据验收胜率调参。

## 探测、观测与网络

`env/sensor.py::sensor_geometry` 是 observation、Blue、potential 共用的三维探测定义：双方存活、距离 ≤3000 m、真实三维 off-boresight ≤90°。`is_detected` 是同一实现的标量入口。各 Red 独立探测，友机完整共享。

66D = self8 + ally4×7 + enemy5×6。self 第八项为剩余弹药 / 6。未探测或死亡敌机的整个六维 slot 为零；死亡 Red 整个观测为零。不会提供隐藏敌机的距离、角度、生命或弹药诊断给 actor。EA/STEA 使用显式 `Linear(8,64)`，旧版本 self7 保持 `Linear(7,64)`；其余网络及 PPO 参数不变。centralized critic 输入为 5×66 + 66 = 396。

新增 `mappo_5v5_v30.yaml`、`rmappo_5v5_v30.yaml`、`ea_mappo_5v5_v30.yaml`、`stea_mappo_5v5_v30.yaml`。与各自原 5v5 配置相比仅 observation_dim=66；EA/STEA 另增加 self_feature_dim=8。checkpoint 的环境版本、dimensions、配置 SHA 和算法版本仍严格校验。MADSAC/MADDPG 的正式支持范围不变。

## 武器与 Blue

武器资格仅要求双方存活、还有弹药、0≤距离≤1000 m、真实三维前向锥≤90°。没有 speed gate 或 rear-aspect gate。命中和未命中均消耗一枚弹药，命中随机性仅使用环境 RNG。

每个 attacker-target pair 保存上一时刻资格，只有 false→true 的 entry 能尝试。每架 attacker 每步至多一次，选择最近目标、距离相同时按目标 index。**同一步所有 entry 都被消费，未选中的 pair 也不能在持续合法期间补射**；必须离开再重入。先生成双方全部 attempt，再同时结算击杀；多个实际命中攻击者平分目标的 +10。

Blue 为无状态、确定性的 sensor-limited pursuit：先 guard，触发时回到圆心并朝高度 3000 m；否则追踪最近可见存活敌机；没有可见目标时朝圆心搜索。速度始终期望 250 m/s。没有 Stern、target lock、evade、复杂论文 Blue、assignment、通信或策略随机性。外部 Blue action override 仍有效。

## 非战斗损失与结局

双方规则一致：altitude≤0 为 ground loss；radius>5000 或 altitude>6000 为 boundary loss。ceiling loss 同时记录为 boundary，但只扣一次 −10；地面与边界不会重复计罚。Blue 非战斗死亡不给 Red kill credit。

双方全灭为 mutual-destruction draw；单方全灭为另一方 elimination win。仅双方仍有存活机且达到 max_steps 时为 timeout，按幸存数量比较胜负，相等为 draw。`red_success == red_win`。终止原因分别为 `draw_mutual_destruction`、`red_win_elimination`、`blue_win_elimination`、`red_win_timeout_survivors`、`blue_win_timeout_survivors`、`draw_timeout_equal_survivors`。

## 四部分奖励

总奖励 `r = event + outcome + adv + safe`，不调用旧状态 shaping。

- `event`：攻击击杀 +10，多个实际攻击者平分；自身战斗/地面死亡 −10，边界死亡 −10，每次自身死亡仅扣一次。
- `outcome`：终局所有五个 Red slot（含死机）胜 +2、败 −2、平 0；其他步为 0。
- `adv = 1 × (0.99 Φ(next) − Φ(current))`，只对各自最近可见敌机计算。无可见敌机或自身死亡时 Φ=0。next 在动态、非战斗损失和同时战斗全部结算后计算，与返回 observation 的状态一致；timeout 仍按该公式使用实际 next-state potential。
- `safe = −0.2 max(0, 1 − d_min/200)`，仅存活 Red，d_min 是自身到任意其他存活 UAV 的真实三维最小距离。死亡实体不参与；没有其他存活实体时为 0。没有碰撞动力学。

`Φ = 0.5 p_d + 0.5 p_a`。距离项：d<200 时 `2d/200−1`；200≤d≤1000 时 1；1000<d<3000 时 `(3000−d)/2000`；d≥3000 时 0。角度项 `p_a=(cos(ATA_3D)+cos(AA_3D))/2`，AA 为目标速度方向与 attacker→target LOS 的夹角；同向尾追为 1、正面对头为 0。AA 仅用于 shaping，不影响开火资格。

潜势及其诊断不使用隐藏敌机。按任务明确指定，soft safety 单独使用所有真实存活 UAV（包括未探测敌机）的间距；它是奖励反馈，不增加 actor 观测字段。没有额外高度、速度优势奖励、原论文距离优势公式、非对称边界或失败正奖励。

`info` 提供四个 explicit per-agent reward、各 component episode totals、current/next Φ 及距离/角度均值诊断。r1/r2/r3/r4 是四个新 component 的兼容别名，只用于已有统计，不重复加入总奖励。

## 验收与复跑

在 WSL Ubuntu `conda activate uav` 后运行，CUDA 检查失败会明确报错：

```bash
python -m pytest tests/test_combat_v30.py -q
python -m pytest -q
python -m compileall -q algorithm env tools tests
python tools/audit_combat_v30.py --episodes 1000 --random-episodes 32 --workers 16 --output outputs/v30_validation/audit
python tools/smoke_formal_8v8.py --env-config configs/combat_environment_v30.yaml --output outputs/v30_validation/cuda_smoke_new
```

audit 对称两侧使用同一确定性 Blue policy，1000 个固定 seed，另做 32 场 random Red；检查 10000 组三维几何、1000 组隐藏状态、弹药守恒、attempt/hit/kill 次序及死亡原因计数。结果位于 `outputs/v30_validation/audit/summary.json`，每场记录同目录保存。短流程每算法仅 512 sampled steps、16 个真实 spawn worker、一次真实 PPO update、保存并严格加载 checkpoint、两场 CUDA evaluation；不是性能实验，也未启动正式 3M 训练。

本次 1000 场对称结果：Red 41.6%、Blue 40.6%、draw 17.8%、timeout 0%，绝对胜率差 1 个百分点，平均长度 114.527 步。双方非战斗损失均为 0；Red 胜局中的 2080 个 Blue 死亡全部来自攻击。random Red 32 场中 Blue 仍无非战斗损失。完整结果以 JSON 和最终验收报告为准；不据此自动调整参数。

验收结果：专项 pytest 63 passed；完整 pytest 782 passed（278.29 s）；compileall 和 git diff --check 均通过。专项包含 RMAPPO/STEA 在 66D 上的 episode-start hidden 清零、death hidden/action 清零。四算法 CUDA smoke 均通过真实更新、有限梯度/参数/value、严格 checkpoint 重载及两场 evaluation。使用 NVIDIA GeForce RTX 5060 Laptop GPU；四个同 seed centralized critic 初始参数完全相同。29 个受保护旧配置及环境核心文件的 SHA256 全部保持一致，旧版本 fixed-seed 与 checkpoint 回归继续通过。
