# Multi-UAV Combat Environment v2.5

v2.5 在最终 v2.4 的纯几何战斗协议上，把正式对局规模从 8v8 降到 5v5。使用独立版本和配置，保留 v2.4 的 8v8 实验、配置及 checkpoint contract；无参数默认环境仍是 v2.3 4v4。

目的是检验中等规模是否降低 8v8 的高 timeout 和多实体复杂度，同时保留尾后攻击难度。它是本项目的中间规模设置，不宣称直接复现某篇论文。

| 项目 | v2.5 |
|---|---|
| Red / Blue | 5 / 5，同性能飞机 |
| observation / action | 65 / 3 |
| 横向 offsets | −600, −300, 0, 300, 600 m |
| 编队中心半径 / 双方间距 | 4000 / 约 8000 m |
| 高度 | 3000 ± 100 m |
| 初速 | 225 ± 10 m/s |
| 航向扰动 | ±5° |
| arena / dt / max_steps | 5000 m / 0.1 s / 1000 |
| 攻击资格 | 0–4000 m；true 3D off-boresight ≤30°；target aspect ≤45° |
| speed gate | 无；速度保留在动作、动力学、观测及诊断中 |
| Blue | 原 nearest-target policy，desired speed 250 m/s |
| fire state | 双方独立 PairFireState [5,5] |

65D = self 7 + 4 ally ×7 + 5 enemy ×6。slot 含义、顺序、归一化及死亡槽清零规则不变。阵型以 0 为中心，保持 300 m 横向间隔，复用现有动态 scenario。

v2.4 和 v2.5 共用 RearAspectWeaponEnvelope 与 pair 入窗逻辑：首次合法且 armed 才尝试，持续合法不重复，离窗 rearm；每 attacker 每步只选最近的合格 armed target，未选 pair 保持 armed。v2.3 继续使用原 FireState 和 forward-cone 资格。

概率命中、随机数消耗、one-hit kill、同时结算、动力学、R1–R4 奖励及 outcome 均继承最终 v2.4。达到 max_steps 且双方存活时仍为 red_failure_timeout，无 terminal reward，无按人数缩放 reward。

`configs/mappo_5v5.yaml` 和 `configs/stea_mappo_5v5.yaml` 使用 65D / 3D / 5 agents、16 env、rollout 256、最多10 PPO epochs、minibatch 512、3M 目标、20 eval episodes、100k eval interval、500k checkpoint interval、CUDA。当前5v5显式采用 state-independent 三维 log_std、初值−0.5、bounds[−5,0.5]、mean head orthogonal gain0.01、entropy coefficient0.001、epoch-level target KL0.015；环境规则不变。MAPPO 使用 [256,256] actor/attention critic 和 2 heads；STEA 保留 entity 64、2 entity heads、spatial 128、GRU 128×1、sequence 32、critic [256,256] 和 2 heads。旧run snapshot保持原协议，详见 `gaussian_policy_protocol.md`。

Checkpoint 记录环境版本、维度、配置指纹和训练 seed，禁止 v2.3/v2.4/v2.5 跨版本加载。MADSAC 保持仅支持 v2.3 4v4/52D。

先运行环境审计，再运行约 512 sampled steps 的 CUDA 闭环 smoke；这不是正式 3M 训练：

```bash
conda activate uav
python tools/audit_combat_v24.py --versions 2.5 --episodes 1000 --seed-base 45000000 --workers 2 --trace-count 10 --output-dir outputs/v25_combat_audit
python tools/review_combat_v24_traces.py --version 2.5 --trace-dir outputs/v25_combat_audit/v25/traces --output-dir outputs/v25_combat_audit/trace_review
python tools/smoke_combat_v24.py --version 2.5 --audit-report outputs/v25_combat_audit/v25/audit_report.json --output-dir outputs/v25_validation/cuda_smoke
```

审计报告和逐局数据分别位于 `outputs/v25_combat_audit/v25/audit_report.json` 与 `episodes.jsonl`。验收结果见 `outputs/v25_validation/report.md`。旧 v2.4 工具默认参数保持原调用方式。
