# v3.39 training readiness

v3.39 环境已完成隔离实现。本次只扩展正式 training / evaluation / checkpoint /
metrics / replay 的版本协议，环境代码、环境 YAML、算法数学、网络和优化参数均未修改。
**没有启动 3M 正式训练。**

| 算法 | 正式配置 | 入口（显式传入两份配置） |
|---|---|---|
| MAPPO | `configs/mappo_4v4_v339.yaml` | `algorithm/train_mappo.py` |
| MAPPO-CA | `configs/mappo_ca_4v4_v339.yaml` | `algorithm/train_mappo.py`（已有 CA dispatch） |
| EA-MAPPO | `configs/ea_mappo_4v4_v339.yaml` | `algorithm/train_eaa_mappo.py`（正式 attention family） |
| EAA-MAPPO | `configs/eaa_mappo_4v4_v339.yaml` | `algorithm/train_eaa_mappo.py` |
| MAAC | `configs/maac_4v4_v339.yaml` | `algorithm/train_maac.py` |
| MADDPG | `configs/maddpg_4v4_v339.yaml` | `algorithm/train_maddpg.py` |
| MADSAC | `configs/madsac_4v4_v339.yaml` | `algorithm/train_madsac.py` |

每份配置以对应正式 `*_v338.yaml` 为唯一 baseline，只有
`environment.environment_version` 改为 `3.39`，team_size 保持 4；其余所有字段
完全相等。MAPPO 的 gae_lambda 保持 0.95，未使用 lambda996195 实验配置。
训练入口原有默认值不变。旧 `train_ea_mappo.py` 属于独立历史 EA 实现；
当前正式 45D EA variant 使用上表的 attention-family 入口。

协议修改范围：MAPPO 身份门禁和版本感知统计、common evaluator 的已有统计门禁、
EAA family 版本对应 baseline / contract / checkpoint family、MAAC/MADDPG/MADSAC
环境维度和架构 allow-list，以及 canonical replay 的显式 family/MADDPG loader。
`split_v338_entities` 名称保留：v3.39 observation layout 与 v3.38 相同，没有改网络。

EAA family 分别使用 `eaa_mappo_v338` / `eaa_mappo_v339`，contract 记录实际环境版本。
各算法 checkpoint 保持 environment version、environment SHA、algorithm SHA、
架构与 schedule 检查。v3.39 MAPPO 正式 resume/evaluation 还严格要求两个 hash
均存在且匹配；v3.38 原有兼容分支不变。旧 checkpoint 不迁移、不改写。
v3.38 和 v3.39 checkpoint 不能互相作为正式同环境 checkpoint 加载。

common evaluator / MAPPO runner 对 v3.39 输出与 v3.38 同样的 R1–R4、paper、
safety、extended、individual_event、team_casualty、boundary、fire/hit/kill、
ceiling/noncombat、Blue nearest/target-switch 指标。旧 controller 诊断仅适用旧
controller 版本，未错误扩展到简化运动学环境。

## CUDA 验证协议

`tools/verify_v339_training.py` 在 Windows `uav` Conda 环境运行（WSL 访问被拒），
CUDA 不可用时直接失败。使用正式网络、原始配置和原优化参数，构造真实
multiprocess vector env、收集 rollout/replay、执行实际 optimizer update、
保存 checkpoint、通过正式 loader 加载，并比较加载前后同 seed 确定性评估。
每算法同时验证 v3.38/v3.39 和双向跨版本拒绝。

仅缩短运行预算、worker 数和 evaluation episode 数，不改配置文件：
PPO family 每组 512 transitions / 2 workers；MAAC 1100 / 4；
MADDPG 10100 / 4（保留正式 10000 warmup）；MADSAC 1200 / 4。
每组评估 1 局，保留原 evaluation seed 起点；另用 31001000 验证 reload 等同。
这些是有严格上限的 readiness runs，使用 formal architecture，因此原生
checkpoint 的 architecture/smoke 标志遵守原协议；独立 metadata 明确标记
`readiness_smoke=true`，不作为正式训练结果或科研性能证据。
MAAC、MADDPG、MADSAC 和 EAA family 不声称支持 exact training resume。

结果保存在 `outputs/v339_training_readiness/`，每组有配置 snapshot、
`readiness_run_config.json`、`readiness_result.json` 和 checkpoint；总表为 `results.json`。

## 冻结 Red 的 Blue 反事实干预

工具：`tools/compare_v338_v339_blue.py`。

主分析只纳入 MAPPO、MAAC、EA-MAPPO、EAA-MAPPO 各 3 个正式完成的 v3.38
训练 seed 的 `best_eval.pt`，共 12 policies。每个 policy 对两个环境各跑
100 局，paired seeds 30001000–30001099，共 2400 局。加载时严格使用训练目录
自己的 v3.38 environment / algorithm snapshots 和正式 validator；随后冻结网络，
以 inference-only policy 放入 target 环境，绝不调用 v3.39 validator 冒认 source。
metadata 同时记录 source checkpoint/environment/algorithm hashes、target environment
version/hash、`cross_environment_inference=true`。每组前后核对权重与 checkpoint hash。

这称为 **frozen v3.38 policy cross-environment intervention**，不是
“v3.39 checkpoint evaluation”。策略确定性推理，Weapon 使用各自原 RNG；
Blue 动作差异会让后续轨迹和 RNG consumption 分叉。
批量 CUDA inference 只减少审计开销，每个环境仍正常调用 canonical observer step；
未改环境或 Reward。overshoot 直接复用 `tools/analyze_red_speed.py::overshoot_proxies`。

结果包含各 algorithm/seed/environment 的 win、return、survivors/losses、
fire/hit/kill、episode length、boundary/ground/ceiling、target switches、
Blue speed 分布、当前追击距离与 far/mid/near occupancy、overshoot。
速度/距离以 step 前存活 Blue 为口径；tier occupancy 按当时最近存活 Red 距离
计算，v3.38 中也按相同距离分箱，不意味着 v3.38 执行三档控制。
distribution histogram 是描述统计；置信区间以 episode 为重采样单位，不把
timestep 当独立样本。delta 定义为 v3.39 减 v3.38，配对 episode bootstrap
2000 次；算法汇总也保留 algorithm/training_seed/episode_seed 的配对关系。
算法和总体汇总会把同一个 scenario seed 下的多个冻结 policy 一起重采样，
因此每组仍是 100 个 scenario bootstrap units，避免将重复场景当独立样本；
区间条件于这些已冻结模型，不代表对所有训练随机性的总体置信区间。
每个 policy 的绝对胜率另给 Wilson 区间；汇总绝对胜率使用 scenario bootstrap。

`weapon_hits` / `attack_kills` 对应 hits / kills。连续汇总指标按 episode 等权；
逐局文件中的速度/距离分位数为该局真实分位数，summary 中这些列是逐局分位数
的均值，并非 pooled timestep 分位数。速度 histogram 另提供全部 live-agent-step
合并计数。far/mid/near occupancy 的主汇总同样按 episode 等权。

原始逐局数据、provenance、summary 位于 `outputs/v339_frozen_blue_comparison/`。
Blue 的 240/220/200 和 3000/1000 参数冻结，不根据结果自动调整。

本机采集命令为 `python -u tools/compare_v338_v339_blue.py --environment-workers 8`；
该模式只有一个 CUDA policy inference 进程，环境采集器仅执行 CPU 环境逻辑。
已有完整结果会校验 provenance 后复用，不重新采样。

## 执行结果

七算法接入与 CUDA 链路验证通过。每组使用相同正式网络和优化参数；以下为原生
actor / critic update counter（MADDPG 为每 agent 同步更新事件计数）：

| 算法 | 每版本 transitions | v3.38 updates | v3.39 updates | save/load/eval、双向隔离 |
|---|---:|---:|---:|---|
| MAPPO | 512 | 10 / 10 | 10 / 10 | PASS |
| MAPPO-CA | 512 | 10 / 10 | 10 / 10 | PASS |
| EA-MAPPO | 512 | 10 / 10 | 10 / 10 | PASS |
| EAA-MAPPO | 512 | 10 / 10 | 10 / 10 | PASS |
| MAAC | 1100 | 4 / 4 | 4 / 4 | PASS |
| MADDPG | 10100 | 1 / 1 | 1 / 1 | PASS |
| MADSAC | 1200 | 4 / 8 | 4 / 8 | PASS |

合计 14 组、28896 transitions，动作和 loss 有限，reload 后同 seed 确定性评估完全相等。
硬件：NVIDIA GeForce RTX 5060 Laptop GPU。正式算法配置 parity、历史配置/环境
未修改证明见 `outputs/v339_training_readiness/config_parity.json`。

所选测试最终结果：**444 passed / 6 existing failures / 1 skipped**（未运行全仓 pytest）。
其中新增专项 57 passed；环境/算法/canonical recorder 回归 358 passed / 1 skipped；
扩展 MAPPO/compact metrics/lambda 回归 29 passed / 6 failed。
跳过项因 FFmpeg/ffprobe 不可用，未声称实际视频编码通过。

6 项既有失败均在修改前 HEAD 的隔离 algorithm package 中重现：

- `tests/test_compact_training_metrics.py::test_one_weighted_summary_after_whole_rollout`
  的 MAPPO、RMAPPO、STEAMAPPO、EAMAPPO 四个参数实例；
- `test_no_episodes_null_and_theoretical_count`；
- `test_accumulation_and_serialization_do_not_consume_rng`。

原因相同：测试通过 `object.__new__` 构造的 bare runner 没有 `env_config`，
旧 controller metric 分支访问该属性时抛出 AttributeError。本次未修改旧测试期望或
额外修复旧 runner 行为。复现日志：`outputs/v339_baseline_failures.log`。
另外曾因同时启动过多 Torch 进程发生 WinError 1455，导致一轮扩展测试在 DLL
导入阶段失败；释放并发后已成功执行测试，上述最终计数不含该未执行测试的收集失败。
审计随后改用单 CUDA inference 进程加 CPU collectors，不改实验条件。

冻结 Red 的完整 A/B 对照已完成，结果见下方。
**当前没有启动 3M 正式训练。**

## 冻结 Red 的完整 A/B 结果

12 个真实完成训练的 best policy × 2 环境 × 100 局，共 **2400 局**。所有 source checkpoint / frozen weight hash 验证通过。
以下 delta 均为 v3.39 − v3.38；方括号为 95% bootstrap CI。每个 policy 每环境 N=100。

| Policy | Red wins A → B | Δ胜率 pp [CI] | Mean return A → B | Δreturn [CI] |
|---|---:|---:|---:|---:|
| MAPPO seed 1 | 51 → 34 | -17.0 [-28.0, -4.0] | 7.55 → -7.55 | -15.11 [-20.97, -9.19] |
| MAPPO seed 2 | 62 → 56 | -6.0 [-17.0, +4.0] | 14.77 → 6.89 | -7.88 [-14.09, -1.95] |
| MAPPO seed 3 | 46 → 47 | +1.0 [-10.0, +13.0] | -1.10 → -0.26 | +0.85 [-6.03, +7.57] |
| MAAC seed 1 | 96 → 92 | -4.0 [-10.0, +1.0] | 36.54 → 33.44 | -3.10 [-6.13, -0.15] |
| MAAC seed 2 | 75 → 57 | -18.0 [-30.0, -6.0] | 11.41 → 0.65 | -10.75 [-15.98, -5.40] |
| MAAC seed 3 | 93 → 89 | -4.0 [-11.0, +2.0] | 33.51 → 29.52 | -3.99 [-7.69, -0.48] |
| EA-MAPPO seed 1 | 83 → 85 | +2.0 [-5.0, +9.0] | 29.05 → 28.25 | -0.79 [-4.09, +2.71] |
| EA-MAPPO seed 2 | 66 → 56 | -10.0 [-23.0, +3.0] | 11.03 → 3.83 | -7.20 [-13.81, -0.58] |
| EA-MAPPO seed 3 | 70 → 68 | -2.0 [-11.0, +7.0] | 19.01 → 16.11 | -2.90 [-8.70, +2.72] |
| EAA-MAPPO seed 1 | 91 → 82 | -9.0 [-18.0, +0.0] | 29.58 → 23.14 | -6.44 [-11.76, -1.41] |
| EAA-MAPPO seed 2 | 68 → 67 | -1.0 [-10.0, +8.0] | 16.06 → 13.14 | -2.92 [-7.22, +1.39] |
| EAA-MAPPO seed 3 | 59 → 48 | -11.0 [-23.0, +0.0] | 7.41 → 0.60 | -6.81 [-13.63, +0.25] |

| Policy | Δepisode steps [CI] | ΔBlue kills [CI] | ΔRed kills [CI] | Δovershoot / episode [CI] |
|---|---:|---:|---:|---:|
| MAPPO seed 1 | +46.11 [+28.60, +65.71] | +0.41 [+0.17, +0.66] | -0.80 [-1.10, -0.50] | +0.020 [-0.090, +0.110] |
| MAPPO seed 2 | +7.25 [-2.24, +16.58] | +0.06 [-0.18, +0.29] | -0.53 [-0.85, -0.22] | -0.080 [-0.310, +0.160] |
| MAPPO seed 3 | +10.25 [-9.16, +30.22] | +0.28 [-0.04, +0.61] | -0.22 [-0.58, +0.16] | -0.300 [-0.530, -0.090] |
| MAAC seed 1 | +20.68 [-8.34, +48.13] | +0.13 [-0.03, +0.29] | -0.05 [-0.11, +0.00] | -0.070 [-0.140, +0.000] |
| MAAC seed 2 | +80.26 [+31.66, +126.64] | +0.28 [-0.02, +0.58] | -0.34 [-0.64, -0.05] | -0.250 [-0.450, -0.050] |
| MAAC seed 3 | +50.88 [+22.38, +85.56] | +0.28 [-0.01, +0.57] | -0.11 [-0.21, -0.02] | +0.010 [-0.090, +0.110] |
| EA-MAPPO seed 1 | +10.21 [-0.01, +20.12] | +0.14 [+0.01, +0.28] | +0.03 [-0.10, +0.16] | +0.240 [+0.100, +0.410] |
| EA-MAPPO seed 2 | +6.44 [-17.74, +31.86] | +0.46 [+0.15, +0.80] | -0.21 [-0.59, +0.17] | +0.190 [-0.030, +0.430] |
| EA-MAPPO seed 3 | -1.07 [-11.98, +11.38] | +0.12 [-0.06, +0.30] | -0.20 [-0.45, +0.06] | +0.000 [-0.060, +0.050] |
| EAA-MAPPO seed 1 | +14.84 [-3.53, +34.95] | +0.43 [+0.16, +0.72] | -0.20 [-0.45, +0.04] | -0.210 [-0.340, -0.090] |
| EAA-MAPPO seed 2 | -0.69 [-12.79, +10.74] | +0.17 [-0.02, +0.37] | -0.09 [-0.30, +0.12] | +0.010 [-0.100, +0.130] |
| EAA-MAPPO seed 3 | +20.34 [-4.83, +47.73] | -0.03 [-0.37, +0.30] | -0.51 [-0.85, -0.20] | +0.040 [-0.180, +0.260] |

每算法含 3 个固定训练 seed，每环境 300 policy-episodes；CI 以共享的 100 个 scenario seed 为重采样单位。

| Algorithm | Red wins A → B (/300) | Δ胜率 pp [CI] | Δreturn [CI] | Blue mean speed A → B (m/s) | Overshoot A → B |
|---|---:|---:|---:|---:|---:|
| MAPPO | 159 → 137 | -7.3 [-14.0, -1.0] | -7.38 [-10.85, -4.00] | 225.00 → 223.19 | 0.380 → 0.260 |
| MAAC | 264 → 238 | -8.7 [-13.7, -3.7] | -5.95 [-8.13, -3.75] | 225.00 → 222.90 | 0.273 → 0.170 |
| EA-MAPPO | 219 → 209 | -3.3 [-9.0, +2.3] | -3.63 [-6.48, -0.59] | 225.00 → 227.16 | 0.143 → 0.287 |
| EAA-MAPPO | 218 → 197 | -7.0 [-12.7, -1.3] | -5.39 [-8.35, -2.26] | 225.00 → 226.46 | 0.287 → 0.233 |

12-policy 总览（每环境 1200 policy-episodes，100 个 scenario bootstrap units）：Red wins **860 → 781**；Δwin **-6.6 [-9.3, -3.8] pp**；Δreturn **-5.59 [-7.12, -4.10]**；Δepisode length **+22.12 [+14.91, +29.20] steps**。

全部分组指标、median return、survivors/losses、fire/hit/kill、boundary/ground/ceiling、target switches、Blue speed/距离分布与 tier occupancy 见：
- `outputs/v339_frozen_blue_comparison/group_summary.csv`（含连续指标 CI）；
- `outputs/v339_frozen_blue_comparison/paired_deltas.csv`；
- `outputs/v339_frozen_blue_comparison/summary.json`；
- `outputs/v339_frozen_blue_comparison/episodes.csv`（2400 局原始统计）；
- 每个 `*_best_v338.json` / `*_best_v339.json` 的 source/target provenance；
- `data_quality.json`、`protocol.json`、`inventory.json`。

![Frozen Red A/B deltas](../outputs/v339_frozen_blue_comparison/win_return_deltas.png)

## 总体行为指标与结论

以下除 median return 外为 episode 等权均值；每环境 1200 policy-episodes。Ceiling 属于 boundary 子类，不能与 boundary 重复相加。

| 指标 | v3.38 | v3.39 |
|---|---:|---:|
| Mean return | 17.9008 | 12.3144 |
| Median return | 30.4252 | 21.3247 |
| Red survivors | 2.4225 | 2.1342 |
| Blue survivors | 0.5983 | 0.8408 |
| Red losses | 1.5775 | 1.8658 |
| Blue losses | 3.4017 | 3.1592 |
| Red fire attempts | 6.3067 | 5.8708 |
| Blue fire attempts | 2.3042 | 2.6117 |
| Red hits | 3.4208 | 3.1408 |
| Blue hits | 1.0175 | 1.2800 |
| Red attack kills | 3.3225 | 3.0533 |
| Blue attack kills | 0.9933 | 1.2208 |
| Episode steps | 358.5658 | 380.6908 |
| Red boundary loss | 0.2775 | 0.2817 |
| Blue boundary loss | 0.0158 | 0.0283 |
| Red ground loss | 0.3067 | 0.3633 |
| Blue ground loss | 0.0633 | 0.0775 |
| Red ceiling loss | 0.2150 | 0.2525 |
| Blue ceiling loss | 0.0158 | 0.0283 |
| Blue target switches | 8.1158 | 8.6658 |
| Blue mean speed (m/s) | 225.0000 | 224.9264 |
| Nearest pursuit distance (m) | 3276.4964 | 3099.1235 |
| Far occupancy | 0.5134 | 0.4750 |
| Mid occupancy | 0.2717 | 0.2774 |
| Near occupancy | 0.2149 | 0.2476 |
| Overshoot / episode | 0.2708 | 0.2375 |

**技术接入已 training-ready：七算法均可通过显式 v3.39 配置开始正式训练。** v3.38 原训练入口默认值与合法 checkpoint 合约保持兼容，v3.38/v3.39 正式 checkpoint 双向严格拒绝。

对冻结旧策略，v3.39 Blue 整体更强：总 Red 胜率由 71.67% 降为 65.08%，Δ=-6.58 pp，95% CI [-9.33,-3.83]；mean return 由 17.90 降为 12.31。MAPPO、MAAC、EAA-MAPPO 的算法汇总胜率差异区间不跨 0；EA-MAPPO 胜率区间跨 0，但其 return 下降区间不跨 0。不能把所有单个训练 seed 一概判为同等程度变强。

平均局长增加 22.125 steps（2.2125 s）；Blue attack kills 平均增加 0.2275，Red attack kills 减少 0.2692。Blue 平均速度几乎不变（225.00 → 224.93 m/s），但阶段速度分配和交战结果已改变。

Overshoot 总体 0.2708 → 0.2375，Δ=-0.0333，95% CI [-0.0808,+0.0108]，不足以认定总体明确下降。分算法 MAPPO/MAAC/EAA 降低，EA-MAPPO 由 0.1433 增至 0.2867。因此不能声称三档速度普遍消除了 overshoot；proxy 定义保持原样，不作意图或因果解释。

**建议进入下一阶段 3M 正式对照训练**，保持本次 Blue 参数与各算法正式配置冻结。MADSAC 的正式配置仍为原来的 8.4M，本任务没有把它改成 3M。以上冻结旧模型结果不能代替 v3.39 重新训练后的性能结论。

**本次未启动任何 3M 或其他正式训练；只执行了有界 CUDA readiness updates 和只读 A/B。**

完整 24 组数据已保存后，长时间后台进程曾在早期汇总函数中触发变量覆盖错误；该错误已修复并增加汇总测试。随后使用现有数据重新验证与汇总成功，没有增加采样或改变参数。最终统计、表格和图表均来自修复后的共享场景 bootstrap 实现。
