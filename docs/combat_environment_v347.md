# Combat environment v3.47

v3.47 是 v3.46 的观测消融版本。`CombatEnvironmentV347` 直接继承
`CombatEnvironmentV346`，仅覆盖版本号与 `_observations()`，调用原始
`build_team_observations_v338(red, blue, config['observation'])`。

每架红机观测为45维：自身7维、三个队友各6维、四个敌机各5维。
字段顺序、归一化、死亡实体清零均使用 v3.38 原始实现，不增加任何观测信息。
动作仍为3维，队伍仍为4v4。v3.44/v3.46 仍为49维。

环境配置 `configs/combat_environment_v347.yaml` 除版本号外与 v3.46 完全一致。
MAPPO、IPPO、MAAC 分别使用 `configs/{mappo,ippo,maac}_4v4_v347.yaml`，
仅修改环境版本号和 `network.observation_dim: 45`。
网络隐藏层、算法超参数、Reward/PBRS、Eq.(25) R4、蓝方策略、动力学、
持续驻留攻击判定及终止语义全部继承 v3.46。

版本注册、维度查询、PBRS gamma 约束、训练器、评估聚合、驻留指标、
IPPO 固定观测诊断以及 Checkpoint 环境/算法指纹校验均识别 v3.47。
旧版本 Checkpoint 不能用于 v3.47；同为45维的旧版本也不能混用。
MAAC 继续保留“不支持精确恢复”的原有约束。

验证使用 Windows PowerShell 的 `uav` Conda 环境：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
conda activate uav
python -m pytest tests/test_combat_v347.py tests/test_combat_v344.py tests/test_combat_v346.py tests/test_maac_v344_v346.py -q
python tools/verify_v347_readiness.py --mode history --output outputs/v347_validation_new/history
python tools/verify_v347_readiness.py --mode cuda --output outputs/v347_validation_new/cuda
```

两个 audit 模式都要求 CUDA，输出目录必须不存在。历史回归将修改前 HEAD
的源码解压到独立目录，比较 v3.44/v3.46 完整回合的状态、观测、奖励、
RNG 和驻留状态哈希；只读加载、评估历史 Checkpoint 并检查跨版本拒绝。
固定动作测试覆盖四种策略、三个种子共12对完整回合，逐步比较 v3.46/v3.47
全部非版本号 info、R1–R4、蓝方动作、状态、终止与 RNG，并验证每一步45维
观测与原始构造函数完全一致；额外覆盖第五步同时毁伤、共享奖励和截断 PBRS。

CUDA 模式仅执行 MAPPO/IPPO 各4096采样及一次 PPO update，MAAC 一次完整
replay cycle（4次 critic、4次 policy、1次 target update），使用正式配置的
网络及超参数。包含梯度有限性、参数更新、Checkpoint 保存/重载、评估和记录。
不进入正式3M训练循环，不写入任何已有3M输出目录。


## 2026-10-23 验收结果

在 PowerShell `uav`（PyTorch 2.11.0+cu128、RTX 5060 Laptop GPU）完成：

- 固定动作/环境测试23项，历史环境与 MAAC 测试170项，Checkpoint 元数据测试12项，共205项通过。
- 四种固定策略 × 三种种子，共12对完整回合：v3.46/v3.47 状态、全部非版本号 info、R1–R4、RNG、攻击与终止事件逐步完全一致；45维观测逐步等于原始 v3.38 构造函数。
- v3.44/v3.46 共8个完整回合与修改前 HEAD `261995a24e174c0527e1736e858b6f8306f20c0d` 的状态、观测、奖励、RNG、驻留哈希完全一致。
- 六个历史 Checkpoint 只读 CUDA 加载/评估通过且拒绝用于 v3.47；文件 SHA256 前后一致。其中 MAPPO/MAAC 使用 v3.44/v3.46 真实3M `best_eval.pt`，IPPO 使用两版本已有诊断 Checkpoint。
- MAPPO/IPPO 各4096采样、1次 PPO update；MAAC 3864采样、4次 critic / 4次 policy / 1次 target update。梯度有限且参数实际更新，保存/重载、评估、记录通过；MAAC 记录 Reward 分解误差为0。
- 三种算法的 v3.47 Checkpoint 拒绝历史版本及损坏的指纹/维度元数据。IPPO 新增的严格维度元数据校验仅适用于 v3.47。
- `git diff --check` 和 Python 编译检查通过。历史环境实现、v3.38 观测构造器及 v3.44/v3.46 YAML 文件未修改。

机器可读总报告：`outputs/v347_validation_20261023/summary.json`。
CUDA 更新原始结果：`cuda/*_v347/summary.json`；补齐 IPPO 维度校验后的最终 Checkpoint 复核汇总：`checkpoint_recheck/cuda_summary.json`。
所有输出均写入独立的新目录，未覆盖已有3M实验输出，未启动正式3M训练。
