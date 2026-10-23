# v3.48：距离相关概率武器消融

v3.48 以 v3.47 为实验基准，只替换武器、弹药及冷却。配置为：

- `configs/combat_environment_v348.yaml`
- `configs/mappo_4v4_v348.yaml`
- `configs/maac_4v4_v348.yaml`

算法 YAML 与对应 v3.47 文件仅环境版本号不同。4v4、45维观测、3维动作、
dt=0.1秒、最大1000步、初始场景、Blue规则、动作/动力学、边界与终止协议不变。
原始 v3.38 观测构造函数直接复用，弹药和冷却不会进入观测。

## 版本与实现隔离

`CombatEnvironmentV348` 使用概率分支的共同祖先 `CombatEnvironmentV343`，
避开 v3.44 专用驻留 hook；直接绑定 v3.47 的 `_observations` 与 v3.46 的
`dense_combat_reward`。其余 R1/R2、PBRS、奖励归约/别名/累积与 v3.47 为相同
方法实现。没有复制训练器、修改网络、损失、优化器或训练参数。

`validate_v348` 严格限定武器及冷却配置；用深拷贝还原 v3.47 的武器字段后调用
其原校验器，保证其他字段满足同一约束，不改变运行时配置。旧环境代码、旧YAML
和测试基准不修改。共享注册/评估/Checkpoint 路径仅增加 v3.48 分支。

## 攻击资格与概率

直接复用 v3.42 `ConeWeapon`：活着的攻击者/目标、剩余弹药>0、距离在
[0,2000]米、前向三维圆锥离轴角≤30°。保持历史浮点比较规则：距离为严格数值
比较，角度沿用 `angle_within`；不为新版本改变原数学容差。

每次允许攻击时，使用 `environment.rng.normal(0,1)` 抽取**一个共享噪声** ε。
设机体系有符号方位偏差为 az，俯仰偏差为 el，角度单位为弧度：

```text
D = 1116.2212531024945
s_az = s_el = 0.9984711359155588
T(d) = pi * exp(-d / D)
hit = abs(az + s_az * epsilon) <= T(d)
      and abs(el + s_el * epsilon) <= T(d)
```

概率等于两个噪声允许区间交集的标准正态质量：

```text
L = max((-T-az)/s_az, (-T-el)/s_el)
U = min(( T-az)/s_az, ( T-el)/s_el)
P = 0 if L > U else Phi(U)-Phi(L)
```

采样和解析概率均直接复用已验证的 `PaperATAHAWeapon` 方法，由 v3.42 分支
传入机体系偏差；不是独立双噪声，也不是只根据距离抽取伯努利随机数。

## 弹药、冷却和同步结算

双方每机初始6次弹药，Reset补满、冷却清零。每次尝试扣1次，命中失败同样扣除
并设置10步冷却。每步先递减冷却，随后判断可否发射，故持续合格且未命中时，
首次在第1步发射，之后第11、21、31、41、51步最多各一次；不会每0.1秒发射。

Red 直接复用既有冷却发射路径，选择距离最近的合格目标，等距按索引。
Blue 直接复用 v3.26 的步首最近追击目标绑定路径；该目标运动后若死亡、
不合格或不可用，本步不换另一个目标发射。双方资源与尝试频率规则相同。
两方尝试均在统一毁伤结算前生成；复用原 resolver 的目标去重及共享击杀奖励。
没有0.5秒驻留状态、计数器或确定性毁伤路径。

## 指标、Checkpoint 与回放

`event_semantics=distance_body_gaussian_probability`，真实统计 attempts、hits、
unique kills、ammo used、双方胜率/损失/存活、全灭平局率、Return与R1–R4。
驻留指标通过 `dwell_metrics_applicable=false` 明确标记不适用，聚合占位为null，
不会生成 `dwell_complete` 事件。MAAC 的 v3.48 recent 指标复用统一聚合器，
旧版本指标分支保持原样。

Checkpoint 要求环境版本及环境/算法指纹、45/3/4维度匹配。跨版本不可混用；
MAAC 仍不支持精确恢复。官方 `tools/replay_combat.py` 支持 v3.48，记录真实
`fire_attempt` 及命中结果，CSV包含三维圆锥资格和机体系概率角度语义。

确定性评估只使策略动作选择确定；环境命中仍消耗环境RNG。同一模型、种子与
动作序列的轨迹/事件应一致，不同种子并不要求命中结果相同。

## 有界验证

在 PowerShell 激活 `uav` 后执行，CUDA不可用即失败，不回退CPU。
测试包含 CUDA 用例；与 smoke 串行执行，避免显存竞争。所有输出目录必须为新目录。

```powershell
$env:PYTHONIOENCODING = 'utf-8'
conda activate uav
python -m pytest tests/test_combat_v348.py tests/test_combat_v342.py tests/test_combat_v344.py tests/test_combat_v346.py tests/test_combat_v347.py tests/test_checkpoint_v347.py tests/test_maac_v344_v346.py -q
python tools/verify_v348_readiness.py --mode history --output outputs/v348_new/history
python tools/verify_v348_readiness.py --mode cuda --output outputs/v348_new/cuda
```

MAPPO 使用正式网络和参数，16×256=4096采样、一次PPO更新。MAAC使用正式
12环境、1024 minibatch、128 hidden、4 heads、16 baseline samples、1M replay，
最多12000采样，完成至少12个回合后仅执行4次critic、4次actor及1次target更新。
检查梯度、参数变化、优化器状态、终止mask、保存重载、固定种子重复评估和回放、
损坏元数据及跨版本拒绝，并只读加载历史训练Checkpoint。

不调用正式3M训练循环，不覆盖历史产物。通过 smoke 仅表示运行条件就绪，
不保证学习性能或最终胜率；性能差异需要后续多种子正式实验。


## 2026-10-23 实测结果

- PowerShell `uav`，PyTorch 2.11.0+cu128，NVIDIA GeForce RTX 5060 Laptop GPU。
- 最终新增环境测试76项，历史/算法/Checkpoint测试286项，共362个不同测试用例全部通过，无遗留失败。JUnit分别保存为 `tests.xml`（初次357项）和 `v348_tests_final_76.xml`（补充后的76项）；以用例标识去重计数。
- 修改前HEAD源码独立解压比对：v3.38/v3.42/v3.44/v3.46/v3.47 × 两个种子 × 两种固定动作，共20个完整回合，状态、观测、奖励、弹药、冷却、RNG、驻留状态哈希完全一致。
- 禁用双方武器事件的完整动态回合中，v3.48与v3.47状态、观测、R1–R4、终止及RNG逐步完全一致。奖励方法身份与观测函数直接复用检查通过。
- MAPPO：4096采样、1次PPO更新；actor/critic 参数变化L2分别约1.62015/1.78115；rollout形状256×16×4×45。
- MAAC：3672采样、至少12个完整回合，4次critic、4次actor、1次target更新；actor/critic 参数变化L2约0.455875/1.116727；actor、critic与attention均收到有限非零梯度。
- 两算法模型重载参数/动作完全一致；缺失指纹、损坏维度/指纹与跨版本Checkpoint被拒绝；确定性评估重复一致，固定种子完整轨迹、动作、随机命中事件重复一致，Reward分解误差为0。
- 只读加载并评估实际 mappo v3.47/v3.46 与 maac v3.46 3M best Checkpoint，SHA256前后一致，并拒绝用于v3.48。
- 官方回放CLI分别为两算法导出PNG、HTML、CSV、报告。实际fire_attempt命中结果、机体系角度和弹药消耗正常，无dwell_complete事件。
- 初次并行运行GPU测试与MAPPO smoke时出现CUDA OOM；该有界尝试退出。等待测试完成后，使用原网络与超参数在新目录串行重跑，两种算法全部通过。未修改算法以绕过显存问题，未回退CPU。
- Python编译检查和 `git diff --check` 通过。旧环境实现、旧观测/概率/Reward公式、旧YAML与历史测试文件未修改。

总报告：`outputs/v348_validation_20261023/summary.json`；成功CUDA结果：
`outputs/v348_validation_20261023/cuda_serial/summary.json`。首次未完成尝试保存在独立的`cuda`目录，不作为通过证据。
正式3M训练的代码与配置已就绪，运行前应释放足够CUDA显存，避免与其他GPU任务并行。没有启动正式3M训练，没有写入或覆盖任何既有训练目录。Smoke只证明运行闭环，不代表策略已学会有效作战。
