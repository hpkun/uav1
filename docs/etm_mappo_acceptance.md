# ETM-MAPPO 独立导航实现与验收报告

日期：2026-10-20。以用户最后一版非武器化二维协作导航任务为准，前一版52/45维观测选择问题已被该边界取代。实现不连接现有任务仿真，不修改已有算法、物理环境、Reward、动作或checkpoint兼容性。

## 四类结论必须分开

1. **网络实现正确**：最终40项专项通过，覆盖ID换槽、失联/重现、空集合、并行隔离、历史反事实、gate消融等。
2. **强化学习闭环正确**：真实CUDA采样、TBPTT PPO、Actor/Critic更新、严格checkpoint重载及确定性评估已完成。
3. **测试环境出现学习**：简单场景的单种子预检出现Return改善；三种子的可重复性按下方实际表格判断，不能用单次成功代替。
4. **稳定优于基线**：本任务不据一次8K预算、三个种子就宣布稳定性能优势。消融中的正负结果全部保留。

## 新增文件与职责

- `algorithm/etm_mappo/memory.py`：ID匹配、独立实体状态和可微重排。
- `algorithm/etm_mappo/networks.py`：类型编码、B0–B3 Actor、attention、门控、集中式Critic及Tanh概率。
- `algorithm/etm_mappo/config.py`：完整配置校验、独立环境/实现版本及指纹。
- `algorithm/etm_mappo/navigation.py`：独立二维物理模拟、局部可见输入与自动reset向量接口。
- `algorithm/etm_mappo/trainer.py`：真实rollout、GAE、chunk/padding、pre-update audit与循环PPO。
- `algorithm/etm_mappo/checkpoint.py`：严格保存/加载、身份保护和恢复能力声明。
- `algorithm/etm_mappo/evaluation.py`：确定性简单/随机场景评估与种子来源。
- `algorithm/etm_mappo/runner.py`：采样状态生命周期、日志、评估、checkpoint和异常记录。
- `algorithm/etm_mappo/benchmark.py`：统一预算的三种子B0–B3与记忆重置对照。
- `algorithm/etm_mappo/analysis.py`：离线配对差异、稳定性核查及导出图表。
- `algorithm/etm_mappo/__init__.py`：独立算法包。
- `algorithm/train_etm_mappo.py`：训练CLI。
- `algorithm/evaluate_etm_mappo.py`：评估CLI。
- `algorithm/benchmark_etm_mappo.py`：有界对照实验CLI。
- `configs/etm_mappo.yaml`：完整独立配置。
- `tests/test_etm_mappo.py`：40项网络、身份、数学、CUDA与环境专项测试。
- `docs/etm_mappo.md`：结构、维度、参数、状态生命周期和使用说明。
- `docs/etm_mappo_acceptance.md`：本实施报告。

没有修改任何已有受保护文件。`frozen_source_audit.json`对实施前记录的621个已跟踪算法/环境/配置/工具文件做原始字节SHA256比对，变化为0。独立进程导入检查确认新Runner没有导入`env.*`、现有MAPPO或STEA运行时。没有添加武器/交战/射击/击毁接口或指标。

## 架构、形状与身份规则

完整模块形状和实测参数见 `docs/etm_mappo.md`、`network_parameter_counts.json`。B3共享Actor176,644，Critic29,953，合计206,597。与STEA空间聚合后单GRU不同，B3先对自身、每个peer、每个obstacle分别执行共享类型GRUCell，再对实体历史做cross-attention和门控。

输入不是扁平固定长度观测，而是self[E,N,Ds]、peer[E,N,Kp,Dp]、obstacle[E,N,Ko,Do]和显式ID/validity/alive/start。默认10/4/5维、2维加速度；通用网络已测试N=1/3/4/7及空实体集合，示范模拟器发出的物理特征接口固定10/4/5/2，不匹配时拒绝运行。

MemoryState保存self[E,N,64]、peer[E,N,Kp,64]、obstacle[E,N,Ko,64]、上一帧对应IDs及B1聚合后128维状态。不同环境、不同观察者、不同实体的状态独立。IDs只参与离散匹配，不进入特征；同类型共享encoder/GRU参数。有效ID在每个观察者/类型内唯一，inactive为-1；非法映射报错。同ID换槽时隐藏跟随；离开后清空，重新出现从零开始；episode reset和agent invalid清空所属历史。

B2保留B3的相同self/context投影与LayerNorm，以固定0.5代替学习门控，仅少32,896个gate参数。将B3 gate参数置0并复制其余权重，B2/B3连续三步输出逐位一致。B3-reset保留全部B3参数，逐步在GRU使用前清除历史。

Attention严格mask，空集合直接零上下文；无外部实体时context projection的bias也被屏蔽。Actor只用本地输入；Critic独立聚合所有机器人状态。自身输入是自身位置/速度、公开目标、指定几何偏移和上步控制，没有不可见机器人的真实质心信息。

## Recurrent PPO与真实预算

GAE先在完整rollout计算；terminated不bootstrap，truncated使用reset前真实后继value bootstrap；两者都停止跨局GAE递归。无效agent不参与归一化和loss。按环境/时间切连续chunk，仅随机排列chunk，起点状态detach，chunk内保持身份对齐的BPTT。尾部padding显式mask，loss算术先选择有效样本，包括exp和square之前；不会让padding污染loss。

默认128×8=1024条联合环境transition/rollout，sequence16，minibatch256 joint transitions，4epoch，分别16次Actor/Critic step。8192预算=8个rollout、128次Actor和128次Critic更新；不乘机器人数量。所有对照相同环境、reward、动作、PPO参数和评估seed。Critic初始化使用独立随机流，同seed的各方法Critic权重逐位相同。

每次优化前重算全部有效old-policy概率；最大ratio误差要求≤1e-4，不满足立即停止。原始Tanh Gaussian latent action和old logp从采样时保存，不用atanh重建饱和动作。PPO数学专项验证ratio clip、value clip、GAE、终止与time-limit边界、梯度传播和padding不变性。

## CUDA闭环证据

PowerShell激活Windows `uav` Conda环境，Python `D:/conda_envs/envs_dirs/uav/python.exe`，PyTorch2.11.0+cu128；GPU NVIDIA GeForce RTX5060 Laptop GPU。CUDA不可用明确失败，没有CPU训练替代。

正式64维实体记忆网络的独立集成测试实际采样66条联合transition、33个vector-step，包含真实环境中途auto-reset和尾部padding；Actor/Critic各1次实际更新。Actor参数变化L2=0.109036848，Critic参数变化L2=0.043247733。完整梯度、隐藏范数、门控等见 `cuda_integration_evidence.json`。

另运行默认完整rollout的B3 seed0预检：8192 transition、1024vector-step、8次rollout update、128次Actor和128次Critic step，评估Return -4.974176→18.740135，完成率0.000，碰撞率0.000。不把Return改善等同于全部机器人已掌握协作导航。预检耗时78.16秒。

保存重载后，同输入确定性动作逐位一致，相同评估种子结果一致；另独立从零重跑的B3 seed0，其Actor/Critic最终参数在本GPU上逐位相同，详见checkpoint_identity_and_rerun_audit.json。与MAPPO、STEA-MAPPO、IPPO的加载器已实际验证双向拒绝错误算法身份；确定性推理不消耗CUDA随机数。错误算法身份、实现版本、架构和指纹拒绝；新checkpoint不会作为现有算法加载。恢复声明为deterministic_evaluation_reload，bit_exact_training_resume=false；没有环境/完整RNG/采样状态恢复功能，--resume明确拒绝。

## 三种子对照

每组8192 joint transition，seed=0/1/2，B0–B3及B3-reset共15个已完成实验，纳入对照的总采样量122,880。无数百万步训练，无自动预算扩大。下表为三个训练种子的均值±样本标准差；完成率和碰撞率来自同一套6个development评估seed。

| 方法 | 初始Return | 最终Return | Return改善种子 | 完成率 | 碰撞率 | 随机布局Return |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.129 ± 2.339 | -153.136 ± 86.668 | 0/3 | 0.000 ± 0.000 | 1.000 ± 0.000 | -192.658 ± 35.563 |
| B1 | -0.819 ± 1.081 | -120.013 ± 112.813 | 1/3 | 0.000 ± 0.000 | 0.667 ± 0.577 | -227.534 ± 123.619 |
| B2 | -0.694 ± 3.278 | -84.166 ± 124.368 | 1/3 | 0.333 ± 0.577 | 0.667 ± 0.577 | -102.858 ± 91.364 |
| B3 | -2.678 ± 2.413 | -72.713 ± 109.105 | 1/3 | 0.000 ± 0.000 | 0.667 ± 0.577 | -94.254 ± 76.925 |
| B3-reset | -3.045 ± 2.994 | -21.372 ± 45.002 | 1/3 | 0.333 ± 0.577 | 0.667 ± 0.577 | -122.070 ± 23.521 |

所有实验的真实更新次数、梯度与完整逐seed数据见 `runs.csv` 和 `ablation_summary.json`，学习曲线/泛化图见 `ablation_results.png`。随机布局结果仅作独立泛化评估，不参与best选择。simple场景的物理布局本身是确定性的，跨seed变化来自初始化、策略采样和槽位打乱。

| 配对差异（左减右） | Return差均值±样本SD | 正差种子 |
|---|---:|---:|
| B1 − B0 | 33.123 ± 27.945 | 3/3 |
| B2 − B1 | 35.848 ± 55.925 | 2/3 |
| B3 − B2 | 11.453 ± 18.375 | 2/3 |
| B3 − B3-reset | -51.341 ± 72.529 | 0/3 |

这些是实测差异，不是显著性或稳定优势证明。学习门控、实体记忆与history-reset的具体正负影响应按配对数据解读，不能只选择成功种子。

## 本次消融的具体解释

- B3的Return改善仅出现于1/3训练种子，三个种子的任务完成率全部为0；所有方法在随机布局评估中的任务完成率也均为0。
- B1相对B0的最终Return在3/3配对种子更高，但B1本身仍有2/3种子相对未训练初始策略退化，不能据此宣布学会协作。
- B2相对B1的配对Return平均差为+35.848，2/3为正，并有一个种子在简单场景完成任务；这不是稳定收益证明。
- 学习gate相对固定gate的配对Return平均差为+11.453，2/3为正，但B3完成率为0，低于B2的三种子平均1/3。门控没有体现一致的任务收益。
- 保留历史的B3在简单场景Return上比B3-reset低51.341，3/3配对种子都更低；这直接不支持当前任务和预算下的时序记忆收益。随机布局上B3平均Return又高于reset，但完成率均为0，说明不能从单一场景排序泛化出记忆有效性结论。
- 目前没有确定导致这种不稳定性的唯一原因；任务历史需求较弱、8K优化不足和循环策略的优化难度均是未验证假设，不作既定解释，也不据此直接扩预算。

## 测试与回归结果

| 检查 | 实际结果 | 日志 |
|---|---|---|
| 阶段一网络/身份 | 19 passed | stage1_network_tests.log |
| 阶段二数学/TBPTT | 27 passed | stage2_ppo_tests.log |
| 最终专项 | 40 passed（14.10秒） | final_40_tests.log |
| 实施起始全仓 | 4507 passed、44 failed、4 skipped（1710.33秒） | baseline_all_tests.log |
| 最终全仓 | 44 failed, 4545 passed, 4 skipped in 1034.57s (0:17:14) | final_all_tests.log / final_suite.xml |

与基线对比，新增失败为0项。失败列表与差异见 `regression_comparison.json`。已有失败断言未修改、删除或跳过。

静态编译`compileall`和`git diff --check`通过。完整全仓运行收集38项ETM测试；之后额外补充gate精确消融和物理可达性两项，最终40项专项全部通过。当前`--collect-only`为4595项；不要把两次运行的重叠测试重复计为独立测试。最终覆盖以全仓回归加两项补测说明。

真实命令（均先conda activate uav）：

```powershell
python -m pytest tests/test_etm_mappo.py -q --tb=short
python -m pytest -q --tb=short
python -m pytest -q --tb=short --junitxml=outputs/etm_validation/final_suite.xml
python -m pytest --collect-only -q
python -m compileall -q algorithm/etm_mappo algorithm/train_etm_mappo.py algorithm/evaluate_etm_mappo.py algorithm/benchmark_etm_mappo.py
python outputs/etm_validation/audit_sources.py
git diff --check
python -u algorithm/train_etm_mappo.py --total-sampled-steps 8192 --output-dir outputs/etm_local_learning_precheck_seed0
python -u algorithm/evaluate_etm_mappo.py --checkpoint outputs/etm_local_learning_precheck_seed0/final.pt --seed-base 20000000 --episodes 6 --scene random --output outputs/etm_validation/random_evaluation_cli.json
python -u algorithm/benchmark_etm_mappo.py --budget 8192 --seeds 0 1 2 --reuse-completed-from outputs/etm_navigation_ablation_3seeds --output-dir outputs/etm_navigation_ablation_final
```

`--reuse-completed-from`只复用早期已完成、配置和checkpoint严格匹配的三个B0实验；不重启中断训练。其余方法使用最终相同协议完整运行；原目录保留，最终目录内的reuse_provenance.json记录来源。开发阶段发现自身特征含真实质心信息后已修正为公开编队偏移，旧schema的512步smoke和中断预检不纳入当前实验结论，当前loader会拒绝缺少local_task_state_v1观测schema的旧开发checkpoint。未据负结果调奖励、修改物理难度或扩大预算。

## 诊断字段与实际例子

`optimization_metrics.jsonl`记录policy/value loss、entropy、KL、clip fraction、ratio审计、advantage/return target均值方差、Actor/Critic裁剪前梯度与裁剪频率、有效/填充样本数量、三个类型隐藏范数、两类attention entropy/top1、gate均值/方差/饱和及128维统计、动作均值/std/饱和和Gaussian sigma。示例：

```json
{
  "pre_update_ratio_max_abs_error": 1.430511474609375e-06,
  "actual_actor_optimizer_steps": 16,
  "actual_critic_optimizer_steps": 16,
  "valid_agent_samples": 4096,
  "actor_gradient_norm": 0.9516244791448116,
  "critic_gradient_norm": 3.992801006883383,
  "gate_mean": 0.5014518164098263,
  "gate_std": 0.03595138504169881,
  "gate_saturation_fraction": 0.0,
  "peer_hidden_norm": 2.653018891811371
}
```

其中optimizer step为本rollout实际次数，checkpoint为累计次数；valid_agent_samples为唯一真实agent-step，不乘epoch。实体无效/集合为空的诊断权重为零，不将attention权重或低Loss直接解释为有效协作。

## 限制、未验证假设与后续重点

- 只有短程8K预算及三个种子。不能把能运行、Loss下降或单次成功当作稳定性能收益；不自动扩预算。
- 当前示范环境的可见实体给出相对速度，simple又是确定性场景，可能并不强依赖历史记忆；history-reset没有劣势时不能硬称记忆有效。
- 失去可见性的实体历史按定义清空，再出现从零开始，不提供遮挡期间持久缓存。
- 碰撞以离散积分后的几何距离检测；没有连续碰撞解析、真实机器人硬件或更复杂传感器验证。随机布局泛化只测试有限seed。
- Actor参数量不同；除gate固定对照和完整参数history-reset对照外，其他比较不能完全分离容量、优化难度和结构收益。
- 不支持bit-exact训练恢复；不保证跨GPU/软件版本的位级训练一致性。chunk起点按保存的状态detach，未额外加入burn-in或辅助损失。
- 回归中原有44项失败仍然存在；本任务保留证据，不用修改历史断言掩盖它们。
- 本次训练日志包含并行回归负载下的运行时间，不能据此作为硬件吞吐量优劣基准。

正式训练前应优先复核：更多独立seed下的可重复学习、目标完成与碰撞率，而非只看Return；检查任务是否确实需要历史、固定ID/失联语义是否符合目标应用；观察ratio审计、GRU/attention/gate统计与梯度是否稳定；验证随机布局下的泛化和环境离散近似。只有明确问题后再决定预算，不为得到期望排序而改环境或奖励。
