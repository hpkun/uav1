# navigation3d-v1 实施与验证报告

验证日期：2026-10-21。执行环境：WSL Ubuntu，`conda activate uav`，PyTorch 2.11.0+cu128，NVIDIA GeForce RTX 5060 Laptop GPU。全部训练、checkpoint 审计与评估实际使用 CUDA。

## 交付范围

新增独立非武器化三维协同导航环境。只执行几何/奖励数学验证、固定控制器预检和 MAPPO/IPPO 各 1024 sampled steps 的短测，没有启动正式长时间强化学习训练。

本次仅新增文件，未编辑既有跟踪文件。历史环境、训练配置、优化器数学和模型文件保持原样。新训练入口、checkpoint family 与现有历史环境分离。

## 文件清单

| 文件 | 用途 |
|---|---|
| `env/navigation3d/__init__.py` | 独立环境公开入口 |
| `env/navigation3d/config.py` | 导航身份、配置校验与维度 |
| `env/navigation3d/geometry.py` | NED/机体导航方向几何 |
| `env/navigation3d/environment.py` | 有序航路点、编队、安全、PBRS、局部观测 |
| `configs/navigation3d_v1.yaml` | 环境全部运动、安全与奖励系数 |
| `configs/mappo_navigation3d_v1.yaml` | 50 维 MAPPO 配置 |
| `configs/ippo_navigation3d_v1.yaml` | 50 维 PS-IPPO 配置 |
| `algorithm/navigation3d/__init__.py` | 导航适配包 |
| `algorithm/navigation3d/trainer.py` | 原 PPO 优化器适配、局部价值网络、严格 checkpoint |
| `algorithm/navigation3d/vector_env.py` | 导航专用长驻 spawn workers |
| `algorithm/navigation3d/runner.py` | 原标准 RolloutBatch 采样 |
| `algorithm/navigation3d/evaluation.py` | 确定性评估、姿态轨迹、checkpoint 重载 |
| `algorithm/train_navigation3d.py` | 默认 1024、上限 8192 步的技术短测 CLI |
| `algorithm/evaluate_navigation3d.py` | 独立 CUDA 评估 CLI |
| `tools/validate_navigation3d.py` | 五种固定控制器同种子预检与分量审计 |
| `tools/render_navigation3d.py` | 保存轨迹的三维位置图、姿态图和 CSV |
| `tests/test_navigation3d_geometry.py` | 独立向量点积、坐标、角度和数值稳定性 |
| `tests/test_navigation3d.py` | 实际 step 奖励、终止、掩码、确定性、安全与反投机 |
| `tests/test_navigation3d_algorithms.py` | CUDA PPO、子进程、checkpoint 与历史超参数一致性 |
| `tests/test_navigation3d_visualization.py` | 已保存轨迹与姿态导出验证 |
| `docs/navigation3d.md` | 公式、状态/动作/观测定义、配置、命令与限制 |
| `docs/navigation3d_acceptance.md` | 本报告 |

## 几何、状态与奖励验证

公式、配置系数和 50 维精确索引布局见 `docs/navigation3d.md`。代码位置：几何 `waypoint_geometry`，奖励 `Navigation3DEnv.potential_components/step`，观测 `Navigation3DEnv.observe`。

独立点积参考覆盖 108 组俯仰、航向、目标高度角和目标方位组合；另覆盖水平 0/15/30/30.001/60/90/180 度、上仰 50 度与下方 20 度航路点形成 70 度夹角、反向下降情况、±180 度航向接缝、正上/正下航路点、重合位置、1e-300 小距离和 1e150 大坐标。前/右/上基的正交性及往返转换、三维夹角与机体系双误差的余弦关系均有断言。非法输入明确拒绝。

状态为每机 NED `(x,y,z,v,theta,psi,alive)`，另有任务阶段、assigned/valid 掩码和剩余时间。动作是三维归一化姿态/速度增量，复用原运动学函数；测试逐元素精确验证其结果。安全检测包括区间内相对最近点，覆盖帧间穿越碰撞。

塑形分量为距离、三维方向、编队误差三个有界非负势函数的 `0.99*Phi(next)-Phi(current)`，最大单机总势为 0.2。安全项为危险接近 -0.02、本人失效 -10、每步 -0.001；团队完成时每架有效飞机 +10。各项只加入 step 返回奖励一次，终止/超时后势函数均为零。自动 reset 的下一回合状态不进入上一回合势函数或 GAE。

为避免切换跳变，势函数固定参考最终指定目的地，中间航路点用于有序任务推进。这是本导航任务的明确设计选择，未宣称它对任意折返航路均给出最优中间引导。配置中的势函数 gamma 与算法 gamma 不同则拒绝构建或评估。

## 固定控制器与奖励投机预检

每种控制器使用同一组 20 个 seed：20000000～20000019。以下是工程预检，不是强化学习算法性能比较。

| 控制器 | 完成 | 平均步数 | 平均仿真秒数 | 团队总回报 | 每机净塑形 | 终止 |
|---|---:|---:|---:|---:|---:|---|
| 航路点跟踪 | 20/20 | 66.00 | 13.20 | 38.651834 | -0.271041 | 成功 20 |
| 零增量直飞 | 0/20 | 145.00 | 29.00 | -22.046406 | -0.366601 | 越界 20 |
| 持续盘旋 | 0/20 | 400.00 | 80.00 | -4.372762 | -0.693190 | 超时 20 |
| 低速盘旋 | 0/20 | 400.00 | 80.00 | -4.416021 | -0.704005 | 超时 20 |
| 远离目标 | 0/20 | 131.95 | 26.39 | -20.629716 | -0.275479 | 越界 20 |

各分量为“每机、每局”的平均累计贡献：

| 控制器 | 距离塑形 | 方向塑形 | 编队塑形 | 危险接近 | 失效 | 时间 | 完成 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 航路点跟踪 | -0.113135 | -0.077489 | -0.080417 | 0 | 0 | -0.066000 | 10 |
| 零增量直飞 | -0.170322 | -0.078672 | -0.117608 | 0 | -5.000000 | -0.145000 | 0 |
| 持续盘旋 | -0.306089 | -0.145257 | -0.241844 | 0 | 0 | -0.400000 | 0 |
| 低速盘旋 | -0.316142 | -0.145703 | -0.242160 | 0 | 0 | -0.400000 | 0 |
| 远离目标 | -0.109532 | -0.054981 | -0.110966 | 0 | -4.750000 | -0.131950 | 0 |

失效惩罚只记入本步真正失效的飞机，因此团队首次失效结束时，按全体四机平均的失效项可以是 -5 或 -4.75；这不是漏记，原始逐机值保存在 episodes.json。

三个塑形分量分别截取正项再相加，20 局中最大单机正贡献总量：跟踪 0.004298（完成奖励的 0.043%）、盘旋 0.200598（2.006%）、低速盘旋 0.199387（1.994%）、直飞和远离目标均为 0。计入负项后的净塑形见表，均小于零；不能把正贡献总量当作净收益。

静止势函数测试证明 `(gamma-1)*Phi<=0`；闭合循环测试证明未折扣循环和为 `(gamma-1)*sum(Phi)<=0`。实际控制器回合验证折扣累计塑形与 `-Phi(initial)` 一致，终止回合不折扣净塑形非正；累计奖励与逐步分量一致。这一保证针对当前非负有界势函数和明确的有限终止约定，不能扩展为“所有可能策略均已经实验穷举”。

原始数据：`outputs/navigation3d_validation/controllers/summary.json`、`episodes.json`、`comparison.csv`。每种控制器首个 seed 的完整轨迹、位置 CSV、三维位置图和姿态图均单独保存。

## MAPPO/IPPO CUDA smoke

两种算法均使用 seed=1、2 个长驻 spawn workers、1024 sampled steps、正式 256×256 网络、原有 rollout=256/epochs=10/minibatch=512；没有为了短测减少网络或改变 PPO 超参数。两次更新均实际改变 Actor/Critic 参数，指标有限。训练步数按环境交互数计，不乘 agent 数。

| 项目 | MAPPO | PS-IPPO |
|---|---:|---:|
| Actor 输入 / 动作 / agents | 50 / 3 / 4 | 50 / 3 / 4 |
| Critic 输入 | 全体 masked 观测 200 + 本机 50 | 本机 50 |
| Actor 参数量 | 79,622 | 79,622 |
| Critic 参数量 | 130,305 | 79,105 |
| sampled steps | 1024 | 1024 |
| PPO 更新 | 2 | 2 |
| Actor / Critic 优化次数 | 20 / 20 | 20 / 20 |
| 第二次更新 Actor 梯度范数 | 0.137352 | 0.130405 |
| 第二次更新 Critic 梯度范数 | 0.194740 | 0.176438 |
| 第二次更新平均近似 KL | 0.005478 | 0.005534 |
| checkpoint 重载动作逐元素一致 | 是 | 是 |
| 5 局确定性评估完成数 | 0/5 | 0/5 |
| 评估平均步数 | 157.8 | 153.8 |
| 评估平均团队回报 | -20.190609 | -20.167776 |

所有短评估均因越界结束。这些模型只验证采样—PPO—梯度—保存—重载—评估链路，尚未学会可靠导航；20/20 的完成率属于固定导航控制器。没有增加预算或挑选更有利 seed 来改变这一结果。

`algorithm/evaluate_navigation3d.py` 也实际执行了 IPPO checkpoint 的独立 CLI 评估，结果 0/5，与短测中的确定性结果一致。`compileall` 静态语法检查通过。checkpoint 还验证了价值输出一致性、优化器状态恢复、参数计数器及篡改身份/指纹的拒绝。

原始证据：`outputs/navigation3d_validation/mappo_smoke/summary.json`、`ippo_smoke/summary.json`、各目录 `optimization_metrics.json`、`final.pt`、配置快照，以及 `ippo_evaluation_cli.json`。

## 测试与历史失败

新增测试：173 passed，0 failed，0 skipped（7.52 秒）。首轮有一项测试将数组列表传给 pytest.approx，触发测试断言本身的数据类型错误；改成实际运动学结果的数组逐元素精确比较后全部通过，没有更改运动学或跳过测试。

修改前完整基线：4565 passed、28 failed、2 skipped，544.72 秒。修改后完整回归：4738 passed、28 failed、2 skipped，510.60 秒。逐个比较失败测试 ID 后，28 项完全一致，新增回归失败为 0；通过数增加 173，与新增测试数一致。日志分别为 `baseline_pytest.log` 与 `final_pytest.log`，机器可读比较结果在 `outputs/navigation3d_validation/verification_summary.json`。

已检查基线失败的实际原因：

- 10 项 runner/compact metrics 测试使用未完整初始化的 runner，缺少 `env_config`，在现有 `_accumulate_rollout_metrics` 抛出 AttributeError。
- 13 项 MADSAC 相关失败涉及旧实现版本断言、先于 environment_version 的实现版本拒绝、checkpoint attention_head_dim 不匹配等历史协议假设。
- 2 项 MADSAC/MAAC 诊断字符串断言仍期望旧版本范围。
- 1 项历史配置目录测试将配置文件列表写死，修改前已不匹配；本次新增导航配置也会列入文件清单。
- 1 项 v3.13 轨迹摘要与既有 JSON reference 不一致，基线已复现；本次不重写参考数据。
- 1 项浮点精确相等断言中，实际 0.39999999999999997 与期望 0.4 不完全相等。

这些历史测试没有删除、改写或新增 skip。两项既有 skip 也保留；本任务没有增加跳过项。具体失败 traceback 与测试 ID 全部保留在日志中。

## 轨迹和未解决问题

代表性 seed=20000000：固定航路点跟踪控制器完成 66 步/13.2 秒，四机先爬升至约 1000 m，再沿第二段航路下降至约 900 m，全程无危险接近/碰撞/越界。三维图有起终点、航路点和运动方向箭头，姿态图包括速度、俯仰和展开后的航向；已实际打开检查。

文件：`outputs/navigation3d_validation/controllers/waypoint/trajectory.png`、`attitudes.png`、`states.csv`、`trace.npz`。终止帧 valid=0 表示回合结束，终止位置仍保存在原始姿态轨迹中。

尚未解决：短测模型 0/5 完成；任意路线的学习效果及更多 seed 泛化尚未验证；零滚转点质点模型、距离式局部可见性、有限同步航路点任务和线性区间碰撞近似的限制仍存在；没有 bit-exact 并行训练恢复能力。详细说明见 `docs/navigation3d.md`。
