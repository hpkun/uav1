# MAAC v3.42 兼容性适配

本轮仅使现有 MAAC 选择原有 v3.42 环境，不优化算法，不修改任何环境机制。

## 实际改动

- `algorithm/maac/protocol.py` 的 `DIMENSIONS` 新增 `'3.42': (45, 3, 4)`，更新对应错误提示的支持范围。
- 新增 `configs/maac_4v4_v342.yaml`，从 `maac_4v4_v341.yaml` 完整复制，只把 `environment.environment_version` 从 `3.41` 改为 `3.42`。
- 新增专用功能测试和有界 CUDA 验证脚本。现有训练入口、并行环境、Replay Buffer、评估和 Checkpoint 指纹逻辑直接复用，不另写实现。

未修改 Actor/Critic、损失、反事实优势、采样策略、目标网络更新、优化器或训练参数。新配置保留共享 Actor、128 宽网络、4 头注意力、1024 minibatch、16 次反事实基线采样和原有 4 critic / 4 actor 更新周期。

## 环境冻结

不修改任何 `env` 源文件及历史环境配置。v3.42 保留原始45维观测及顺序、3维动作、直接增量运动学、Reward、Blue策略、场景、30°三维攻击圆锥、2000米射程、原有概率命中模型、6发弹药、1秒冷却和原有胜负判定。

不引入 v3.43/v3.48/v3.50 的 Reward 或武器修改。v3.41 与 v3.42 的环境差异仍仅来自既有版本设计。

## 开火率的精确定义

- 训练近期 `fire_rate`：近期已完成回合中，`red_first_attempt_step is not None` 的回合比例。
- 确定性评估 `fire_rate`：`red_attempt_episode_rate` 的别名，也是至少有一次实际 Red 开火尝试的回合比例。
- `red_fire_window_episode_rate`：进入开火窗口的回合比例，独立保留，不作为 `fire_rate`。
- `red_step_fire_attempts` 是当前环境步的尝试次数；`red_fire_attempts` 是当前回合累计尝试次数。概率未命中也属于真实尝试。

专用测试构造了“满足攻击几何、窗口已出现，但冷却仍未结束”的状态，验证窗口比例为1而训练/评估开火率均为0。

## CUDA 验证范围

脚本 `tools/verify_maac_v342.py --stage smoke --output <新的空审查目录>`：

1. 使用完整原配置网络及12个并行环境，采样1200个transition，验证Replay形状为 `[1024,4,45]` / `[1024,4,3]`。
2. 只执行一个原有更新周期：4次Critic、4次Actor、1次Target；检查梯度、损失、参数及Adam状态有限，在线/目标权重确实更新。
3. 保存并经官方加载器重载Checkpoint，检查全部网络权重及确定性动作精确一致。
4. 两个固定种子的确定性评估重复运行一致，Checkpoint拒绝缺失身份字段及跨版本混用，并能记录对局trace。
5. 不进入正式2M/3M训练循环。`training_smoke=false`表示使用完整配置网络，不代表运行正式预算。

`--stage history` 用于比对输出目录中的修改前 `baseline_source` 快照：v3.41/v3.42/v3.48/v3.50 的完整固定动作轨迹、状态、观测、Reward、武器事件、RNG和v3.41/v3.48/v3.50的MAAC身份元数据必须完全一致。
运行历史比对前需保存本次修改前的 `env/algorithm/configs` 快照。审查目录须独立保存，不能覆盖已有实验结果。

所有命令在 `uav` Conda 环境执行；WSL不可用时按仓库政策使用PowerShell。训练更新和模型审查必须使用CUDA。
检查脚本将PyTorch导入限定在主进程，避免Windows spawn环境子进程不必要地重复加载CUDA运行库；不改变生产算法或并行采样行为。

## Checkpoint 能力边界

环境版本、环境/算法配置SHA256、45/3/4维度和原MAAC实现身份必须匹配。相同网络维度不允许跨v3.41/v3.42/v3.48/v3.50加载。
保留原有保存/评估重载行为；精确续训仍按原实现拒绝，因为完整Replay、环境和RNG状态没有保存，本轮不扩展此能力。

实测结果与日志保存在本轮独立的 `outputs/maac_v342_compat_*` 审查目录，详见其中验收报告。短程功能通过只说明兼容链条可用，不证明策略性能或学习效果改善。
