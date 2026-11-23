# 全局可行性匹配基线

本实验将 Huang 等论文的缓冲区边界拦截点和到达时间可行性约束，用于现有的全局一对一目标分配，并沿用项目的 MADDPG 控制器。它不是论文中的局部 Hungarian + MASAC 完整算法。

对直冲模式的进攻机 `j`，沿其当前位置到目标区中心的名义直线轨迹，计算进入突防圆的预测点 `q_j`。仅在防御机 `i` 当前能感知 `j` 且 `η T_D(i,j) ≤ T_A(j)` 时建立可分配边。到达时间按照当前速度在目标方向的有符号投影、最大加速度和最大速度，估计连续的“先加速后匀速”时间；反向飞行须先减速和掉头。`η` 为 `assignment_safety_factor`，默认 1.1。这个数值是本项目的实验设定，不是从论文提取的固定参数。估计忽略横向转向和离散动力学误差；攻击机的脚本还有凝聚、避碰项，实际轨迹可能偏离名义直线。因此它是匹配启发式，不能视为精确轨迹预测或物理可行性的严格证明。

- `configs/interception_huang_global.yaml`：可行边的代价为 `||p_i - q_j||`，对应论文的边界点距离式。
- `configs/interception_huang_time.yaml`：可行边的代价为 `T_D(i,j)`，是额外的时间代价对照实验。

分配使用现有五对五精确最大基数、最小总代价求解器，得到的全局最优匹配与 Hungarian 对同一代价矩阵的目标相同，且不依赖 SciPy。初始化、每 20 步、系统首次检测新目标以及捕获或突防事件会触发重分配。Actor 的 54 维观测、Critic 的 75 维状态、奖励和 MADDPG 更新流程保持原有定义。新方法仅支持 `direct_rush`；分散进攻需要单独设计多段轨迹的拦截点预测。

正式训练使用新配置和原 MADDPG 配置，在 Linux 上运行：

```bash
python -u -m experiment.cli train \
  --env-config configs/interception_huang_global.yaml \
  --algorithm-config configs/maddpg_interception.yaml \
  --algorithm maddpg --attack-mode direct_rush \
  --total-sampled-steps 2000000 --num-envs 4 \
  --seed 7 --device cuda \
  --output-dir runs/maddpg_huang_global_direct_seed7
```

建议将该结果标为 `Huang-inspired global matching + MADDPG`，并与 `distance_matching + MADDPG` 使用相同训练预算和评估种子比较。

## 36 维 Actor 观测对照

`configs/interception_huang_global_target_velocity.yaml` 使用独立的 `Interception/3` 合同。Actor 输入为自身位置和速度 4 维、4 个友机与 5 个进攻机各自的相对位置及有效位共 27 维、保持原样的分配目标相对位置/相对速度/有效位 5 维，总计 36 维。普通可见对象的速度不进入 Actor；分配器仍按真实速度计算到达时间。MAPPO 和 MADDPG 的 Critic 仍访问相同的 75 维全局状态（双方位置、速度、存活位和分配矩阵），MADDPG Critic 还接收联合动作。这遵循 CTDE：执行时的局部信息可以简化，而训练时的价值估计继续使用全局动力学信息。原 54 维 `Interception/2` 配置及 checkpoint 合同不变。新旧观测版本的模型不能互换加载。
