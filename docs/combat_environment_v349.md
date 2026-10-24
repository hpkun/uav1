# v3.49：仅R4的三维战术引导候选

这是MAAC v3.48退化专项审查中的独立、默认不启用实验，尚未证明能解决学习失败。
运行时须显式指定v3.49配置；所有旧环境、Checkpoint和正式训练输出保持原样。

新增环境继承v3.48，观测仍直接使用v3.38原始45维，动作3维、4v4、dt=.1、
最大1000步、Blue规则、概率武器、6次弹药和1秒冷却全部不变。
R1、R2、R3/PBRS的方法与参数不变；仅覆盖dense_combat_reward。
MAPPO/MAAC配置仅修改环境版本身份，网络和算法参数与v3.48一致。

源码审查必须区分v3.48实际运动学与历史控制分支：当前路径为v3.32直接
航向/俯仰/速度增量，nz和滚转phi不适用，不应伪造这些控制量或将兼容零值解释为实测。

## 改动依据与公式

原R4要求AA/ATA/HA满足论文分段门槛，但概率武器使用三维前向圆锥。
1500米正对接敌时，三维alpha=0而AA约pi，存在武器合格、R4进攻为0的明确反例。
20个开发种子的Best策略资格样本中，也未观察到R4正奖励。
这证明几何目标存在差异，不证明其为MAAC失败的唯一原因。

复用现有 `v343_reward.continuous_tactical_rewards`，不复制数学实现：

```text
w(d) = clip((4000-d)/2000, 0, 1)
a(alpha) = clip(1-alpha/pi, 0, 1)
offense_i = max_live_j 0.02 * w(d_ij) * a(alpha_i_to_j)
threat_i = min_live_j -0.01 * w(d_ij) * a(alpha_j_to_i)
R4_i = offense_i + threat_i
```

无AA尾后门槛，无进攻优先覆盖防御。独立分量使用既有tactical_offense、
tactical_defense字段，另提供候选分量episode总计；完整R1–R4分解保持一致。
单步每机R4界限[-.01,.02]。此项是普通战术密集奖励，不是PBRS。
R3仍为gamma*Phi(next)-Phi(current)，gamma=.99，终止/截断势为0。

研究假设：连续三维对准区分信号可帮助学习形成合法攻击窗口。
副作用：刷态势/保持距离、奖励累积、迎头互毁及更激进接敌；未新增安全奖励，
不能保证防止越界、撞地或超高。绝不能用不同Reward定义下更高Return宣称性能改善。

## 可复现审查与有界对照

在PowerShell激活uav，CUDA不可用即失败。输出路径必须为新目录。

```powershell
$env:PYTHONIOENCODING='utf-8'
conda activate uav
python tools/audit_maac_v348_failure.py --output outputs/new_failure_audit/diagnostics_clean
python -m pytest tests/test_maac_failure_math_audit.py tests/test_combat_v349.py -q
python tools/compare_v348_v349_bounded.py --output outputs/new_failure_audit/bounded_comparison_verified
python tools/report_maac_v348_failure.py --output outputs/new_failure_audit
```

诊断涵盖Best/Final确定性与随机策略、规则追击各21个种子，共105回合；
原20开发种子与20000071分别报告。CSV中仅记录器观察实际执行过程；
规则探针使用独立控制器，避免污染Blue诊断计数，不调用optimizer.step。
Critic探针明确为Soft Q/有限采样离策略诊断，不把未折扣Return当Q真值。

有界对照为v3.48/v3.49各6144采样，seed=1、12环境、正式网络和超参数，
初始Actor/Critic参数相同；同5个开发种子评估。每组192次Actor、192次Critic、
48次Target更新，含有限性、参数变化、保存重载、跨版本拒绝及回放检查。
Windows诊断驱动仅在主进程导入CUDA库，避免spawn子进程重复加载导致页面文件不足；
没有改动算法或环境worker协议。

44项新增数学/候选测试、387项历史相关测试通过；16个历史完整回合与HEAD精确一致。
完整科研报告、数据与图表位于：
`outputs/maac_v348_failure_review_20261024/科研分析报告.md`。
本轮没有启动正式3M训练。功能通过不是学习有效的证据。
