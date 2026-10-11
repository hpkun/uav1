# Combat Environment v3.12

v3.12 是 v3.11 的 **observation-only** 独立版本，维度为56/3/5。简化运动、初始化、Radar、4km weapon、单Gaussian Equation(8)、6枚弹药、每step一次发射、R1/R2/R3/R4/safety、PBRS关闭、零terminal reward、Blue全局最近追击、Blue边界返回、outcome全部继承v3.11同一个实现。只有 actor observation 及相关输入接口改变；不修改历史builder。

## Equation (24)信息结构与项目约定

论文直接定义的结构为 Self `(p_e,v,phi,psi,theta)`、Allies `(p_e,v,psi,theta)`、Enemies `(d,v,AA,ATA,HA)`。将4v4自然扩展为5v5后：self7 + 4×ally6 + 5×enemy5 = **56D**。

当前实现约定：own位置使用arena/world coordinates，使agent获知相对于边界的位置；ally位置使用当前agent flight-path/body-frame 的相对位置；采用固定identity slots、死亡槽全零、明确数值归一化；这些不是论文额外公开的归一化/死亡打包规定。

`phi exists in the MADSAC observation schema, but is identically zero under the simplified v3.11/v3.12 kinematic model.` 没有恢复bank/roll/controller。死亡agent整条56D row也全零，算法继续使用现有独立Red alive mask处理loss/action/GRU reset。

## 精确布局和归一化

切片为Python左闭右开：

| 切片 | 实体 | 特征顺序和固定归一化 |
|---|---|---|
| 0:7 | Self | x/5000, y/5000, altitude/6000, (v−225)/75, 0, psi/π, theta/(π/3) |
| 7:13 | 第1 ally | relative_forward/10000, relative_right/10000, relative_up/10000, (v_j−225)/75, psi_j/π, theta_j/(π/3) |
| 13:19 | 第2 ally | 同上，6D |
| 19:25 | 第3 ally | 同上，6D |
| 25:31 | 第4 ally | 同上，6D |
| 31:36 | Blue0 | d/10000, (v_k−225)/75, AA/π, ATA/π, HA/π |
| 36:41 | Blue1 | 同上，5D |
| 41:46 | Blue2 | 同上，5D |
| 46:51 | Blue3 | 同上，5D |
| 51:56 | Blue4 | 同上，5D |

Allies按Red index升序跳过自身，如Red2对应Red0/1/3/4；Blue始终0/1/2/3/4。不按distance、visibility、alive或threat重排。Ally的psi是友机绝对world heading，并非旧版的relative heading。

归一化不clip输入；合法3D长对角距离可超过10000m，因此normalized distance可以略大于1。builder使用float64保留geometry精度以供审查；vector/trainer沿用既有float32网络输入转换，不改变算法数值协议。

## Ally body-frame

直接复用 `env.observation.flight_path_frame` 和 `_relative_position`，没有创建另一套左右/上下定义。在NED下：

```
delta_p = p_j - p_i
f = [cos(theta_i)cos(psi_i), cos(theta_i)sin(psi_i), -sin(theta_i)]
r = [-sin(psi_i), cos(psi_i), 0]
u = normalize(cross(r,f))
relative_forward = dot(delta_p,f)
relative_right   = dot(delta_p,r)
relative_up      = dot(delta_p,u)
```

heading0/pitch0：+x为forward、+y为right、−z为up；headingπ/2后frame随之旋转。

## Enemy geometry统一

直接调用 `env.v36_geometry.combat_geometry`，与v3.11 Equation(7)/(8)、R3/R4使用同一个helper。Observation不调用旧`engagement_geometry`来生成HA。

原始helper：d是真实3D Euclidean距离；ATA=LOS水平heading与own heading的圆周角差；AA=LOS水平heading与target heading的圆周角差；HA=NED LOS elevation与own pitch的圆周角差。统一helper本身已经输出非负圆周角差[0,π]，v3.12直接暴露该输出，未新增绝对值处理或观测专用signed geometry，也未修改Weapon/Reward几何。

因此AA/ATA/HA统一除以π。HA数学helper范围[0,π]；当前pitch限制±π/3、LOS elevation±π/2下，其实际合法上界为5π/6。不能把它当作旧HA的[−π/2,π/2]而继续除以π/2。多个人工几何反归一化后d/AA/ATA/HA与combat输出误差≤1e−8。

## Radar、ammo和死亡槽

只要Blue存活，其5D slot就填充完整geometry，不受10km radar range或120° FOV影响。目标在机尾、侧方、超过radar range也仍可观测；有物理意义的零角度允许为0，不强制每个元素非零。builder完全没有sensor/visible依赖。Radar配置10km/half60°保持，仅供diagnostics/replay/visible-pair statistics。

所有ammo、alive flag、radar flag从actor input删除；实际弹药系统仍6枚且完全冻结。死亡ally6D/Blue5D槽全零，identity位置保留，其它实体不前移：`dead-slot zeroing is a fixed-size implementation convention, because the paper does not specify post-destruction packing.`

EA/STEA输入encoder只适配self7/ally6/enemy5，保持entity64、heads2、fusion128、原GRU128和其它隐藏宽度。实体attention有效mask遵循整个slot是否非零的输入约定，不添加flag、外部Blue存活信息或隐藏额外观测。Legacy布局继续使用原alive-feature mask。STEA表示诊断工具同步使用同一个mask helper。

## 算法与checkpoint

MAPPO flat input56；RMAPPO flat encoder input56；EA/STEA合计输入56，分别以7/6/5维encoder处理实体。共同centralized MLP critic仍采用原global+local拼接协议，输入(5+1)×56=336，隐藏层256/256不变。

四份`*_5v5_v312.yaml`仅改observation_dim和必要entity layout描述，训练、隐藏维度、学习率、gamma/lambda/clip/entropy/rollout/minibatch/epochs/evaluation与checkpoint cadence同v3.11。checkpoint合同保留版本、维度、SHA、implementation和architecture验证；v3.11 66D checkpoint被拒绝用于v3.12，无截断、padding、填零或state_dict改写。v3.12需要从头训练，本次未启动正式训练。

历史v3.0–v3.11继续原66D或各自旧维度、ammo/alive特征和radar mask。验收包括原完整轨迹reference、旧实体网络参数/输出reference、相同state/actions下v3.11与v3.12武器/RNG/reward/outcome/运动逐步完全一致、vector reset、四算法各512 transitions/1次真实update/save-reload/2局CUDA evaluation；结果见`outputs/v312_validation/report.md`。
