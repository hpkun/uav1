# Combat Environment v3.11

v3.11 是基于 v3.10 的独立 5v5 简化三维 MADSAC-style 环境。旧版本源码与配置保持原样。四个 PPO 算法的网络、超参数、evaluation/checkpoint cadence 与 v3.10 完全一致。正式配置见 `configs/combat_environment_v311.yaml`。

## 直接运动

每步将连续 action `[a_psi,a_theta,a_v]` clip 到 `[-1,1]^3`，计算：

```
Δψ = π a_psi
Δθ = (π/3) a_theta
Δv = 50 a_v
ψ' = wrap(ψ + Δψ)
θ' = clip(θ + Δθ, -π/3, π/3)
v' = clip(v + Δv, 150, 300)
x' = x + v' cos(θ') cos(ψ') dt
y' = y + v' cos(θ') sin(ψ') dt
z' = z - v' sin(θ') dt
dt = 0.1 s
```

增量按每个 simulation step 应用；不再乘 dt 或加入动作滞后。位置使用更新后的姿态与速度。死亡飞机不推进。v3.11 不构造或调用 PointMassDynamics/RK4/inverse controller，不使用 nx/nz、bank dynamics、load factor、PID、autopilot、turn/acceleration limits。配置只保留 `action.command` 三个增量上限；validator 拒绝 controller 和旧工程参数。

Observation 仍为 66D：self8 + ally7×4 + enemy6×5，实体顺序、ammo、死亡 mask 完全沿用 v3.10。`last_executed_phi`、`executed_red_phi` 和内部双方 phi 兼容数组始终为0；self 的 phi 槽是0-based index 4。本版不显式建模 bank angle，Observation 将来另行设计。

## 雷达

仅 alive 的 observer/target 且真实三维距离 ≤10000m、飞机 forward vector 与 LOS 的三维夹角 ≤π/3（60°）时 visible。总前向圆锥 FOV 为120°。不分离 azimuth/elevation，不加入 noise、tracking memory、lock delay、概率探测或 RCS。重用原简单3D sensor，仅改变参数。雷达只服务于 observation/诊断，不参与 Blue 目标选择、武器资格、R3/R4。

## 初始化

每次 reset 只采样一次 `alpha~Uniform(-π,π)`。令 `u=[cosα,sinα]`、`t=[-sinα,cosα]`：

```
C_red = -4800 u
C_blue = +4800 u
lateral_i = [-400,-200,0,200,400] m
p_red_i = C_red + lateral_i t
p_blue_i = C_blue + lateral_i t
```

编队中心相距9600m，各机位于5000m半径内。双方所有飞机初始 altitude=3000m、v=225m/s、pitch=0；Red heading=α，Blue heading=wrap(α+π)。无 chord placement、lateral rejection sampling、高度/速度/heading扰动。初始化后该 RNG 仅供原武器 Gaussian 采样；Blue策略无 RNG。

## 开火与 Equation (8)

沿用项目的 ATA=水平 heading 与 LOS 的绝对角差、HA=pitch 与 NED LOS elevation 的绝对角差。资格仅为 alive attacker/target、ammo>0、0≤真实3D距离≤4000m、ATA≤30°、HA≤30°。AA、radar、速度不参与资格。角度沿用原 `angle_within` 的1e-12弧度数值舍入容差。

每步每架攻击者从所有合法目标中选真实3D距离最近者，精确相同则选较小 target index，最多发射1枚并消耗1枚弹药。初始每架6枚。持续合法可每步重复发射直到弹药耗尽；不存在 previous_eligible、strict pair-entry、cooldown、rearm、lock timer 或导弹飞行时间。继承 step 调用中的 `fire_states` 兼容参数为None，射击实现完全不读取它。

双方完成各自全部 fire/hit sampling 后统一结算击毁，保留同一步互相攻击。每次 attempt 只抽一次 `epsilon~N(0,1)`，两条不等式共享同一个值：

```
ATA + c4 epsilon <= π exp(-d / D_hit)
HA  + c5 epsilon <= π exp(-d / D_hit)
c4 = c5 = 0.9984711359155588
D_hit = 4000/ln(6) = 2232.442506204989 m
```

`P(hit|4000m,0,0)≈0.70` 是透明项目标定，c4/c5/D_hit 没有被表述为论文公开数值。没有另加 Bernoulli 概率抽样。

## 奖励与终局

R1 kill +10、Red destroyed -10；R2 Red horizontal/ceiling boundary loss -10，ground loss沿用R1。R3：任一存活Blue满足 d≥4000、ATA≤30°、HA≤30°，该存活Red获得+0.001，否则0，不受雷达距离/FOV限制。

R41：d≤4000、AA_r≤30°，ATA/HA同时落在30°/15°/5°内分别+0.01/+0.02/+0.10。R42 使用Blue为中心的reverse geometry，d≤4000、AA_b≤30°，ATA_b/HA_b同时落在30°/15°/5°内分别−0.015/−0.025/−0.15。只取 strongest positive target 与 strongest negative threat，R4=offense+defense。双方死亡 mask 保留，没有 visible gate。

注：ATA/HA≤30° 的目标无法处于当前60°雷达半角外（最大3D偏离约41.4°）。专项测试用强制不可见/禁止调用sensor，以及R3目标超过10km雷达距离，验证奖励与雷达真正解耦，未改变几何定义来构造不可能场景。

PBRS关闭；win/lose/draw terminal reward均0，team casualty为0。保留 safe_distance=200m、safety_penalty_scale=0.2；`paper_reward=R1+R2+R3+R4`，`training_reward=paper_reward+safety`。safety是5v5碰撞扩展，不属于MADSAC Equation (25)。没有新增中心/高度/生存/编队shaping。

只要Blue alive count=0就判Red win，包括双方同时全部击毁；Red=0且Blue>0判Blue win；timeout且Blue>0判Blue win。没有draw或timeout survivor-count comparison，terminal reward仍为0。

## Blue 与边界

沿用 v3.10 `PaperNearestTargetBluePolicy`：每步全局最近存活Red的真实3D位置，精确距离tie选小index，立即重选，没有 radar gate、锁定、escape/guerrilla/centripetal或随机数。desired_speed=250m/s是项目参数。原追击输出的归一化增量直接交给新运动模型。

Blue水平、ground、ceiling触发后的位置/heading/pitch修正和计数直接继承v3.10的同一个方法，不改为死亡。Red边界失效及ground优先级同样继承。arena保持半径5000m、高度0–6000m、max_steps1000。

验收仅运行专项/full pytest、compileall、vector auto-reset、四算法各512 transitions/1次真实update/save-reload/2局CUDA evaluation，以及ZERO/PURSUIT各3个固定seed。结果见 `outputs/v311_validation/report.md`；不根据sanity结果调参，不启动正式训练。
