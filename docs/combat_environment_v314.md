# Combat Environment v3.14

v3.14 只改变动作执行语义，并恢复 self Observation 的实际控制 phi。v3.13 保留历史 instantaneous direct-motion，旧实验和录像继续可复现。

## 控制链

Actor 仍输出 `[-1,1]^3` 的 `[a_psi,a_theta,a_v]`。Equation (23) desired-state 映射复用 `env/control.py`：

```text
Δpsi=π*a_psi; Δtheta=(π/3)*a_theta; Δv=50*a_v
psi_d=wrap(psi+Δpsi); theta_d=theta+Δtheta; v_d=v+Δv
```

动作范围没有缩小。增量表示当前 decision step 的相对期望目标，不是必须瞬间执行的真实状态增量。每步根据当前真实状态重新计算目标；没有跨 step 持久 target。

`action_to_control` 调用现有 `action_to_target` 和 `target_to_control`，输出 `ControlCommand(nx,nz,phi)`。现有 controller 对垂向 A 进行非负投影，对 A/B 同比例投影以满足 normal load cap。没有新增 PID、气动模型或执行器动态。

沿用的透明模拟参数：heading/pitch/speed time constants 均为 **2.0 s**，`normal_load_max=8.0`。**这些 time constants 和 8-g cap 是已有模拟实现的参数，不是 MADSAC 论文公布的参数。**

完整 ±π desired heading 属于最短角方向的平局：历史 `wrap_angle` 使用 `[-π,π)`，因此 +1 和 -1 均可选 -π 方向。非平局正负命令保持左右对称。

## 点质量动力学与 RK4

继承的 `_advance` 调用现有 `PointMassDynamics` 和 `RK4Integrator(dt=0.1)`，固定该 step 的控制量，执行四个 RK4 stage。积分前不将状态赋为 desired state。

Equation (2) 保持原公式：

```text
v_dot     = g*(nx-sin(theta))
theta_dot = g/v*(nz*cos(phi)-cos(theta))
psi_dot   = g*nz*sin(phi)/(v*cos(theta))
x_dot     = v*cos(theta)*cos(psi)
y_dot     = v*cos(theta)*sin(psi)
z_dot     = -v*sin(theta)
```

RK4 继续负责 speed clip `[150,300]`、theta clip `[-π/3,+π/3]` 和 heading wrap。Red/Blue 都由同一个 `_advance` 接口执行相同控制链，死亡飞机不推进且 phi 为 0。

## 56D Observation

保持 `7 + 4*6 + 5*5 = 56` 和原有 fixed-slot ordering。调用 v3.13 权威 builder，仅替换存活 Red 的 self index 4：

```text
0 → red_last_executed_phi[i]/(π/2)
```

phi 是上一 transition 实际送入动力学的控制量，不是新增 AircraftState 动态状态。reset phi=0，dead row 全零。v3.14 保留 phi 的 float64 精度，避免 `π/2` 经 float32 舍入后使归一化值略超过 1；历史版本继续使用原有 float32 phi。

其余 55 个元素完全复用 v3.13。ally 保持 flight-path forward/right/up，不引入 roll 旋转。enemy 保持 distance/speed/AA/ATA/HA 和 `HA/(π/2)`，几何完全复用 `combat_geometry_v313`。

## 冻结合同与 checkpoint

Reward R1–R4、Safety 及所有系数完全不变；weapon eligibility、Equation (8) shared Gaussian sampler、ammo、repeat fire 和 simultaneous resolution 不变。Scenario、Blue global-nearest pursuit/desired speed 250、boundary return、outcome 和 timeout 逻辑不变。

环境配置仅新增 `action.controller` 和版本 3.14。validator 先剥离 controller 并映射为 3.13 调用旧 validator，再严格冻结 controller schema/数值。四算法 v314 YAML 与 v313 YAML 逐字节一致，centralized critic 与所有超参数不变。

虽然两版均为 56D，transition dynamics 不兼容，**v3.13 checkpoint 必须被 v3.14 resume/evaluation 拒绝**。重新训练使用 v3.14 环境快照、版本与 hash。

## 验证与有界运行

在 PowerShell 的 `uav` Conda 环境执行；checkpoint/evaluation/smoke 审计要求 CUDA，不回退 CPU：

```powershell
Import-Module 'C:\Users\HPK\anaconda3\shell\condabin\Conda.psm1'
conda activate uav
python -m pytest tests/test_combat_v314.py -q
python tools/audit_combat_v314.py
python tools/validate_combat_v314.py
python tools/smoke_formal_8v8.py --env-config configs/combat_environment_v314.yaml --output outputs/v314_validation/cuda_smoke
python -m pytest -q -ra
python -m compileall -q env algorithm tools tests
```

smoke 为每算法 512 transitions、实际 parameter update、save/reload 和两局 deterministic evaluation；保留正式网络及 PPO 配置。既有 smoke 输出不覆盖，复跑需另指定输出目录。停止于验证，不自动开始 pilot、3M 或搜索。

完整验收报告与响应 CSV/图位于 `outputs/v314_validation`。
