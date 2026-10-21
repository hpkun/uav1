# navigation3d-v1：独立三维协同导航

## 范围与入口

本环境研究三维航路点导航、编队与飞行安全。环境身份为 `navigation3d-v1`，观测 schema 为 `local_navigation3d_20_plus_10_per_peer_v1`。没有武器、敌机、命中、锁定或交战任务接口。

配置入口：`configs/navigation3d_v1.yaml`。环境入口：`env.navigation3d.Navigation3DEnv`。算法入口：`algorithm/train_navigation3d.py` 与 `algorithm/evaluate_navigation3d.py`。

现有 combat factory、历史配置、算法优化器实现和模型文件保持不变。新适配器复用 `MAPPOTrainer.update`、`IPPOTrainer.update/_ppo_update`、`RolloutBatch` 和原有 `compute_gae`。导航不通过历史 combat 训练入口注册，避免放宽历史环境身份限制。

## 几何与运动学

世界坐标采用 NED：x 向北、y 向东、z 向下；高度 h = -z。俯仰 theta 向上为正；航向 psi 从北向东增加；模型没有滚转和独立侧滑。

运动方向：

\[
f=(\cos\theta\cos\psi,\cos\theta\sin\psi,-\sin\theta).
\]

对导航航路点 q，r=q-p，距离 d=||r||，水平距离为 hypot(rx,ry)，高度差为 -rz。

\[
\alpha=\operatorname{atan2}(\|f\times(r/d)\|,\operatorname{clip}(f\cdot(r/d),-1,1)).
\]

机体导航坐标轴采用前、右、上：right=(-sin psi, cos psi, 0)，up=right×f。这里是零滚转导航投影基，不能把“上”当成 NED 的向下轴。

令 (l_f,l_r,l_u)=(f,right,up) r/d，则机体方位误差为 atan2(l_r,l_f)，俯仰误差为 atan2(l_u,hypot(l_f,l_r))。满足 cos alpha = cos(body_elevation) cos(body_azimuth)。航向误差 wrap(atan2(ry,rx)-psi)。

使用 `hypot` 避免平方范数的过早上溢/下溢；空间夹角用 atan2 形式稳定处理接近零与 pi 的情况。完全重合时没有唯一视线方向，约定距离和方向误差均为零；超出浮点表示范围的位移明确报错。测试以独立三维向量点积的 acos 结果作为参考。

动作 `[heading_increment,pitch_increment,speed_increment]` 均为 [-1,1]。调用原有 `env.v311_kinematics.advance_aircraft`：每一步先更新姿态与速度，再按 `v*f*dt` 积分位置。增量是每一步的增量，不额外乘 dt。

| 参数 | 配置值 |
|---|---:|
| dt / 最大步数 | 0.2 s / 400 |
| 速度范围 / 初始速度 | 80～250 / 180 m/s |
| 单步最大航向 / 俯仰增量 | 5° / 3° |
| 单步最大速度增量 | 5 m/s |
| 俯仰范围 | ±60° |
| 水平安全区域 | x,y 各 ±4000 m |
| 高度安全区域 | 100～2500 m，边界失效 |
| 航路点半径 | 60 m |
| 队友可见距离 | 1500 m |

四机初始中心为 (-1200,0,-800)，各机获得固定方形编队偏移 (±75,±75,0)，初始化加入独立 seed 决定的 ±5 m 扰动。两个航路点中心为 (0,0,-1000)、(1100,400,-900)，每机目标等于中心加自身偏移。全体被分配的有效飞机同时进入当前航路点半径后切换下一阶段，最后一阶段完成即成功。任一有效飞机发生碰撞或越界则团队回合失败；达到时限则 truncated。

安全距离使用每个线性积分区间内的相对轨迹最近点检测，可检测端点之间的穿越碰撞。20 m 内判为碰撞，60 m 内产生危险接近惩罚。

## 势函数与奖励

三项势函数始终参考固定最终目的地，避免航路点切换时改变奖励基线；当前航路点与最终目的地都出现在观测中。中间航路点用于有序任务进度，不单独发放额外奖励。

\[
\Phi_d=0.1(1-\operatorname{clip}(d_{final}/6000,0,1)),
\]

\[
\Phi_a=0.05(1-\alpha_{final}/\pi),
\]

\[
e_i=\operatorname{mean}_{j\ne i,valid}\|(p_j-p_i)-(offset_j-offset_i)\|,
\quad \Phi_f=0.05(1-\operatorname{clip}(e_i/200,0,1)).
\]

没有有效队友时 e_i=0。单机势函数总和处于 [0,0.2]。三个塑形分量各自只计算一次：

\[
F_k=0.99\Phi_k(s_{next})-\Phi_k(s).
\]

总奖励为：

\[
R=F_d+F_a+F_f
-0.02\,I_{danger}
-10\,I_{own\ failure}
-0.001\,I_{valid\ before}
+10\,I_{team\ success}I_{valid\ before}.
\]

全部系数都记录在环境 YAML 中。gamma 与算法配置的 gamma 必须严格相等。完成、失败和时限都属于本任务的有限时域终止，后继势函数统一为 0，时间剩余量出现在观测中。训练 GAE 沿用现有对 terminated/truncated 的截止约定，不跨自动 reset 接续轨迹。

终止回合的折扣塑形和满足：

\[
\sum_{t=0}^{T-1}\gamma^tF_t=-\Phi(s_0).
\]

不折扣的完整回合塑形和满足：

\[
\sum F_t=-\Phi(s_0)-(1-\gamma)\sum_{t=1}^{T-1}\Phi(s_t)\le0.
\]

因此静止保持同一非零势函数会产生非正塑形奖励；闭合循环不能获得正净塑形奖励。单步可以有正奖励，正贡献总量需要与负贡献一起读取，不能只挑选正项。每回合记录各分量的总和、正项和负项；`step` 的返回奖励与分量和有实际验证。

最终完成奖励为每架有效飞机 +10，团队合计 +40；单机势函数幅度最多 0.2，不会以一个势函数边界跳变代替任务完成。理论约束和固定控制器检查不等于已穷举所有学习策略。

## 50 维局部观测

所有槽位稳定排序。相对位置和给定编队偏移按 6000 m 归一化，相对速度按 2*v_max 归一化；最终数值裁剪到 [-1,1]。

| 零起始索引 | 含义 |
|---|---|
| 0:3 | 自身 NED 位置，按水平区域/高度上界归一化 |
| 3 | 自身速度 / v_max |
| 4:6 | sin(heading), cos(heading) |
| 6:8 | sin(pitch), cos(pitch) |
| 8:11 | 当前指定航路点相对位置 |
| 11:14 | 最终指定航路点相对位置 |
| 14 | 当前航路点三维方向夹角 / pi |
| 15 | 当前航路点机体方位误差 / pi |
| 16 | 当前航路点机体俯仰误差 / pi |
| 17 | 当前航路点索引 / max(1,航路点数量-1) |
| 18 | 自身有效标记 |
| 19 | 剩余时间比例 |
| 20:30、30:40、40:50 | 三名队友槽位，每槽 10 维 |

队友槽内顺序：相对位置 3、相对速度 3、公开指定的相对编队偏移 3、可见且有效标记 1。队友按固定全局编号排序并略过自身；不可见、未分配或无效的槽位全零。自身无效时整行全零。维度公式为 20+10*(N-1)，默认 N=4，实际输入 50 维；动作维度为 3。

物理 alive、任务 assigned 和训练 valid 分开：未分配槽不参与运动、碰撞或奖励；回合结束时 valid 全零，物理位置仍保存在轨迹中。并行采样返回明确的终止观测/掩码，再返回新回合的 reset 观测/掩码，二者不混用。

Actor 只使用自身局部观测；没有真实全局质心或不可见队友状态特征。MAPPO Critic 使用各机观测与掩码拼接的集中式输入；IPPO Critic 只读取本机 50 维观测。

## 算法与 checkpoint

两个导航算法配置均从现有正式配置复制，只改变环境身份和 observation_dim。training、implementation 字段及训练超参数保持逐项一致。网络均为正式 [256,256]；短测通过运行时预算限制减少采样，不减网络宽度、PPO epochs 或 minibatch 设置。

新 adapter 使用原始 MAPPO/IPPO 更新函数，保留 GAE、优势归一化、tanh Gaussian/latent action、PPO Clip、Value Clip、KL early stop、Adam、梯度裁剪与参数共享。IPPO 的新增局部 Critic 与原局部网络为同一两层 MLP 结构，仅输入维度不同。构建时使用通用 MAPPO 初始化入口后安装局部 Critic；实际 IPPO 更新只使用局部 Critic。

并行采样沿用长驻 spawn 子进程、同步 Pipe batch 和 seed 调度。worker 只构建导航环境。`RolloutBatch` 对接现有优化器，显式校验采样动作与执行动作一致。

checkpoint 使用 `checkpoint_family=navigation3d-v1`、`algorithm=navigation3d_mappo/navigation3d_ippo`，同时验证环境/算法配置快照与 SHA256、观测 schema、网络结构、优化器源实现版本、参数量、Gaussian 协议和 tensor 形状/有限性。历史 checkpoint 被拒绝。保存优化器和计数器，支持验证性重载；没有承诺 bit-exact 恢复并行环境、随机流或训练中间状态。

## 短测命令

在 WSL Ubuntu 中先 `source /home/hpk/anaconda3/etc/profile.d/conda.sh`、`conda activate uav`，进入项目目录后执行。训练、评估和 checkpoint 审计必须有 CUDA，没有 CPU 回退。

```bash
python -m pytest -q tests/test_navigation3d*.py
python tools/validate_navigation3d.py --output-dir outputs/navigation3d_precheck_new --episodes 20
python algorithm/train_navigation3d.py --algorithm-config configs/mappo_navigation3d_v1.yaml --output-dir outputs/navigation3d_mappo_new --sampled-steps 1024 --num-envs 2 --seed 1
python algorithm/train_navigation3d.py --algorithm-config configs/ippo_navigation3d_v1.yaml --output-dir outputs/navigation3d_ippo_new --sampled-steps 1024 --num-envs 2 --seed 1
python algorithm/evaluate_navigation3d.py --checkpoint outputs/navigation3d_ippo_new/final.pt --episodes 5 --output outputs/navigation3d_ippo_new/evaluation.json
python tools/render_navigation3d.py --trace outputs/navigation3d_precheck_new/waypoint/trace.npz --output-dir outputs/navigation3d_render_new
```

训练 CLI 专用于短规模技术验证，预算上限 8192；默认 1024。本次没有启动正式长时间训练。详细实际结果见 `docs/navigation3d_acceptance.md` 和 `outputs/navigation3d_validation`。

## 限制

- 点质点、零滚转、机头与运动方向重合假设；不是完整六自由度飞行器动力学。
- 最终目的地势函数保障航路点切换没有奖励基线跳变，但不保证它对任意折返路线都给出最优的中间导航引导。
- 本版要求所有有效飞机同时进入当前航路点半径，未实现不同机型、异步任务分配或停车等待。
- 局部可见性采用距离门限，没有噪声、遮挡或真实机载传感器验证。
- 碰撞使用每个积分区间的线性相对运动近似；碰撞时冻结在该步积分后位置，未回溯到连续首次接触时刻。
- 短测验证了更新和保存重载，尚未证明 RL 已经学会可靠导航。
