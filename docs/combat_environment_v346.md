# v3.46：Li（2023）公式（25）R4 对照

实施日期：2026-10-22。实际命令均在 **PowerShell 的 uav Conda 环境**执行，PyTorch 2.11.0+cu128，CUDA GPU 为 NVIDIA GeForce RTX 5060 Laptop GPU。没有启动 500K、3M 或 7M 正式训练。验证产物位于 `outputs/v346_implementation/`，历史输出未删除或改写。

## 文献核对与实现范围

直接读取仓库中的 Li et al. (2023), *Multi-UAV Cooperative Air Combat Decision-Making Based on Multi-Agent Double-Soft Actor-Critic*, Aerospace 10(7),574，DOI [10.3390/aerospace10070574](https://doi.org/10.3390/aerospace10070574)。检查 PDF 第14页 Section 3.3.4 / Eq.(25) 的全文与渲染图，并检查第5–6页 Section 2.3 / Figure 2 的角度定义。核对图保存在验证目录 `paper_page_14.png`、`paper_page_6.png`。

本次仅采用论文的 R4；R3 仍为既有 PBRS，不声称复现论文整个奖励、环境、武器系统或 MADSAC 算法。

| 论文 R4 内容 | 实现对应 |
|---|---|
| Red 攻击几何 | `engagement_geometry(red, blue)` 的 AA、ATA、HA、三维距离 |
| Blue 威胁几何 | 交换角色为 `engagement_geometry(blue, red)`；配对矩阵仍按 Red/Blue 索引 |
| 分支前提 | `paper_branch`：绝对 AA ≤ π/6，distance ≤ 4000 |
| 最严格档优先 | 按 5°、15°、30° 顺序，ATA 与 HA 同时满足才返回；不累加 |
| R41 | 5°=+0.1；15°=+0.02；30°=+0.01 |
| R42 | 5°=−0.15；15°=−0.025；30°=−0.015 |
| 距离、角度权重 | 没有连续距离衰减、tail、offense_base、defense_scale 或 alpha 代替 ATA/HA |

AA 是目标航向相对于水平视线的姿态角；ATA 是攻击者航向相对于水平视线的攻击角；HA 是视线相对于水平面的高度角，不减攻击者俯仰。项目使用带符号角，式（25）取绝对值。`env/geometry.py` 未修改。

## 论文与工程补充的边界

论文明确给出上述门限和六个数值，但没有完整给出多个目标、进攻与防御重叠、无匹配条件时的单值归约规则。本实现明确采用用户规定的工程补充：对每架 Red 计算所有存活配对，取最大进攻奖励及最负威胁惩罚；有正进攻时仅应用最大进攻，无正进攻时应用最负威胁，两者均无则为零。不能把此优先级归于论文明确规定。

默认零、死亡实体过滤、空目标集合、最严格档优先、角度边界的既有 1e-12 rad 浮点比较容差，属于数值/工程解释。距离使用几何结果的字面 `<=4000`，不加距离权重或容差。恰好4000的轴向状态通过，4000+1e-8被拒绝；斜向三角构造可能产生约4.5e-13 m的舍入误差，测试分别隔离角度和字面距离边界，未放宽距离条件。

`tactical_offense` / `tactical_defense` 日志记录**实际应用贡献**，因此进攻存在时 defense 为零。候选分支与已应用分支的触发率在固定策略审计中另列，不将候选惩罚再次加到奖励。

## 相对 v3.44 的精确差异

`CombatEnvironmentV346` 继承 `CombatEnvironmentV344`，只覆盖 `dense_combat_reward()`，调用独立 `env/v346_reward.py`。另外新增环境身份、配置校验与正式配置，以及必要的训练/评估身份白名单。

环境 YAML 的解析差异只有：environment_version 从3.44改为3.46；research_reward 删除旧连续R4字段，仅保留原有5项PBRS参数。reward 中已有的论文六档及AA/距离门限成为新R4的唯一有效参数来源。历史保留的 guide_reward/guide_distance_min 对新R3/R4不生效；R3仅读取research_reward中的PBRS参数，不增加论文的guide项。

新校验器严格拒绝把连续R4系数放回research_reward，严格冻结论文档位及4000m/π/6，验证副本才重建历史schema。运行时不注入旧系数，也不放宽v3.44或历史校验。gamma仍为0.99且必须与算法一致。

以下全部沿用：4v4、真实三维30°攻击锥、2000m实际失效范围、0.5秒/5次采样、dt=0.1、max_steps=1000、49D/3D、Blue、场景、动力学、观测、终止规则、目标选择、同步失效、共享归属。R1击毁+10/损失−10；R2水平越界−10；R3为 `0.99*Phi(next)-Phi(current)`，终止势函数归零。不增加R5/R6或终局奖励。

R4仍在运动后、同步失效前计算：已死亡实体不产生新奖励，当前transition交战失效可以保留该步失效前R4，与基线时序相同。没有复制或修改step核心、DwellEnvelope、DwellState或事件结算。

MAPPO/IPPO/MAAC各自v3.46 YAML与其v3.44正式配置仅环境身份不同，全部网络、学习率、Rollout/GAE/PPO、共享策略、本地IPPO Critic、MAAC Attention/Replay/反事实基线参数不变。

## 文件交付与退役

新增代码：`env/v346_reward.py`、`env/combat_v346.py`、`env/v346_config.py`。

新增配置：`configs/combat_environment_v346.yaml`、`configs/mappo_4v4_v346.yaml`、`configs/ippo_4v4_v346.yaml`、`configs/maac_4v4_v346.yaml`。

新增/重组测试：`tests/test_combat_v344.py`（保留基线并对共同机制补入v3.46）、`tests/test_combat_v346.py`、`tests/test_maac_v344_v346.py`。

新增/重组工具：`tools/preflight_v344_v346.py`、`tools/verify_v346_readiness.py`、`tools/verify_maac_v344_v346.py`、`tools/audit_v346_history.py`。

新增/重组文档：本文、`docs/combat_environment_v344.md`、`docs/maac_v344_v346.md`；更新README。

删除原v3.45可执行文件：`env/combat_v345.py`、`env/v345_config.py`，以及combat_environment/mappo_4v4/ippo_4v4/maac_4v4四份v345 YAML。

重组后删除旧组合文件：`tests/test_combat_v344_v345.py`、`tests/test_maac_v344_v345.py`；`tools/preflight_v344_v345.py`、`tools/verify_v344_v345_readiness.py`、`tools/verify_maac_v344_v345.py`；`docs/combat_environment_v344_v345.md`、`docs/maac_v344_v345.md`。v3.44覆盖保留，旧45°可执行参数化不保留。

调整共享入口：`env/combat_env.py`、`env/config.py`、`env/v343_config.py`；`algorithm/common/checkpoint.py`、`algorithm/common/evaluator.py`；`algorithm/mappo/runner.py`、`algorithm/mappo/evaluation.py`；`algorithm/ippo/runner.py`、`algorithm/ippo/protocol.py`、`algorithm/ippo/diagnostics.py`；`algorithm/maac/protocol.py`、`algorithm/maac/metrics.py`。改动限版本路由、身份/指纹/维度/gamma与统计分类。

录像/回放原本按事件语义和通用工厂分发，无需加算法或环境实现。运行代码、配置中的3.45支持分支已移除。剩余v345字样仅用于历史产物路径、明确拒绝测试或退役说明；未删除其他历史版本独立的45°参数，也未删除既有模型和录像。

## 测试与隔离证据

- 专项：`new_tests.log`，**170 passed**。包含3°/4°、10°/12°、25°/20°；正负号、5°/15°/30°闭边界、超限、AA门限、4000与极小超距、低于2000、零距离、死亡/空目标、角色交换、多目标最大/最负、进攻优先、槽位重排、同步失效、共享归属、无连续衰减、无tail、ATA/HA独立于alpha、严格配置和算法参数冻结。
- 相关历史/入口/回放：`targeted_tests.log`，**436 passed, 1 skipped, 2 failed**。两个失败是旧MADSAC/MAAC错误信息范围正则已过时，在原HEAD副本复现，证据`baseline_known_failures.log`。
- 扩展回归：`regression.log`，**638 passed, 2 failed**。两个失败是v3.32旧MADSAC夹具期望environment_version错误，但当前原实现先拒绝旧implementation；在原HEAD复现，证据`baseline_legacy_failures.log`。本次没有为迁就旧断言改变算法协议。
- `compileall`与`git diff --check`通过；初轮三个新增测试构造问题已修正，最终专项全部通过。
- `protected_sources.json`：**384份**未改环境/配置与修改前快照逐字节相同；**523份**既有环境、配置及算法文件与HEAD在规范化行尾后内容相同。包括v3.44、Dwell与v343_reward，以及各算法核心。不同于Windows git archive的CRLF规范化，384份比较使用实际修改前工作区字节。
- `history_verified.json`：11个旧版本（3.1/3.2/3.7/3.9/3.30/3.32/3.36/3.41/3.42/3.43/3.44）×2种子×Zero/oscillating，共**44个完整回合**，状态、观测、奖励、RNG及适用的驻留数组哈希与修改前快照完全相同。

## 完整回合账务与固定策略

`preflight_verified/`包含每版本每策略20局，共160局，80对完整回合；固定种子0–19，动作只计算为共同的外部序列，同时断言两环境按状态计算的策略动作完全相同。初始状态、全部运动/观测、双方事件、驻留、目标选择、终止、RNG与逐步R1/R2/R3全部精确相同。每回合同时核对四项累计、总奖励与PBRS望远镜和。

逐步 `reward=R1+R2+R3+R4` 误差 **0**；总Reward差与R4差的最大浮点残差 **8.87e-16**；完整回合累计账务误差 **3.11e-15**。

下表胜负/损失/长度对v3.44和v3.46相同，R4为每回合四个Red累计的均值；每行每版本20局。

| 策略 | Red胜/Blue胜/平局 | 全灭平局 | Red/Blue损失 | 步数 | v3.44 R4 | v3.46 R4 | R4非零比例（旧/新） |
|---|---|---|---|---|---|---|---|
| Zero Action | 0%/0%/100% | 100% | 4/4 | 144 | −0.636366 | 0 | 41.4815%/0% |
| Nearest Pursuit | 0%/0%/100% | 100% | 4/4 | 144 | −0.632141 | 0 | 41.4815%/0% |
| Lateral | 0%/100%/0% | 0% | 4/1 | 143 | −0.744618 | 0 | 41.5896%/0% |
| Vertical | 0%/100%/0% | 0% | 4/3 | 147 | −0.916542 | 0 | 41.9118%/0% |

| 策略 | R1 | R2 | R3（两版一致） | Red驻留开始/完成/唯一目标 | Blue驻留开始/完成/唯一目标 |
|---|---|---|---|---|---|
| Zero | 0 | 0 | −0.58525546 | 5/4/4 | 5/4/4 |
| Pursuit | 0 | 0 | −0.58787172 | 5/4/4 | 5/4/4 |
| Lateral | −30 | 0 | −0.58484636 | 2/1/1 | 7/5/4 |
| Vertical | −10 | 0 | −0.57485322 | 7/4/3 | 7/4/4 |

四种策略的新R4进攻/防御5°、15°、30°候选及应用触发比例全部为 **0%**；所有初始Red的论文R4也为零。态势按交战前所有存活配对的绝对AA统计：迎头≥150°、尾后≤30°、其余侧向；四策略均为迎头100%、侧向0%、尾后0%。触发率分母为交战前存活Red agent步，详细原始行与分母保存在episodes.csv/comparison.csv/summary.json。

## CUDA与Checkpoint

`cuda_verified/summary.json`记录三个原正式参数有界验收，使用CUDA且无CPU回退：

| 算法 | 样本 | 正式网络 | 更新 |
|---|---|---|---|
| MAPPO | 4096，16×256 | Actor/Critic 256×256，49/3/4 | 1次完整PPO更新，Actor/Critic各80优化步 |
| IPPO | 4096，16×256 | 原共享Actor/本地Critic 256×256，49/3/4 | 1次完整PPO更新，Actor/Critic各80优化步 |
| MAAC | 3768，12个完整回合 | hidden128、4heads、batch1024、MC16 | 原4次Critic+4次Actor+1次目标更新 |

有效梯度、Actor/Critic实际参数变化、参数/优化器有限、Checkpoint保存/重载一致、确定性评估重复、Reward统计、驻留事件账务及所有Worker正常关闭均通过。MAPPO/IPPO保持Rollout256、10个PPO epoch、batch512；MAAC保持Replay1,000,000及全部正式参数。有限样本预算与少量诊断评估种子不是超参数调优。

历史实际Checkpoint评估：MAPPO v3.43、IPPO v3.43、MAPPO v3.44、IPPO v3.44、MAAC v3.44，各在CUDA完成2局；原文件SHA256前后一致，均被拒绝用于v3.46。新v3.46 Checkpoint拒绝用于v3.44。实际旧v3.45 MAPPO Checkpoint拒绝用于3.44/3.46；3.45配置不再被工厂接受。证据为`cuda_verified/historical_checkpoints.json`。

MAAC/IPPO既有精确续训能力边界不变，不能把评估重载成功解释为支持保存完整Replay/环境RNG后的精确续训。

三个算法均通过正式`tools/replay_combat.py`入口生成`replay_verified_mappo/`、`replay_verified_ippo/`、`replay_verified_maac/`，每份包括MP4、PNG、交互HTML、逐步状态CSV、驻留事件CSV、NPZ轨迹和Markdown账务报告。保存的轨迹分项账务误差均为0，驻留完成/唯一失效与元数据一致，无伪造fire_attempt。三份MP4已用FFmpeg完整解码验证，分别147/146/150帧、14.7/14.6/15.0秒，包含初始状态帧，因此比14.6/14.5/14.9秒的仿真时长多1帧，符合既有回放协议。结果在`replay_validation.json`。

现有外部FFmpeg缺少rawvideo，首轮编码失败；为完成验证，仅下载独立imageio-ffmpeg wheel并解压二进制到`tmp/v346_media/`，没有安装或改变Conda包。最终三份视频和全部其他格式均成功。回放图已目视检查。最后MAAC回归另有**23 passed, 1 skipped**；旧MP4单测因要求PATH中同时有ffmpeg/ffprobe而跳过，实际三个版本视频由上述完整解码/帧率/时长检查独立验证，没有伪造ffprobe或降低单测门限。

## 风险与正式训练条件

工程上已具备v3.46采样、更新、评估、身份隔离、记录与回放的接入条件。但本次四种固定策略下论文R4全程为零，AA门限在迎头阶段不能提供R4信号；实际三维驻留允许迎头同时失效，两者几何本来就不同。Zero/Pursuit为100%双方全灭平局。PBRS R3仍有信号，但不能据此保证策略会探索到R4有效区域。

不同初始种子主要旋转当前对置编队，固定策略结果相同，20局数量不能代替更广态势覆盖或学习收敛证据。因此不建议无条件进入大规模正式训练；这不是代码故障，也未靠增加奖励、改变优先级、扩大角度或调整Blue来掩盖。后续若启动严格论文R4对照，应接受此稀疏性作为实验结果，保留v3.44匹配对照和各档触发统计。本次没有启动正式训练。

## 复现

验证在PowerShell激活uav后执行：

```powershell
conda activate uav
python -m pytest tests/test_combat_v346.py tests/test_combat_v344.py tests/test_maac_v344_v346.py -q
python tools/preflight_v344_v346.py --output outputs/v346_preflight_new --seeds 20 --workers 2
python tools/verify_v346_readiness.py --output outputs/v346_cuda_new
```

以下是用户要求的简洁Ubuntu正式训练命令，**仅供后续使用，本次未执行**；使用新输出目录避免覆盖历史记录：

```bash
conda activate uav
cd /mnt/c/Users/HPK/Desktop/uav1
python algorithm/train_mappo.py --env-config configs/combat_environment_v346.yaml --algorithm-config configs/mappo_4v4_v346.yaml --device cuda --seed 1 --output-dir outputs/mappo_v346_seed1
python algorithm/train_ippo.py --env-config configs/combat_environment_v346.yaml --algorithm-config configs/ippo_4v4_v346.yaml --device cuda --seed 1 --output-dir outputs/ippo_v346_seed1
python algorithm/train_maac.py --env-config configs/combat_environment_v346.yaml --algorithm-config configs/maac_4v4_v346.yaml --device cuda --seed 1 --output-dir outputs/maac_v346_seed1
```
