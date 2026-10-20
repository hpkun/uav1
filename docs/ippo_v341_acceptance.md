# PS-IPPO v3.41 实施与验收报告

日期：2026-10-20。所有 Python、训练、评估、审计和测试使用 PowerShell 中的 Windows `uav` Conda 环境；解释器 `D:/conda_envs/envs_dirs/uav/python.exe`，PyTorch 2.11.0+cu128。未启动300k、3M或其他正式长训练。

## 文件交付

新增：
- `algorithm/ippo/__init__.py`
- `algorithm/ippo/networks.py`
- `algorithm/ippo/trainer.py`
- `algorithm/ippo/factory.py`
- `algorithm/ippo/runner.py`
- `algorithm/ippo/protocol.py`
- `algorithm/ippo/evaluation.py`
- `algorithm/ippo/diagnostics.py`
- `algorithm/train_ippo.py`
- `algorithm/evaluate_ippo.py`
- `algorithm/diagnose_ippo.py`
- `configs/ippo_4v4_v341.yaml`
- `tests/test_ippo.py`
- `docs/ippo_v341.md`
- `docs/ippo_v341_acceptance.md`

修改：`tools/record_combat_episode.py`，仅新增IPPO加载分支、IPPO录像seed范围验证及IPPO完整飞行诊断输出。其他算法分支保留原行为。

保护范围：现有MAPPO源码和正式配置、环境动力学/观测/Blue/奖励/武器源码、MADDPG/MATD3/MADSAC/MAAC源码未修改。`outputs/ippo_validation/unchanged_source_audit.json` 对188个受保护文件与HEAD做LF标准化比对，188/188一致；原始SHA同时保留，Windows换行差异不当作行为变化。`git diff --check`通过。历史配置、checkpoint和实验结果未覆盖。新实验使用独立新目录。

## 网络结构、参数和共享证据

MAPPO v3.41正式Critic为集中式MLP，输入225维=(4+1)*45。IPPO为独立 `LocalValueCritic`：45→256→256→1，两个ReLU，没有联合状态、agent index或其他隐式特征。数学实现位于 `algorithm/ippo/networks.py:LocalValueCritic.forward`，即对输入最后一维直接共享MLP，输出 `V_phi(o_i)`。

Actor直接使用现有 `SharedMAPPOActor`，45→256→256→3 Tanh-Gaussian，state_independent log_std初始化-0.5、裁剪[-5,0.5]、mean head gain0.01。

实际PyTorch计数：Actor78,342，Critic77,825，总计156,167。Trainer中各只有一个actor和critic对象、一个Actor Adam和一个Critic Adam；没有按agent复制的ModuleList。重复相同局部观测时4架输出完全一致；只修改其他独立输入观测时目标价值逐位不变；225维输入被拒绝；完整[T,E,4,45]输入正确返回[T,E,4]。

## 数学验证和公平协议

专项测试验证：GAE手工递推、当步死亡奖励保留、next_alive=0不bootstrap、done阻断跨局、全死安全跳过、死亡样本不影响归一化和参数更新、PPO ratio/clip、Tanh log Jacobian双精度数值、极端饱和logprob有限、value clipping、Actor/Critic梯度隔离、确定性动作不消耗CUDA RNG、完整epoch KL early stop。

PPO内核与原版MAPPO做严格对照：相同局部网络权重、rollout、minibatch排列和CUDA RNG，更新后所有Actor/Critic参数逐位一致，所有公共指标完全相同。原版MAPPO算法文件未改动。

每轮采样256×16=4096条联合transition，展平[4096,4,...]；512条联合transition/minibatch。没有以agent-step重新划分minibatch。无early stop时每epoch8次、10epoch80次共享Actor/Critic更新。3M理论上限经测试为732个完整rollout+108vector-step尾部，共733次rollout、58,600次Actor和Critic optimizer step。运行日志记录真实计数而不是理论计数。

## 死亡和reset边界

采样使用现有ParallelVectorEnv。当前alive mask用于loss；逐智能体原始奖励保留；next_alive控制bootstrap。`terminated | truncated`合并为done，两者都不bootstrap。next_observations保存真实 `transition_next_observations`，reset后观测仅用于下一次动作。测试实际跑到环境自动reset，确认终止transition与下一局开局观测不同。存储的动作与每步 `executed_red_actions`逐位核对；不一致直接报错，避免latent action与旧概率失配。

## Checkpoint、评估和种子

IPPO checkpoint独立身份algorithm='ippo'、ippo_impl_version=1、parameter_sharing=true、critic_scope=local_observation。包含完整Actor/Critic及优化器、架构、参数数目、算法完整配置、环境/算法SHA256、Gaussian/PPO/GAE/价值裁剪参数、实际sampled/vector/rollout/optimizer计数、训练seed、development seed来源、评估历史和best选择信息。

严格评估拒绝环境/算法指纹变化、架构变化、信息范围变化、不完整元数据和不兼容算法身份。测试使用真实创建的IPPO、MAPPO和MATD3 checkpoint，验证双向拒绝。checkpoint重载后确定性动作逐位一致；相同评估种子的所有战斗指标完全一致。

训练seed<10M；development在[10M,20M)，默认10M起20局；recording在[20M,30M)；holdout>=70M且显式split=holdout。训练只用development选择best，使用公共evaluation_selection_key。holdout仅在训练完成后做入口验证，不用于调参或checkpoint选择。

恢复声明为evaluation_reload_only、bit_exact_training_resume=false。没有提供精确训练恢复入口，也不声称保存了全部环境、RNG和采样状态。

## 真实CUDA证据

GPU：NVIDIA GeForce RTX 5060 Laptop GPU。真实环境16个spawn worker，rollout形状[256, 16, 4, 45]，正式256×256网络，采样4096环境transition、256vector-step，训练seed321。

| 指标 | 实测 |
|---|---:|
| Actor参数变化global L2 | 1.726393700 |
| Critic参数变化global L2 | 1.995722651 |
| Actor optimizer steps | 80 |
| Critic optimizer steps | 80 |
| effective PPO epochs | 10 |
| valid alive agent-step | 14104 |
| actor loss | -0.003262175 |
| value loss | 1.720398283 |
| approximate KL | 0.003536829 |
| actor preclip grad norm，minibatch平均 | 0.118622049 |
| critic preclip grad norm，minibatch平均 | 0.084144184 |

完整指标见 `outputs/ippo_validation/cuda_test_evidence.json`。另通过训练CLI跑seed0的独立4096步smoke，同样80次Actor和80次Critic更新，末尾使用20个development seed评估，产物位于 `outputs/ippo_cuda_smoke/`。不以smoke胜率验收。

## Canonical recorder与诊断

使用原有官方 `tools/replay_combat.py`，直接走IPPO严格checkpoint loader和原SharedMAPPOActor。测试逐步核对Actor动作、info实际执行动作、Red/Blue每个物理状态；录制环境与普通环境结束后RNG state完全相同。两次独立官方录像（同checkpoint、seed20000071）全部19个trace字段逐位相同，见 `replay_trace_equality.json`。

官方MP4、PNG、HTML、CSV、报告已生成：`outputs/ippo_validation/official_replay_mp4/`。该场景148步，14.8秒。首次进入武器射程和首次开火均12.0秒，首次击毁12.4秒，首次开火ATA=0.00106795rad、HA=-0.11369690rad；仅为只读诊断，没有据此改训练。

原环境未配置FFmpeg。首次MP4命令明确失败；之后将imageio-ffmpeg 0.6.0安装到独立验证目录 `outputs/ippo_validation/video_deps`，指定其中FFmpeg v7.1后官方MP4导出成功；没有修改uav环境依赖文件或仓库视频代码。

优化、动作、飞行和固定观测字段及单位详见 `docs/ippo_v341.md`。实际输出包括 `optimization_metrics.jsonl`、`action_diagnostics.jsonl`、`ippo_flight_diagnostics.json`、`fixed_policy_verified.json`。固定观测来自首次训练rollout并保留seed/环境SHA来源；没有使用holdout观测。支持多checkpoint确定性动作均值/方差、差异、持续饱和，以及同seed轨迹变化。单轮smoke的best/final可能相同，不据动作不变认定收敛。

## 测试命令和结果

以下均先执行：

```powershell
& 'C:\Users\HPK\anaconda3\shell\condabin\conda-hook.ps1'
conda activate uav
```

| 命令 | 真实结果/日志 |
|---|---|
| `python -m pytest tests/test_ippo.py -q --tb=short` | 最终专项22 passed（28.07秒）；`outputs/ippo_validation/ippo_final_22_tests.log` |
| `python -m pytest -q --tb=short`（实施起始运行） | 4485 passed，44 failed，4 skipped；`baseline_tests.log`。此运行启动早于编辑；为排除运行期间目录变化，全部失败另在独立未修改HEAD快照复现 |
| `python -m pytest -q --tb=short`（最终全仓） | 4504 passed，44 failed，4 skipped（926.88秒）；与起始失败清单完全相同，0新增失败；`outputs/ippo_validation/final_all_tests.log` |
| MAPPO/公共环境专项回归，下方完整命令 | 732 passed，9 failed，1 skipped；`regression_tests.log` |
| 未修改HEAD快照复现9个专项历史失败 | 22 passed，9 failed；`historical_failures_clean.log` |
| 未修改HEAD快照复现全仓失败nodeids | 44 failed，全部在未修改HEAD中复现（86.78秒）；`historical_all_failures_clean.log` |
| `python -m compileall -q algorithm/ippo algorithm/train_ippo.py algorithm/evaluate_ippo.py algorithm/diagnose_ippo.py tools/record_combat_episode.py` | 通过 |
| `git diff --check` | 通过 |
| `python outputs/ippo_validation/audit_unchanged.py` | 188/188受保护文件与HEAD一致（标准化换行） |

最终全仓运行收集了4552项（当时IPPO专项19项）。其后补充显式0环境、0预算和不能整除预算三个入口边界测试，最终IPPO专项22项全部通过。`pytest --collect-only -q`确认当前4555项；全仓回归与补充边界测试的合并覆盖为4507通过、44历史失败、4跳过。该合并数不是一次pytest运行输出。对比证据为 `outputs/ippo_validation/regression_comparison.json`，新增失败/未复现历史失败均为空。

专项回归完整命令：

```powershell
python -m pytest tests/test_mappo.py tests/test_mappo_lambda.py tests/test_mappo_environment_identity.py tests/test_gaussian_protocol.py tests/test_critic_protocol.py tests/test_training_protocol.py tests/test_compact_training_metrics.py tests/test_combat_v341.py tests/test_combat_protocol.py tests/test_combat_environment.py tests/test_combat_replay_pipeline.py tests/test_combat_visualization.py tests/test_v341_training_configs.py -q --tb=short
```

历史失败复现使用 `git archive --format=zip --output=outputs/ippo_validation/unchanged_baseline.zip HEAD` 提取独立快照；`python outputs/ippo_validation/reproduce_historical.py` 从起始全仓日志提取44个真实nodeid，在快照cwd执行同一uav解释器的pytest。未修改、删除或跳过任何历史测试断言；失败清单为 `historical_failed_nodeids.json`。专项9项为两项诊断报错文本旧范围、六项bare runner缺env_config、一个历史固定配置文件清单。

开发过程的初次专项测试有缺导入和测试标量/冻结配置处理错误，已修复并重跑；原始 `ippo_tests.log`、`ippo_tests_2.log` 保留，不计为通过。

实际训练、评估、录像和诊断命令：

```powershell
python -u algorithm/train_ippo.py --smoke --output-dir outputs/ippo_cuda_smoke
python -u algorithm/evaluate_ippo.py --checkpoint outputs/ippo_cuda_smoke/final.pt --seed-base 10000100 --episodes 2 --split development --output outputs/ippo_validation/development_evaluation.json
python -u algorithm/evaluate_ippo.py --checkpoint outputs/ippo_cuda_smoke/final.pt --seed-base 70000100 --episodes 1 --split holdout --output outputs/ippo_validation/holdout_evaluation.json
python -u tools/replay_combat.py --checkpoint outputs/ippo_cuda_smoke/final.pt --env-config configs/combat_environment_v341.yaml --episode-seed 20000071 --output-dir outputs/ippo_validation/official_replay --formats png html csv report
python -m pip install --target outputs/ippo_validation/video_deps --no-deps imageio-ffmpeg
python -u tools/replay_combat.py --checkpoint outputs/ippo_cuda_smoke/final.pt --env-config configs/combat_environment_v341.yaml --episode-seed 20000071 --output-dir outputs/ippo_validation/official_replay_mp4 --formats mp4 png html csv report --ffmpeg C:/Users/HPK/Desktop/uav1/outputs/ippo_validation/video_deps/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe
python algorithm/diagnose_ippo.py --checkpoints outputs/ippo_cuda_smoke/best_eval.pt outputs/ippo_cuda_smoke/final.pt --observation-bank outputs/ippo_cuda_smoke/diagnostic_observations.npz --recordings outputs/ippo_validation/official_replay outputs/ippo_validation/official_replay_mp4 --output outputs/ippo_validation/fixed_policy_verified.json
```

以上均实际完成；首次没有FFmpeg的MP4命令和网络受限的依赖安装尝试失败，分别保留 `replay_console.log`、`ffmpeg_setup.log`。随后指定FFmpeg后的成功记录为 `replay_mp4_console.log`，安装成功为 `ffmpeg_setup_retry.log`。

## 正式3M训练命令（未运行）

```powershell
& 'C:\Users\HPK\anaconda3\shell\condabin\conda-hook.ps1'
conda activate uav
python -u algorithm/train_ippo.py --env-config configs/combat_environment_v341.yaml --algorithm-config configs/ippo_4v4_v341.yaml --device cuda --seed 0 --num-envs 16 --total-sampled-steps 3000000 --output-dir outputs/ippo_3m_seed0_new
```

## 已知限制和验收边界

1. 支持完整保存和确定性评估重载，不支持bit-exact训练恢复。
2. 正式3M未运行，不对长期收敛、胜率或实际长训KL停止频率作结论。
3. 历史全仓测试原本存在失败；最终差异单独比较，不宣称仓库全绿，也不为本次任务修改历史断言。
4. 训练rollout中由float32自身观测反解角度/速度，只在有效存活后继上统计，局内相邻动作变化限当前rollout；完整连续转弯、死亡当步物理状态、开火几何和损失原因来自独立完整canonical诊断轨迹。
5. 首次开火ATA/HA为原canonical hook记录的开火瞬间、命中结算之前几何，不对环境额外步进。持续同向转弯与饱和均为描述指标，不作为自动失败或收敛判据。
6. 显式0环境、0预算及不能整除环境数的预算均在环境启动前拒绝，专项三项已通过。
7. CUDA强制；正式网络固定256×256；精确采样预算要求可被环境数整除；smoke有界且使用相同网络与PPO调度。
