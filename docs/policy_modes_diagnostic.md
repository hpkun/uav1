# v2.5 策略部署模式诊断

`tools/evaluate_policy_modes.py` 对用户指定的单个 checkpoint 和共同 environment seeds 比较 deterministic `tanh(mu)` 与 stochastic `tanh(N(mu,sigma))`。它不选择 checkpoint、不训练、不修改正式 evaluator 或实验参数，仅支持 CUDA 上的 v2.5 / 65D / 5 agents / 3 actions，执行原有 checkpoint contract 并要求两个配置 SHA 完全匹配。

每局 environment seed 为 `seed_base+i`，policy sampling seed 为 `stochastic_policy_seed+i`。每局使用独立、显式设种的 Torch CPU/CUDA RNG，上下文恢复 RNG 状态，cudnn 使用 deterministic 模式。可复现性指相同 checkpoint、配置、软件/硬件及 seed；不宣称跨设备或跨 PyTorch 版本逐位一致。

MAPPO 调用当前 actor 的 `distribution()`；STEA 调用 `distribution_step()`，每局 hidden 从零开始、死亡后清零、按现有 recurrent 语义更新。直接读取实际执行的 `distribution.scale.log()`，在浮点精度内核验 `sigma == exp(log_std)`；无额外策略 forward，不依赖 Linear hook，同时支持 legacy state-dependent head 和新的 state-independent Parameter。

输出 JSON 包含两种模式的行为 metrics、逐局记录、每局执行动作 SHA256、三个动作维度 heading/pitch/speed 的方差统计、checkpoint metadata 和 `mode_gap`。行为聚合直接使用现有 `aggregate_combat_records()` / `episode_return_metrics()`；R1–R4 为每局 Red 团队累计分量的均值，与原 evaluator 相同。

每种模式只统计决策前存活 Red agent，每个 agent-decision 等权：mean/median/population-std log_std；mean/median/p10/p90 sigma；mean absolute mu；mean absolute `(sampled_raw_action-mu)`。deterministic 使用的 raw action 就是 mu，偏离值为零。两种模式的轨迹可能不同，统计分别来自各自访问到的状态，不是逐状态配对实验。

`mode_gap` 始终为 stochastic−deterministic，包含 win rate、team return、Red/Blue loss、episode length 和 timeout rate，不自动判定部署方式优劣。输出使用独立 JSON 路径；同命令重复得到完全相同结果时保留已有文件，已有不同结果则拒绝覆盖。

以下是后续手动选择 checkpoint 后运行的示例，工具验收不会执行这些 50–100 局命令。配置必须使用该训练目录保存的版本；该实验的 2M algorithm config 与仓库 3M 正式配置 SHA 不同。

```bash
conda activate uav
python tools/evaluate_policy_modes.py --algorithm mappo \
  --checkpoint outputs/mappo_v25_5v5_seed1_2m/best_eval.pt \
  --env-config outputs/mappo_v25_5v5_seed1_2m/env_config.yaml \
  --algorithm-config outputs/mappo_v25_5v5_seed1_2m/algorithm_config.yaml \
  --seed-base 60000000 --episodes 50 --device cuda \
  --stochastic-policy-seed 12345 --output outputs/policy_modes/mappo_best_50.json
```

手动评估同一目录的 `checkpoint_2000000.pt` 时替换 checkpoint，并使用不同 output 路径，例如 `mappo_2m_50.json`。

STEA 训练完成后，将下例目录替换为实际训练目录，手动选择 best 或 2M checkpoint：

```bash
RUN_DIR=outputs/stea_mappo_v25_5v5_seed1_2m
python tools/evaluate_policy_modes.py --algorithm stea-mappo \
  --checkpoint "$RUN_DIR/best_eval.pt" \
  --env-config "$RUN_DIR/env_config.yaml" \
  --algorithm-config "$RUN_DIR/algorithm_config.yaml" \
  --seed-base 60000000 --episodes 50 --device cuda \
  --stochastic-policy-seed 12345 --output outputs/policy_modes/stea_best_50.json
```

需要100局时仅将 `--episodes` 改为100；共同environment seeds及policy seed保持相同，output文件名相应区分。不要用512步smoke的性能推断正式训练的 mean-policy deployment gap。
