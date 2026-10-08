# 5v5 Gaussian/PPO 稳定性协议

本次只修正 MAPPO/STEA 的 Gaussian distribution 和 PPO epoch 停止协议。环境、R1–R4、武器、Blue、场景、critic 架构、STEA attention/GRU、GAE 和 clipping 公式均不变，正式 evaluation 仍执行 `tanh(mu)`。

两个5v5正式配置统一设置：

```yaml
implementation:
  policy_std_mode: state_independent
  log_std_init: -0.5
  mean_head_init_gain: 0.01
  log_std_min: -5.0
  log_std_max: 0.5
training:
  entropy_coefficient: 0.001
  target_kl: 0.015
  ppo_epochs: 10
```

新 actor 只有 `log_std_parameter[3]`，在所有state/agent/env/time共享；heading、pitch、speed分别可学习。sigma为clamped Parameter的指数，初始exp(−0.5)=0.60653066，最大exp(0.5)=1.64872127。mean head使用orthogonal gain0.01、zero bias。tanh、保存latent raw action和exact tanh Jacobian保留。

缺少新字段的旧config默认 `state_dependent`：保留原Linear log_std head、原初始化、原bounds和entropy，没有target-KL停止。旧state dict名称和legacy STEA architecture字典格式保持不变，旧checkpoint只能使用对应run的原配置快照读取；不能和新的5v5配置混用。4v4/8v8仓库配置不修改。

每个epoch遍历全部有效minibatch后，对它们的approx_kl求算术均值；严格超过target_kl才停止后续epochs，不在单个minibatch停止、不增加KL loss。MAPPO和STEA共用同一个判断与统计方法。STEA pre-update recurrent ratio audit保留。

每次update保存configured/effective PPO epochs、kl_early_stop、last_epoch_mean_kl、target_kl，以及新协议三个维度的实际clamped log_std/sigma。新协议更新记录同时写入optimization_metrics.jsonl和training_metrics.jsonl，后者用`record_type=ppo_update`区分原step记录。legacy仍保留visited-state mean log_std；为保持数值metrics兼容，disabled KL在metrics中记target_kl=0、target_kl_enabled=0，而checkpoint/config的真实值是None。

Checkpoint extra增加policy_std_mode、log_std_init、mean_head_init_gain、target_kl；新STEA architecture额外记录前三项。校验同时检查actor state dict的std类型、metadata和config SHA。原implementation version保留，通过显式新协议字段和state dict区分模式，旧文件无需迁移或改写。

独立policy_modes工具从实际distribution.scale读取sigma及其log，不再依赖Linear hook，可以诊断两套模式。sigma指数往返验证允许浮点舍入误差，统计始终来自实际执行的distribution。

修改前后旧checkpoint双种子结果、针对性测试和两个512步CUDA smoke结果见 `outputs/gaussian_protocol_validation/report.md`。短smoke只验证更新闭环，不能据此认定正式训练已消除mean-policy deployment gap。没有自动开始新的2M实验。
