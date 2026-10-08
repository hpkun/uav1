# 正式 5v5 centralized critic 协议

正式 `configs/mappo_5v5.yaml` 定义 **shared-parameter MAPPO with a centralized MLP critic**；`configs/stea_mappo_5v5.yaml` 定义 **STEA-MAPPO with entity-attention + GRU actor and the same centralized MLP critic**。两者共享环境、奖励、Gaussian policy、PPO 和 centralized critic 架构，主要结构差异是 actor representation。

这不是对 MADSAC 论文 MAPPO baseline 的逐字复现。根据本项目任务提供的 Table 3 描述，该 baseline 为 Attention=no、Shared=no；本研究针对同构 UAV、公平控制变量采用 shared-parameter actor。

两者实际使用同一个 `CentralizedMLPCritic` 类。5 个 65D masked observations 拼成 325D global state，再拼接 focal agent 的 65D observation，得到 390D 输入。网络为 Linear(390,256)→ReLU→Linear(256,256)→ReLU→Linear(256,1)，每个 agent 输出一个 value。死亡槽位先清零，最终 value 乘 alive mask；没有 Q/K/V、attention 或 GRU。参数量均为 **166,145**，Adam、critic LR 3e-4、value loss coefficient 0.5、value clipping 与原 PPO 实现一致。

MAPPO actor 的两层宽度来自 `actor_hidden_layers`，critic 两层宽度来自 `critic_hidden_layers`，两者必须为两个等宽正整数。`MAPPOTrainer.hidden_dim` 保留为旧 actor 入口；新增 `critic_hidden_dim`，缺省时等于 `hidden_dim`。Factory 正式构造分别读取两个配置。旧 checkpoint 评估从保存的 tensor 推导各自实际宽度，兼容早期缩小宽度的 smoke。STEA actor 的 entity_dim=64、entity_attention_heads=2、spatial_hidden_dim=128、gru_hidden_dim=128、GRU 单层、sequence=32 均独立保留。

正式 MAPPO actor 参数量 **83,462**；正式 STEA actor 参数量 **167,110**。STEA 的 ally/enemy attention 和 GRU 保留。同 seed、相同 256D temporary baseline actor 初始化顺序下，两种算法的 critic 初始权重也完全一致；不对初始化协议做额外重排。

两份正式 5v5 配置仅改变 critic_type。Gaussian 仍为 state-independent Parameter[3]、log_std 初值 -0.5、bounds [-5,0.5]、mean head gain 0.01；entropy coefficient=0.001、target KL=0.015、最多 10 epochs。GAE、ratio、clip、tanh log-prob、raw latent action 保存、环境、reward 和 weapon 均不变。

保留 `CentralizedValueCritic` 和 `critic_type=attention`，用于原 v2.3/v2.4/旧 v2.5 run snapshots。旧 attention checkpoint 使用自己的 env_config.yaml 和 algorithm_config.yaml 只读评估；不编辑原 checkpoint 或 snapshot，不使用新 MLP 配置加载旧 attention checkpoint。

新 MAPPO checkpoint metadata 显式记录 actor_hidden_dim、critic_hidden_dim、critic_type 和各自参数量；network_architecture 使用两个独立宽度。STEA architecture 动态记录真实 critic_type/critic_hidden_dim，attention 模式仍生成原 architecture 字典，MLP 模式不含 critic_attention_heads。两份配置保留 attention_heads 字段供 legacy 构造使用，但该字段不参与 MLP 构造；MLP 日志显示 heads=n/a，STEA 单独显示 actor 的 entity_attention_heads。

Checkpoint 校验先拒绝 attention/mlp 类型不符，配置 SHA、环境版本、维度、算法 implementation version 和 metadata/tensor 检查继续保护协议。MAPPO/STEA checkpoint 不能互换。

本次验收仅包括完整测试、compileall、四类旧 checkpoint 的各两局修改前后 deterministic 对照，以及每种算法 512 sampled steps + 两局 deterministic CUDA smoke。没有正式长训练或大规模对局审计。详细实测结果见 `outputs/critic_protocol_validation/`。
