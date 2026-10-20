# ETM-MAPPO：独立二维协作导航

本实现只连接新建的二维协作导航环境。算法包不导入现有仿真环境、MAPPO或STEA运行时，也不修改任何已有算法、配置、物理环境或奖励。ETM-MAPPO是待验证结构，不预设比基线更好。

## 结构与消融

Actor参数由所有机器人共享，各机器人及其每个可见实体维护独立状态。

| 方法 | 编码/聚合 | 时序记忆 | 融合 |
|---|---|---|---|
| B0 MLP-MAPPO | 类型编码后masked mean，无注意力 | 无 | 拼接MLP |
| B1 Post-Aggregation GRU | 类型编码、当前实体cross-attention | 聚合后单个128维GRUCell | 拼接MLP |
| B2 Entity-GRU Attention | 类型编码、逐实体64维GRUCell、cross-attention | 自身/协作/障碍独立历史 | 与B3相同投影，固定g=0.5 |
| B3 ETM-MAPPO | 同B2 | 同B2 | 学习门控g，LayerNorm |
| B3-reset | B3全部参数保留 | 每步在使用历史前清零 | 同B3 |

B2与B3的共享参数可以逐项复制，B3门控置零时sigmoid=0.5，两者连续多步输出逐位相同；参数差异仅为32,896个门控参数。B1与B0、B2与B1的结构/参数量变化较多，比较时不把全部差异归因于单一组件。

## 输入和模块形状

单步输入均有环境/观察者前缀[E,N]。N和实体容量不是网络固定常量；示范环境每次运行保持固定最大容量，实体可见性通过mask变化。

| 字段 | 形状 | 含义 |
|---|---|---|
| self_features | [E,N,Ds] | 自身及公开任务指令 |
| peer_features | [E,N,Kp,Dp] | 可见协作实体相对状态 |
| obstacle_features | [E,N,Ko,Do] | 可见障碍相对状态 |
| peer_ids / obstacle_ids | [E,N,Kp] / [E,N,Ko] | int64身份；无效位置为-1 |
| peer_valid / obstacle_valid | 对应ID形状 | 显式bool有效性 |
| agent_alive | [E,N] | 当前观察者有效性 |
| episode_start | [E] | 当前帧是回合首帧 |

有效性从显式mask读取，不以全零特征推断。有效位置ID必须非负且每个观察者/类型内唯一；无效位置必须为-1。重复ID、dtype和形状不一致均报错。ID只参与匹配，从不当作连续特征。

默认网络输入维度Ds=10、Dp=4、Do=5，动作维度2；网络支持其他维度，示范环境的真实输出接口固定10/4/5/2，Runner会拒绝不匹配，不能仅修改维度数字后继续训练。

| B3模块 | 输入→输出 | 实际参数 |
|---|---|---:|
| SelfEncoder，Linear+LayerNorm+ReLU | 10→64 | 832 |
| PeerEncoder，同类型共享 | 4→64 | 448 |
| ObstacleEncoder，同类型共享 | 5→64 | 512 |
| Self / Peer / Obstacle GRUCell | 每实体64→64 | 各24,960 |
| Peer / Obstacle TemporalAttention | 64维，2头，每头32 | 各12,480 |
| Self Projection | 64→128 | 8,320 |
| Context Projection | 128→128 | 16,512 |
| Gate，sigmoid | 256→128 | 32,896 |
| Fusion LayerNorm | 128→128 | 256 |
| Policy Hidden，Linear+ReLU | 128→128 | 16,512 |
| Mean / Log Std头 | 128→2 | 各258 |

B3 Actor合计176,644，集中式Critic29,953，总计206,597。Critic采用共享10→64编码、独立2头attention聚合所有有效机器人、128→128→1价值头，只在Critic跨机器人聚合。Actor的神经网络计算对[E,N]逐行独立。

| 方法 | Actor | Critic | 总计 |
|---|---:|---:|---:|
| B0 | 43,780 | 29,953 | 73,733 |
| B1 | 167,812 | 29,953 | 197,765 |
| B2 | 143,748 | 29,953 | 173,701 |
| B3 / B3-reset | 176,644 | 29,953 | 206,597 |

实际PyTorch计数文件位于验证目录。所有方法同seed下Critic初始权重逐位相同；Critic使用独立初始化随机流，避免Actor构造消耗不同数量随机数造成比较混杂。

## 身份状态管理与生命周期

`memory.py`将离散身份匹配与可微隐藏重排分开。`MemoryState`保存：自身[E,N,64]、peer[E,N,Kp,64]、obstacle[E,N,Ko,64]、对应上一帧ID，以及B1用的聚合后[E,N,128]状态。

对每个环境和每个观察者独立比较当前有效ID与上一帧ID，得到current-slot→previous-slot索引。无梯度部分只计算离散索引，`gather`保留隐藏值的时间梯度。同ID换槽时记忆跟随；新ID、上一帧不可见后重现的ID，从零状态开始。当前无效实体输出隐藏值和ID分别为零与-1。没有跨失联期的长期缓存，不会将离开的实体记忆分配给新实体。

自身无效时，所有属于该观察者的隐藏状态和动作清零。episode_start在GRU使用前清除该环境所有历史；Runner在自动reset之后也立即清零这些状态。不同环境或观察者之间不共享状态，只有参数共享。

Attention在无效位置权重精确为零；K=0直接返回零上下文；全部mask为假时先保护softmax再显式清零。无外部实体时，context projection在Linear之后再乘集合非空mask，防止bias形成虚假消息。

B3计算 u=W_self*h_self，v=W_context*[peer_context;obstacle_context]，g=sigmoid(W_gate*[u;v])，z=LayerNorm(g*u+(1-g)*v)，再接128维ReLU策略层。Tanh-Gaussian使用state-dependent log std，初始化-0.5，范围[-5,2]；log概率直接由保存的raw latent action计算稳定Jacobian，不经过atanh。

## Recurrent PPO协议

默认：rollout128、sequence16、PPO epochs4、minibatch256条联合环境transition、8个环境、gamma0.99、lambda0.95、clip0.2、entropy0.01、value coefficient0.5、Actor/Critic Adam各3e-4、独立global L2裁剪0.5。

Rollout包含结构化当前/真实后继观测（含ID和mask）、动作/raw/old logp、逐机器人奖励、采样时value和next value、terminated/truncated、动作前MemoryState和ID映射、episode_start、真实时间有效mask。

GAE先在完整真实rollout上计算。真正terminated或next_alive=false时不bootstrap；time-limit truncated使用reset前真实后继value bootstrap，但两类done都阻断GAE递归，绝不跨局。与历史环境的truncation约定不同之处仅存在于本独立算法，原有代码不改动。

然后按环境划分连续chunk，chunk内时间不打乱，只打乱chunk顺序。每个chunk从保存的动作前状态/ID开始并detach，内部重新按时间执行所有编码、身份匹配、GRU和注意力。尾部显式padding，ID=-1、alive/valid=false、episode_start=true。Padding/无效agent不参与loss、entropy、advantage统计或有效样本计数；无效entity不参与attention。损失算术先选有效样本，避免padding中的任意数值进入exp/square。

正式默认每个rollout=128×8=1024条联合transition，4个minibatch/epoch，4epoch=16次Actor和16次Critic更新；8192步=8个rollout，最多128次。sampled_steps不乘机器人数量。

每次更新前，重新计算所有真实有效样本的logp，严格检查 max|exp(new-old)-1|≤1e-4。失败立即报错。标准PPO clipped policy objective、MC有界策略entropy、max平方误差value clipping；无其他RL损失。可选target_kl默认null，启用时按完整epoch平均KL决定停止后续epoch。

## 独立二维环境

`navigation.py`为新建的二维加速度模拟器，没有导入现有环境。默认4机器人、3障碍、dt0.12、每局80步，速度上限1m/s、加速度上限1m/s²，动作在[-1,1]²。机器人有固定ID，障碍有固定ID；每次观测可随机打乱槽位，距离1.25m外实体不提供特征。槽位随机流与物理初始化随机流分离，打乱槽位不会改变物理轨迹。

自身10维：[自身位置2，自身速度2，公共目标相对位置2，指定编队几何偏移2，上步控制2]。编队偏移是公开任务指令，不是其他机器人的真实状态；不可见机器人的位置不能改变观察者自身特征。Peer4维为相对位置/速度；Obstacle5维为相对位置/速度/半径。只有Critic接收所有机器人的自身状态。

奖励仅四项：朝各自指定目标区域的距离进度、相对指定几何编队的误差、机器人/障碍碰撞惩罚、相邻控制变化惩罚。参数在所有对照中完全一致。成功条件为所有机器人进入各自目标区域且本局未碰撞；碰撞不会销毁机器人。Time limit单独标为truncated。

simple场景使用确定性初始编队、目标与静态离轴障碍，作为学习预检；random评估随机化初始位置、目标和缓慢移动障碍。默认训练只用simple，random仅评估泛化，不根据其结果选择best或调整训练。

## 产物、指标和恢复边界

新实验拒绝覆盖非空目录。输出config快照、run_config、training_metrics.jsonl、optimization_metrics.jsonl、episode_metrics.jsonl、evaluation_history.csv/jsonl、latest.pt、best_eval.pt、定期checkpoint、final.pt、run_summary.json及train.log。异常写exceptions.jsonl。

Checkpoint明确记录ETM-MAPPO身份、实现版本、环境/观测schema、完整配置和SHA256、网络架构/参数量、训练seed、Actor/Critic/Adam状态、sampled/vector/rollout/实际optimizer次数、评估/最佳模型来源。加载严格匹配身份/版本/配置/架构/参数量；确定性评估重载支持，精确训练恢复不支持，CLI --resume明确拒绝。不能宣称恢复了环境、实体历史和全部RNG状态。

主要诊断字段：policy_loss/actor_loss、value_loss、entropy、approximate_kl、clip_fraction、ratio_mean/min/max、pre_update_ratio_max_abs_error、actor/critic_gradient_norm、gradient_clipped_ratio、advantage_mean/std、return_target_mean/std、valid_agent_samples、valid/padded_joint_transitions、self/peer/obstacle_hidden_norm、peer/obstacle_attention_entropy/top1、gate_mean/std/saturation_fraction、gate_dim_i_mean/std、action_mean/std/saturation_i、policy_sigma_i。有效样本数量是本rollout唯一agent-step，optimizer次数为真实执行值；梯度范数为裁剪前值，饱和阈值为|action|>0.99和gate<0.05或>0.95。

B0没有attention，其权重为零占位；B0/B1没有gate，内部gate张量为零占位，派生饱和比例会为1，这些统计不适用，不能解释为模块塌缩。B2的gate是固定0.5。可学习门控统计仅对B3及B3-reset适用，具体模块是否存在由variant和参数清单确定。

任务指标为Return、完成率、碰撞率、超时率、目标/编队误差和四项奖励分解；本环境没有对抗方、射击或击毁指标。

## PowerShell用法

```powershell
& 'C:/Users/HPK/anaconda3/shell/condabin/conda-hook.ps1'
conda activate uav
python -m pytest tests/test_etm_mappo.py -q --tb=short
python -u algorithm/train_etm_mappo.py --smoke --output-dir outputs/etm_smoke_new
python -u algorithm/train_etm_mappo.py --total-sampled-steps 8192 --seed 0 --num-envs 8 --variant B3 --output-dir outputs/etm_short_new
python -u algorithm/evaluate_etm_mappo.py --checkpoint outputs/etm_short_new/final.pt --scene random --seed-base 20000000 --episodes 6 --output outputs/etm_random_new.json
python -u algorithm/benchmark_etm_mappo.py --budget 8192 --seeds 0 1 2 --output-dir outputs/etm_ablation_new
```

可通过--algorithm-config/--config指定完整配置；--env-config只接受独立导航环境配置。所有运行均要求CUDA。默认预算8192；smoke默认512、最多16384；benchmark最多16384/组，不会因结果不佳自动扩预算。benchmark的--reuse-completed-from仅复用完整结束且checkpoint与配置严格匹配的已完成实验，不恢复任何中断训练状态。

当前网络正确性、训练闭环、短程学习迹象和稳定性能收益四类结论须分开；实际结果与限制见 `docs/etm_mappo_acceptance.md`。
