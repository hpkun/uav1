# MAPPO λ-only GAE：实现、固定 rollout 对照与 CUDA smoke

日期：2026-10-17。正式pilot尚未启动。本次只执行针对性测试、冻结checkpoint采样与两组各4096 sampled steps的链路smoke。所有诊断工件存放项目外：C:/Users/HPK/.codex/visualizations/2026/10/14/01a13a26-30ef-7722-9212-1adb0c28ae9a/mappo_lambda_only_20261017。

## 1. 实际修改内容

新增 `configs/mappo_4v4_v338_lambda996195.yaml`，Control直接使用原始baseline配置。新增12个归一化前统计字段，写入现有optimization_metrics.jsonl；追加λ身份和底层配置检查；补summary的ceiling与正确noncombat汇总；新增固定rollout工具和针对性测试。没有修改环境、reward、Blue、网络、LR、γ、PPO epochs、target KL、entropy系数、std模式或正式evaluation协议。

相关实现：[trainer.py](C:/Users/HPK/Desktop/uav1/algorithm/mappo/trainer.py)、[factory.py](C:/Users/HPK/Desktop/uav1/algorithm/mappo/factory.py)、[runner.py](C:/Users/HPK/Desktop/uav1/algorithm/mappo/runner.py)、[checkpoint.py](C:/Users/HPK/Desktop/uav1/algorithm/common/checkpoint.py)、[evaluator.py](C:/Users/HPK/Desktop/uav1/algorithm/common/evaluator.py)、[train_mappo.py](C:/Users/HPK/Desktop/uav1/algorithm/train_mappo.py)、[诊断工具](C:/Users/HPK/Desktop/uav1/tools/diagnose_mappo_lambda.py)、[测试](C:/Users/HPK/Desktop/uav1/tests/test_mappo_lambda.py)。

现有explained_variance、preclip grad norm、actual epochs/stop、epoch KL/ratio、σ/logstd、饱和率，以及训练任务/奖励分量/Blue switch等日志直接复用。training_metrics在有完成episode时本来就通过aggregate_combat_records可靠提供red/blue_ceiling_losses，因此没有再造第二套training ceiling计数。

## 2. Baseline 数学是否保持不变

compute_gae函数与修改前保存源码的AST完全相同；递推顺序、continuation=(1−done)×next_alive、死亡即时reward、alive mask、terminal/truncated处理、returns及原adv normalization表达式均未改。新增函数在torch.no_grad中，仅索引/读取既有tensor并返回Python float，不修改数据、不用RNG、不进入loss。

测试同时验证两种λ与冻结旧递推逐位一致；diagnostics开/关时Actor/Critic参数更新与所有既有优化指标逐值一致；标准化公式逐位一致。数学不变指基线算法计算，没有声称日志文件字节或运行耗时不变。新增metadata仅影响身份校验，不改权重/optimizer数学状态。163个受保护历史结果、baseline配置/环境配置/网络文件哈希均保持不变。

## 3. Treatment 配置差异和身份

机器比对整个YAML字典，唯一差异为 `training.gae_lambda: 0.95 → 0.996195`。Control未复制；两份配置的training budget仍为3M，pilot用同一个CLI预算覆盖。网络Actor78342/Critic123905参数、MLP256×256、obs45/action3/4agents、16env/256rollout/512minibatch/10epochs上限等完全一致。

startup/summary原有gamma与gae_lambda直接复用；run_config增加training_gae_lambda，checkpoint.extra增加training_gae_lambda（training_gamma由trainer规范写入），algorithm_config snapshot完整保留。factory产生的trainer携带配置hash，底层load亦检查；上层evaluation/resume继续使用既有严格hash/Gaussian/架构协议，并显式核验γ/λ。历史checkpoint缺少新增λ字段时保留兼容，但有配置hash的旧正式checkpoint不能跨Control/Treatment混载。没有伪造旧checkpoint的λ metadata，也未迁移已有文件。

## 4. 新增诊断指标定义

全部在normalization之前，对整个rollout `alive_masks>.5` 样本统计一次；不是每minibatch重新统计，也不对dead slots平均。std均采用population std（unbiased=False），分位数为torch.quantile默认线性插值。

| 字段 | 定义 |
|---|---|
| raw_advantage_mean/std | GAE原始A的均值/总体标准差 |
| raw_advantage_abs_mean | mean(|A|) |
| raw_advantage_abs_p50/p90/p99 | |A|的50/90/99百分位 |
| raw_advantage_min/max | A的真实全rollout存活样本最小/最大值 |
| return_target_mean/std | 未标准化的固定GAE return target均值/总体std |
| old_value_mean/std | 本次优化前、用于GAE的V(s)均值/总体std |


本次未增加可选td_error_mean/std：保持compute_gae原函数逐位不动，不另抄一份δ公式或额外重复GAE。上述12个必需字段已覆盖。没有增加归一化后std≈1的重复指标，也没有改explained_variance的原口径（更新前V对fixed targets）。

非战斗损失：`boundary_exits + ground_losses`，**ceiling已属于boundary子类，不再相加**。公共helper集中提供此定义；aggregate增加average_red/blue_noncombat_losses，training复用aggregate，summary按同一定义汇总，并补average_red/blue_ceiling_losses。summary/training/eval的这些completed-episode指标是每局平均；不是step事件率。真实resolver测试构造2个boundary（其中1个ceiling）+1个ground，正确noncombat=3而非4。

## 5. 固定 rollout λ 对照

使用原MAPPO seed1 best_eval.pt（step2203648），模型冻结，CUDA执行采样；16env先warmup128steps=2048 transitions，再取256steps×16=4096 transitions。同一份rewards/values/next_values/dones/alive/next_alive分别调用未修改compute_gae，两次之间只有λ不同。没有optimizer step。数据形状256×16×4，13652个alive action samples；每个输入数组SHA256保存在comparison.json，结果前后未改变。

| metric | control | treatment | ratio |
|---|---|---|---|
| raw_advantage_mean | 0.280626 | 1.081648 | 3.854416 |
| raw_advantage_std | 1.680904 | 3.620100 | 2.153663 |
| raw_advantage_abs_mean | 0.677360 | 1.974295 | 2.914691 |
| raw_advantage_abs_p50 | 0.151760 | 0.572354 | 3.771449 |
| raw_advantage_abs_p90 | 1.557251 | 6.543405 | 4.201895 |
| raw_advantage_abs_p99 | 8.479383 | 14.584698 | 1.720019 |
| raw_advantage_min | -11.989060 | -12.293410 | 1.025386 |
| raw_advantage_max | 16.396416 | 22.190607 | 1.353382 |
| return_target_mean | -1.513433 | -0.712411 | 0.470725 |
| return_target_std | 3.269406 | 4.923441 | 1.505913 |
| old_value_mean | -1.794059 | -1.794059 | 1.000000 |
| old_value_std | 2.709472 | 2.709472 | 1.000000 |


| metric | value |
|---|---|
| alive_samples | 13652.000000000 |
| pearson | 0.793469525 |
| spearman | 0.724956682 |
| sign_disagreement_fraction | 0.219088778 |
| abs_mean_ratio | 2.914690941 |
| mean_sample_abs_ratio_nonzero_control | 23.595110976 |
| median_sample_abs_ratio_nonzero_control | 2.628959771 |
| zero_control_samples | 0.000000000 |


raw std约2.154倍，|A|P99约1.720倍，return-target std约1.506倍；raw advantage符号不一致约21.91%，rank/Pearson相关约.725/.793。固定输入下这些数值变化可明确归于λ的trace权重；但它不是任务层面的因果胜率实验，也不是总体统计显著性检验。样本在时间/agent内相关，不能把13652当独立重复实验计算伪精确p值。

mean(|A_T|)/mean(|A_C|)=2.915与逐样本比值均值23.595是不同量；后者被control接近0的分母放大，主要参考整体绝对均值比和逐样本比中位数2.629。raw sign不一致也不等于标准化后policy gradient方向必然同比改变，标准化还会减去各组均值。

事件分层用同一agent、同一episode内距下一次事件的step数；不跨death/terminal/reset。kill标签为正individual kill credit（R1加回nonboundary死亡−10项后重建），因此34是有kill credit的agent-step数，不是34架Blue；own_death指非boundary死亡（weapon或ground）15次，boundary_death21次（包含ceiling）。分层区间含0秒的即时event，其他行是此前若干秒；多个事件类可能重叠，不当独立因果组。

| event | seconds_from | seconds_to | n | control_std | treatment_std | control_abs_mean | treatment_abs_mean | sign_disagreement |
|---|---|---|---|---|---|---|---|---|
| kill | 0.000000 | 1.000000 | 340 | 2.840679 | 4.397212 | 7.739055 | 11.496659 | 0.000000 |
| kill | 1.000000 | 2.000000 | 290 | 1.849883 | 4.189649 | 3.880270 | 9.776007 | 0.000000 |
| kill | 2.000000 | 5.000000 | 684 | 1.279993 | 3.844315 | 1.451771 | 7.763129 | 0.065789 |
| kill | 5.000000 | 10.000000 | 970 | 0.474133 | 2.565906 | 0.340227 | 4.544440 | 0.306186 |
| kill | 10.000000 | 20.000000 | 614 | 0.372246 | 1.067930 | 0.256093 | 1.958011 | 0.340391 |
| own_death | 0.000000 | 1.000000 | 150 | 3.633638 | 4.183341 | 3.873228 | 4.612033 | 0.006667 |
| own_death | 1.000000 | 2.000000 | 150 | 2.721712 | 4.453632 | 2.646988 | 4.677255 | 0.000000 |
| own_death | 2.000000 | 5.000000 | 450 | 1.066158 | 2.920765 | 1.102208 | 3.558084 | 0.037778 |
| own_death | 5.000000 | 10.000000 | 565 | 1.913319 | 3.982158 | 0.739807 | 2.899987 | 0.111504 |
| own_death | 10.000000 | 20.000000 | 501 | 0.519141 | 3.258987 | 0.226654 | 1.841859 | 0.532934 |
| boundary_death | 0.000000 | 1.000000 | 210 | 0.694356 | 0.787741 | 0.758204 | 0.924381 | 0.028571 |
| boundary_death | 1.000000 | 2.000000 | 210 | 0.425058 | 0.668430 | 0.355834 | 0.672478 | 0.071429 |
| boundary_death | 2.000000 | 5.000000 | 630 | 0.360654 | 0.571508 | 0.267375 | 0.505095 | 0.293651 |
| boundary_death | 5.000000 | 10.000000 | 1050 | 1.084644 | 1.738719 | 0.384462 | 0.783066 | 0.260000 |
| boundary_death | 10.000000 | 20.000000 | 1628 | 0.310189 | 1.021785 | 0.122326 | 0.432732 | 0.393735 |


增大λ通常让更早时刻的事件相关advantage绝对量更大，但表中分层n不均、使用的是已有baseline近似V而非真实Q/V。当前数据不能测得真实estimator bias，也不能证明bias已下降；能够直接测量的是raw A/target分布、rank/sign改变及尾部放大。提高λ减少短步bootstrap混合、增加MC成分的理论倾向应与可能增大的方差一起考虑；rollout尾部仍bootstrap，不是纯完整episode MC。

![固定rollout与trace](C:/Users/HPK/.codex/visualizations/2026/10/14/01a13a26-30ef-7722-9212-1adb0c28ae9a/mappo_lambda_only_20261017/fixed_rollout_and_trace.png)

## 6. Residual trace 时间尺度

| seconds | steps | control | treatment |
|---|---|---|---|
| 1 | 10 | 0.5414869555 | 0.8705536131 |
| 2 | 20 | 0.2932081230 | 0.7578635933 |
| 5 | 50 | 0.0465521769 | 0.5000087583 |
| 10 | 100 | 0.0021671052 | 0.2500087584 |


λ_exact=.5^(1/50)/.99≈.9961946510；按要求写入.996195，实际(.99×.996195)^50=0.5000087583185194，与.5差约8.7583e−6。测试容差1e−5通过。表描述显式TD residual的权重，不是严格“reward credit截止线”；V bootstrap仍可承载更远期信息。γ=.99完全未变，因此没有把discount objective本身改成长γ。

## 7. CUDA smoke

两个fresh运行均seed1、16env、4096 sampled steps，保留正式256×256网络、rollout256、10epochs、minibatch512。没有使用旧CLI --smoke，因为它会缩hidden/epochs/minibatch；因此启动日志mode=formal表示正式数学设置，**运行预算只有4096，科学用途是smoke，不是正式pilot结果**。两个output均在本报告目录独立子目录，未写进历史outputs。

| arm | sampled_steps | ppo_updates | actor_changed | critic_changed | metrics_finite | roundtrip_actions_equal | actor_parameters | critic_parameters |
|---|---|---|---|---|---|---|---|---|
| Control | 4096 | 1 | True | True | True | True | 78342 | 123905 |
| Treatment | 4096 | 1 | True | True | True | True | 78342 | 123905 |


| metric | control | treatment | delta |
|---|---|---|---|
| raw_advantage_mean | -0.410503 | -1.628452 | -1.217949 |
| raw_advantage_std | 1.493745 | 2.565614 | 1.071870 |
| raw_advantage_abs_p50 | 0.013555 | 0.125889 | 0.112334 |
| raw_advantage_abs_p90 | 1.104473 | 5.848619 | 4.744145 |
| raw_advantage_abs_p99 | 8.358315 | 9.637934 | 1.279618 |
| return_target_std | 1.495188 | 2.567952 | 1.072764 |
| old_value_mean | 0.051623 | 0.051623 | 0.000000 |
| old_value_std | 0.020356 | 0.020356 | 0.000000 |
| actor_grad_norm | 0.114918 | 0.118805 | 0.003886 |
| critic_grad_norm | 0.049251 | 0.144648 | 0.095397 |
| approx_kl | 0.001746 | 0.001684 | -0.000062 |
| clip_fraction | 0.008938 | 0.007239 | -0.001699 |
| effective_ppo_epochs | 10.000000 | 10.000000 | 0.000000 |
| kl_early_stop | 0.000000 | 0.000000 | 0.000000 |
| explained_variance | 0.001929 | 0.001820 | -0.000110 |
| entropy | 1.799382 | 1.797600 | -0.001782 |


按正确runtime seed=1重建初始模型后，Control Actor/Critic最大参数绝对变化分别为0.02380729/0.02882954；Treatment分别为0.02389816/0.02943782。

每侧1个真实PPO update，均实际执行10epochs（80个Actor和80个Critic optimizer steps），所有新增/原有optimization数值有限。checkpoint metadata/load/evaluation/resume验证通过，保存再加载后的固定观测deterministic action逐值相等；各侧使用同一个开发seed10000000做单局deterministic链路检查，修正初始参数验证器后各重复检查一次，共4次单局执行；不作为20局正式eval，不据此比较胜率。正式evaluation配置仍是固定20seed且未修改。

本次smoke：raw std 1.494→2.566、P99 8.358→9.638、critic preclip grad norm .0493→.1446，actor norm仅.1149→.1188；KL .001746→.001684，clip .008938→.007239，二者稍降。两次更新前采样的训练episode统计相同。**raw variance变大不等于PPO更新必然更大**：alive normalization、A形状/排序、Adam、clipping、状态分布共同影响。这里的变化不是“失败”，也不是“Treatment更好”；smoke太短且只有1update。

## 8. Tests 和静态检查

83 passed in 15.61s。运行tests/test_mappo_lambda.py、test_mappo.py、test_gaussian_protocol.py、test_eaa_mappo.py，均在uav/CUDA可用环境；pytest临时目录和日志位于项目外，禁用cache，未覆盖历史输出。覆盖baseline/treatment递推、5秒数学、death/terminal、normalization、诊断无梯度/不消耗RNG/不改变loss和参数、metadata与错误配置拒绝、fresh output保护、ceiling不重复、所有新增metric finite。git diff --check通过。

## 9. 新实现问题与边界

没有发现新的GAE/PPO数学错误。发现并补齐两个可观测性/身份缺口：summary未显式汇总ceiling；底层factory trainer.load在历史checkpoint缺新λ字段时也需要配置hash把关（上层既有Gaussian protocol原本已检查hash，直接复用，未重复实现另一套eval身份）。metadata是附加字段，未升数学实现版本。

固定rollout统计初版调用Series Spearman依赖环境中缺失的SciPy，已改为平均秩后的Pearson，无需安装依赖；失败尝试目录保留，权威完成工件是fixed_final，不把失败尝试混入结果。所有诊断脚本遇到CUDA不可用显式失败，不回退CPU。

smoke初始参数验证已显式使用CLI相同runtime seed=1，未通过改config.seed来间接设置，因为通用evaluation factory原本默认seed0。最终control_validation_final/treatment_validation_final重检后，两侧Actor/Critic相对正确初始模型均有非零参数变化；这只修正验证器，未修改训练随机种子、未重跑smoke。

## 10. 正式 paired pilot 推荐预算

推荐每侧 **2,400,256 sampled steps**，即586个完整4096-step rollout，相比原733个update/3M预算约少20%。不是100k/200k，也不直接要求完整3M。历史vanilla MAPPO seed1到2.203648M才出现最高.70；在2.301952M降至.55、2.400256M降至.30。若只到1M或2M，会错过这次高点及最早明显回落；2.400256M能覆盖它们，又不继续到2.6M恢复和2.8M第二个低谷。

| sampled_steps | win_rate | average_return |
|---|---|---|
| 2002944 | 0.250000 | -12.985828 |
| 2101248 | 0.300000 | -8.055623 |
| 2203648 | 0.700000 | 22.150916 |
| 2301952 | 0.550000 | 10.536882 |
| 2400256 | 0.300000 | -11.355235 |


预算取full-rollout边界，而非整数2.4M，避免末轮短rollout额外改变最后一次GAE/minibatch。历史同seed路径在新增纯诊断下数学不变，但不能承诺跨平台/CUDA执行逐局bitwise重现；pilot仍是一个seed的机制筛查，不能给总体显著性或最终3M保持率结论。原训练没有可靠wall-clock成本记录，因此只报告采样/更新预算节省，不能虚报节省小时；Treatment可能改变KL-stop实际epochs与耗时。

## 11. 两条正式启动命令（只提供，未执行）

当前WSL访问被拒，使用已授权的Windows PowerShell uav fallback。以下先进入项目目录并激活环境；不传--resume、不传--smoke，两个fresh独立目录，Control直接读取baseline。目录一旦有训练工件，入口会拒绝复用。

```powershell
Set-Location C:\Users\HPK\Desktop\uav1
conda activate uav

python -u algorithm/train_mappo.py `
  --env-config configs/combat_environment_v338.yaml `
  --algorithm-config configs/mappo_4v4_v338.yaml `
  --device cuda `
  --seed 1 `
  --num-envs 16 `
  --total-sampled-steps 2400256 `
  --output-dir outputs/mappo_v338_seed1_lambda095_pilot2400256

python -u algorithm/train_mappo.py `
  --env-config configs/combat_environment_v338.yaml `
  --algorithm-config configs/mappo_4v4_v338_lambda996195.yaml `
  --device cuda `
  --seed 1 `
  --num-envs 16 `
  --total-sampled-steps 2400256 `
  --output-dir outputs/mappo_v338_seed1_lambda996195_pilot2400256
```

如果之后WSL可正常访问，按仓库政策先在Ubuntu conda activate uav，再在/mnt/c/Users/HPK/Desktop/uav1运行相同Python参数；PowerShell续行反引号需改为bash反斜杠。不要同时启动两侧来制造GPU资源竞争；可依次人工启动。

## 12. Pilot 完成后的比较与结论边界

| 类别 | 比较要求 |
|---|---|
| 性能 | train/eval return分开，win、RedLoss/BlueLoss、窗口fire与实际attempt、kill，best/final及best→final gap；best沿既有(win,R,−RedLoss)规则，不改选择目标 |
| GAE | raw_advantage_std/abs mean/P50/P90/P99/extrema、return_target_std、old_value_std、explained_variance；阶段对齐，区分raw尺度与normalize后的实际更新 |
| PPO | KL、clipfrac、effective epochs、early-stop标记率与实际<10epochs比例、Actor/Critic preclip grad norm；不要把update均KL误当epoch stop判断 |
| Exploration | squashed entropy、各动作σ/logstd、.9/.99/.999饱和率；不能把负H当异常 |
| 环境行为 | boundary、ground、ceiling、noncombat=boundary+ground、timeout、Blue switches；ceiling不重复累计，raw switches需注意episode长度 |
| 统计设计 | 同sampled-step网格、预先固定时间段，train任务按完成episode数加权、eval同20开发seed；一个训练seed不支持总体显著性结论，也不从best单点决定胜负 |


优先比较：Treatment是否持续改变raw优势尾部/target尺度、这些变化是否伴随critic EV与grad、实际epochs改变，以及非战斗死亡和高性能保留是否同向变化。若方差升而保持率改善，可能有有益credit变化；若方差升且估值/更新压力恶化，也不是简单“λ理论错误”，需要bias-variance证据链。两侧轨迹在首次update后分叉，后续差异是该λ干预的闭环总效应，无法再假定每个rollout输入相同。

**当前实验只检验“λ / GAE estimator”这一变量，不能用结果直接判断 gamma、reward、GRU 或环境边界设计是否正确。**本次尚未获得pilot性能结论，也没有声称Treatment减少了真实bias或解决best→final degradation。

工件：[comparison.json](C:/Users/HPK/.codex/visualizations/2026/10/14/01a13a26-30ef-7722-9212-1adb0c28ae9a/mappo_lambda_only_20261017/fixed_final/comparison.json)、[fixed_rollout.npz](C:/Users/HPK/.codex/visualizations/2026/10/14/01a13a26-30ef-7722-9212-1adb0c28ae9a/mappo_lambda_only_20261017/fixed_final/fixed_rollout.npz)、[event_strata.csv](C:/Users/HPK/.codex/visualizations/2026/10/14/01a13a26-30ef-7722-9212-1adb0c28ae9a/mappo_lambda_only_20261017/event_strata.csv)、[smoke_statistics.csv](C:/Users/HPK/.codex/visualizations/2026/10/14/01a13a26-30ef-7722-9212-1adb0c28ae9a/mappo_lambda_only_20261017/smoke_statistics.csv)、[verification.json](C:/Users/HPK/.codex/visualizations/2026/10/14/01a13a26-30ef-7722-9212-1adb0c28ae9a/mappo_lambda_only_20261017/verification.json)、[测试日志](C:/Users/HPK/.codex/visualizations/2026/10/14/01a13a26-30ef-7722-9212-1adb0c28ae9a/mappo_lambda_only_20261017/tests_identity.log)。
