# Combat replay

## 唯一正式入口

```bash
conda activate uav
python -u tools/replay_combat.py --checkpoint outputs/maac_v337_seed1_7m/best_eval.pt --env-config outputs/maac_v337_seed1_7m/env_config.yaml --episode-seed 1 --output-dir outputs/maac_replay_seed1 --device cuda
```

遵循仓库策略：优先 WSL Ubuntu 的 uav，不可用时使用 Windows PowerShell 的 uav。
真实 checkpoint 录像强制 CUDA，无 CPU fallback。CPU 只用于无需正式 checkpoint 的环境/observer 单元测试。
算法自动识别，支持 MAPPO、MAAC、MADSAC、RMAPPO、EA-MAPPO、STEA-MAPPO；每种均使用其正式 validator、factory/trainer 构造和 load，不提供另一套 actor。支持的环境组合仍由各算法协议决定，不为录像放宽版本限制。
默认读取 checkpoint 同目录 algorithm_config.yaml；缺失时必须用 --algorithm-config 明确指定。环境/算法指纹不匹配、checkpoint 在加载时变化都明确报错，不静默使用其他配置。

可选：--algorithm-config PATH；--playback-speed 1（默认正常速度）；--formats mp4,png,html,csv,report（也接受空格分隔）；--formats trace 只保留基础记录；--ffmpeg PATH（也可配置 FFMPEG_BINARY 或 PATH）。输出目录必须为空。没有 FFmpeg 时明确报错，不能将 Pillow 文件当 MP4；也可只选择 png,html,csv,report。

## 一次 rollout，唯一数据源

```
one deterministic rollout -> episode_trace.npz + metadata.json
                         -> MP4 / PNG / HTML / CSV / Markdown
```

所有派生模块只读 trace + metadata，不构造环境、不重新调用策略。metadata.events 是 canonical event stream，NPZ 是 canonical numerical timeline，二者对应完全同一局。
默认生成 episode_trace.npz、metadata.json、episode.mp4、trajectory.png、interactive.html、all_uav_states.csv、fire_events.csv、detailed_report.md。--formats 可省略派生文件，前两个始终生成。

## Schema 3

T=transition 数，N=每队飞机数，动态支持当前4v4及历史5v5/8v8。

| Field | Shape | Meaning |
|---|---|---|
| trace_schema_version / dt | scalar / scalar | integer 3 / seconds |
| red_kinematics / blue_kinematics | (T+1,N,6) | NED x,y,z,v,theta,psi |
| red_alive / blue_alive | (T+1,N) | boolean |
| steps / time_s | (T+1,) | 0…T / step*dt |
| red_actions / blue_actions | (T,N,3) | actual executed actions |
| local_rewards | (T,N) | environment-returned reward |
| reward_components | (T,4,N) | transition, component, Red agent |
| terminated / truncated | (T,) | boolean |
| blue_pursuit_target_indices | (T,N) | authoritative info, zero-based |
| red_ammo / blue_ammo | (T,N) | post-transition ammo |
| red_cooldown / blue_cooldown | (T,N), optional | post-transition public cooldown, simulation steps |

frame[0] 是 reset 后状态；transition[t] 把 frame[t] 转为 frame[t+1]。Transition t 的 target 是该次 step 实际绑定值，ammo/cooldown 是该 step 结束后的值。初始 ammo 在 metadata.initial_ammo。
当前 v3.37 的 Blue action 直接来自 info.executed_blue_actions，target 直接来自 info.blue_pursuit_target_indices。禁止从轨迹或距离推断。旧环境没有 executed_blue_actions 时只观察 _advance 已执行输入，不再调用 policy。旧环境没有公开 target 时 metadata 标为 unavailable，数组填 -1；缺失 ammo 同理，初值为 None；不存在的 cooldown 不伪造。

Reward 顺序固定 [R1,R2,R3,R4]；metadata.reward_component_axis=1、agent_axis=2。报告明确区分 Reward-R1 与 Red UAV R1。当前 v3.37 safety=0，测试验证 local_reward=Reward-R1+Reward-R2+Reward-R3+Reward-R4。历史版本按实际返回值保存，不强改公式。
严格检查 T+1/T、两队 shape、steps/time、单调步长、dt、所有动作/reward/target/ammo/cooldown 长度与 shape、finite、索引/类型及最终结束状态；始终 allow_pickle=False。Schema2 仅兼容历史渲染，不补造缺失字段或导出冒充完整 schema3 的报告。

## 坐标、射击和损失

NED z 向下，altitude=-z，仅转换一次。状态 theta/psi 和 event geometry 为 radians；CSV 另有明确 _deg 列。事件/CSV agent 索引从0开始，画面飞机编号从1开始。
真实 _entry_attempts 返回多少 (attacker,target,hit)，就记录多少；不补充、删除或重新决定资格。每条 fire_attempt 保存 step/time_s、side、完整双方状态、start/end，以及 authoritative engagement_geometry 全部字段：distance、horizontal_distance、line_of_sight、ata、ha、aa、hca、off_boresight、boresight_azimuth_error、boresight_elevation_error。当前 PaperATAHAWeapon 另存 range_pass/ata_pass/ha_pass，body Weapon 存其 body pass flags。这些仅用于诊断，绝不影响环境。

noncombat_loss 保存 side/agent/cause/state/time_s 及 horizontal_exceeded、ceiling_exceeded、ground_exceeded。分类依据原 resolver 实际返回桶及计数：当前 v3.37 ground 优先；boundary 桶中原计数标记为 ceiling 的超高状态归为 ceiling（即使同时超水平边界），否则是 horizontal_boundary。原 boundary_exits 仍包含 ceiling。历史版本水平优先于 ground 时仍按原桶；Blue 被原环境返场但未损失时不虚构 loss。无法确认时标为 unclassified。

## 播放时间

正式默认 stride=1，MP4 fps=1/dt（dt=.1 时10fps）；--playback-speed 倍率相应改变 fps。保留 reset/末状态，编码时长约 T*dt，端点保持最多一个输出 frame 的误差。
HTML 不再为 event 插额外 frame。内部 stride>1 preview 保持规则采样，只补最终状态，后一个 sampled frame 展示前一区间内全部事件，并明确标为 downsampled preview。轻量 scheduler 按相邻实际 time_s 差值调度，包括不足一个 stride 的最后区间；暂停/拖动/倍速不增加 event 保持时间。默认 stride=1 可逐 step 查看事件。浏览器性能会影响实际渲染速度，标称时间表不变。

## 模块和历史范围

唯一正式用户 CLI 为 tools/replay_combat.py。record_combat_episode.py、combat_visualization.py、combat_trace.py、combat_replay_report.py、combat_video.py 和两个 render 模块均为内部实现，不再提供独立正式 CLI。
create_combat_v24_videos.py 与 render_combat_replays_realtime.py 保留为 LEGACY v2.4/8v8 实验；旧 off_boresight/target_aspect 审计有 v2.4 guard，不能用于当前环境，当前入口也不 import 该逻辑。
报告只描述位置、轨迹、真实事件、目标绑定、计数和最终状态，不自动推断诱敌、包夹或战术意图。单局录像用于科研审计，不能代替正式 evaluator 的多 seed 统计评估。
