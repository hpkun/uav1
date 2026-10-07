# Combat episode recording

`RecordingCombatEnv` 继承 `MultiUAVCombatEnv`。它调用原环境的事件方法并记录返回结果，不增加随机数采样、不改变 step 顺序和武器结算。

`record_combat_episode.py` 根据 checkpoint 的算法标识加载 MAPPO 或 MADSAC；默认使用正式环境配置，可通过 `--algorithm-config` 与 `--env-config` 指定运行快照。Checkpoint recording 要求 CUDA。策略以 deterministic 模式部署。

## Trace schema 2

`episode_trace.npz` 只包含数值数组，加载时禁止 pickle。

| Field | Shape | Meaning |
|---|---|---|
| red_kinematics / blue_kinematics | `(T+1,4,6)` | NED `x,y,z,v,theta,psi` |
| red_alive / blue_alive | `(T+1,4)` | 存活状态 |
| steps / time_s | `(T+1,)` | 初始帧与每次 step 结束的时间 |
| red_actions | `(T,4,3)` | 裁剪后执行动作 |
| local_rewards | `(T,4)` | 各 Red agent 奖励 |
| reward_components | `(T,4,4)` | transition、R1–R4、agent |
| terminated / truncated | `(T,)` | episode 结束类型 |

`metadata.json` 包含环境与算法配置指纹、checkpoint SHA-256、episode seed、双方存活/损失、回报和终止原因。`events` 保存真实射击尝试的 attacker/target、hit 与起止坐标；击杀事件保存共同攻击者列表；越界和撞地事件保存实体索引。事件索引从 0 开始，画面标签从 1 开始。

## Rendering

`render_combat_episode.py` 支持 PNG、GIF 和 MP4。展示轨迹、终端存活/死亡状态、射击连线及时间。NED 的 `z` 转为 altitude=`-z`。GIF 使用 Pillow；MP4 使用 FFmpeg。

`render_combat_episode_interactive.py` 生成内嵌 Plotly 的独立 HTML，支持 Play/Pause、时间滑块、camera rotation 和 hover 中的 alive、speed、pitch、heading。绿色射击线表示命中，橙色表示未命中。帧采样保留每个事件 step；已死亡飞机的轨迹在死亡位置结束。

记录输出必须写入空目录。采用正式 checkpoint 做定量评估时，使用独立的 evaluate 入口及一致的配置/seed 协议。
