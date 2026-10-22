# v3.44 连续驻留基线

v3.44 保留原有 4v4、NED 点质点运动、初始场景、Blue 最近目标控制器、49D 观测、3D 动作和终止规则。相对 v3.43，事件机制改为距离 0–2000 m、真实三维半角 30° 区域内连续 0.5 秒驻留后确定性失效；dt=0.1 秒，对应连续 5 次有效采样。旧资源与冷却限制不适用。

`env/v344_dwell.py` 的 `DwellEnvelope` / `DwellState` 是唯一权威实现。Red 累计当前最近合法目标；Blue 累计 step 开始确定的追击目标。离区、切换、死亡与回合结束使驻留段失效。双方事件先收集、后同步结算；多人完成同一目标时 +10 按既有规则分摊。完成次数与去重后的实际失效目标数分别记录。

每步先运动、非战斗死亡，再计算 R4，然后同步交战，最后计算事件奖励及 PBRS 下一势函数。死亡所在 transition 可保留交战前 R4；已经死亡的实体不再产生新 R4。终止和截断的 shaping 下一势函数归零。

v3.44 的 R1/R2/PBRS R3/连续 R4 完全继承 v3.43。历史配置、`env/combat_v344.py`、`env/v344_config.py`、`env/v344_dwell.py`、`env/v343_reward.py` 均未被本次奖励实验修改。共享注册器更新不会改写历史配置指纹。

`event_semantics=continuous_dwell_neutralization`；fire_attempts、weapon_hits、ammo_used 为不适用的零兼容占位，相应首次事件为 None。录像使用 `dwell_complete` 和 `attack_kill`，不伪造发射或概率命中。回放工具按事件语义识别驻留版本。

原 45° v3.45 实验于 2026-10-22 从可执行代码与配置中退役；历史模型、结果和录像保留，不允许换版本身份加载。新奖励对照见 [v3.46](combat_environment_v346.md)。旧实验记录仍在原 `outputs/` 目录，不能据其路径推断该环境仍受支持。

基线测试现为 `tests/test_combat_v344.py`，保留原几何、状态机、共享归属、同步失效、配置与回合重复性覆盖；适用的共同机制同时覆盖 v3.46。
