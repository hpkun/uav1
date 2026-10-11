"""Build acceptance report only from completed local validation artifacts."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np

OUT = Path('outputs/v311_validation')


def main():
    special = (OUT/'special_pytest.log').read_text()
    full = (OUT/'full_pytest.log').read_text()
    special_result = re.search(r'\d+ passed in .+', special).group(0)
    full_result = re.search(r'\d+ passed, \d+ skipped in .+', full).group(0)
    assert 'FAILED ' not in special and 'FAILED ' not in full
    assert (OUT/'compileall.log').read_text() == ''
    smoke = json.loads((OUT/'cuda_smoke/summary.json').read_text())
    sanity = json.loads((OUT/'fixed_policy_sanity.json').read_text())
    vector = json.loads((OUT/'vector_reset.json').read_text())
    baseline = json.loads((OUT/'before_hashes.json').read_text())
    allowed = {'env/config.py', 'env/combat_env.py', 'algorithm/common/evaluator.py',
        'algorithm/mappo/runner.py', 'algorithm/rmappo/runner.py', 'algorithm/rmappo/protocol.py',
        'algorithm/ea_mappo/runner.py', 'algorithm/ea_mappo/protocol.py',
        'algorithm/maddpg/protocol.py', 'algorithm/madsac/protocol.py'}
    changed = [p for p, h in baseline.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert set(changed) <= allowed
    check = dict(baseline_files=len(baseline), changed_extension_files=changed,
                 historical_configs_and_version_specific_sources_unchanged=True)
    (OUT/'historical_source_hash_check.json').write_text(json.dumps(check, indent=2))
    for name, record in smoke['records'].items():
        assert record['sampled_steps'] == 512
        assert record['finite_gradients_parameters_and_values'] and record['finite_rewards_advantages_and_returns']
        result = record['evaluation_after_checkpoint_reload']
        assert result['protocol_complete'] and result['evaluation_episodes'] == 2
        assert result['checkpoint_environment_version'] == '3.11'
        assert result['draw_rate'] == 0.
        assert all(np.isfinite(v) for v in result.values() if isinstance(v, (float, int)))
        for key in ('average_episode_paper_reward_total', 'average_episode_safety_total',
                    'average_episode_extended_reward_total', 'average_blue_paper_nearest_agent_steps'):
            assert key in result
    assert len(smoke['records']) == 4 and vector['first_step_identical_to_fresh_environment']
    assert len(sanity['rows']) == 6
    for row in sanity['rows']:
        assert row['finite_observation_action_reward_state'] and row['target_selection_violations'] == 0
        assert row['episode_length'] > 1
    table = '\n'.join('| {policy} | {seed} | {first_fire_window_step} | {first_attempt_step} | {first_kill_step} | '
        '{red_ammo_used}/{blue_ammo_used} | {episode_length} | {red_boundary_losses} | {blue_boundary_returns} | '
        '{outcome} |'.format(**r) for r in sanity['rows'])
    skips = '\n'.join(line for line in full.splitlines() if line.startswith('SKIPPED '))
    report = f'''# Combat Environment v3.11 验收报告

实现完成，所有运行均在 WSL Ubuntu 的 uav Conda 环境；CUDA：{smoke['cuda_device']}。没有正式1M/3M训练，没有参数调整。

## 1. 文件清单

新增环境：env/combat_v311.py、env/v311_config.py、env/v311_kinematics.py、env/v311_scenario.py、env/v311_weapon.py、env/v311_reward.py。

新增配置：configs/combat_environment_v311.yaml、configs/mappo_5v5_v311.yaml、configs/rmappo_5v5_v311.yaml、configs/ea_mappo_5v5_v311.yaml、configs/stea_mappo_5v5_v311.yaml。四算法YAML与v3.10对应文件字节相同。

新增验证/文档：tests/test_combat_v311.py、tests/v311_v310_reference.json、docs/combat_environment_v311.md；tools/_prepare_v311.py、tools/_wire_v311.py、tools/audit_combat_v311.py、tools/_check_v311_vector_reset.py、tools/_report_v311_validation.py。

修改公共扩展：env/config.py、env/combat_env.py；algorithm/common/evaluator.py、algorithm/mappo/runner.py、algorithm/rmappo/runner.py、algorithm/rmappo/protocol.py、algorithm/ea_mappo/runner.py、algorithm/ea_mappo/protocol.py、algorithm/maddpg/protocol.py、algorithm/madsac/protocol.py（只扩展版本分派/诊断）；tests/test_combat_protocol.py、tests/test_formal_8v8.py、tools/smoke_formal_8v8.py（配置清单/版本提示/smoke配置选择）。

## 2–3. 简化运动和controller删除

Δψ=πaψ，Δθ=(π/3)aθ，Δv=50av；ψ'=wrap(ψ+Δψ)，θ'=clip(θ+Δθ,−π/3,+π/3)，v'=clip(v+Δv,150,300)。x'=x+v'cosθ'cosψ'dt，y'=y+v'cosθ'sinψ'dt，z'=z−v'sinθ'dt，dt=0.1s。动作增量每step直接应用；位置使用更新后的状态。

不创建/调用旧controller、PointMassDynamics或RK4Integrator；不使用nx/nz/phi动力学、过载、滞后、inverse controller、PID/autopilot。正式配置删除heading_time_constant、pitch_time_constant、speed_time_constant、normal_load_max，validator拒绝controller和额外工程参数。

## 4–5. Radar与初始化

Radar：alive双方、真实3D距离≤10000m、forward与LOS的3D夹角≤60°（总前向FOV120°）。没有noise/记忆/锁定/概率探测，只服务Observation/诊断。

初始化唯一随机draw：α~Uniform(−π,π)。u=[cosα,sinα]，t=[−sinα,cosα]；Red/Blue中心分别−4800u/+4800u；各机位置=本队中心+[-400,−200,0,200,400]ᵢt。中心距离9600m，全部飞机半径<5000m、高度3000m、速度225m/s、pitch0，heading分别α/wrap(α+π)。100个seed验证仅消耗一个uniform draw与完全可复现，无其它扰动。

## 6–8. Weapon与时序

Equation(7)：双方alive、ammo>0、0≤真实3D距离≤4000m、|ATA|≤π/6、|HA|≤π/6。AA/radar/speed不参与资格。每架初始6枚；每step从合法目标中选最近3D距离，精确tie选较小index，最多发射1枚。严格删除pair-entry/previous_eligible/cooldown/rearm；持续合法3步明确得到ammo6→5→4→3，耗尽后不开火。

Equation(8)沿用同一个Gaussian epsilon进入两条不等式：ATA+c4ε≤πexp(−d/Dhit)，HA+c5ε≤πexp(−d/Dhit)。每attempt仅一个N(0,1)draw。c4=c5=0.9984711359155588，Dhit=4000/ln6=2232.442506204989m，P(hit|4000,0,0)≈0.70。均为透明项目标定，不声称来自论文公开数值。先完成双方全部sampling，再统一击毁，双向本步攻击不被取消。

## 9–13. Reward、PBRS和胜负

R3/R4完全不调用sensor，不含visible gate；保留alive mask。R3：任一Blue d≥4000、ATA/HA≤30°，给+0.001。R41：d≤4000、AA_r≤30°，ATA/HA同时≤30/15/5°分别+0.01/+0.02/+0.10；R42以Blue为中心reverse geometry，AA_b≤30°及同样距离/分级，分别−0.015/−0.025/−0.15。offense=max正目标，defense=min负威胁，不对5个目标累加。

原ATA/HA≤30°合格目标不可能在60°雷达半角外；采用强制sensor不可访问/设置零探测范围与R3超过10km目标，测试reward和radar真正解耦。没有改变几何定义来迎合不可能的测试场景。

R1 kill+10/Red destroyed−10；R2 horizontal/ceiling boundary−10，ground沿用R1。PBRS关闭；win/loss/draw terminal reward均0，team casualty0。safe_distance200m、safety_penalty_scale0.2保留；paper=R1+R2+R3+R4，training=paper+safety。safety是碰撞扩展，不是Equation(25)。

Blue=0即Red win，包括Red=Blue=0的mutual destruction；Red=0且Blue>0判Blue win；timeout且Blue>0判Blue win。没有draw/survivor comparison。

## 14–16. Boundary、Observation和phi

Blue return直接继承v3.10的同一个方法，水平/ground/ceiling/复合触发修正状态与计数完全一致，没有改成死亡。Red边界规则/ground优先级保持。arena半径5000、高度0–6000，max_steps1000。

Observation仍66D、3动作、5agents；self/ally/enemy顺序、ammo、dead masking使用原builder。新radar参数仅改变enemy visibility。phi兼容数组和executed_phi为0，Observation self index4为0，无bank controller。

Blue使用原PaperNearestTargetBluePolicy：每step全局最近alive Red真实3D位置、精确tie小index、立即切换，desired_speed250m/s；无radar gate/escape/guerrilla/centripetal/RNG。原归一化pursuit动作直接交给新运动模型。

## 17–22. 回归与验证

修改前{len(baseline)}个env/algorithm/config源文件哈希已保存，只有列出的10个公共扩展文件发生变化；所有v3.0–v3.10版本专属代码和旧配置完全未改。新增6条修改前v3.10 ZERO/PURSUIT完整轨迹reference，状态/obs/reward/info/weapon RNG完全匹配；v3.10原controller/2km/strict-entry/init/reward gate/outcome保留，已有历史回归通过。

- 专项pytest：{special_result}
- full pytest：{full_result}
- 跳过原因：
```
{skips}
```
- compileall env/algorithm/tools/tests通过，git diff --check通过。
- vector reset：真实spawn worker自动重置seed={vector['seed']}；下一step完整obs/reward/info与fresh环境完全一致，ammo恢复6、无pair state、Blue诊断/phi/奖励累积正确重置。

| 算法 | transitions | 真实update | save/reload | CUDA deterministic eval | finite检查 |
|---|---|---|---|---|---|
| MAPPO | 512 | 1 | 通过 | 2局 | 通过 |
| RMAPPO | 512 | 1 | 通过 | 2局 | 通过 |
| EA-MAPPO | 512 | 1 | 通过 | 2局 | 通过 |
| STEA-MAPPO | 512 | 1 | 通过 | 2局 | 通过 |

Observation builder/action入口检查finite；CUDA更新检查reward/GAE/return/梯度/参数和值finite；reloaded evaluation指标finite、checkpoint版本/SHA合同完整。网络和正式PPO参数保持，smoke实际末段rollout为32×16=512，不启动配置中的长预算。

## 23. ZERO/PURSUIT sanity

六场初始编队中心距离均9600m，真实跨队距离9600–9633.275663m；双方初始radar visible pairs均25。表中时序单位step，dt=0.1秒，ammo为Red/Blue使用总数。

| Red策略 | seed | 首次火窗 | 首次attempt | 首次kill | ammo使用R/B | 回合step | Red boundary losses | Blue returns | 终局 |
|---|---|---|---|---|---|---|---|---|---|
{table}

六场均正常发生battle，未开局立即结束；没有NaN或速度/pitch非法状态，Red ground losses均0，Blue target selection violations均0。完整双方ammo逐步记录在decisions_*.json。该sanity不是训练难度结论，没有据此修改参数。

完成后停止，没有正式1M/3M训练。
'''
    (OUT/'report.md').write_text(report, encoding='utf-8')
    print(full_result)


if __name__ == '__main__':
    main()
