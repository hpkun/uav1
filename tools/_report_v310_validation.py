"""Reviewable report from completed bounded v3.10 validation artifacts."""
import hashlib
import json
from pathlib import Path
import re


def main():
    out = Path('outputs/v310_validation')
    before = json.loads((out / 'before_hashes.json').read_text())
    allowed = {'env/config.py', 'env/combat_env.py', 'algorithm/common/evaluator.py',
        'algorithm/mappo/runner.py', 'algorithm/ea_mappo/protocol.py', 'algorithm/ea_mappo/runner.py',
        'algorithm/rmappo/protocol.py', 'algorithm/rmappo/runner.py',
        'algorithm/maddpg/protocol.py', 'algorithm/madsac/protocol.py'}
    changed = [p for p, digest in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
    assert set(changed) <= allowed, changed
    assert not any(p.startswith('configs/') for p in changed)
    (out / 'historical_source_hash_check.json').write_text(json.dumps(dict(
        baseline_files=len(before), changed_extension_files=changed,
        historical_configs_and_version_specific_sources_unchanged=True), indent=2))
    special = (out / 'special_pytest.log').read_text()
    full = (out / 'full_pytest.log').read_text()
    assert re.search(r'\d+ passed', special) and re.search(r'\d+ passed', full)
    assert ' failed' not in special and ' failed' not in full
    assert (out / 'compileall.log').exists() and not (out / 'compileall.log').read_text().strip()
    sanity = json.loads((out / 'fixed_policy_sanity.json').read_text())
    vector = json.loads((out / 'vector_reset.json').read_text())
    smoke = json.loads((out / 'cuda_smoke/summary.json').read_text())
    lines = ['# Combat Environment v3.10 验收报告', '',
        '全部运行在 WSL Ubuntu 的 uav Conda 环境。CUDA：' + smoke['cuda_device'] + '。没有对照实验、参数扫描或正式1M/3M训练。', '',
        '## 文件清单', '',
        '新增：env/v310_policy.py、env/v310_config.py、env/combat_v310.py；configs/combat_environment_v310.yaml；configs/mappo_5v5_v310.yaml、rmappo_5v5_v310.yaml、ea_mappo_5v5_v310.yaml、stea_mappo_5v5_v310.yaml；tests/test_combat_v310.py、tests/v310_v39_reference.json；docs/combat_environment_v310.md；tools/_prepare_v310.py、_wire_v310.py、audit_combat_v310.py、_check_v310_vector_reset.py、_report_v310_validation.py。', '',
        '修改：' + '、'.join(changed) + '；tests/test_combat_protocol.py 的 YAML 清单、tests/test_formal_8v8.py 的版本错误文本、tools/smoke_formal_8v8.py 的配置选择。', '',
        '## Blue 精确流程', '',
        '每个 simulation step 在 dynamics 前，对每架存活 Blue，从所有存活 Red 真实状态选择 `(d², index)` 最小者；d²=(xr−xb)²+(yr−yb)²+(zr−zb)²。距离 tie 选较小 index。每步重新计算，没有 target lock、迟滞、保持时间或概率切换；目标死亡后当前 step 重选。Blue死亡或无存活Red才返回zero action。', '',
        '直接复用 NearestTargetPursuitPolicy 的 nearest_target_index、action、action_toward；wrapper的team_actions调用原team_actions，只加diagnostics。100个随机合法状态场景的action与原策略逐值严格相同，target index也一致，强于atol≤1e−7。', '',
        'Blue目标选择不使用radar visible mask；FOV外、无可见目标、Blue人数劣势、Red高度分散都仍nearest pursuit。唯一正式决策为GLOBAL_NEAREST_PURSUIT。escape/guerrilla/centripetal/质心/RMS/random-target/close_distance/dispersion_threshold全部从v3.10正式决策与配置移除。', '',
        'desired_speed=250 m/s：当前场景固定策略实现参数，不是论文给出的唯一巡航速度。NED elevation与heading/controller完全复用原实现。论文Section2.5描述每步全Red位置信息、最近追击、最近改变就切换、attack phase继续追击、attack model决定开火命中；3D距离/tie/250m/s/连续控制是项目实现约定。', '',
        '新Blue policy无rng成员；reset(seed)不创建或消耗随机数。父类临时构造旧对象被替换后不参与任何step；运行时只有新policy。相同seed与前5步相同Red actions的Blue actions/轨迹完全可复现，初始化RNG状态保持；显式Blue override下双方真实Gaussian attempt与命中/环境RNG和v3.9完全一致。', '',
        '## 诊断与sanity', '',
        'info保留blue_paper_nearest_agent_steps、blue_target_switches、blue_last_target_indices；正式训练/evaluation记录平均pursuit agent-steps和switches，不再输出不存在的escape/guerrilla/centripetal统计。首次获取和死亡变空不计switch，两个非空index之间改变才计switch；诊断不参与action。', '',
        '仅运行v3.10 ZERO/PURSUIT，每种3个固定seeds。逐Blue agent-step检查使用该步dynamics前所有alive Red的真实3D几何；未用旧版本做对照实验。', '',
        '| Red策略 | Seed | Blue switches | Pursuit agent-steps | 首次火窗 | 首次attempt | 首次kill | 回合长 | Blue末次target indices |',
        '|---|---|---|---|---|---|---|---|---|']
    for row in sanity['rows']:
        assert row['target_selection_violations'] == 0
        values = [row[k] for k in ('policy', 'seed', 'blue_target_switches', 'blue_paper_nearest_agent_steps',
            'first_fire_window_step', 'first_attempt_step', 'first_kill_step', 'episode_length', 'blue_last_target_indices')]
        lines.append('| ' + ' | '.join(str(v) for v in values) + ' |')
    checked = sum(r['episode_length'] * 5 for r in sanity['rows'])
    pursued = sum(r['blue_paper_nearest_agent_steps'] for r in sanity['rows'])
    lines += ['', f'共检查 {checked} 个Blue agent-step（包括死亡/无目标的空选择），其中 {pursued} 个存活且有目标的pursuit decisions；target selection violations=0。',
        '时序单位step，dt=0.1秒。完整每一步selected indices与switches保存在blue_decisions_*.json；末次indices对应终局step开始时状态，不是终局命中后状态。六局均Blue获胜，只用于行为sanity，不用于难度或训练结论，不根据结果调参。', '',
        '## 冻结项逐项验收', '',
        '| 项目 | v3.10实际设置与验收 |', '|---|---|',
        '| Reward | v3.9 R1/R2/R3/R4/safety/paper/extended全部继承；guide/tactical距离4000m，代码未改 |',
        '| Sensor | v3.9 10000m / FOV π/2，完全未改；仅Blue目标选择不以它过滤 |',
        '| Weapon | 2000m、ATA/HA30°、ammo6、Gaussian Dhit/c4/c5、pair-entry、同时结算完全未改 |',
        '| Initialization | 原chord 10km diameter、lateral/altitude/speed/heading/RNG完全未改 |',
        '| Blue boundary | 原强制水平/高度return方法本身继承，四种return原因单测与v3.9状态/计数一致 |',
        '| Red observation/action | 66/3/5；原Red sensor-limited observation，无额外全局Blue信息 |',
        '| Dynamics/controller/termination | 方法全部继承；dt0.1、max_steps1000、arena5000m/0–6000m，与v3.9配置相同 |',
        '| Algorithms | 四份YAML与v3.9字节相同；网络/GRU/attention/所有PPO参数与评估/保存计划未改 |', '',
        'CombatEnvironmentV310只定义version和__init__替换fixed_policy；没有复制step。validator只接受desired_speed=250，拒绝close_distance/dispersion_threshold及未定义行为字段。MAPPO/RMAPPO/EA/STEA及MADDPG/MADSAC allowlists均支持3.10；checkpoint仍检查环境版本和SHA。', '',
        '## 历史与测试', '',
        'v3.0–v3.9配置和版本专属源码哈希全部未改；v3.9仍为PaperTieredBluePolicy并保留escape/guerrilla/centripetal。新增8条修改前完整v3.9ZERO/MIRROR轨迹reference，状态、观测、reward、info、初始化/weapon RNG和Blue独立RNG均完全匹配；已有v3.7/v3.8等历史reference继续通过。', '',
        '- v3.10专项：' + re.search(r'\d+ passed', special).group(0) + '。',
        '- full pytest：' + re.search(r'\d+ passed(?:, \d+ skipped)? in [^\r\n]+', full).group(0) + '。',
        '- 跳过原因：' + '; '.join(line for line in full.splitlines() if line.startswith('SKIPPED')),
        '- compileall：env/algorithm/tools/tests通过。',
        f"- vector reset：真实spawn worker，auto-reset seed={vector['seed']}，下一步完整obs/reward/info与fresh环境一致，Blue targets/switches/pursuit计数及paper/safety/extended累计清零。", '',
        '| 算法 | Transitions | 真实update | Save/reload | Deterministic eval | Finite checks |',
        '|---|---|---|---|---|---|']
    for name, row in smoke['records'].items():
        assert row['finite_gradients_parameters_and_values'] and row['finite_rewards_advantages_and_returns']
        assert row['sampled_steps'] == 512
        evaluation = row['evaluation_after_checkpoint_reload']
        assert evaluation['evaluation_episodes'] == 2 and evaluation['protocol_complete']
        assert evaluation['checkpoint_environment_version'] == '3.10'
        for field in ('average_blue_paper_nearest_agent_steps', 'average_blue_target_switches',
            'average_episode_paper_reward_total', 'average_episode_extended_reward_total'):
            assert field in evaluation
        assert not any(word in key for key in evaluation for word in ('escape', 'guerrilla', 'centripetal'))
        training = json.loads((out / 'cuda_smoke' / name / 'training_metrics.jsonl').read_text())
        assert 'average_blue_target_switches' in training and 'average_blue_paper_nearest_agent_steps' in training
        assert not any(word in key for key in training for word in ('escape', 'guerrilla', 'centripetal'))
        lines.append(f"| {row['algorithm']} | 512 | 1 | 通过 | 2回合 | 通过 |")
    lines += ['', '验证完成即停止；没有进行正式1M/3M训练或版本对照实验。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
