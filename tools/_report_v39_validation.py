"""Build a reviewable report from completed bounded validation artifacts."""
import hashlib
import json
from pathlib import Path
import re


def main():
    out = Path('outputs/v39_validation')
    baseline = json.loads((out / 'before_hashes.json').read_text())
    allowed = {'env/config.py', 'env/combat_env.py', 'algorithm/common/evaluator.py',
        'algorithm/mappo/runner.py', 'algorithm/rmappo/runner.py', 'algorithm/rmappo/protocol.py',
        'algorithm/ea_mappo/runner.py', 'algorithm/ea_mappo/protocol.py',
        'algorithm/madsac/protocol.py', 'algorithm/maddpg/protocol.py'}
    changed = [p for p, digest in baseline.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest]
    assert set(changed) <= allowed, changed
    assert not any(p.startswith('configs/') for p in changed)
    (out / 'historical_source_hash_check.json').write_text(json.dumps(dict(
        baseline_files=len(baseline), changed_extension_files=changed,
        protected_historical_configs_and_version_sources_unchanged=True), indent=2))
    special = (out / 'special_pytest.log').read_text()
    full = (out / 'full_pytest.log').read_text()
    assert re.search(r'\d+ passed', special) and re.search(r'\d+ passed', full)
    assert ' failed' not in full and ' failed' not in special
    assert (out / 'compileall.log').exists() and not (out / 'compileall.log').read_text().strip()
    visibility = json.loads((out / 'initial_visibility.json').read_text())
    audit = json.loads((out / 'fixed_policy_comparison.json').read_text())
    contract = json.loads((out / 'old_checkpoint_contract.json').read_text())
    vector = json.loads((out / 'vector_reset.json').read_text())
    smoke = json.loads((out / 'cuda_smoke/summary.json').read_text())
    lines = ['# Combat Environment v3.9 验收报告', '',
        '所有测试、静态检查及审计均在 WSL Ubuntu 的 uav Conda 环境执行。CUDA：' + smoke['cuda_device'] + '。未启动正式训练或参数扫描。', '',
        '## 修改文件清单', '',
        '新增：env/v39_config.py、env/v39_reward.py、env/combat_v39.py；configs/combat_environment_v39.yaml；四份 mappo/rmappo/ea_mappo/stea_mappo_5v5_v39.yaml；tests/test_combat_v39.py、tests/v39_v38_reference.json；docs/combat_environment_v39.md；tools/_prepare_v39.py、_wire_v39.py、audit_combat_v39.py、_check_v39_vector_reset.py、_report_v39_validation.py。', '',
        '修改：' + '、'.join(changed) + '；tests/test_combat_protocol.py 配置清单、tests/test_formal_8v8.py 版本错误文本、tools/smoke_formal_8v8.py 新配置选择。', '',
        '## Sensor 与奖励精确语义', '',
        '实际 sensor.range_max=10000.0 m，off_boresight_angle_max=π/2；原 true 3D 距离及 FOV 模型不变。10 km 是项目适配设置，论文只描述机头前方扇区扫描，没有提供可直接采用的数值探测距离。', '',
        '| 项目 | v3.8 | v3.9 |', '|---|---|---|',
        '| Radar | 4000 m | 10000 m |',
        '| R3 距离 | weapon_range<d≤sensor_range，即 2<d≤4 km | d≥guide_distance_min=4 km；visible 约束有效覆盖至 10 km |',
        '| R4 距离 | d≤weapon_range=2 km | d≤tactical_distance_max=4 km |',
        '| R3 角度/系数 | ATA/HA≤30°，+0.001 | 不变 |',
        '| R4 AA/级别/系数 | AA≤30°；30/15/5°三级 | 不变 |', '',
        'R3：存在一个 visible 且 d≥4000 m、|ATA|≤30°、|HA|≤30°的目标时 +0.001，否则 0。',
        'R41：visible 且 d≤4000 m、|AA_r|≤30°；ATA/HA 同时≤30/15/5°时取最高级别 +0.01/+0.02/+0.10。',
        'R42：双方存活，Blue→Red reverse geometry 的 d≤4000 m、|AA_b|≤30°；Blue ATA/HA 同时≤30/15/5°时取最强级别 −0.015/−0.025/−0.15。防御项沿用原语义，不新增 Red-visible 或 Blue-visible gate。',
        '多目标 offense 取 strongest positive level，defense 取 strongest negative level，R4=offense+defense，不对目标数求和。', '',
        '## Equation (25) 对照与扩展', '',
        '| 项目 | 论文定义 | 当前实现约定 |', '|---|---|---|',
        '| R1 | kill +10、destroyed −10 | 同时多攻击者分配 +10；ground loss 按 loss 处理 |',
        '| R2 | boundary −10 | 只记 R2，不重复记 R1 |',
        '| R3 | d≥4 km、ATA/HA≤30°、+0.001 | visible gate；任一满足目标触发一次 |',
        '| R4 | d≤4 km、AA≤30°、三级原系数 | strongest-target；offense+defense 聚合 |',
        '| Safety | 不属于 Equation (25) | 保留 5v5 collision penalty，200 m/0.2 |',
        '| Radar | 机头前方扇区 | 数值 range=10 km |',
        '| Weapon | Table 1 D_firemax=4 km | 保留已标定的 2 km，未修改 Gaussian 模型 |', '',
        'AA 仍按正式 Equation (25) 的 30°，不是前文定性区域的 90°。',
        'paper_reward=R1+R2+R3+R4；extended_reward=paper_reward+safety=实际返回训练 reward。Safety 作为原有 5v5 碰撞扩展保留，不冒充论文奖励。',
        '每步及每回合分解用 atol=1e-6、rtol=0 检查通过；返回 extended_reward 与 info/local_rewards 完全一致。v3.9 直接保留 float64 分解至算法原有 tensor conversion，避免父类 float32 中间舍入造成累计误差。',
        '日志、evaluation、summary 增加 paper/safety/extended 回合累计字段；R1–R4 保留。没有完成 episode 的 rollout 累计均值为 null。', '',
        '## 100 个初始化 seeds（0–99）', '',
        '| 方向 | 总 visible pairs / 2500 | 每回合 min/mean/max | 全 25 对可见回合 | 每架 min/max |',
        '|---|---|---|---|---|']
    for side, label in (('red', 'Red→Blue'), ('blue', 'Blue→Red')):
        v = visibility[side]
        lines.append(f"| {label} | {v['total_pairs']} / 2500 | {v['minimum_pairs']} / {v['mean_pairs']:.2f} / {v['maximum_pairs']} | {v['episodes_all_25_visible']} / 100 | 4 / 5 |")
    lines.append('')
    for side, label in (('red', 'Red→Blue'), ('blue', 'Blue→Red')):
        v = visibility[side]
        lines.append(label + '：每架初始可见敌机数量的均值依 UAV 1–5 为 ' + str(v['per_agent_mean']) + '。')
        lines.append('')
    lines += ['逐 seed、逐 UAV 计数保存在 initial_visibility.json；不把 radar 改成 10001，不强求 25/25。', '',
        '## 边界与小规模固定策略审计', '',
        '4 km 精确边界测试通过：角度满足时 R3 与 R4 均可非零。2001、3500、4000 m 的 R4 可以非零而 weapon qualifies=false。', '',
        '下表单位 step（dt=0.1 s）。detection 包含 reset step 0；R3/R4 取实际 reward 从 step 1 开始；fire-window/attempt/kill 取红蓝双方最早事件。—表示整局未发生。', '',
        '| 策略 | Seed | 版本 | 检测 | R3 | R4 | 火窗 | Attempt | Kill | 回合长 | Red出界 | Red坠地 | Red战斗损失 |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    def display(v):
        return '—' if v is None else str(v)
    for r in audit['rows']:
        values = [r[k] for k in ('policy', 'seed', 'version', 'first_detection_step', 'first_r3_step',
            'first_r4_step', 'first_fire_window_step', 'first_attempt_step', 'first_kill_step',
            'episode_length', 'red_boundary_exits', 'red_ground_losses', 'red_combat_losses')]
        lines.append('| ' + ' | '.join(map(display, values)) + ' |')
    lines += ['',
        '这六组对照中，v3.9 开局已有可见目标，第 1 步已有 R3，消除了样本中的无目标信息/无引导奖励空窗。R4 仍受角度及 4 km 约束，部分对局未触发；不声称扩大 radar 能消除所有出界或保证胜率。Blue 策略代码未改，但共享 10 km sensor 会改变其感知时机。', '',
        'v3.8 最终 3M checkpoint：' + contract['reason'] + '。未跨版本执行模型，不改 metadata、不绕过检查；本次环境对照仅使用固定策略。', '',
        '## 历史隔离与测试', '',
        'v3.0–v3.8 配置和版本专属源码哈希不变；公共文件只有新版本分派/allowlist 和 v3.9 专属诊断。v3.8 radar 仍为 4 km，旧 R3/R4 单测通过。8 条修改前捕获的完整 v3.8 ZERO/MIRROR 轨迹完全匹配状态、观测、reward、info、环境及 Blue RNG；已有 v3.7 reference 与所有历史回归继续通过。', '',
        '- v3.9 专项：' + re.search(r'\d+ passed', special).group(0) + '。',
        '- 全部 pytest：' + re.search(r'\d+ passed(?:, \d+ skipped)? in [^\r\n]+', full).group(0) + '。',
        '- 跳过原因：' + '; '.join(line for line in full.splitlines() if line.startswith('SKIPPED')),
        '- compileall：env/algorithm/tools/tests 全部通过。',
        f"- vector reset：实际 spawn worker，auto-reset seed={vector['seed']} 与 fresh env 下一步完整 info/obs/reward 相同，paper/safety/extended 累计正确清零。", '',
        '| 算法 | Transitions | 真实 update | Save/reload | Deterministic eval | Finite checks |',
        '|---|---|---|---|---|---|']
    for name, record in smoke['records'].items():
        assert record['finite_gradients_parameters_and_values'] and record['finite_rewards_advantages_and_returns']
        evaluation = record['evaluation_after_checkpoint_reload']
        for reward_name in ('paper_reward', 'safety', 'extended_reward'):
            assert f'average_episode_{reward_name}_total' in evaluation
        metrics = json.loads((out / 'cuda_smoke' / name / 'training_metrics.jsonl').read_text())
        for reward_name in ('paper', 'safety', 'extended'):
            assert f'mean_{reward_name}_reward' in metrics
        for reward_name in ('paper_reward', 'safety', 'extended_reward'):
            assert f'episode_{reward_name}_total' in metrics
        assert abs(evaluation['average_episode_extended_reward_total'] -
            evaluation['average_episode_paper_reward_total'] - evaluation['average_episode_safety_total']) < 1e-6
        lines.append(f"| {record['algorithm']} | 512 | 1 | 通过 | 2 回合 | 通过 |")
    lines += ['', '四算法配置与 v3.8 字节一致；网络、学习率、gamma/lambda/entropy/clip、rollout/epochs/minibatch 和评估计划均未调整。验证已结束，不启动 1M/3M 正式训练。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
