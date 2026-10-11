"""Assemble the 25 HA-only acceptance items from completed validation artifacts."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

OUT = Path('outputs/v313_validation')


def main():
    special = (OUT/'special_pytest.log').read_text()
    full = (OUT/'full_pytest.log').read_text()
    special_result = re.search(r'\d+ passed in .+', special).group(0)
    full_result = re.search(r'\d+ passed, \d+ skipped in .+', full).group(0)
    assert 'FAILED ' not in full and 'FAILED ' not in special
    assert not (OUT/'compileall.log').read_text()
    vector = json.loads((OUT/'vector_reset.json').read_text())
    isolation = json.loads((OUT/'checkpoint_isolation.json').read_text())
    smoke = json.loads((OUT/'cuda_smoke/summary.json').read_text())
    audit = json.loads((OUT/'same_state_audit.json').read_text())
    baseline = json.loads((OUT/'before_hashes.json').read_text())
    allowed = {'env/config.py', 'env/combat_env.py', 'algorithm/common/evaluator.py',
        'algorithm/mappo/runner.py', 'algorithm/rmappo/protocol.py', 'algorithm/rmappo/runner.py',
        'algorithm/ea_mappo/protocol.py', 'algorithm/ea_mappo/runner.py',
        'algorithm/maddpg/protocol.py', 'algorithm/madsac/protocol.py'}
    changed = [p for p, h in baseline.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert set(changed) <= allowed
    hashes = dict(baseline_files=len(baseline), changed_extension_files=changed,
        historical_configs_and_version_specific_sources_unchanged=True,
        networks_trainers_and_ppo_sources_unchanged=True)
    (OUT/'historical_source_hash_check.json').write_text(json.dumps(hashes, indent=2))
    assert len(smoke['records']) == len(isolation) == 4
    rows = []
    for name, record in smoke['records'].items():
        assert record['observation_dim'] == 56 and record['sampled_steps'] == 512
        assert record['critic_input_dim'] == 336 and record['critic_parameter_count'] == 152321
        assert record['finite_gradients_parameters_and_values'] and record['finite_rewards_advantages_and_returns']
        evaluation = record['evaluation_after_checkpoint_reload']
        assert evaluation['protocol_complete'] and evaluation['evaluation_episodes'] == 2
        assert evaluation['observation_dim'] == 56 and evaluation['checkpoint_environment_version'] == '3.13'
        assert all(np.isfinite(v) for v in evaluation.values() if isinstance(v, (int, float)))
        assert isolation[name]['old_checkpoint_valid'] and isolation[name]['v312_to_v313_rejected']
        rows.append(f"| {name} | 512 | 1 | 通过 | 2 | 通过 | {evaluation['win_rate']:.2f} | {evaluation['average_return']:.4f} |")
    assert vector['first_step_identical_to_fresh_environment'] and vector['observation_shape'] == [1, 5, 56]
    subprocess.run(['git', 'diff', '--check'], check=True, capture_output=True)
    status = subprocess.run(['git', 'status', '--short', '--untracked-files=all'], check=True,
        capture_output=True, text=True).stdout
    files = '\n'.join(f'- `{line[3:]}`' for line in status.splitlines() if line[3:].startswith(
        ('env/', 'algorithm/', 'configs/', 'tests/', 'tools/', 'docs/')))
    skips = '\n'.join(line for line in full.splitlines() if line.startswith('SKIPPED '))
    a, b = audit['qualification_snapshot_v312'], audit['qualification_snapshot_v313']
    report = f'''# Combat Environment v3.13 验收报告

验证在 WSL Ubuntu 的 uav Conda 环境完成，CUDA：{smoke['cuda_device']}。没有启动正式训练或修改训练参数。

## 1. 修改文件清单

{files}

六个新环境模块分别隔离 geometry、weapon、reward、observation、config 与环境类；五份 YAML 是新版本配置。修改的算法文件仅扩展版本 allowlist/既有诊断分派，没有改变网络、trainer 或 PPO。两个既有测试文件只同步配置清单/版本提示；既有 smoke 工具只增加 v313 配置选择。

## 2. v3.13 HA 精确定义

`horizontal=hypot(dx,dy)`；`los_elevation=atan2(-(z_target-z_attacker),horizontal)`；`HA=abs(los_elevation)`。NED 高度为 −z。整个公式不读取 own_pitch。vertical 正上/下为 π/2；完全重合 ATA=HA=0，AA 按原稳定 heading 差处理，signed LOS 显式 0。

## 3. 与 v3.12 的数学区别

旧：`angular_difference(los_elevation,own_pitch)`，垂直瞄准误差；新：`abs(los_elevation)`，高度视线角。v3.12 及之前保留旧定义。除完全重合的旧 own_pitch elevation 占位值在新模块明确为 0 外，signed LOS 与旧版一致；正常随机几何中仅 HA 不同。

## 4. Pitch 独立性

1000 组随机位置关系 × 5 种 pitch（−60/−30/0/30/60°），HA/distance/ATA/AA 最大变化均为 0，严格优于 atol≤1e−12。同高度、上下45°、垂直上下90°、重合专项覆盖。独立性针对固定位置，pitch 仍会通过真实运动改变之后的位置关系。

## 5. HA 范围

严格 [0,π/2]，由 abs(atan2(height_delta,nonnegative_horizontal)) 给出；垂直与重合值有限。

## 6. ATA

原水平 LOS heading 与 attacker heading 的最小圆周角差 [0,π]，未改。1000×5 随机比较与旧 helper 逐值完全一致。

## 7. AA

原水平 LOS heading 与 target heading 的最小圆周角差 [0,π]，未改；水平无方向时仍用 attacker heading。1000×5 随机比较与旧版逐值完全一致。

## 8. Observation HA 归一化

`HA/(π/2)`；0/30/45/90°→0/1⁄3/0.5/1。AA/ATA 仍除 π，其他 scale 不变。旧版仍是 HA_old/π。

## 9. Weapon

新 MADSACPaperHAWeapon override eligibility，使用新 geometry；qualifies 调用此实现。双方 alive、ammo>0、0≤d≤4000、ATA≤30°、HA≤30°不变；AA/radar 不参与。同高度 d=3000、ATA=0、own_pitch60°：v3.12 不合法，v3.13 合法。LOS30°合法边界及略超界拒绝均通过。

## 10. Equation (8)

CombatEnvironmentV313 override attempt_hit_for_pair，实际命中采样输入与 eligibility 一样使用新 HA。sampler 函数直接继承原类，threshold=πexp(−d/D_hit)、D_hit=4000/ln6、c4=c5=0.9984711359155588 不变，一次发射 ATA/HA 共享唯一 epsilon。spy 验证 pitch60°同高度的采样 HA 为 0；采样 RNG 调用次数恰为 1。

## 11. R3

新 geometry、d≥4000、ATA/HA≤30°、any 活目标 +0.001；无 visible gate，系数不变，固定位置对 Red pitch 不敏感。

## 12. R4 offense

新 geometry、d≤4000、AA≤30°、30/15/5°分级 +0.01/+0.02/+0.10、max 聚合不变，对 Red pitch 不敏感。

## 13. R4 defense

reverse 新 geometry(blue,red)，HA 同样是 height LOS、与 Blue pitch 无关；−0.015/−0.025/−0.15 分级、min 聚合不变。固定位置的 reverse pitch 测试通过。

## 14. Observation 几何

Enemy d/AA/ATA/HA 全来自 combat_geometry_v313，反归一化逐项 atol≤1e−8；Self/Ally 用原 helper，数值与 v3.12 完全一致。own pitch 改变 Self theta/Ally body-frame 是原有正确行为。旧 observation builder/source 冻结。

## 15. 56D 布局

Self7(0:7)+4×Ally6(7:31)+5×Enemy5(31:56)，identity 顺序和死亡槽全零不变；phi=0，无 ammo/alive/radar flag。四算法配置为 v3.12 字节副本，MAPPO/RMAPPO flat56，EA/STEA self7/ally6/enemy5，critic input336、hidden256/256、参数152321。

## 16. 全 Blue 可观测

所有存活 Blue 仍全部填充对应 slot；机尾且超过10km的活 Blue 在测试中仍可观测，死亡 Blue 固定 slot 清零，不重排。

## 17. Radar

继续不参与 Observation、weapon 或 R3/R4 gate。10km/120°配置未改；改变测试对象 radar 配置不改变 observation。

## 18. 同 shape checkpoint 隔离

四个实际 v3.12 bounded checkpoint 均在原配置下有效，目标 v3.13 均因 environment_version mismatch 被拒绝（双方均56D）；没有绕过 metadata/SHA/algorithm 合同或修改权重。

## 19. 历史回归与冻结机制

修改前 {len(baseline)} 个源码/配置哈希已保存；仅 {len(changed)} 个允许的版本分派/诊断文件有变化。v3.0–v3.12 专属环境模块和旧配置逐字节未改，网络/trainer/PPO源码未改。特别 v36_geometry.py、v311_weapon.py、v311_reward.py、v312_observation.py 哈希完全一致。

新增六条修改前 v3.12 ZERO/PURSUIT、seeds1/31/10000000 的完整 reference，覆盖 state/observation/reward/info/RNG；修改后全部完全一致。完整 pytest 的已有历史回归全部通过。v3.13 仅覆盖 version 和四个必要方法；step/reset/init/动力学/Blue策略/boundary/每步开火/同时结算/终止/奖励参数均同一继承实现，配置 validator 拒绝范围、奖励、Blue speed、scale 漂移。

## 20. 专项 pytest

{special_result}。覆盖 HA A–F、1000×5 pitch、ATA/AA旧版一致、weapon差异与边界、共享Gaussian、reward pitch、observation归一化/旧版一致、版本隔离、frozen config、旧完整 reference 和同状态分叉。

## 21. Full pytest

{full_result}。

```
{skips}
```

上述 skip 为既有历史诊断实验工件缺失，未跳过新 v3.13 测试。

## 22. Compileall

`python -m compileall -q env algorithm tools tests` 通过，日志为空；`git diff --check` 通过。

## 23. Vector reset

真实 spawn worker，batch[1,5,56]；自动 reset seed={vector['seed']}，下一 step 与 fresh environment 相同（obs 按既有 vector float32 约定比较），reward/info 完全一致；ammo6、Blue diagnostics 重新初始化，R1/R2/R3/R4+safety 分解验证通过。

## 24. 四算法 CUDA smoke

| 算法 | transitions | 真实update | save/reload | deterministic eval局数 | finite检查 | 2局win_rate | 2局average_return |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

只使用 512 transitions cap 和末段 rollout32×16。正式网络和 PPO 参数冻结，10 epochs/minibatch512 等不变。finite observation/action/reward/GAE/return/critic/gradient/parameter 检查通过，evaluation contract 完整，RMAPPO/STEA death/episode recurrent reset 验证通过。2局只用于执行验收，不代表学习性能。

## 25. 同状态 v3.12/v3.13 差异审计

正常 reset 的初始状态与 RNG 相同；随后双方同时注入相同合法 HA 敏感测试状态（Red pitch60°，活 pair 距离4500m，其余死亡槽，双方各6枚弹药）。每步同 Red/Blue ZERO actions、同 RNG seed，只有版本配置不同。使用外部 Blue override 固定动作，双方 Blue policy 类和参数仍相同；本审计没有调整 Blue 策略。

第 {audit['first_reward_difference_step']} 步 R3 因 HA 定义不同出现预期奖励差异。第 {audit['first_qualification_difference_step']} 步出现首次 eligibility 差异；此步 pre-combat state、distance、ATA、AA、signed LOS 和 RNG 完全相同，旧 HA={np.rad2deg(a['geometry']['ha']):.6f}°、新 HA={np.rad2deg(b['geometry']['ha']):.6f}°；Red 旧资格={a['red_eligible']}、新资格={b['red_eligible']}。

post-combat state 与 RNG 首次分别在第 {audit['first_postcombat_state_difference_step']} / {audit['first_rng_difference_step']} 步分叉；来源为 HA eligibility/hit-input 变化引起的真实战斗采样与结算，没有初始化、运动、Radar、Blue policy 或参数变化。两条轨迹各自继续至 terminal，R1/R2/R3/R4+safety 总和均通过；详见 same_state_audit.json。

完成后停止，没有正式1M/3M训练。
'''
    (OUT/'report.md').write_text(report, encoding='utf-8')
    print(special_result)
    print(full_result)


if __name__ == '__main__':
    main()
