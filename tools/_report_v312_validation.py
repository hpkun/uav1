"""Report the 25 observation-only acceptance items from finished artifacts."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np

OUT = Path('outputs/v312_validation')


def main():
    special = (OUT/'special_pytest.log').read_text()
    full = (OUT/'full_pytest.log').read_text()
    special_result = re.search(r'\d+ passed in .+',special).group(0)
    full_result = re.search(r'\d+ passed, \d+ skipped in .+',full).group(0)
    assert 'FAILED ' not in full and 'FAILED ' not in special
    assert not (OUT/'compileall.log').read_text()
    vector = json.loads((OUT/'vector_reset.json').read_text())
    isolation = json.loads((OUT/'checkpoint_isolation.json').read_text())
    smoke = json.loads((OUT/'cuda_smoke/summary.json').read_text())
    baseline = json.loads((OUT/'before_hashes.json').read_text())
    allowed = {'env/config.py','env/combat_env.py','algorithm/common/control_factory.py',
        'algorithm/common/evaluator.py','algorithm/mappo/runner.py',
        'algorithm/rmappo/networks.py','algorithm/rmappo/runner.py','algorithm/rmappo/protocol.py',
        'algorithm/ea_mappo/networks.py','algorithm/ea_mappo/factory.py','algorithm/ea_mappo/trainer.py',
        'algorithm/ea_mappo/protocol.py','algorithm/ea_mappo/runner.py',
        'algorithm/stea_mappo/networks.py','algorithm/stea_mappo/factory.py','algorithm/stea_mappo/trainer.py',
        'algorithm/stea_mappo/protocol.py','algorithm/maddpg/trainer.py','algorithm/maddpg/protocol.py',
        'algorithm/madsac/protocol.py'}
    changed = [p for p,h in baseline.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert set(changed) <= allowed
    (OUT/'historical_source_hash_check.json').write_text(json.dumps(dict(baseline_files=len(baseline),
        changed_extension_files=changed,historical_configs_and_version_specific_sources_unchanged=True),indent=2))
    assert len(smoke['records']) == len(isolation) == 4
    for name, record in smoke['records'].items():
        assert record['observation_dim'] == 56 and record['sampled_steps'] == 512
        assert record['critic_input_dim'] == 336 and record['critic_parameter_count'] == 152321
        assert record['finite_gradients_parameters_and_values'] and record['finite_rewards_advantages_and_returns']
        evaluation = record['evaluation_after_checkpoint_reload']
        assert evaluation['protocol_complete'] and evaluation['evaluation_episodes'] == 2
        assert evaluation['observation_dim'] == 56 and evaluation['checkpoint_environment_version'] == '3.12'
        assert all(np.isfinite(v) for v in evaluation.values() if isinstance(v,(int,float)))
        assert isolation[name]['old_checkpoint_valid'] and isolation[name]['v311_to_v312_rejected']
    assert vector['first_step_identical_to_fresh_environment'] and vector['observation_shape'] == [1,5,56]
    skips = '\n'.join(line for line in full.splitlines() if line.startswith('SKIPPED '))
    report = f'''# Combat Environment v3.12 验收报告

实现与验证完成。所有命令运行于WSL Ubuntu的uav Conda环境；CUDA：{smoke['cuda_device']}。没有正式1M/3M训练，也没有调整环境或PPO参数。

## 1. 文件清单

新增环境：env/v312_observation.py、env/v312_config.py、env/combat_v312.py。

新增配置：configs/combat_environment_v312.yaml、configs/mappo_5v5_v312.yaml、configs/rmappo_5v5_v312.yaml、configs/ea_mappo_5v5_v312.yaml、configs/stea_mappo_5v5_v312.yaml。

新增输入协议：algorithm/common/entity_layout.py。

新增验证/文档：tests/test_combat_v312.py、tests/v312_v311_reference.json、tests/v312_legacy_actor_reference.json、docs/combat_environment_v312.md；tools/_prepare_v312.py、tools/_wire_v312.py、tools/_check_v312_vector_reset.py、tools/_audit_v312_checkpoints.py、tools/_report_v312_validation.py。

修改版本/诊断分派：env/config.py、env/combat_env.py、algorithm/common/evaluator.py、algorithm/mappo/runner.py、algorithm/rmappo/runner.py、algorithm/rmappo/protocol.py、algorithm/ea_mappo/runner.py、algorithm/ea_mappo/protocol.py、algorithm/madsac/protocol.py、algorithm/maddpg/protocol.py、algorithm/maddpg/trainer.py。

修改必要输入接口：algorithm/common/control_factory.py、algorithm/rmappo/networks.py；algorithm/ea_mappo及algorithm/stea_mappo下的networks.py/factory.py/trainer.py/protocol.py（协议文件上面已列EA）。旧默认布局保持，PPO更新数学不变。

修改工具与清单：tools/diagnose_stea_representation.py使用统一entity mask helper；tools/smoke_formal_8v8.py新增配置选择、56D encoder检查；tests/test_combat_protocol.py、tests/test_formal_8v8.py更新配置inventory/版本提示。

## 2–5. 精确56D布局

| slice | 实体 | 归一化特征 |
|---|---|---|
| 0:7 | Self7 | x/5000,y/5000,altitude/6000,(v−225)/75,phi=0,psi/π,theta/(π/3) |
| 7:13 | Ally1 | forward/10000,right/10000,up/10000,(v_j−225)/75,psi_j/π,theta_j/(π/3) |
| 13:19 | Ally2 | 同样6D |
| 19:25 | Ally3 | 同样6D |
| 25:31 | Ally4 | 同样6D |
| 31:36 | Blue0 | d/10000,(v_k−225)/75,AA/π,ATA/π,HA/π |
| 36:41 | Blue1 | 同样5D |
| 41:46 | Blue2 | 同样5D |
| 46:51 | Blue3 | 同样5D |
| 51:56 | Blue4 | 同样5D |

Self保留绝对world位置以描述arena boundary；ally的psi_j为绝对world heading，非旧relative heading。

## 6. Body-frame公式

直接复用原flight_path_frame与_relative_position。delta=p_j−p_i；f=[cosθcosψ,cosθsinψ,−sinθ]，r=[−sinψ,cosψ,0]，u=normalize(r×f)；forward/right/up分别dot(delta,f/r/u)。heading0及π/2下的前、右、上专项测试均通过，未创建新符号体系。

## 7–8. Enemy geometry及HA统一

全部直接调用env.v36_geometry.combat_geometry，与Weapon Equation(7)/(8)、R3/R4一致；不使用旧observation专用engagement_geometry。前/左/右/上/下/迎头/尾追/斜向等8种人工几何反归一化d/AA/ATA/HA与combat helper逐项atol≤1e−8，包含HA。

当前helper的AA/ATA/HA本身输出非负圆周角差[0,π]；Observation直接暴露该原始输出，没有新绝对值转换或signed定义。HA实际合法state的上界5π/6，采用HA/π而非旧HA/(π/2)。没有修改combat geometry或Weapon/Reward角度。

## 9–14. Radar、ammo、alive、死亡槽、phi

Radar完全退出actor observation；机尾、侧方和超过10km的存活Blue依旧完整填充5D geometry。patch sensor helper和legacy geometry为raise时builder仍成功。Radar10km/half60°配置及诊断保持。

ammo不进入任何actor slot；改变双方ammo不改变observation，但实际6枚弹药系统保持。所有alive/radar flag删除。死亡ally6D/Blue5D全零，保持identity，其他实体不前移；死亡Red整row全零。EA/STEA实体mask按全槽非零判断，HA=0的存活敌机不再被错误当作dead flag；原legacy仍使用原alive列。没有额外隐藏flag。

phi在Self index4恒为0，即使兼容数组被人为填非零也不影响builder，不重新引入bank dynamics。

## 15–16. Normalization和ordering

严格使用上表固定scale，不clip合法值。builder以float64保留几何审查精度；vector/trainer仍原float32 tensor输入约定。主要常规范围约±1，合法3D长对角distance可>1且保留。

Allies按Red index升序跳过self，例如Red2对应0/1/3/4；Enemies固定Blue0..4。死亡、最近距离和radar visibility不改变排序。

## 17–18. 实际维度与网络

环境每agent56D、action3、agents5；vector batch[env,5,56]。MAPPO首层Linear input56；RMAPPO flat_encoder input56；EA/STEA输入总56，self/ally/enemy encoder输入7/6/5。entity64、heads2、fusion128、GRU128等隐藏结构不变。四算法centralized critic仍原global+local协议，input=(5+1)×56=336，hidden256/256，参数152321，seeded critic初值相同。

四份新配置与v3.11逐项比较：仅observation_dim及必要self_feature_dim/observation_layout改变，其余网络隐藏参数、training、implementation、logging、evaluation/checkpoint cadence完全一致。

## 19. Checkpoint隔离

使用四个真实v3.11 bounded smoke checkpoint做只读CUDA审计：原3.11配置下全部有效；目标3.12配置下全部因environment_version mismatch被拒绝。无padding/截断/state_dict重写/合同绕过。v3.12新checkpoint保存56D、3.12版本、SHA及新entity layout，save/reload和正式evaluation合同完整通过。

## 20. 历史和冻结机制

修改前{len(baseline)}个源文件/配置哈希已保存；所有历史环境版本专属源码和旧配置哈希未改。新增6条修改前v3.11完整ZERO/PURSUIT轨迹reference，state/observation/reward/info/RNG完全一致；8组旧EA/STEA布局52/65/66/104的参数+forward输出hash完全一致。已有历史回归通过。

v3.12类只有version与_observations覆盖；__init__/reset/step/kinematics/initialization/weapon/hit/reward/Blue policy/boundary/outcome均继承同一个v3.11实现。相同seed与相同ZERO/PURSUIT/RANDOM actions的9条完整轨迹逐步比较weapon eligibility、Gaussian RNG、state、reward decomposition、info（仅去除版本号）和outcome全部完全一致，只有返回的Observation不同。v3.11仍66D，ammo/alive和radar enemy mask保留。

## 21–25. 测试与CUDA验收

- 专项pytest：{special_result}
- full pytest：{full_result}
- skip原因：
```
{skips}
```
- compileall env/algorithm/tools/tests通过；git diff --check通过。
- vector reset：真实spawn worker，[1,5,56]；自动reset seed={vector['seed']}，下一stepobs按原float32转换后与fresh完全一致，reward/info完全一致，ammo与Blue诊断重置正确。
- replay wrapper实际生成56D，完整回合state/reward/RNG与原环境相同、事件记录正常；无需修改trace schema或旧版本录像默认行为。

| 算法 | 实际input | transitions | 真实update | save/reload | deterministic CUDA eval | finite检查 |
|---|---|---|---|---|---|---|
| MAPPO | 56 | 512 | 1 | 通过 | 2局 | 通过 |
| RMAPPO | 56 | 512 | 1 | 通过 | 2局 | 通过 |
| EA-MAPPO | 56（7/6/5） | 512 | 1 | 通过 | 2局 | 通过 |
| STEA-MAPPO | 56（7/6/5） | 512 | 1 | 通过 | 2局 | 通过 |

Observation builder/vector与action入口验证finite；CUDA真实更新验证reward/GAE/return/gradients/parameters/critic values finite，reload evaluation指标finite。PPO参数与网络隐藏宽度冻结，只有512 budget与末段rollout32×16用于验证。

完成后停止，无正式1M/3M训练。
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    print(full_result)


if __name__ == '__main__':
    main()
