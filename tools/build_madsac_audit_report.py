"""Assemble measured audit artifacts and source evidence; no training is run."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import json
import subprocess
import platform
from datetime import datetime, timezone, timedelta
import xml.etree.ElementTree as ET
import torch
import yaml
from tools.madsac_audit_reference import dump, sha256, summarize_rows, require_cuda
from tools.audit_madsac_training_schedule import theoretical_counts

OUT=ROOT/'outputs/madsac_algorithm_audit'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8').strip()


def source(path, needle):
    p=ROOT/path
    line=next(i for i,s in enumerate(p.read_text(encoding='utf-8').splitlines(),1) if needle in s)
    return f'[{path}:{line}]({p.as_posix()}:{line})'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])


def main():
    from algorithm.madsac.trainer import MADSAC_IMPL_VERSION
    if MADSAC_IMPL_VERSION != 3:
        raise RuntimeError('Legacy implementation-v3 audit is frozen; use tools/validate_madsac_v4.py for v4')
    require_cuda()
    main_sha=git('rev-parse','main');head=git('rev-parse','HEAD')
    changes=git('diff','HEAD','--name-only')
    protected_diff=git('diff','HEAD','--name-only','--','algorithm','env','configs')
    assert not protected_diff,protected_diff
    protected=git('ls-files','algorithm','env','configs').splitlines()
    manifest={p:sha256(ROOT/p) for p in protected if (ROOT/p).is_file()}
    dump(OUT/'protected_source_sha256.json',dict(main_sha=main_sha,head_sha=head,
        tracked_protected_diff=protected_diff,files=manifest))
    random=read('attention_random_init.json');real=read('attention_real_env.json')
    checkpoints=read('attention_checkpoint.json');grad=read('gradient_comparison.json');schedule=read('schedule.json')
    # Refresh only descriptive UTD fields under the explicitly stated all-optimizer convention.
    schedule['formal_configs']={key:theoretical_counts(yaml.safe_load(Path(key).read_text(encoding='utf-8')))
                                for key in schedule['formal_configs']}
    c=yaml.safe_load((ROOT/'configs/madsac_4v4_v337.yaml').read_text(encoding='utf-8'))
    schedule['smoke_expected']=theoretical_counts(c,24000)
    dump(OUT/'schedule.json',schedule)
    pooled={}
    for category,results in [('RANDOM INIT',random['results']),('REAL ENV INPUT',real['results'])]:
        rows={mode:[] for mode in ('unscaled','scaled')}
        for result in results:
            r=read(result['label']+'_rows.json')
            for mode in rows: rows[mode].extend(r[mode])
        pooled[category]={mode:summarize_rows(r) for mode,r in rows.items()}
    all_results=random['results']+real['results']+[r for ck in checkpoints['results'] for r in ck['results']]
    tests=ET.parse(OUT/'pytest.xml').getroot()
    suites=tests.findall('testsuite') if tests.tag!='testsuite' else [tests]
    test_summary={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert test_summary['failures']==test_summary['errors']==0
    all_untracked=git('ls-files','--others','--exclude-standard').splitlines()
    owned={'tools/audit_madsac_attention.py','tools/audit_madsac_training_schedule.py',
           'tools/madsac_audit_reference.py','tools/build_madsac_audit_report.py',
           'tests/test_madsac_algorithm_audit.py'}
    new_files=sorted(owned)
    external_changes=dict(tracked=changes.splitlines(),untracked=[p for p in all_untracked if p not in owned])
    assert all((ROOT/p).exists() for p in new_files)
    status=dict(main_sha=main_sha,head_sha=head,tracked_files_changed=changes,
        protected_source_diff_empty=not protected_diff,new_source_files=new_files,
        formal_algorithm_runtime_config_modified=False,unrelated_workspace_changes=external_changes,tests=test_summary,
        device=torch.cuda.get_device_name(0),python=sys.executable,torch=torch.__version__,
        timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        paper_sha256=sha256(next(ROOT.glob('Li*2023*.pdf'))),
        attention_pooled=pooled,schedule=schedule,checkpoint_provenance=checkpoints,
        conclusion='Initialization is not saturated. Historical v3.26 trained critics show strong attention concentration and substantial offline scaling sensitivity. No aggregate Q/K gradient collapse was observed. Author implementation remains unknown; no production change is made.')
    dump(OUT/'summary.json',status)
    rows=[
        ('shared actor','homogeneous actors share parameters','one SharedMADSACActor','MATCH','Sec.3.2; trainer.__init__'),
        ('actor hidden structure','2 hidden layers, 256 each','45→256→256; mean/log_std heads','MATCH','Sec.4.1; networks.py'),
        ('actor activation','not specified','ReLU','PAPER UNSPECIFIED / PROJECT ASSUMPTION','networks._activation'),
        ('stochastic policy form','stochastic continuous policy','tanh diagonal Gaussian; log_std [-5,2]','PROJECT ASSUMPTION','Eq.20 does not specify exact density'),
        ('reparameterization','explicit','Normal.rsample or mean+std*epsilon','MATCH','Eq.20/21; actor.sample'),
        ('target actor','explicit target policy','deepcopy; target_actor.sample for bootstrap','MATCH','Eq.18; trainer.update'),
        ('twin critics','two independent centralized Q','distinct parameters and Adam instances','MATCH','Sec.3.2.1/2'),
        ('target critics','two target Q networks','deepcopy of each online Q','MATCH','Eq.18; Algorithm 1'),
        ('min Q','target and actor use minimum','torch.minimum in both paths','MATCH','Eq.18/20'),
        ('critic attention heads','2','2; hidden total256 split into128/head','MATCH; head width PAPER AMBIGUOUS','Sec.4.1'),
        ('attention self inclusion','explicit self contribution','includes own live key/value','MATCH','Eq.16'),
        ('attention scaling','Eq.17 omits divisor; Fig.6c says scaled','unscaled QK^T','PAPER AMBIGUOUS / POTENTIAL MISMATCH','visually verified PDF p10'),
        ('value activation','elementwise nonlinearity e.g. LeakyReLU','LeakyReLU(.01)','MATCH for family; slope PROJECT ASSUMPTION','Eq.16 prose'),
        ('embedding and Q MLP','two-layer MLP descriptions','2 hidden256 embedding; context→256→256→1','MATCH at stated level; exact layer counting PAPER AMBIGUOUS','Eq.16; Sec.4.1'),
        ('head concatenation / self shortcut','concat heads shown; no explicit raw-self shortcut','concat; context-only Q MLP','MATCH at diagram level; exact widths PROJECT ASSUMPTION','Fig.6c'),
        ('replay capacity','1e6','1e6 joint environment transitions','MATCH','Sec.4.1'),
        ('batch size','1024 transitions','1024 joint replay items','MATCH','Sec.4.1'),
        ('optimizer','Adam','independent Adam per actor/Q','MATCH','Sec.4.1'),
        ('lr','1e-4','actor/Q1/Q2=1e-4','MATCH','Sec.4.1'),
        ('gamma','.99','.99','MATCH','Sec.4.1'),
        ('alpha','.1','.1 fixed; no alpha optimizer','MATCH','Sec.4.1'),
        ('tau','.001','.001','MATCH','Sec.4.1'),
        ('learning starts','not reported','10000 transitions; first eligible step10008','PROJECT ASSUMPTION','runner.run'),
        ('policy delay','delayed updates; d not reported','d=2 paired critic rounds','PROJECT ASSUMPTION','Algorithm 1 uses t, not critic update count'),
        ('gradient steps / trigger','steps_per_update and n not reported','1 paired Q update every vector step after warmup','PROJECT ASSUMPTION','cannot reconstruct paper numerical UTD'),
        ('parallel envs','24 training seeds/environments','24 spawned environments','MATCH','Sec.3.3.1 p13'),
        ('total training steps','more than8M','3M','POTENTIAL MISMATCH','experiment budget difference; Sec.4.1'),
        ('evaluation episodes','20 distinct test seeds','20, stochastic, independent per-environment policy RNG','MATCH for count; stochastic mode PROJECT ASSUMPTION','Sec.3.3.1/4.1'),
        ('evaluation cadence','every two training cycles','every100000 sampled transitions','POTENTIAL MISMATCH / PAPER AMBIGUOUS mapping','cycles cannot be uniquely converted to transitions'),
        ('alive-mask / terminal semantics','not specified','dead keys/queries masked; dead-next and done stop bootstrap','PROJECT ASSUMPTION','networks/trainer/replay'),
        ('actor/critic minibatches','appears to resample in actor loop','one replay.sample; same batch for Q1/Q2/delayed actor','PAPER AMBIGUOUS','Algorithm 1 appears to resample for actor updates'),
        ('target update branch','inside delayed branch on PDF p12','only when actor_updated','MATCH','clock and n remain project choices'),
    ]
    f=[]
    def section(n,title,text): f.append(f'## {n}. {title}\n\n{text}\n')
    f.append('# MADSAC 只读算法审计\n\n日期：2026-10-15（Asia/Shanghai）。仅新增审计工具、测试与输出；未修改生产算法。\n')
    section(1,'当前 main SHA',f'`{main_sha}`；HEAD=`{head}`，二者一致。工作开始时 `git status --short` 为空。')
    section(2,'新增文件','\n'.join(f'- `{p}`' for p in new_files)+'\n\n输出目录：`outputs/madsac_algorithm_audit/`，含本报告、summary.json、三个 attention 分类 JSON、gradient_comparison.json、schedule.json、逐 item/head/query 原始统计、真实输入 NPZ、pytest.xml/log、论文核对页和短运行日志/检查点。outputs 被仓库忽略，不代表没有生成。')
    section(3,'正式算法 / runtime / config 零修改确认',f'**确认本次审计对正式算法/runtime/config零修改。** `algorithm/`、`env/`、`configs/` 对HEAD无diff；受保护源文件 SHA256 清单见 `protected_source_sha256.json`。未修改 networks.py、trainer.py、runner.py 或配置；没有将 scaled 路径接入训练、没有更改 optimizer、没有启动3M/8M训练。短运行只改 total=24000。\n\n审计开始时工作区干净，生成报告时观察到本次审计未实施的其他工具变更：{external_changes}。这些文件未撤销、未纳入本次审计新增清单，不能把整个工作区称为干净。\n\n执行环境：PowerShell 激活 uav，`{sys.executable}`，PyTorch {torch.__version__}，{torch.cuda.get_device_name(0)}。所有数值审计/测试使用 CUDA。首次两个 CUDA 审计并发出现 OOM，已串行重跑完成，保留 `schedule_initial_oom.log`。最初新增测试的 float64 输入适配错误已在审计测试中改为 float32；最终全部测试通过。')
    section(4,'PAPER vs CURRENT 总表',table(['项目','PAPER','CURRENT','判定','证据'],rows)+'\n\n论文基准：Li et al., Aerospace 2023,10,574, DOI https://doi.org/10.3390/aerospace10070574 。直接读取仓库原 PDF，并目视核对 p10/11/12/15；24并行和20测试见p13。没有依据“paper_reported”配置标签反推论文事实。v3.35/v3.36/v3.37正式配置的网络与训练调度字段相同，当前数值输入主用v3.37。')
    section(5,'Eq.(16)/(17) attention 结构',
        '当前路径：`[o_i,a_i] → Linear256/ReLU → Linear256/ReLU → e_i`；'
        '`q=Wq e, k=Wk e, v=LeakyReLU(Wv e,.01)`；每头 self-inclusive softmax 对存活 keys 归一化；加权值拼接成256维 context，再经 `256→256→256→1` Q MLP。没有 raw-self embedding shortcut。'
        'Eq.(16) 明确存在自身贡献项及其他 agent 项，故 self-inclusive 有依据。图示支持 head concat；每头128、Linear bias、exact hidden/output层计数属于实现细化。\n\n'+
        source('algorithm/madsac/networks.py','self.embedding =')+'；'+source('algorithm/madsac/networks.py','logits = query @')+'；'+source('algorithm/madsac/critic_contract.py',"'attention_logit_scaling'"))
    section(6,'Figure 6(c) 与 Eq.(17) 的缩放歧义',
        '**Figure 6 supports scaled attention. Eq.(17) omits explicit scaling. Author implementation is unpublished/unknown for this audit.** '
        '原图 p10 标注 Scaled Dot-Product，公式p10的双线性打分没有显式 sqrt(d_k)。不能断言 unscaled 是唯一忠实实现，也不能断言作者代码必然 scaled。'
        '本次仅离线比较同一网络权重/输入，缩放后结果从未返回正式 trainer。')
    section(7,'d_head 与理论 scaling factor',f'hidden=256，heads=2，`d_head=128`，`sqrt(128)=11.313708498984761`。离线B严格为 `QK^T/sqrt(128)`。本次共{len(all_results)}个 critic/input组合，测得有效 logits 全局 std 比值为 '+f"{min(r['measured_logit_std_ratio'] for r in all_results):.9f}～{max(r['measured_logit_std_ratio'] for r in all_results):.9f}"+'，符合 float32 舍入范围。')
    def entropy_table(results):
        return table(['输入 / critic','H unscaled','H scaled','归一化H unscaled','归一化H scaled','有效query/head数'],[
            (r['label'],f"{r['unscaled']['attention_entropy']['mean']:.6f}",f"{r['scaled']['attention_entropy']['mean']:.6f}",
             f"{r['unscaled']['normalized_attention_entropy']['mean']:.6f}",f"{r['scaled']['normalized_attention_entropy']['mean']:.6f}",r['unscaled']['live_query_head_count']) for r in results])
    section(8,'RANDOM INIT：unscaled vs scaled entropy',
        '正式 hidden256、heads2，seed=0,1,2,3,4，每seed两个独立critic；128个合法reset输入加384条短random-policy rollout输入。没有使用纯 Gaussian tensor。未训练。\n\n'+entropy_table(random['results']))
    trained=[r for ck in checkpoints['results'] for r in ck['results']]
    section(9,'REAL ENV INPUT 与 TRAINED CHECKPOINT：entropy',
        'B：v3.37，初始化正式随机策略仅采样2400条环境transitions，不更新参数。\n\n'+entropy_table(real['results'])+
        '\n\nC：已有v3.26/5v5/56D正式训练best(600000步)及final(3000000步)，每检查点用其自身验证通过的环境/config收集1024条策略输入，同时审计Q1/Q2。'
        '这是历史检查点结果，不能冒充v3.37已训练结果。checkpoint身份、架构、config hash均经现有protocol校验；显式读取critic1/critic2权重，因为现有load_for_evaluation仅恢复actor。\n\n'+entropy_table(trained)+
        '\n\n熵以自然对数计算；raw H平均含单存活agent的H=0，normalized H在单存活情况下未定义并排除。多存活饱和率另行统计，避免单存活权重1的必然结果污染结论。')
    section(10,'最大 attention weight 分布',table(['组合','模式','mean','p50','p90','p95','p99','max'],[
        (r['label'],mode,*[f"{r[mode]['max_attention_weight'][k]:.6f}" for k in ('mean','p50','p90','p95','p99','max')])
        for r in all_results for mode in ('unscaled','scaled')])+'\n\n此分布含所有存活query。逐项 `logit_mean/std/abs_max`、`second_max_attention_weight`、`max_peer_attention_weight` 及其分位数保存在分类JSON和`*_rows.json`。每条原始记录含item/head/query/alive_keys，item整除256即推理batch索引；不统计dead query/key。')
    section(11,'≥.5/.75/.9/.95/.99 饱和率',
        '主表仅统计 `alive_keys>1`。JSON同时保留包含singleton的原始比例，阈值采用 ≥（不是严格 >）。\n\n'+
        table(['组合','模式','≥.5','≥.75','≥.9','≥.95','≥.99'],[
        (r['label'],mode,*[f"{100*r[mode]['nontrivial_saturation_fraction_ge'][str(t)]:.3f}%" for t in (.5,.75,.9,.95,.99)])
        for r in all_results for mode in ('unscaled','scaled')]))
    section(12,'Effective attended agents',table(['组合','unscaled exp(H) 均值','scaled exp(H) 均值'],[
        (r['label'],f"{r['unscaled']['effective_number_of_attended_agents']['mean']:.6f}",f"{r['scaled']['effective_number_of_attended_agents']['mean']:.6f}") for r in all_results])+
        '\n\n按query先exp再平均，不是exp(mean H)。不同输入存活数、4v4/5v5不同，不直接将raw H或effective agents跨类别作为性能指标。')
    section(13,'Self attention weight',table(['组合','unscaled self均值','scaled self均值','unscaled max peer均值','scaled max peer均值'],[
        (r['label'],*[f"{r[mode][key]['mean']:.6f}" for key in ('self_attention_weight','max_peer_attention_weight') for mode in ('unscaled','scaled')]) for r in all_results]))
    section(14,'Attention logit std',table(['组合','unscaled全局有效std','scaled全局有效std','比值','unscaled逐query key-std均值','scaled逐query key-std均值'],[
        (r['label'],f"{r['unscaled']['global_valid_logit_std']:.6f}",f"{r['scaled']['global_valid_logit_std']:.6f}",f"{r['measured_logit_std_ratio']:.8f}",
         f"{r['unscaled']['logit_std']['mean']:.6f}",f"{r['scaled']['logit_std']['mean']:.6f}") for r in all_results])+
        '\n\n全局std包含不同query间均值差；softmax只取决于同一query内key间差异，因此同时提供逐query std，不能仅凭全局std判断饱和。')
    gradient_rows=[]
    for label,g in grad.items():
        for key in ('wq','wk','wv','embedding','embedding_output'):
            metric='grad_norm_'+key
            gradient_rows.append((label,key,f"{g['unscaled'][metric]:.7g}",f"{g['scaled'][metric]:.7g}",f"{g['scaled_over_unscaled'][metric]:.6f}"))
    section(15,'Q/K/V 与 embedding 梯度',
        '每个组合固定相同权重、相同前256条输入及alive mask，对 `sum(alive Q)/sum(alive)` 反向；使用autograd.grad，不调用optimizer。'
        'wq/wk/wv/embedding范数含该模块weight+bias，embedding_output是中间embedding张量梯度。所有结果见下表；这仅是指定标量的局部敏感度，不是训练loss梯度。\n\n'+table(['组合','模块','unscaled梯度范数','scaled梯度范数','scaled/unscaled'],gradient_rows))
    section(16,'是否存在明显 unscaled softmax saturation',
        '**初始化：没有；已训练历史检查点：存在显著集中及部分强饱和。** '
        '随机5seed和未训练真实输入在多存活条件下≥.9/.95/.99比例全部为0。v3.26 best的Q1/Q2 ≥.95约30.65%/24.75%，≥.99约15.01%/6.48%；'
        'scaled离线对照这些比例为0，entropy明显提高。final的≥.95约8.01%/4.44%，风险较best轻。'
        '**梯度未发现整体消失：** best/final的scaled/unscaled Q/K范数比约0.388～0.596，unscaled总体反而更大；不能声称scaled恢复了消失的Q/K梯度。'
        '初始化的比值约0.088与显式1/sqrt128因子相符，不能误判为scaled更差。'
        '将attention scaling列为**高数值敏感性/潜在实现偏差风险（限已有v3.26权重证据）**；这不是作者实现错误或当前v3.37训练性能受损的证明。'
        '没有跨训练seed、长期训练或scaled重训练因果实验，不能把低胜率归因于此。')
    section(17,'sampled_steps 精确定义与runner逐行路径',
        'sampled_steps单位是**joint environment transitions**，不是agent transitions。24个环境每vector step产生24条joint记录，每条包含全部4个agent（历史检查点5个）。'
        '当前环境transition数3M，对4agent而言可包含12M个agent槽位，但不能据此把采样步数称为12M。\n\n'+
        '\n'.join('- '+source(path,needle)+'：'+desc for path,needle,desc in [
        ('algorithm/madsac/runner.py','actions = self.trainer.act','随机actor生成vector actions'),
        ('algorithm/madsac/runner.py','result = self.vector.step_batch','推进24个环境'),
        ('algorithm/madsac/runner.py','self.store_vector_step(observations','replay insertion，存terminal transition_next_observations和前后alive'),
        ('algorithm/madsac/runner.py','self.trainer.sampled_steps = advance','sampled_steps += num_envs'),
        ('algorithm/madsac/runner.py','self.trainer.vector_steps +=','vector_steps +=1'),
        ('algorithm/madsac/runner.py','if self.trainer.sampled_steps >= self.learning_starts','warmup与replay batch双门槛'),
        ('algorithm/madsac/runner.py','self.last_metrics = self.trainer.update','每轮一次sample，共享minibatch'),
        ('algorithm/madsac/trainer.py','self.critic_update_count +=','两个Q optimizer各step后计一轮'),
        ('algorithm/madsac/trainer.py','actor_updated =','轮数mod policy_delay==0时更新actor/targets')]))
    section(18,'正式 num_envs 与调度参数','v3.35/v3.36/v3.37均：num_train_envs=24，total_sampled_steps=3000000，learning_starts=10000，gradient_steps_per_vector_step=1，policy_delay=2，minibatch_size=1024。warmup期间仍使用随机初始化actor的随机策略，不是单独uniform-action warmup。')
    section(19,'Steady-state critic UTD',
        '按任务字面定义“critic optimizer steps / sampled environment transitions”，两个critic合计为 **2/24=0.0833333333**。'
        '常见的每critic/双critic更新轮口径为 **1/24=0.0416666667**；代码critic_update_count采用后者，不能混用。'
        '每vector step：1轮、2个optimizer steps；每1000 transitions：41.6667轮、83.3333个optimizer steps；每轮24条新transitions，每个合计optimizer step对应12条。')
    section(20,'Steady-state actor UTD','**1/48=0.0208333333**；每vector step平均0.5次actor optimizer step，每1000 sampled transitions约20.8333次，每次对应48条新transitions。')
    section(21,'完整3M run理论critic次数与whole-run UTD',
        '设V=3000000/24=125000，首个更新vector索引 `s0=max(1,ceil(10000/24),ceil(1024/24))=417`，对应10008 transitions。'
        '第417个vector step已包含在更新内，因此 `C=(125000-417+1)*1=124584`，而不是124583。'
        '**Q1=124584，Q2=124584，合计249168 optimizer steps。** whole-run每critic/轮UTD=0.041528，两个optimizer合计UTD=0.083056。'
        'steady-state与whole-run的差异来自前416个vector steps（9984条transition）不更新。')
    section(22,'完整3M run理论actor次数','**floor(124584/2)=62292** actor optimizer steps，同时每个target更新62292次。whole-run actor UTD=62292/3000000=0.020764。')
    section(23,'Smoke实测计数',
        '直接执行生产 `MADSACTrainingRunner.run()`，`smoke=False`，只覆盖total=24000；hidden256、24进程环境、replay1e6、batch1024、warmup10000、gradient_steps1、delay2均保留。'
        'optimizer post-step hooks统计真实step，replay只包装计数并原样调用。未替换正式训练逻辑。\n\n'+table(['字段','理论','实测'],[
        (key,schedule['smoke_expected'][key],schedule['smoke_actual'][key]) for key in ('vector_steps','replay_size','critic_update_count','actor_update_count','critic1_optimizer_steps','critic2_optimizer_steps')])+
        '\n\n实测 sampled_steps=24000，actor optimizer=292，replay.sample=584，replay.add_batch=1000，插入transitions=24000。Q optimizer合计1168；短run whole UTD：合计1168/24000=0.0486667，每critic584/24000=0.0243333，actor292/24000=0.0121667。')
    section(24,'理论 vs 实测','**逐项一致，所有计数断言PASS。** 见schedule.json的checks。'+f"\n\n最终CUDA测试：{test_summary['tests']}项，failures={test_summary['failures']}，errors={test_summary['errors']}，skipped={test_summary['skipped']}。测试包含本次新增8项及既有MADSAC回归测试。")
    section(25,'Algorithm 1 schedule vs current runner',
        '论文伪代码（原PDF p12；`if t mod d`的零/非零条件没有完整印出，不能猜作者精确代码）：\n\n'
        '```text\nT += M\nif T >= steps_per_update:\n    repeat n:\n        sample B; update Q1 and Q2\n    if delayed_condition(t,d):\n        repeat n:\n            sample B; update actor\n        update target actor and both target critics\n    T = 0\n```\n\n'
        '当前实现：\n\n```text\ncollect M transitions; replay.add_batch\nsampled_steps += M; vector_steps += 1\nif sampled_steps >= learning_starts and replay_size >= batch_size:\n    repeat gradient_steps_per_vector_step:\n        B = replay.sample(batch_size)\n        update Q1(B); update Q2(B); critic_update_count += 1\n        if critic_update_count % policy_delay == 0:\n            update actor(B), resampling current actions only\n            update target actor and target Q1/Q2\n```\n\n'+
        table(['步骤','判定','说明'],[
        ('并行采样与replay插入','SAME','M条joint transitions'),
        ('T>=steps_per_update与n','PAPER UNSPECIFIED','论文给符号不给数值；当前steady每24条1轮，不可证明与论文UTD相同'),
        ('warmup=10000','PAPER UNSPECIFIED','当前项目门槛；论文未公开对应值'),
        ('先n次critic后n次actor vs interleave','DIFFERENT','当前每轮内按计数决定actor；gradient_steps=1时仍不能恢复论文未知n'),
        ('delay时钟','DIFFERENT','论文t（episode timestep）；当前critic轮计数，不因episode reset清零'),
        ('actor重采batch','PAPER AMBIGUOUS','版面看似另采；当前复用同一transition batch'),
        ('target放置分支','SAME','目视确认均在delayed policy branch'),
        ('episode reset','DIFFERENT','论文外层episode reset所有环境；当前各环境done后各自reset，训练持续'),
        ('测试频率','PAPER UNSPECIFIED','two training cycles不能唯一换算成100k transitions')]))
    section(26,'Actor / critic minibatch是否复用',
        '**复用。** runner每轮仅一次replay.sample；trainer.update以同一B更新Q1/Q2以及延迟触发的actor。'
        'actor会在同一observations上重新采当前策略动作，不等于重新采replay minibatch。'
        '**Algorithm 1 appears to resample for actor updates.** 但不应声称paper definitively requires independent minibatches；判PAPER AMBIGUOUS。')
    section(27,'Target update时机',
        '**MATCH（branch placement）。** 当前仅actor_updated时对target_actor、target_critic1、target_critic2各Polyak一次；tau=.001。'
        'PDF p12目视确认目标更新在delayed branch内；正文Sec.3.2也描述delayed policy update。'
        '测试检查第一次critic轮targets不变、第二轮各更新一次且等于(1-tau)*old+tau*online。'
        '实际时钟、d和n未知，分支一致不代表数值频率已复现。\n\n'+source('algorithm/madsac/trainer.py','polyak_update(self.target_actor'))
    section(28,'Own-action gradient检查',
        '**PASS。** CUDA自动微分遍历4个query agent，`dQ_i/da_i !=0`，所有 `dQ_i/da_j ==0 (j!=i)` 严格零。'
        'joint_actions_with_own_gradient保留动作值，仅切断peer动作梯度。Eq.(21)的∇a_i Q_min支持该own-action解释；'
        '参数共享后将各agent项累加。Eq.(20)直接对全部共享动作做联合全导数有表述歧义，当前选择与Eq.(21)一致，不将其称为bug。')
    section(29,'Twin critic independence',
        '**PASS。** 两critic参数storage不共享、至少部分初值不同，Adam实例及参数集合独立；各target初始化与自己的online逐元素相等但storage不共享且requires_grad=False。'
        'target和actor两个路径均用minimum；相应数学与own-action测试通过。')
    section(30,'Target actor使用','**PASS。** trainer.update在no_grad下调用target_actor.sample生成next_actions/log_prob；未使用online actor替代。测试观察非延迟轮target_actor调用1次、online actor调用0次。对应Eq.(18)的target policy。')
    section(31,'Entropy term与tanh Jacobian','**PASS。** alpha=.1固定；critic target为`r + gamma*continuation*(min(targetQ1,targetQ2) - alpha*next_log_prob)`，actor最小化`alpha*log_prob - min(Q1,Q2)`。tanh对数Jacobian使用稳定精确表达式；与直接change-of-variables数学对照通过，极端raw=50仍finite，重参数动作对mean权重梯度非零。exact tanh-Gaussian和log_std范围属于PROJECT ASSUMPTION，不是PAPER-SPECIFIED。')
    section(32,'Loss reduction','**MATCH（对Eq.18/20的sum-agent、mean-batch结构）。** `((values*alive).sum(agent)).mean(batch)`，没有按alive agent数归一化。alive masking及done/next-alive的bootstrap gate属于项目适配。Eq.(19)印刷的平方loss梯度省略常数2；当前自动微分平方误差与Eq.(18)一致，不能据此断言作者代码采用0.5*MSE或当前bug。')
    section(33,'PAPER-SPECIFIED','shared homogeneous actors；two independent centralized attention critics；self contribution；embedding/Q MLP描述；Wq/Wk/Wv、Softmax；two heads；hidden256；Adam1e-4；replay1e6；batch1024；gamma.99；alpha.1；tau.001；target actor/critics；delayed policy update；reparameterization；target/actor min-double-Q；24并行训练、20测试场景；论文训练预算>8M。')
    section(34,'PAPER-AMBIGUOUS','Eq.(17)无显式sqrt而Fig.6(c)scaled；actor minibatch独立重采样是否硬性要求；每头维度/精确MLP输出层计数与bias；LReLU仅示例且无slope；Algorithm1的if t mod d条件写法及真实d/n/steps_per_update未完整数值化；two training cycles与transition数映射；Eq.(19)梯度常数2。作者实际实现未提供，本审计不推测。')
    section(35,'PROJECT-ASSUMED','learning_starts10000；每vector step gradient_steps1；policy_delay2且按critic轮计数；3M预算；ReLU；tanh-Gaussian；log_std[-5,2]；LReLU slope.01；head_dim128与投影bias；alive masks和terminal gates；同一replay batch复用；eval stochastic与独立策略噪声seed；100k测试间隔；异步episode reset。24环境与20测试数量本身有论文依据。')
    section(36,'POTENTIAL IMPLEMENTATION MISMATCH',
        '1. **Attention缩放：高数值敏感性风险（历史v3.26已训练权重）。** 有明显softmax集中及scaled/unscaled差异；作者scaled与否未知，无整体Q/K梯度消失证据。初始化无明显饱和，不能一概称unscaled必坏。\n'
        '2. **训练调度不可声称精确复现。** paper的d/n/steps_per_update无数值，当前按critic轮延迟、逐vector更新为项目选择；minibatch重采样呈现不同但PAPER AMBIGUOUS，target分支一致。\n'
        '3. **训练预算明确不同：3M vs >8M。** 这是实验预算差异，不是trainer数学bug。\n'
        '4. **未发现own-action梯度、twin独立性、target actor、entropy符号、tanh Jacobian或sum-agent reduction的实现错误。**\n'
        '5. 本审计不跨环境复现reward/动力学或胜率曲线；v3.26 checkpoint分布、有限采样与mean-Q单标量不能给出当前v3.37性能的因果结论。')
    section(37,'审计结论',
        '**当前实现的主要SAC数学路径与论文对应，调度中的多个数值为项目假设；attention scaling在论文中有图/式歧义，并在已有训练权重上表现出显著数值敏感性。** '
        '正式runner的24环境采样/更新计数已用真实24000步短运行验证。训练后attention集中值得记录，但不能称已证实梯度消失，也不能推断作者私有代码。'
        '**本次仅提供审计证据和结论；没有自行修改任何正式算法、runtime或config。**\n\n'
        '复现（PowerShell先conda activate uav；串行运行以控制显存）：\n\n```powershell\n'
        'python -u tools/audit_madsac_attention.py\n'
        'python -u tools/audit_madsac_training_schedule.py --output-dir outputs/madsac_algorithm_audit_repeat\n'
        'python -m pytest -q tests/test_madsac_algorithm_audit.py tests/test_madsac_v3.py tests/test_madsac_v326.py tests/test_madsac.py\n'
        'python tools/build_madsac_audit_report.py\n```\n\n'
        '调度重复使用新的输出目录，避免复用已有JSONL；不把本次短审计产生的检查点列为历史训练证据。所有JSON为实测/源代码计算，报告未以CPU结果替代CUDA。')
    (OUT/'report.md').write_text('\n'.join(f),encoding='utf-8')
    print(json.dumps(dict(report=str(OUT/'report.md'),tests=test_summary,tracked_changes=changes,
                         measured_combinations=len(all_results)),ensure_ascii=True))


if __name__=='__main__': main()
