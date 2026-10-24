"""Build research report/figures from completed audits; read-only model probes."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.record_combat_episode import load_recording_policy
from tools.audit_maac_v348_failure import stats,dump
import yaml


def read_csv(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;diag=out/'diagnostics_clean';source=ROOT/'outputs/maac_v348_seed1_3m'
    assert torch.cuda.is_available();torch.set_num_threads(1)
    groups=json.loads((diag/'all_episode_summaries.json').read_text());summary={};frames={}
    for group,episodes in groups.items():
        development=[r for r in episodes if r['seed']!=20000071]
        for r in episodes:
            rows=read_csv(diag/group/f"{r['seed']}.csv");frames[group,r['seed']]=rows
            r['discounted_return']=sum(.99**(int(x['step'])-1)*sum(float(x[k]) for k in ('r1','r2','r3','r4')) for x in rows)
            r['discounted_r3']=sum(.99**(int(x['step'])-1)*float(x['r3']) for x in rows)
        keys=['return_sum','discounted_return','steps','attempts','hits','kills','red_losses','blue_losses','boundary','ceiling','ground','alignment_rate_in_range','error_reduction_ratio']
        s={k:float(np.mean([r[k] for r in development])) for k in keys}
        s.update(win_rate=float(np.mean([r['red_win'] for r in development])),fire_episode_rate=float(np.mean([r['attempts']>0 for r in development])),
            saturation=np.mean([r['saturation'] for r in development],axis=0).tolist(),
            r4_sign=np.mean([r['r4_positive_negative_zero'] for r in development],axis=0).tolist(),
            reward_totals=np.mean([r['reward_totals'] for r in development],axis=0).tolist(),
            longest_same_sign_seconds=np.max([r['longest_same_sign_seconds'] for r in development],axis=0).tolist(),
            qualification_episodes=sum(r['first_qualification_s'] is not None for r in development),
            cone_samples=sum(r['cone_agent_steps'] for r in development),cone_with_positive_r4=sum(r['cone_with_positive_r4'] for r in development))
        s['failure_categories']=dict(never_in_range=sum(r['range_agent_steps']==0 for r in development),
            in_range_no_qualification=sum(r['range_agent_steps']>0 and r['first_qualification_s'] is None for r in development),
            attempted_but_zero_hits=sum(r['attempts']>0 and r['hits']==0 for r in development),
            noncombat_loss=sum(r['boundary']+r['ground']>0 for r in development))
        summary[group]=s
    dump(out/'development_comparison.json',summary)
    original_evaluations=[json.loads(line) for line in (source/'evaluation_metrics.jsonl').read_text().splitlines()]
    for label,step in (('best',300000),('final',3000000)):
        old=next(r for r in original_evaluations if r['sampled_steps']==step)
        for diagnostic,original in (('return_sum','average_return'),('attempts','red_fire_attempts'),('hits','red_weapon_hits'),('red_losses','average_red_loss'),('blue_losses','average_blue_loss')):
            assert abs(summary[label+'_det'][diagnostic]-old[original])<1e-8,'observer changed original evaluation'
    dump(out/'observer_invariance.json',dict(original_20_seed_evaluation_reproduced=True,
        checked=['return','attempts','hits','red_losses','blue_losses'],red_blue_policy_controllers_separate=True))
    telescoping=[]
    for seed in range(10000000,10000020):
        values=[next(r['discounted_r3'] for r in rs if r['seed']==seed) for rs in groups.values()]
        assert max(values)-min(values)<1e-10
        telescoping.append(dict(seed=seed,discounted_r3=values[0],max_cross_policy_error=max(values)-min(values)))
    dump(out/'pbrs_discounted_telescoping.json',telescoping)
    # Fixed live-observation bank: same states and other agents' actions for both models.
    datasets=[np.load(diag/f'{a}_det/critic_samples.npz') for a in ('best','final')]
    for label,data in zip(('best','final'),datasets):
        result=json.loads((diag/f'{label}_det/critic_diagnostics.json').read_text())
        expected=int(data['alive'].sum())
        for key in ('q','target_q','td_residual','advantage'):
            assert result[key]['n']==expected,'nonfinite values must not be silently omitted'
            result[key]['variance']=result[key]['std']**2
            result[key]['nonfinite_count']=0
        for key in ('actor_grad','critic_grad'):
            assert result[key]['n']==int(np.ceil(len(data['observation'])/64))
        dump(diag/f'{label}_det/critic_diagnostics_with_variance.json',result)
    o=np.concatenate([d['observation'][::max(1,len(d['observation'])//128)] for d in datasets])
    a=np.concatenate([d['actions'][::max(1,len(d['observation'])//128)] for d in datasets])
    m=np.concatenate([d['alive'][::max(1,len(d['observation'])//128)] for d in datasets])
    rule=np.concatenate([d['rule_actions'][::max(1,len(d['observation'])//128)] for d in datasets])
    ec=yaml.safe_load((source/'env_config.yaml').read_text());ac=yaml.safe_load((source/'algorithm_config.yaml').read_text())
    probes={};bank=o[m>0];np.save(out/'fixed_live_observation_bank.npy',bank)
    for label,path in [('best',source/'best_eval.pt'),('final',source/'final.pt')]:
        t,_,_=load_recording_policy(path,ec,ac,'cuda');obs=t.tensor(o);act=t.tensor(a);mask=t.tensor(m)
        deltas=[];evade_deltas=[];unsafe_deltas=[];hold_deltas=[]
        with torch.no_grad():
            for i in range(4):
                good=act.clone();bad=act.clone();good[:,i]=t.tensor(rule[:,i]);bad[:,i]=good[:,i];bad[:,i,:2]*=-1
                delta=(t.critic(obs,good,mask)[:,i,0]-t.critic(obs,bad,mask)[:,i,0])[mask[:,i]>0]
                deltas.extend(delta.cpu().numpy().tolist())
                baseq=t.critic(obs,good,mask)[:,i,0]
                evade=good.clone();evade[:,i,0]*=-1;evade[:,i,1]=0;evade[:,i,2]=1
                unsafe=good.clone();unsafe[:,i,1]=torch.where(obs[:,i,2]>.5,1.,-1.);unsafe[:,i,2]=1
                hold=good.clone();hold[:,i]=0
                for candidate,values in ((evade,evade_deltas),(unsafe,unsafe_deltas),(hold,hold_deltas)):
                    values.extend((t.critic(obs,candidate,mask)[:,i,0]-baseq)[mask[:,i]>0].cpu().numpy().tolist())
            d=t.actor.distribution(t.tensor(bank));mu=d.mean.cpu().numpy();sd=d.scale.cpu().numpy()
        probes[label]=dict(rule_minus_reversed_heading_pitch_q=stats(deltas),rule_ranked_above_reversed_fraction=float(np.mean(np.array(deltas)>0)),
            turn_away_level_fast_minus_rule_q=stats(evade_deltas),pitch_toward_nearer_vertical_boundary_minus_rule_q=stats(unsafe_deltas),hold_minus_rule_q=stats(hold_deltas),
            fixed_live_mu=[stats(mu[:,i]) for i in range(3)],fixed_live_std=[stats(sd[:,i]) for i in range(3)])
    dump(out/'matched_live_bank_critic_actor.json',probes)
    snapshots=[]
    for path in [source/'best_eval.pt']+list(source.glob('checkpoint_*.pt'))+[source/'final.pt']:
        t,state,_=load_recording_policy(path,ec,ac,'cuda')
        with torch.no_grad():d=t.actor.distribution(t.tensor(bank));mu=d.mean.cpu().numpy();sd=d.scale.cpu().numpy()
        snapshots.append(dict(name=path.name,steps=state['sampled_steps'],std=sd.mean(0).tolist(),mu=mu.mean(0).tolist()))
    snapshots.sort(key=lambda x:x['steps']);dump(out/'fixed_live_bank_snapshots.json',snapshots)
    # Existing optimization gradients are means of four updates per cycle, not per-update records.
    train=[dict(samples=0,attempts=0,hits=0,kills=0,actor_grad=[],critic_grad=[],adv=[]) for _ in range(3)]
    previous=0
    for line in (source/'training_metrics.jsonl').open():
        r=json.loads(line);stage=min(2,int(r['sampled_steps'])//1000000);b=train[stage]
        b['samples']+=int(r['sampled_steps'])-previous;previous=int(r['sampled_steps'])
        for k,field in [('attempts','red_step_fire_attempts'),('hits','red_step_weapon_hits'),('kills','red_step_attack_kills')]:b[k]+=r.get(field,0)
    for line in (source/'optimization_metrics.jsonl').open():
        r=json.loads(line);b=train[min(2,int(r['sampled_steps'])//1000000)]
        for k,field in [('actor_grad','actor_grad_norm'),('critic_grad','critic_grad_norm'),('adv','mean_abs_counterfactual_advantage')]:b[k].append(r[field])
    for b in train:
        b['red_attempts_per_1000_samples']=1000*b['attempts']/b['samples'];b['red_hits_per_1000_samples']=1000*b['hits']/b['samples']
        b['actor_cycle_mean_above_clip_fraction']=float(np.mean(np.array(b['actor_grad'])>.5));b['critic_cycle_mean_above_clip_fraction']=float(np.mean(np.array(b['critic_grad'])>50))
        for k in ('actor_grad','critic_grad','adv'):b[k]=stats(b[k])
    dump(out/'historical_training_stages.json',train)
    # Reproducible charts; English axes avoid machine-specific CJK fonts.
    names=list(summary);x=np.arange(len(names));fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].bar(x,[summary[n]['attempts'] for n in names]);axes[0].set_title('Red shots / episode')
    axes[1].bar(x,[summary[n]['alignment_rate_in_range'] for n in names]);axes[1].set_title('Cone qualification / in-range agent steps')
    axes[2].bar(x,[summary[n]['boundary']+summary[n]['ground'] for n in names]);axes[2].set_title('Red noncombat losses / episode')
    for ax in axes:ax.set_xticks(x,names,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'policy_comparison.png',dpi=160);plt.close(fig)
    representative=json.loads((diag/'representative_seeds.json').read_text())['best_fires_final_not'][0]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for col,g in enumerate(('best_det','final_det','best_rule')):
        rows=frames[g,representative]
        for agent in range(1,5):
            own=[r for r in rows if int(r['agent'])==agent]
            axes[0,col].plot([float(r['x']) for r in own],[float(r['y']) for r in own],label=f'R{agent}')
            axes[1,col].plot([float(r['time_s']) for r in own],[float(r['altitude']) for r in own])
        blue_rows=read_csv(diag/g/f'{representative}_blue.csv')
        for agent in range(1,5):
            own=[r for r in blue_rows if int(r['agent'])==agent]
            axes[0,col].plot([float(r['x']) for r in own],[float(r['y']) for r in own],color='steelblue',linestyle='--',alpha=.5)
        axes[0,col].add_patch(plt.Circle((0,0),5000,fill=False,color='gray'));axes[0,col].set_xlabel('X (m)');axes[0,col].set_ylabel('Y (m)');axes[0,col].set_aspect('equal');axes[0,col].set_title(f'{g}, seed {representative}');axes[0,col].legend()
        axes[1,col].axhline(6000,color='gray',linestyle='--');axes[1,col].axhline(0,color='gray',linestyle='--');axes[1,col].set_xlabel('Time (s)');axes[1,col].set_ylabel('Altitude (m)')
    fig.tight_layout();fig.savefig(out/'representative_trajectories.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,axis in zip(axes,('heading','pitch','speed')):
        for g in ('best_det','final_det'):
            values=[float(r[axis+'_action']) for seed in range(10000000,10000020) for r in frames[g,seed]]
            ax.hist(values,bins=50,range=(-1,1),density=True,histtype='step',label=g)
        ax.set_title(axis+' actual normalized action');ax.legend()
    fig.tight_layout();fig.savefig(out/'action_distributions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for i,n in enumerate(('heading','pitch','speed')):ax.plot([r['steps'] for r in snapshots],[r['std'][i] for r in snapshots],marker='o',label=n)
    ax.set_xlabel('Checkpoint sampled steps');ax.set_ylabel('Mean pre-tanh standard deviation');ax.set_title('Same live observation bank; saved snapshots only');ax.legend();fig.tight_layout();fig.savefig(out/'fixed_bank_std.png',dpi=160);plt.close(fig)
    comparison=json.loads((out/'bounded_comparison_verified/comparison.json').read_text())
    lines=['# MAAC v3.48训练失败专项审查与独立Reward候选实验','',
        '## 结论与证据边界','',
        '**现有证据支持多因素共同作用：动作均值的持续方向及目标对准退化、价值学习对战术动作的区分较弱，以及R4几何失配/稀疏、折扣下延迟损失的激励共同存在。不能证明Reward是唯一或主要因果因素。未在本次可解析测试中证实MAAC核心梯度/Mask/TD/软更新存在实现Bug。**',
        '', '105个完整回合：原20个开发种子10000000–10000019，外加20000071；Best/Final分别确定性与随机策略，另有相同环境的Red规则追击。随机策略torch种子=40000000+对局种子；环境RNG保持原种子。随机策略每个环境种子仅抽一个策略随机种子，不覆盖随机策略的完整方差。',
        '', '## 1. 多种子行为结果（下表仅原20开发种子）','',
        '| 策略 | Red胜率 | 有开火回合率 | 每局开火/命中 | Red/Blue损失 | 每局非战斗损失 | 射程内对准率 | Return | 折扣Return |',
        '|---|---:|---:|---|---|---:|---:|---:|---:|']
    for n,s in summary.items():lines.append(f"| {n} | {s['win_rate']:.1%} | {s['fire_episode_rate']:.1%} | {s['attempts']:.2f}/{s['hits']:.2f} | {s['red_losses']:.2f}/{s['blue_losses']:.2f} | {s['boundary']+s['ground']:.2f} | {s['alignment_rate_in_range']:.1%} | {s['return_sum']:.3f} | {s['discounted_return']:.3f} |")
    lines += ['', '四类失败机制分开计数（每组20回合；可重叠，不能相加当总失败数）：','',
        '| 策略 | 从未进入射程 | 进入射程但无资格 | 已开火但全未命中 | 存在非战斗损失 |', '|---|---:|---:|---:|---:|']
    for n,s in summary.items():lines.append('| '+n+' | '+' | '.join(str(v) for v in s['failure_categories'].values())+' |')
    lines += ['', '射程内对准率：post-motion/pre-settlement红机有至少一个合格目标的agent-step数 / 至少一个目标进入2000米的agent-step数，先逐回合计算再等权平均；不是击杀率。对准率仅检查存活与几何条件，忽略弹药以免将资源耗尽混作对准失败；CSV另存实际含弹药资格。',
        f'',f'代表性种子：{representative}；Best可开火而Final不能开火的完整种子列表见diagnostics_clean/representative_seeds.json。',
        '', '![多策略对照](policy_comparison.png)', '', '![代表性轨迹](representative_trajectories.png)',
        '', '## 2. 动作、实际控制与安全失效','',
        '**源码事实：v3.48使用CombatEnvironmentV332._advance → v311_kinematics.advance_aircraft，直接执行航向增量、俯仰增量、速度增量；不是运行中的点质点/RK4/滚转过载分配路径。nz和phi在该机制中没有物理定义，CSV明确标记N/A，不从兼容零值虚构载荷。**',
        '', '每步航向变化=π/40×a_heading，俯仰变化在±π/3限幅前为π/60×a_pitch，速度变化在[150,300]m/s限幅前为2.5×a_speed。即使动作远离±1，持续同号仍能持续转弯、下降或爬升，不能把“无动作饱和”等同于飞行安全。', '',
        '| 策略 | 航向/俯仰/速度近±1比例 | 最长同号航向/俯仰秒 | 实际一步离轴角改善比例 |', '|---|---|---|---:|']
    for n,s in summary.items():lines.append(f"| {n} | {'/'.join(f'{v:.3%}' for v in s['saturation'])} | {s['longest_same_sign_seconds']} | {s['error_reduction_ratio']:.1%} |")
    lines += ['', '种子20000071的具体损失与失效前1秒实际动作（H/P/V）：','',
        '| 模型 | 飞机 | 时刻s | 最后(x,y,h)米 | 前1秒动作均值H/P/V |', '|---|---|---:|---|---|']
    for n in ('best_det','final_det'):
        r=next(r for r in groups[n] if r['seed']==20000071)
        for loss in r['loss_precursors']:
            commands='/'.join(format(loss['last_1s_actions'][axis],'.4f') for axis in ('heading','pitch','speed'))
            lines.append(f"| {n} | R{loss['agent']} | {loss['time_s']:.1f} | {', '.join(f'{x:.1f}' for x in loss['last_position'])} | {commands} |")
    lines += ['', '![动作分布](action_distributions.png)']
    lines += ['', '“实际一步改善”用真实执行后的Red位置/姿态对同一个冻结Blue目标位置重新计算三维离轴角，排除仅凭动作符号判断。该局部比例不等于整体战术质量或长期胜率；远距离转弯改善也可能随后越界。',
        '', '每个CSV记录实际角度/速度变化、Actor高斯均值/标准差/log_std、确定性动作、最近目标体轴误差、边界/地面/上限裕量、速度/俯仰限幅、资格和累计攻击计数。每次死亡之前1秒的动作均值与位置已保存在逐回合summary的loss_precursors。',
        '', 'Final的确定性与随机策略都未恢复开火，因此本组不支持“训练攻击完全依靠噪声、均值不攻击”作为唯一解释；Best随机策略比确定性策略更差，也不能据此推断全部历史随机样本。',
        '', '![固定观测标准差](fixed_bank_std.png)',
        '', '固定存活观测库显示的是所保存Checkpoint在相同观测上的分布变化，不是在线训练期间完整std轨迹；未保存的中间状态无法恢复。较后期标准差并非坍缩到下限。最终均值行为仍表现为不建立资格，随机扰动并未补救。',
        '', '## 3. Critic/反事实优势诊断','',
        'Q、TD、反事实优势使用相应模型自己的确定性轨迹采样（每10步及事件步）；不同模型的状态分布不同，不能把均值差直接归因于参数变化。另提供匹配观测、相同其他智能体动作的单智能体替代动作探针，减少这一混杂。', '',
        '| 模型 | Q均值±std | TD残差均值±std | 反事实优势std | 诊断Actor/critic裁剪批比例 | 匹配库规则动作Q高于反向动作比例 |', '|---|---|---|---:|---|---:|']
    for n in ('best','final'):
        q=json.loads((diag/f'{n}_det/critic_diagnostics.json').read_text());lines.append(f"| {n} | {q['q']['mean']:.3f}±{q['q']['std']:.3f} | {q['td_residual']['mean']:.3f}±{q['td_residual']['std']:.3f} | {q['advantage']['std']:.4f} | {q['actor_clip_fraction']:.1%}/{q['critic_clip_fraction']:.1%} | {probes[n]['rule_ranked_above_reversed_fraction']:.1%} |")
    lines += ['', '完整分位数、极值、Online/Target Q、终止/开火/击杀/普通步TD残差见各critic_diagnostics.json。反事实采样数16；TD target为单次目标动作样本，存在采样噪声。诊断梯度为保存策略在新诊断mini-batch上的梯度，没有调用optimizer.step，不能替代原训练梯度。原训练日志是4次更新的cycle均值，只能报告cycle均值超限比例，不能还原逐更新裁剪比例。',
        '', '规则动作与反向航向/俯仰动作的Q排序仅是动作敏感性探针，不是Critic准确率：立即对准更好并不必然等于长期Soft Q更高。结果显示缺少稳定、强的区分证据，不能据此证明Q数学实现错误。',
        '', 'Critic估计熵正则化Soft Q。本报告没有把普通未折扣MC Return当作Q真值。表中的折扣episode Return只用于展示奖励/生存时间激励，不作Soft Q误差标签。',
        '', '## 4. 算法审查分类','',
        '- A 已证实实现错误：本次未证实。不能由Loss下降/上升或失败胜率推出Bug。',
        '- 已通过受控测试：两动作解析熵正则score-function梯度、连续tanh积分梯度、log_prob雅可比、raw detach、基线stop-gradient、仅替换指定智能体动作、死亡信息隔离、无其他存活者的零注意力、终止/个体死亡TD mask、熵符号、Polyak在线权重、共享Actor单optimizer无重复参数、共享损失按存活者归约及Replay环形时序。',
        '- B 合理但可能不稳定：离策略分布变化、有限16样本反事实噪声、稀疏动作价值差、熵与微小优势同尺度；45维观测及Critic联合输入不含弹药/冷却，存在部分可观测性与资源状态别名风险，但不能用它直接解释Final从未开火。Final轨迹的优势std下降和历史训练平均梯度下降是事实，尚不构成因果证明。',
        '- C 工程适配：连续tanh高斯、同质共享Actor、16样本Monte Carlo反事实估计、死亡Mask与有限回合终止处理。MAAC原文Eq.(5)–(9)支持动作相关注意力query、own embedding shortcut、Soft TD、score-function与固定其他动作的反事实基线；这些结构本身不能列为偏离原文的Bug。原文实验策略未采用本项目的同质共享Actor方案。参见[MAAC原文](https://proceedings.mlr.press/v97/iqbal19a/iqbal19a.pdf)。本轮MDPI原文页面返回429，未重新逐式核验MADSAC适配全文，相关仓库协议不当作独立外部证据。现行运动学简化也应作为实验前提记录。',
        '- D 未确认：Critic系统性高估逃离、优化器错误、随机策略训练异常、Reward为唯一主因。现有证据不足，未实施任何算法修复。',
        '', '## 5. R1–R4的信号与几何失配','',
        'R1中敌方击毁和撞地损失、R2中水平越界/超高损失，直接每机约-10；击杀奖励共享规则保持原样。等大直接惩罚无法仅靠事件幅度区分“有攻击机会的战斗死亡”和“无效越界”，并且γ=.99会折扣较晚发生的损失。实际开发集未折扣Return由Best约-36.21恶化到Final约-40.47，但同一轨迹的gamma=.99折扣Return由约-6.19提高到约-1.92。训练目标与未折扣评估指标的差异有明确数值证据；这仍不能证明Actor有意拖延死亡。',
        '', 'R3=γΦ(s′)-Φ(s)，scale=.1、距离/三维角度权重均.5，终止/截断势为0。对完整回合，折扣R3 telescopes到-Φ(s0)，因此不应为了增大未折扣Return随意放大scale；它可以改善局部学习信号，但不会在理想条件下改变最优折扣策略。',
        '', '| 策略 | R4正/负/零agent-step比例 | R1/R2/R3/R4每局合计 | 武器资格样本中R4为正 |', '|---|---|---|---|']
    for n,s in summary.items():lines.append(f"| {n} | {'/'.join(f'{v:.2%}' for v in s['r4_sign'])} | {'/'.join(f'{v:.3f}' for v in s['reward_totals'])} | {s['cone_with_positive_r4']}/{s['cone_samples']} |")
    lines += ['', '源码/数值反例：距离1500米、双方机头正对，三维α=0且武器合格，但AA≈±π，原R4的|AA|≤30°门槛使进攻与防御分支都为0。故“能瞄准/能开火”与“R4进攻奖励”并不等价。原防御分支同样可能遗漏真实三维威胁；高俯仰时ATA/HA分段与三维圆锥亦不同。原offense优先还会抑制并存的其他目标威胁。',
        '', '## 6. 独立v3.49候选（待验证假设）','',
        '只替换R4，直接复用v343_reward.continuous_tactical_rewards。令w(d)=clip((4000-d)/2000,0,1)，a(α)=clip(1-α/π,0,1)：',
        '', '```text\nO_i = max_live_j 0.02 w(d_ij) a(alpha_i_to_j)\nD_i = min_live_j -0.01 w(d_ij) a(alpha_j_to_i)\nR4_i = O_i + D_i\n```',
        '', '不再使用AA尾后门槛；进攻与威胁各自独立统计，不设进攻优先。R4单步每机界限[-.01,.02]；目标死亡过滤。R1/R2/R3、45维观测、武器、Blue策略、动作/运动学、种子协议、网络与超参数全部不变。战术项不是PBRS，明确不声称保持原最优策略。没有行为克隆、直接输出目标动作、新Actor输入或放宽攻击资格。',
        '', '研究假设：使向目标三维对准的动作获得连续、与武器几何一致的区分信号，减少合法接敌无进攻奖励的情况。副作用：最多1000步可累积显著正回报，可能产生保持距离/刷态势或迎头互毁；防御权重变小也可能过度激进。没有新增安全项，因此不能保证消除越界/超高。候选默认不启用，须显式选择v3.49。',
        '', '冻结场景72点网格见bounded_comparison_verified/frozen_reward_grid.json，包含距离500–5000米、不同Red方向与Blue姿态。头对头1500米：原R4=0，候选R4=.01；同向尾追1500米：候选进攻.02、威胁0。',
        '', '## 7. 同预算有界MAAC对照','',
        '两组均seed=1、12环境、6144采样，正式网络/训练参数不变、初始Actor/Critic逐参数相同；仅环境Reward版本不同。原训练总预算参数仍存于配置，诊断override单独记录为6144。评估使用同5个开发种子10000000–10000004。没有3M训练。', '',
        '| 环境 | 采样 | Actor/Critic更新 | Red胜率 | Return | 开火/命中 | Red/Blue损失 |', '|---|---:|---|---:|---:|---|---|']
    for v,r in comparison.items():
        e=r['evaluation'];lines.append(f"| {v} | {r['sampled_steps']} | {r['actor_updates']}/{r['critic_updates']} | {e['win_rate']:.1%} | {e['average_return']:.3f} | {e['red_fire_attempts']:.2f}/{e['red_weapon_hits']:.2f} | {e['average_red_loss']:.2f}/{e['average_blue_loss']:.2f} |")
    lines += ['', '不同Reward定义的Return不能直接用于性能优劣比较。仅一个训练种子、5个评估回合和6144步，不足以支持性能改善结论；此对照用于运行闭环和初步信号检查。参数更新、有限性、保存重载、版本拒绝已检查，不能等同于学习有效。',
        '', '## 8. 测试、产物与下一步建议','',
        '44项新增数学/候选测试、387项原环境/算法相关回归测试通过；16个完整历史回合与修改前HEAD精确一致。原v3.48配置/Reward及历史模型只读，模型SHA256前后一致。诊断代码与训练逻辑分离，既有MAAC网络/损失/优化过程未修改。',
        '', '**不建议直接将本候选提升为下一轮单一正式3M方案。** 建议先审阅几何失配与动作/价值诊断，再明确是否接受v3.49的刷态势/激进风险；若继续，应预注册多种子基线/候选消融，并用胜率、攻击资格、非战斗损失而非跨Reward的Return作主指标。本轮没有证明Reward改动已解决MAAC退化。',
        '', '可复现入口：tools/audit_maac_v348_failure.py、tools/compare_v348_v349_bounded.py、tools/report_maac_v348_failure.py；详细协议和配置在各输出子目录。逐步轨迹/动作分布CSV按策略/种子保存；关键统计JSON、Critic采样NPZ、模型快照固定观测探针、冻结Reward网格和图表均随报告提供。']
    (out/'科研分析报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Chinese report and charts saved',flush=True)


if __name__=='__main__':main()
