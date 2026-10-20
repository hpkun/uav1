"""Offline paired-seed summaries and exportable figures; no training changes."""
import json
from pathlib import Path
import numpy as np


def summarize_experiments(directory):
    directory=Path(directory)
    summary=json.loads((directory/'ablation_summary.json').read_text(encoding='utf-8'))
    methods=['B0','B1','B2','B3','B3-reset']
    comparisons=[]
    for left,right,label in [('B1','B0','post-aggregation memory/attention vs MLP'),
                              ('B2','B1','entity memory vs post-aggregation memory'),
                              ('B3','B2','learned gate vs fixed gate'),
                              ('B3','B3-reset','history vs per-step history reset')]:
        differences=[]
        for seed in summary['seeds']:
            a=next(r for r in summary['runs'] if r['method']==left and r['seed']==seed)
            b=next(r for r in summary['runs'] if r['method']==right and r['seed']==seed)
            differences.append(dict(seed=seed,return_delta=a['episode_return']-b['episode_return'],
                completion_delta=a['task_completion_rate']-b['task_completion_rate'],
                collision_delta=a['collision_rate']-b['collision_rate'],
                random_return_delta=a['random_return']-b['random_return']))
        values=[r['return_delta'] for r in differences]
        comparisons.append(dict(left=left,right=right,interpretation=label,paired_seed_differences=differences,
            mean_return_delta=float(np.mean(values)),sample_std_return_delta=float(np.std(values,ddof=1)),
            positive_difference_seeds=sum(v>0 for v in values),stable_superiority_established=False))
    stability={}
    curves={method:[] for method in methods}
    for row in summary['runs']:
        run=directory/f"{row['method']}_seed{row['seed']}"
        metrics=[json.loads(line) for line in (run/'optimization_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        numeric=[value for record in metrics for value in record.values() if isinstance(value,(int,float))]
        stability[run.name]=dict(max_ratio_audit_error=max(r['pre_update_ratio_max_abs_error'] for r in metrics),
            max_actor_gradient_norm=max(r['actor_gradient_norm'] for r in metrics),
            max_critic_gradient_norm=max(r['critic_gradient_norm'] for r in metrics),
            all_finite=bool(np.isfinite(numeric).all()),
            actor_steps=sum(r['actual_actor_optimizer_steps'] for r in metrics),
            critic_steps=sum(r['actual_critic_optimizer_steps'] for r in metrics))
        evaluations=[json.loads(line) for line in (run/'evaluation_history.jsonl').read_text(encoding='utf-8').splitlines()]
        curves[row['method']].append(evaluations)
    report=dict(comparisons=comparisons,stability=stability,
        cautions=['Three training seeds and one short budget do not establish stable superiority.',
                  'The simple scene is deterministic; random-layout evaluation is a separate generalization check.',
                  'B1-vs-B0 and B2-vs-B1 change more than parameter count; causal attribution is limited.',
                  'B3-vs-B2 changes the learned gate; B3-vs-reset preserves all network parameters.'])
    (directory/'paired_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#718096','#3182ce','#38a169','#805ad5','#dd6b20']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for method,color in zip(methods,colors):
        steps=np.array([e['sampled_steps'] for e in curves[method][0]])
        values=np.array([[e['episode_return'] for e in series] for series in curves[method]])
        mean=values.mean(0);std=values.std(0,ddof=1)
        axes[0].plot(steps,mean,marker='o',label=method,color=color)
        axes[0].fill_between(steps,mean-std,mean+std,alpha=.1,color=color)
    axes[0].set(xlabel='Sampled joint environment transitions',ylabel='Deterministic evaluation team return',
        title='Simple scene: mean ± sample SD across 3 training seeds')
    axes[0].legend();axes[0].grid(alpha=.2)
    for i,(method,color) in enumerate(zip(methods,colors)):
        values=[r['random_return'] for r in summary['runs'] if r['method']==method]
        axes[1].bar(i,np.mean(values),yerr=np.std(values,ddof=1),color=color,alpha=.65,capsize=4)
        axes[1].scatter(np.full(len(values),i),values,color=color,edgecolors='white',zorder=3)
    axes[1].set(xticks=range(len(methods)),xticklabels=methods,ylabel='Team return',
        title='Random layouts: final-policy generalization')
    axes[1].grid(axis='y',alpha=.2)
    fig.savefig(directory/'ablation_results.png',dpi=180)
    plt.close(fig)
    return report
