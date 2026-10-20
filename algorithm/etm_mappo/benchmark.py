"""Bounded matched-budget B0--B3 and history-reset experiments (three seeds)."""
import copy
from contextlib import redirect_stdout
import csv
import json
from pathlib import Path
import numpy as np
import shutil
from .config import runtime_config,fingerprint
from .checkpoint import load_checkpoint
from .runner import ETMTrainingRunner


def run_benchmark(config,output_dir,seeds=(0,1,2),budget=8192,reuse_completed_from=None):
    if budget<=0 or budget>16384:raise ValueError('short diagnostic budget must be in (0,16384]')
    if len(set(seeds))<3:raise ValueError('at least three distinct seeds required')
    output=Path(output_dir)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    rows=[]
    for label,variant,reset in [('B0','B0',False),('B1','B1',False),('B2','B2',False),('B3','B3',False),('B3-reset','B3',True)]:
        for seed in seeds:
            effective=runtime_config(config,seed=int(seed),variant=variant,total_sampled_steps=budget,memory_reset_each_step=reset)
            directory=output/f'{label}_seed{seed}'
            previous=Path(reuse_completed_from)/directory.name if reuse_completed_from else None
            if previous is not None and (previous/'run_summary.json').is_file() and (previous/'final.pt').is_file():
                # Reuse only fully finished, strictly validated matched runs;
                # never resume an interrupted training state or alter its logs.
                _,state=load_checkpoint(previous/'final.pt',config=effective)
                summary=json.loads((previous/'run_summary.json').read_text(encoding='utf-8'))
                if summary['sampled_steps']!=budget or summary['config_sha256']!=fingerprint(effective):
                    raise RuntimeError('completed run provenance/budget mismatch')
                shutil.copytree(previous,directory)
                (directory/'reuse_provenance.json').write_text(json.dumps(dict(source=str(previous.resolve()),training_resume=False)),encoding='utf-8')
            else:
                runner=ETMTrainingRunner(effective,directory)
                with (directory/'train.log').open('w',encoding='utf-8') as stream:
                    with redirect_stdout(stream):summary=runner.run()
            final=summary['final_evaluation'];random=summary['random_layout_evaluation']
            row=dict(method=label,seed=int(seed),sampled_steps=summary['sampled_steps'],
                initial_return=summary['initial_evaluation']['episode_return'],episode_return=final['episode_return'],
                return_change=summary['return_change'],task_completion_rate=final['task_completion_rate'],
                collision_rate=final['collision_rate'],random_return=random['episode_return'],
                random_completion_rate=random['task_completion_rate'],random_collision_rate=random['collision_rate'],
                actor_optimizer_steps=summary['actor_optimizer_steps'],critic_optimizer_steps=summary['critic_optimizer_steps'],
                actor_gradient_norm=summary['last_optimization']['actor_gradient_norm'],
                critic_gradient_norm=summary['last_optimization']['critic_gradient_norm'],
                ratio_error=summary['last_optimization']['pre_update_ratio_max_abs_error'],
                elapsed_seconds=summary['elapsed_seconds'])
            rows.append(row)
            with (output/'runs.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            print(f"[ABLATION] {label} seed={seed} return={row['initial_return']:.3f}->{row['episode_return']:.3f} completion={row['task_completion_rate']:.2f}",flush=True)
    aggregates={}
    for method in dict.fromkeys(r['method'] for r in rows):
        group=[r for r in rows if r['method']==method]
        aggregates[method]={key:dict(mean=float(np.mean([r[key] for r in group])),std=float(np.std([r[key] for r in group],ddof=1)))
            for key in group[0] if key not in ('method','seed')}
        aggregates[method]['positive_return_change_seeds']=sum(r['return_change']>0 for r in group)
    result=dict(budget_per_run=budget,seeds=list(seeds),runs=rows,aggregates=aggregates,
        total_sampled_steps=sum(r['sampled_steps'] for r in rows),
        protocol='same environment/reward/action/PPO/evaluation seeds/budget; actors and history-reset flag differ',
        automatic_budget_expansion=False,stable_superiority_claim=False)
    (output/'ablation_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

