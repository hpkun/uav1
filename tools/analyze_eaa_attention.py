"""Trace one deterministic EA/EAA episode. Entity slot weights are not causal explanations."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.eaa_mappo.evaluation import load_checkpoint
from env.factory import make_combat_environment


def trace_attention(checkpoint, env_config, algorithm_config, episode_seed, output_dir, device='cuda'):
    trainer = load_checkpoint(checkpoint,algorithm_config,env_config,device)
    if trainer.actor_type != 'entity_attention':
        raise ValueError('self/ally/enemy entity trace requires EA-MAPPO or EAA-MAPPO')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):raise FileExistsError('trace output must be fresh')
    output.mkdir(parents=True,exist_ok=True)
    env = make_combat_environment(env_config)
    obs,_ = env.reset(int(episode_seed))
    rows=[];heatmap=[];step=0;returns=np.zeros(4)
    while True:
        alive=env.red_alive_mask.copy()
        with torch.no_grad():
            _,weights=trainer.actor.distribution(torch.as_tensor(obs,dtype=torch.float32,device=device),return_attention=True)
        weights=weights.cpu().numpy()
        for agent in np.flatnonzero(alive):
            # Head-average for summary; retain all heads and all eight weights in JSON.
            w=weights[agent].mean(0);slot=int(w.argmax())
            rows.append(dict(step=step,time=step*env_config['simulation']['dt'],red_agent=int(agent),
                self_mass=float(w[0]),ally_mass=float(w[1:4].sum()),enemy_mass=float(w[4:8].sum()),
                strongest_entity_slot=slot,strongest_attention_weight=float(w[slot])))
            heatmap.append(dict(step=step,red_agent=int(agent),self_query_weights_per_head=weights[agent].tolist()))
        actions=trainer.act(obs,alive,deterministic=True)
        obs,reward,terminated,truncated,info=env.step(actions)
        returns+=reward;step+=1
        if terminated or truncated:break
    with (output/'attention_trace.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata=dict(algorithm=trainer.variant,implementation_version=1,episode_seed=int(episode_seed),
        checkpoint=str(Path(checkpoint).resolve()),steps=step,rows=len(rows),team_return=float(returns.sum()),
        slot_semantics='Observation slots, not persistent tactical identities. ally slots enumerate team indices excluding focal Red; enemy slots enumerate opponent array indices; no slot compaction on death in v3.38.',
        token_order=['self','ally_slot1','ally_slot2','ally_slot3','enemy_slot1','enemy_slot2','enemy_slot3','enemy_slot4'])
    (output/'attention_trace.json').write_text(json.dumps(dict(metadata=metadata,rows=rows,heatmap_data=heatmap),indent=2),encoding='utf-8')
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--env-config',default='configs/combat_environment_v338.yaml')
    p.add_argument('--algorithm-config')
    p.add_argument('--episode-seed',type=int,required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    config_path=Path(a.algorithm_config) if a.algorithm_config else Path(a.checkpoint).parent/'algorithm_config.yaml'
    print(json.dumps(trace_attention(a.checkpoint,yaml.safe_load(Path(a.env_config).read_text(encoding='utf-8')),
        yaml.safe_load(config_path.read_text(encoding='utf-8')),a.episode_seed,a.output_dir,a.device),indent=2))


if __name__=='__main__':main()
