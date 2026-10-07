"""Raw-environment evaluation with one canonical recurrent-state lifecycle."""
import numpy as np
from env.factory import make_combat_environment
from algorithm.common.evaluator import episode_return_metrics,persistent_mission_metrics
from algorithm.modules.wave_survival_pbrs import mission_context_numpy

def per_wave_episode_diagnostics(info,total_waves=3):
 """Flatten only terminal fields recorded by the environment (no inference)."""
 records={int(row["wave_index"]):row for row in info.get("per_wave_metrics",[])};out={}
 mappings={"red_survivors_after_wave":"red_survivors_end","blue_survivors_after_wave":"blue_survivors_end","red_attack_kills_after_wave":"red_attack_kills","red_boundary_losses_after_wave":"red_boundary_exits","red_ground_losses_after_wave":"red_ground_losses"}
 for wave in range(1,total_waves+1):
  row=records.get(wave)
  out[f"wave_{wave}_recorded"]=row is not None
  out[f"wave_{wave}_cleared"]=False if row is None else bool(row.get("wave_cleared",False))
  for name,source in mappings.items():out[f"{name}_{wave}"]=None if row is None else row.get(source)
  out[f"wave_{wave}_entry_step"]=None if row is None else row.get("start_step")
  out[f"wave_{wave}_duration"]=None if row is None else row.get("duration_steps")
 return out

def aggregate_per_wave_diagnostics(records,total_waves=3):
 """Aggregate explicit record- and clear-conditioned wave diagnostics."""
 result={};names=("red_survivors_after_wave","blue_survivors_after_wave","red_attack_kills_after_wave","red_boundary_losses_after_wave","red_ground_losses_after_wave")
 for wave in range(1,total_waves+1):
  for name in names:
   values=[r[f"{name}_{wave}"] for r in records if r[f"wave_{wave}_recorded"]]
   clear_values=[r[f"{name}_{wave}"] for r in records if r[f"wave_{wave}_cleared"]]
   value=float(np.mean(values)) if values else None;clear_value=float(np.mean(clear_values)) if clear_values else None
   result[f"average_{name}_{wave}_conditional_on_record"]=value
   result[f"average_{name}_{wave}_conditional_on_clear"]=clear_value
   result[f"{name}_{wave}"]=value
  for name in (f"wave_{wave}_entry_step",f"wave_{wave}_duration"):
   values=[r[name] for r in records if r[name] is not None];result[name]=float(np.mean(values)) if values else None
   clear_values=[r[name] for r in records if r[f"wave_{wave}_cleared"]]
   result[name+"_conditional_on_clear"]=float(np.mean(clear_values)) if clear_values else None
 return result

def evaluate_modular_episode(trainer, env_config, seed, include_trace=False):
 env=make_combat_environment(env_config);obs,_=env.reset(int(seed));alive=env.red_alive_mask.copy()
 ah,ch=trainer.initial_hidden(1);wave=1;total=int(env_config.get("persistent_waves",{}).get("total_waves",1));ret=np.zeros(4);ep=np.zeros(1,np.float32)
 hta=bool(getattr(getattr(trainer,"hierarchical_temporal_abstraction",None),"enabled",False))
 options=trainer.manager_act(obs[None],alive[None],True) if hta else None
 manager_duration=0;actions_trace=[];wave_trace=[];option_trace=[];manager_decision_trace=[];phase_flags=np.ones(1,bool);phase_trace=[];wave_context_trace=[];pre_hidden_norm_trace=[]
 if hta and include_trace:manager_decision_trace.append({"step":0,"reason":"episode_start","options":options[0].copy()})
 while True:
  ctx=mission_context_numpy(
   trainer,np.asarray([wave]),np.asarray([total]),env.blue_alive_mask[None],
   np.asarray([env.steps]),env.max_steps)
  ctx=trainer.actor_context_numpy(np.asarray([wave]),alive[None],ctx)
  if include_trace and trainer.recurrent.state_memory:
   pre_hidden=trainer.prepare_actor_hidden(ah,alive[None],np.asarray([wave]),phase_flags)
   pre_hidden_norm_trace.append(np.linalg.norm(pre_hidden[0],axis=-1));wave_context_trace.append(ctx[0].copy())
  if hta:
   actions,ah=trainer.act(obs[None],alive[None],True,False,ctx,ah,ep,option_ids=options,wave_indices=np.asarray([wave]))
   _,ch=trainer.values_step(obs[None],alive[None],ctx,ch,ep,option_ids=options)
  else:
   kwargs={"actor_phase_reset_flags":phase_flags} if trainer.recurrent.wave_boundary_reset else {}
   actions,ah=trainer.act(obs[None],alive[None],True,False,ctx,ah,ep,wave_indices=np.asarray([wave]),**kwargs)
   _,ch=trainer.values_step(obs[None],alive[None],ctx,ch,ep)
  if include_trace:
   actions_trace.append(actions[0].copy());wave_trace.append(wave);phase_trace.append(bool(phase_flags[0]))
   if hta:option_trace.append(options[0].copy())
  obs,reward,terminated,truncated,info=env.step(actions[0]);ret+=reward
  manager_duration+=1
  alive=np.asarray(info["red_alive_mask"],np.float32)
  ah=trainer.recurrent.apply_alive(ah,alive[None]);ch=trainer.recurrent.apply_alive(ch,alive[None])
  ep[:]=1;wave=int(info.get("wave_index",1));total=int(info.get("total_waves",total))
  phase_flags[:]=bool(info.get("spawned_next_wave",False))
  if hta and not (terminated or truncated) and (bool(info.get("spawned_next_wave",False)) or manager_duration>=trainer.hierarchical_temporal_abstraction.decision_interval_steps):
   reason="wave_transition" if info.get("spawned_next_wave",False) else "periodic"
   options=trainer.manager_act(obs[None],alive[None],True);manager_duration=0
   if include_trace:manager_decision_trace.append({"step":int(env.steps),"reason":reason,"options":options[0].copy()})
  if terminated or truncated:
   team,agent=episode_return_metrics(ret);record={"episode_return":team,"mean_agent_episode_return":agent,**info,**per_wave_episode_diagnostics(info,total)}
   if include_trace:
    record.update({"action_trace":np.asarray(actions_trace),"wave_trace":np.asarray(wave_trace)})
    if trainer.recurrent.wave_boundary_reset:record["actor_recurrent_phase_reset_trace"]=np.asarray(phase_trace)
    if trainer.recurrent.state_memory:record.update({"actor_wave_one_hot_trace":np.asarray(wave_context_trace),"actor_hidden_before_step_norm_trace":np.asarray(pre_hidden_norm_trace)})
    if hta:record.update({"option_trace":np.asarray(option_trace),"manager_decision_trace":manager_decision_trace})
   return record

def evaluate_modular(trainer,env_config,seeds):
 records=[evaluate_modular_episode(trainer,env_config,seed) for seed in seeds]
 mean=lambda k:float(np.mean([r[k] for r in records]))
 result={"average_return":mean("episode_return"),"average_agent_return":mean("mean_agent_episode_return"),"win_rate":mean("red_success"),"loss_rate":mean("blue_win"),"draw_rate":mean("draw"),"timeout_rate":float(np.mean([r["termination_reason"]=="red_failure_timeout" for r in records])),"average_red_loss":mean("red_losses"),"average_blue_loss":mean("blue_losses"),"average_red_attack_kills":mean("red_attack_kills"),"average_blue_attack_kills":mean("blue_attack_kills"),"average_red_boundary_exits":mean("red_boundary_exits"),"average_blue_boundary_exits":mean("blue_boundary_exits"),"evaluation_boundary_exit_rate":float(np.mean([r["red_boundary_exits"]>0 for r in records])),"average_red_ground_losses":mean("red_ground_losses"),"average_blue_ground_losses":mean("blue_ground_losses"),"average_episode_length":mean("episode_length"),"evaluation_episodes":len(records),**{f"{side}_{event}_episode_rate":float(np.mean([r[f"{side}_first_{event}_step"] is not None for r in records])) for side in ("red","blue") for event in ("fire_window","attempt","hit","kill")},**{f"average_episode_{name}_total":mean(f"episode_{name}_total") for name in ("r1","r2","r3","r4")},**persistent_mission_metrics(records)}
 result.update(aggregate_per_wave_diagnostics(records,int(env_config.get("persistent_waves",{}).get("total_waves",1))))
 return result
__all__=["aggregate_per_wave_diagnostics","evaluate_modular","evaluate_modular_episode","per_wave_episode_diagnostics"]
