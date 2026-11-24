"""Compact, algorithm-neutral terminal progress for interception training."""
from collections import deque
import math
import time


def _number(value,digits=3):
    return '--' if value is None or not math.isfinite(value) else f'{value:.{digits}f}'


def _signed(value,digits=2):
    return '--' if value is None or not math.isfinite(value) else f'{value:+.{digits}f}'


def _percent(value):
    return '--' if value is None else f'{100*value:.1f}%'


def _duration(seconds):
    if seconds is None or not math.isfinite(seconds):return '--'
    seconds=max(0,int(seconds))
    hours,rem=divmod(seconds,3600)
    minutes,sec=divmod(rem,60)
    return f'{hours}h{minutes:02d}m' if hours else f'{minutes}m{sec:02d}s'


class ProgressLogger:
    def __init__(self,algorithm,total_steps,interval,*,attack_mode,window=100,clock=time.monotonic,
                 attacker_count=5):
        if interval<1 or attack_mode not in ('direct_rush','split_attack','jia_straight'):
            raise ValueError('Invalid log interval or attack mode')
        self.algorithm=algorithm
        self.attack_mode=attack_mode
        self.attacker_count=attacker_count
        self.total_steps=total_steps
        self.interval=interval
        self.clock=clock
        self.started=clock()
        self.last_print=self.started
        self.next_step=interval
        self.episodes=deque(maxlen=window)
        self.metrics=deque(maxlen=window)

    def episode(self,row):
        if row['attack_mode']!=self.attack_mode:
            raise ValueError('Episode attack mode differs from training mode')
        self.episodes.append(row)

    def update(self,row):
        self.metrics.append(row)

    def ready(self,step,*,final=False):
        now=self.clock()
        if not (final or step>=self.next_step or now-self.last_print>=30):return False
        self.last_print=now
        self.next_step=(step//self.interval+1)*self.interval
        return True

    def snapshot(self,step,episodes,updates,*,replay_size=None,replay_capacity=None,warmup=None):
        elapsed=max(self.clock()-self.started,1e-9)
        speed=step/elapsed
        rows=list(self.episodes)
        n=len(rows)
        recent={self.attack_mode:dict(episodes=n,
                all_capture_rate=sum(row['outcome']=='all_captured' for row in rows)/n if n else None,
                failure_rate=sum(row['outcome']!='all_captured' for row in rows)/n if n else None,
                penetration_rate=sum(row.get('actual_penetrated_count',row.get('penetrated_count',0))>0
                                     for row in rows)/n if n else None,
                time_limit_rate=sum(row.get('time_limit_occurred',row['outcome']=='time_limit')
                                    for row in rows)/n if n else None,
                capture_efficiency=sum(row['captured_count'] for row in rows)/(self.attacker_count*n) if n else None,
                mean_team_return=sum(row['team_return'] for row in rows)/n if n else None)}
        keys=('actor_loss','critic_loss','entropy','approx_kl','td_abs',
              'actor_grad_norm','actor_clip_rate','critic_grad_norm','critic_clip_rate','active_critic_count',
              'critic_grad_norm_mean_agent',
              'critic_grad_norm_max_agent','policy_log_std_mean','policy_std_mean',
              'advantage_raw_mean','advantage_raw_std','q_mean','q_std','target_q_mean',
              'explained_variance','clip_fraction','value_norm_std',
              'per_beta','per_is_weight_mean','per_is_weight_min','per_is_ess',
              'per_priority_mean','per_priority_max','per_critic_grad_clipped',
              'per_buffer_capture_fraction','per_sample_capture_fraction',
              'per_buffer_first_breach_fraction','per_sample_first_breach_fraction',
              'per_buffer_boundary_fraction','per_sample_boundary_fraction',
              'per_buffer_ordinary_fraction','per_sample_ordinary_fraction')
        learning={key:sum(values)/len(values) for key in keys
                  if (values:=[float(row[key]) for row in self.metrics
                               if row.get(key) is not None and math.isfinite(float(row[key]))])}
        return dict(schema='InterceptionProgress/1',algorithm=self.algorithm,
                    attack_mode=self.attack_mode,step=step,total_steps=self.total_steps,
                    percent=step/self.total_steps,episodes=episodes,updates=updates,
                    elapsed_s=elapsed,steps_per_s=speed,eta_s=(self.total_steps-step)/speed if speed>0 else None,
                    recent_window=len(self.episodes),recent=recent,learning=learning,
                    replay_size=replay_size,replay_capacity=replay_capacity,warmup=warmup)

    @staticmethod
    def lines(row):
        header=(f"[train {row['algorithm']}] {row['step']:,}/{row['total_steps']:,} ({_percent(row['percent'])})"
                f" | episodes {row['episodes']} | updates {row['updates']}"
                f" | {row['steps_per_s']:.0f} steps/s")
        if row['replay_size'] is not None:
            header+=f" | replay {row['replay_size']:,}"
            if row['replay_capacity'] is not None:
                header+=f"/{row['replay_capacity']:,}"
            if row['warmup'] is not None and row['replay_size']<row['warmup']:
                header+=f"/{row['warmup']:,} warmup"
        mode=row['attack_mode']
        label={'direct_rush':'direct','split_attack':'split','jia_straight':'jia'}[mode]
        value=row['recent'][mode]
        recent=(f"  recent {row['recent_window']}: {label} n={value['episodes']}"
                f" all={_percent(value['all_capture_rate'])} cap={_percent(value['capture_efficiency'])}"
                f" fail={_percent(value['failure_rate'])}"
                f" return={_number(value['mean_team_return'],2)}")
        if value['time_limit_rate']:
            recent+=f" timeout={_percent(value['time_limit_rate'])}"
        learning=row['learning']
        if learning:
            keys=(('actor_loss','critic_loss','td_abs','q_mean') if row['algorithm']=='maddpg' else
                  ('actor_loss','critic_loss','entropy','approx_kl','explained_variance'))
            detail='  learn: '+' '.join(f'{key}={_number(learning.get(key),4)}' for key in keys if key in learning)
        else:
            detail='  learn: waiting for first update'
        lines=[header,recent,detail]
        if row['algorithm']=='maddpg' and learning:
            lines.append('  diag: actor_grad_norm='+_number(learning.get('actor_grad_norm'),4)+
                         ' actor_clip='+_percent(learning.get('actor_clip_rate'))+
                         ' critic_grad_norm='+_number(learning.get('critic_grad_norm'),4)+
                         ' critic_clip='+_percent(learning.get('critic_clip_rate'))+
                         ' active_critics='+_number(learning.get('active_critic_count'),1))
        elif row['algorithm']!='maddpg' and learning:
            lines.append('  diag: '+' '.join(f'{key}={_number(learning[key],4)}' for key in
                         ('actor_grad_norm','critic_grad_norm','policy_std_mean','value_norm_std')
                         if key in learning))
        return tuple(lines)

    @staticmethod
    def evaluation(step,summaries):
        if len(summaries)!=1:
            raise ValueError('Periodic evaluation must contain only the training attack mode')
        mode,row=next(iter(summaries.items()))
        label={'direct_rush':'direct','split_attack':'split','jia_straight':'jia'}[mode]
        status=(f'[eval {step:,}] {label}: all={_percent(row["all_capture_rate"])}'
                f' cap={_percent(row["capture_efficiency"])}'
                f' fail={_percent(row.get("failure_rate",1-row["all_capture_rate"]))}'
                f' return={_number(row.get("mean_team_return"),2)} ({row["episodes"]} eps)')
        if row.get('time_limit_rate'):
            status+=f' timeout={_percent(row["time_limit_rate"])}'
        components=row.get('mean_reward_components',{})
        names=('capture','penetration','approach','heading','boundary','danger')
        detail=' '.join(f'{name}={_signed(components[name])}' for name in names if name in components)
        return f'{status}\n  reward/ep (undiscounted): {detail}'
