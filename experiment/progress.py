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
    def __init__(self,algorithm,total_steps,interval,*,attack_mode,window=100,clock=time.monotonic):
        if interval<1 or attack_mode not in ('direct_rush','split_attack'):
            raise ValueError('Invalid log interval or attack mode')
        self.algorithm=algorithm
        self.attack_mode=attack_mode
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

    def snapshot(self,step,episodes,updates,*,replay_size=None,warmup=None):
        elapsed=max(self.clock()-self.started,1e-9)
        speed=step/elapsed
        rows=list(self.episodes)
        n=len(rows)
        recent={self.attack_mode:dict(episodes=n,
                all_capture_rate=sum(row['outcome']=='all_captured' for row in rows)/n if n else None,
                penetration_rate=sum(row['outcome']=='penetrated' for row in rows)/n if n else None,
                time_limit_rate=sum(row['outcome']=='time_limit' for row in rows)/n if n else None,
                capture_efficiency=sum(row['captured_count'] for row in rows)/(5*n) if n else None,
                mean_team_return=sum(row['team_return'] for row in rows)/n if n else None)}
        keys=('actor_loss','critic_loss','entropy','approx_kl','td_abs')
        learning={key:sum(values)/len(values) for key in keys
                  if (values:=[float(row[key]) for row in self.metrics
                               if row.get(key) is not None and math.isfinite(float(row[key]))])}
        return dict(schema='InterceptionProgress/1',algorithm=self.algorithm,
                    attack_mode=self.attack_mode,step=step,total_steps=self.total_steps,
                    percent=step/self.total_steps,episodes=episodes,updates=updates,
                    elapsed_s=elapsed,steps_per_s=speed,eta_s=(self.total_steps-step)/speed if speed>0 else None,
                    recent_window=len(self.episodes),recent=recent,learning=learning,
                    replay_size=replay_size,warmup=warmup)

    @staticmethod
    def lines(row):
        header=(f"[train {row['algorithm']}] {row['step']:,}/{row['total_steps']:,} ({_percent(row['percent'])})"
                f" | episodes {row['episodes']} | updates {row['updates']}"
                f" | {row['steps_per_s']:.0f} steps/s | ETA {_duration(row['eta_s'])}")
        mode=row['attack_mode']
        label='direct' if mode=='direct_rush' else 'split'
        value=row['recent'][mode]
        recent=(f"  recent {row['recent_window']}: {label} n={value['episodes']}"
                f" all={_percent(value['all_capture_rate'])} breach={_percent(value['penetration_rate'])}"
                f" cap={_percent(value['capture_efficiency'])} timeout={_percent(value['time_limit_rate'])}"
                f" return={_number(value['mean_team_return'],2)}")
        learning=row['learning']
        if learning:
            detail='  learn: '+' '.join(f'{key}={_number(learning.get(key),4)}' for key in
                ('actor_loss','critic_loss','entropy','approx_kl','explained_variance',
                 'value_norm_std','policy_log_std_mean') if key in learning)
            if 'td_abs' in learning:detail+=f" td_abs={_number(learning['td_abs'],4)}"
        else:
            detail='  learn: waiting for first update'
        if row['replay_size'] is not None:
            detail+=f" | replay={row['replay_size']:,}"
            if row['warmup'] is not None and row['replay_size']<row['warmup']:
                detail+=f"/{row['warmup']:,} warmup"
        return header,recent,detail

    @staticmethod
    def evaluation(step,summaries):
        if len(summaries)!=1:
            raise ValueError('Periodic evaluation must contain only the training attack mode')
        mode,row=next(iter(summaries.items()))
        label='direct' if mode=='direct_rush' else 'split'
        status=(f'[eval {step:,}] {label}: all={_percent(row["all_capture_rate"])}'
                f' breach={_percent(row["penetration_rate"])}'
                f' cap={_percent(row["capture_efficiency"])} ({row["episodes"]} eps)')
        components=row.get('mean_reward_components',{})
        names=('capture','penetration','approach','boundary','danger')
        detail=' '.join(f'{name}={_signed(components[name])}' for name in names if name in components)
        return (f'{status}\n  reward/ep (undiscounted): {detail}'
                f' total={_signed(row.get("mean_team_return"))}')
