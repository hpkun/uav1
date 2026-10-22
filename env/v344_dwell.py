"""Deterministic, sampled continuous-dwell state for an abstract simulation.

Sampling is post-motion/pre-event. The first valid sample counts as one dt;
completion is inclusive at ceil(dwell_seconds/dt). No interpolation, ammunition,
cooldown, target attitude gate, or random hit law is part of this mechanism.
"""
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math
import numpy as np
from .geometry import engagement_geometry
from .v36_geometry import angle_within

EVENT_SEMANTICS = 'continuous_dwell_neutralization'


def required_dwell_steps(dwell_seconds, dt):
    for value in (dwell_seconds,dt):
        if isinstance(value,bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('dwell_seconds and dt must be finite and positive')
    # Decimal strings avoid .3/.1 accidentally rounding upward to four.
    return int((Decimal(str(dwell_seconds))/Decimal(str(dt))).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class DwellEnvelope:
    range_min: float
    range_max: float
    off_boresight_max: float
    ammo_per_aircraft = 0  # allocation compatibility only; never an event limit

    def eligibility(self, attackers, targets, ammo=None):
        shape=(len(attackers),len(targets))
        eligible=np.zeros(shape,dtype=bool);distance=np.zeros(shape,dtype=float)
        for s in (*attackers,*targets):
            if not np.isfinite(s.as_array()).all():
                raise FloatingPointError('nonfinite state in dwell geometry')
        for i,a in enumerate(attackers):
            for j,b in enumerate(targets):
                g=engagement_geometry(a,b);distance[i,j]=g.distance
                eligible[i,j]=(a.alive and b.alive and self.range_min<=g.distance<=self.range_max
                               and angle_within(g.off_boresight,self.off_boresight_max))
        return eligible,distance

    def qualifies(self, attacker, target, ammo=None):
        return bool(self.eligibility([attacker],[target])[0][0,0])


class DwellState:
    def __init__(self, size, required):
        self.size=size;self.required=required
        self.targets=np.full(size,-1,dtype=np.int64)
        self.counts=np.zeros(size,dtype=np.int64)
        self.eligible=np.zeros(size,dtype=bool)
        self.sampled_targets=self.targets.copy();self.sampled_counts=self.counts.copy()
        self.completed_this_step=np.zeros(size,dtype=bool)
        self.metrics=dict(dwell_eligible_pair_steps=0,dwell_active_agent_steps=0,dwell_started=0,
            dwell_interrupted=0,dwell_target_switches=0,dwell_completed=0,dwell_neutralizations=0,
            first_dwell_start_step=None,first_dwell_complete_step=None,first_dwell_geometry_step=None,
            max_dwell_duration_steps=0)
        self.duration_sum=0;self.finished_segments=0

    def _close(self, i, interrupted):
        if self.counts[i]:
            self.duration_sum+=int(self.counts[i]);self.finished_segments+=1
            self.metrics['dwell_interrupted']+=int(interrupted)
        self.targets[i]=-1;self.counts[i]=0;self.eligible[i]=False

    def update(self, eligible, distance, step, selected=None):
        eligible=np.asarray(eligible,dtype=bool);distance=np.asarray(distance,dtype=float)
        if eligible.shape!=distance.shape or eligible.ndim!=2 or eligible.shape[0]!=self.size or not np.isfinite(distance).all():
            raise ValueError('invalid dwell pair arrays')
        if selected is not None:
            selected=np.asarray(selected)
            if selected.shape!=(self.size,) or selected.dtype.kind not in 'iu' or np.any((selected < -1)|(selected>=eligible.shape[1])):
                raise ValueError('invalid dwell target identities')
        self.completed_this_step.fill(False)
        n=int(eligible.sum());self.metrics['dwell_eligible_pair_steps']+=n
        if n and self.metrics['first_dwell_geometry_step'] is None:self.metrics['first_dwell_geometry_step']=int(step)
        events=[]
        for i in range(self.size):
            if selected is None:
                js=np.flatnonzero(eligible[i])
                j=int(min(js,key=lambda j:(distance[i,j],j))) if js.size else -1
            else:
                j=int(selected[i]);j=j if j>=0 and eligible[i,j] else -1
            old=int(self.targets[i])
            if j!=old:
                self.metrics['dwell_target_switches']+=int(old>=0 and j>=0)
                self._close(i,interrupted=True)
            if j<0:continue
            if self.counts[i]==0:
                self.metrics['dwell_started']+=1
                if self.metrics['first_dwell_start_step'] is None:self.metrics['first_dwell_start_step']=int(step)
            self.targets[i]=j;self.eligible[i]=True
            self.counts[i]+=1;self.metrics['dwell_active_agent_steps']+=1
            self.metrics['max_dwell_duration_steps']=max(self.metrics['max_dwell_duration_steps'],int(self.counts[i]))
            if self.counts[i]==self.required:
                self.completed_this_step[i]=True;self.metrics['dwell_completed']+=1
                if self.metrics['first_dwell_complete_step'] is None:self.metrics['first_dwell_complete_step']=int(step)
                events.append((i,j,True))
        self.sampled_targets=self.targets.copy();self.sampled_counts=self.counts.copy()
        # A completion ends its segment even in an offline nonlethal probe.
        # sampled_* retains the completion identity/count after progress clears.
        for i,_,_ in events:
            self._close(i,interrupted=False)
        return events

    def invalidate(self, attackers, targets, episode_end=False):
        for i,j in enumerate(self.targets.copy()):
            if j>=0 and (episode_end or not attackers[i].alive or not targets[j].alive):
                self._close(i,interrupted=True)

    def diagnostics(self):
        return dict(self.metrics,mean_dwell_duration_steps=self.duration_sum/max(1,self.finished_segments),
                    dwell_finished_segments=self.finished_segments)
