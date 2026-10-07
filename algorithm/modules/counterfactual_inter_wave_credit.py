"""CAIW-MAPPO V2: held-out, action-conditioned inter-wave credit support."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import numpy as np

from .base import CapabilityModule

CAIW_MAPPO_VERSION = 1
W1_TO_W2 = "w1_to_w2"
W1_TO_W3 = "w1_to_w3"
W2_TO_W3 = "w2_to_w3"
CAIW_TASKS = (W1_TO_W2, W1_TO_W3, W2_TO_W3)
TASK_SOURCE_WAVE = {W1_TO_W2: 1, W1_TO_W3: 1, W2_TO_W3: 2}
TASK_HEAD = {W1_TO_W2: 0, W1_TO_W3: 1, W2_TO_W3: 1}


def binary_auroc(labels, predictions):
    """Dependency-free AUROC using average ranks for ties; NA for one class."""
    y=np.asarray(labels,dtype=np.int64);score=np.asarray(predictions,dtype=np.float64)
    positive=int((y==1).sum());negative=int((y==0).sum())
    if positive==0 or negative==0:return None
    order=np.argsort(score,kind="mergesort");ranks=np.empty(len(score),dtype=np.float64)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and score[order[end]]==score[order[start]]:end+=1
        ranks[order[start:end]]=(start+1+end)/2.0;start=end
    return float((ranks[y==1].sum()-positive*(positive+1)/2)/(positive*negative))


def prior_corrected_probability(raw_logits, prior, epsilon=1e-4):
    """Case-control correction from balanced-training logits to a recent prior."""
    import torch
    p=float(np.clip(prior,epsilon,1.0-epsilon));bias=math.log(p/(1.0-p))
    return torch.sigmoid(raw_logits+bias)


def freshness_mask(log_ratios, alive_masks, low=.8, high=1.2):
    """A state is fresh iff every alive agent is inside the ratio interval."""
    values=np.asarray(log_ratios,dtype=np.float64);alive=np.asarray(alive_masks)>0.5
    valid=(values>=math.log(low))&(values<=math.log(high))
    return np.all(valid|~alive,axis=-1)


class CounterfactualInterWaveCreditModule(CapabilityModule):
    name="counterfactual_inter_wave_credit"

    def __init__(self,config=None):
        super().__init__(config);c=self.config;self.version=CAIW_MAPPO_VERSION
        self.max_waves=int(c.get("max_waves",3));self.outcome_critic_learning_rate=float(c.get("outcome_critic_learning_rate",3e-4))
        self.train_segments_per_class=int(c.get("train_segments_per_class",128));self.validation_segments_per_wave=int(c.get("validation_segments_per_wave",256))
        self.prior_window_segments_per_wave=int(c.get("prior_window_segments_per_wave",512));self.max_states_per_segment=int(c.get("max_states_per_segment",64))
        self.train_states_per_class_per_task=int(c.get("train_states_per_class_per_task",64));self.critic_updates_per_rollout=int(c.get("critic_updates_per_rollout",4))
        self.min_train_segments_per_class=int(c.get("min_train_segments_per_class",8));self.freshness_ratio_low=float(c.get("freshness_ratio_low",.8));self.freshness_ratio_high=float(c.get("freshness_ratio_high",1.2))
        self.min_fresh_states_per_validation_segment=int(c.get("min_fresh_states_per_validation_segment",4));self.validation_interval_updates=int(c.get("validation_interval_updates",10))
        self.min_validation_segments=int(c.get("min_validation_segments",64));self.min_validation_positive_segments=int(c.get("min_validation_positive_segments",16));self.min_validation_negative_segments=int(c.get("min_validation_negative_segments",16))
        self.min_validation_auroc=float(c.get("min_validation_auroc",.60));self.min_validation_brier_skill=float(c.get("min_validation_brier_skill",0.));self.readiness_consecutive_passes=int(c.get("readiness_consecutive_passes",3))
        self.counterfactual_samples=int(c.get("counterfactual_samples",4));self.counterfactual_batch_size=int(c.get("counterfactual_batch_size",512));self.auxiliary_gradient_ratio_cap=float(c.get("auxiliary_gradient_ratio_cap",.25))
        self.gradient_projection=str(c.get("gradient_projection","asymmetric_tactical_preserving"));self.prior_probability_epsilon=float(c.get("prior_probability_epsilon",1e-4))
        if self.enabled:
            if self.max_waves!=3 or self.counterfactual_samples!=4:raise ValueError("CAIW V2 requires three waves and K=4")
            if self.gradient_projection!="asymmetric_tactical_preserving":raise ValueError("unsupported CAIW projection")
            if not 0<self.auxiliary_gradient_ratio_cap<=1:raise ValueError("invalid CAIW trust cap")
        self.train_replay={1:{key:deque(maxlen=self.train_segments_per_class) for key in ("00","10","11")},2:{key:deque(maxlen=self.train_segments_per_class) for key in ("0","1")}}
        self.validation={w:deque(maxlen=self.validation_segments_per_wave) for w in (1,2)}
        self.prior_window={w:deque(maxlen=self.prior_window_segments_per_wave) for w in (1,2)}
        self.validation_pass_streak={task:0 for task in CAIW_TASKS};self.task_ready={task:False for task in CAIW_TASKS}
        self.first_ready_sampled_steps={task:None for task in CAIW_TASKS};self.ready_activation_count={task:0 for task in CAIW_TASKS};self.ready_deactivation_count={task:0 for task in CAIW_TASKS}
        self.validation_check_count=0;self.next_segment_id=0

    @staticmethod
    def labels(waves_cleared):
        c=int(waves_cleared);return {"c2":int(c>=2),"c3":int(c>=3)}

    @staticmethod
    def episode_split(episode_group_id):return "validation" if int(episode_group_id)%5==0 else "train"

    def cap_segment(self,rows,source_wave,waves_cleared,episode_group_id,source_env_id):
        if not rows:raise ValueError("empty CAIW segment")
        wave=int(source_wave);labels=self.labels(waves_cleared)
        if labels["c3"] and not labels["c2"]:raise RuntimeError("invalid CAIW outcome class 01")
        count=len(rows)
        indices=list(range(count)) if count<=self.max_states_per_segment else np.linspace(0,count-1,self.max_states_per_segment,dtype=np.int64).tolist()
        forced=[i for i,row in enumerate(rows) if row.get("wave_cleared_this_step",False)]
        for index in forced:
            if index not in indices:indices[-2]=index
        indices=sorted(set([0,count-1,*indices]));chosen=[rows[i] for i in indices]
        segment={"segment_id":self.next_segment_id,"episode_group_id":int(episode_group_id),"source_env_id":int(source_env_id),"source_wave":wave,"split":self.episode_split(episode_group_id),"label_c2":labels["c2"],"label_c3":labels["c3"],"episode_waves_cleared":int(waves_cleared),"original_state_count":count,"sample_indices":np.asarray(indices,dtype=np.int64)}
        for dst,src,dtype in (("observations","observation",np.float32),("alive_masks","alive_mask",np.float32),("actions","action",np.float32),("raw_actions","raw_action",np.float32),("behavior_log_probs","behavior_log_prob",np.float32)):
            segment[dst]=np.asarray([row[src] for row in chosen],dtype=dtype)
        for dst,src,dtype in (("remaining_horizons","remaining_horizon",np.float32),("collection_ppo_update_ids","collection_ppo_update_id",np.int64),("collection_sampled_steps","collection_sampled_steps",np.int64),("wave_cleared_flags","wave_cleared_this_step",bool),("spawned_next_wave_flags","spawned_next_wave",bool)):
            segment[dst]=np.asarray([row[src] for row in chosen],dtype=dtype)
        self.next_segment_id+=1;return segment

    def ingest(self,segments):
        for segment in segments or ():
            wave=int(segment["source_wave"]);c2=int(segment["label_c2"]);c3=int(segment["label_c3"])
            if c3 and not c2:raise RuntimeError("invalid CAIW outcome class 01")
            if segment["split"]=="validation":
                self.validation[wave].append(deepcopy(segment))
            else:
                self.train_replay[wave][f"{c2}{c3}" if wave==1 else str(c3)].append(deepcopy(segment))
                self.prior_window[wave].append({"c2":c2,"c3":c3,"segment_id":int(segment["segment_id"])})
            self.next_segment_id=max(self.next_segment_id,int(segment["segment_id"])+1)

    def task_classes(self,task):
        if task==W1_TO_W2:return [*self.train_replay[1]["10"],*self.train_replay[1]["11"]],[*self.train_replay[1]["00"]]
        if task==W1_TO_W3:return [*self.train_replay[1]["11"]],[*self.train_replay[1]["00"],*self.train_replay[1]["10"]]
        if task==W2_TO_W3:return [*self.train_replay[2]["1"]],[*self.train_replay[2]["0"]]
        raise KeyError(task)

    def task_label(self,task,segment):return int(segment["label_c2"] if task==W1_TO_W2 else segment["label_c3"])

    def recent_prior(self,task):
        wave=TASK_SOURCE_WAVE[task];rows=self.prior_window[wave]
        if not rows:return .5
        key="c2" if task==W1_TO_W2 else "c3";return float(np.mean([row[key] for row in rows]))

    def apply_validation(self,task,metrics,sampled_steps):
        passed=(metrics["validation_segments"]>=self.min_validation_segments and metrics["validation_positive"]>=self.min_validation_positive_segments and metrics["validation_negative"]>=self.min_validation_negative_segments and metrics["validation_auroc"] is not None and metrics["validation_auroc"]>=self.min_validation_auroc and metrics["validation_brier_skill"] is not None and metrics["validation_brier_skill"]>self.min_validation_brier_skill)
        old=self.task_ready[task];self.validation_pass_streak[task]=self.validation_pass_streak[task]+1 if passed else 0
        new=passed and self.validation_pass_streak[task]>=self.readiness_consecutive_passes;self.task_ready[task]=new
        if new and not old:
            self.ready_activation_count[task]+=1
            if self.first_ready_sampled_steps[task] is None:self.first_ready_sampled_steps[task]=int(sampled_steps)
        if old and not new:self.ready_deactivation_count[task]+=1
        return passed

    def state_dict(self):
        return {"version":self.version,"train_replay":{w:{k:list(v) for k,v in rows.items()} for w,rows in self.train_replay.items()},"validation":{w:list(v) for w,v in self.validation.items()},"prior_window":{w:list(v) for w,v in self.prior_window.items()},"validation_pass_streak":deepcopy(self.validation_pass_streak),"task_ready":deepcopy(self.task_ready),"first_ready_sampled_steps":deepcopy(self.first_ready_sampled_steps),"ready_activation_count":deepcopy(self.ready_activation_count),"ready_deactivation_count":deepcopy(self.ready_deactivation_count),"validation_check_count":self.validation_check_count,"next_segment_id":self.next_segment_id}

    def load_state_dict(self,state):
        if not state:return
        if int(state.get("version",-1))!=self.version:raise RuntimeError("CAIW replay version mismatch")
        tr=state["train_replay"];self.train_replay={w:{k:deque(tr.get(w,tr.get(str(w)))[k],maxlen=self.train_segments_per_class) for k in (("00","10","11") if w==1 else ("0","1"))} for w in (1,2)}
        self.validation={w:deque(state["validation"].get(w,state["validation"].get(str(w))),maxlen=self.validation_segments_per_wave) for w in (1,2)}
        self.prior_window={w:deque(state["prior_window"].get(w,state["prior_window"].get(str(w))),maxlen=self.prior_window_segments_per_wave) for w in (1,2)}
        for name in ("validation_pass_streak","task_ready","first_ready_sampled_steps","ready_activation_count","ready_deactivation_count"):setattr(self,name,deepcopy(state[name]))
        self.validation_check_count=int(state["validation_check_count"]);self.next_segment_id=int(state["next_segment_id"])


__all__=["CAIW_MAPPO_VERSION","W1_TO_W2","W1_TO_W3","W2_TO_W3","CAIW_TASKS","TASK_SOURCE_WAVE","TASK_HEAD","binary_auroc","prior_corrected_probability","freshness_mask","CounterfactualInterWaveCreditModule"]
