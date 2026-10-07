"""Modular actor/critic; the disabled topology is state-dict compatible with MAPPO."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from algorithm.mappo.networks import SharedMAPPOActor, CentralizedValueCritic


class ModularMAPPOActor(SharedMAPPOActor):
    def __init__(self, observation_dim=52, action_dim=3, hidden_dim=256,
                 log_std_min=-5.0, log_std_max=2.0, activation="relu",
                 context_dim=0, recurrent_hidden_dim=0,
                 entity_attention_config=None, mission_film_config=None,
                 hierarchical_temporal_abstraction_config=None, wave_segmented_actor_gru=False):
        self.base_observation_dim=int(observation_dim); self.context_dim=int(context_dim)
        self.recurrent_hidden_dim=int(recurrent_hidden_dim)
        self.wave_segmented_actor_gru=bool(wave_segmented_actor_gru)
        film=dict(mission_film_config or {});self.mission_film_enabled=bool(film.get("enabled",False))
        super().__init__(observation_dim if self.mission_film_enabled else observation_dim+self.context_dim, action_dim, hidden_dim,
                         log_std_min, log_std_max, activation)
        if self.mission_film_enabled:
            if self.context_dim<=0:raise ValueError("mission_film requires actor context")
            self.mission_film_mode=str(film["mode"]);self.mission_encoder_hidden_dim=int(film["encoder_hidden_dim"])
            self.mission_film_alpha=float(film["alpha"]);self.mission_film_augmented_residual=bool(film["augmented_residual"])
            self.mission_film_identity_init=bool(film["identity_init"])
            # Mission-only construction must not perturb the RNG stream used by
            # the subsequently-created baseline critic.
            with torch.random.fork_rng(devices=[]):
                self.mission_encoder=nn.Sequential(nn.Linear(self.context_dim,self.mission_encoder_hidden_dim),nn.ReLU(),
                                                   nn.Linear(self.mission_encoder_hidden_dim,self.mission_encoder_hidden_dim),nn.ReLU())
                self.gamma_head=nn.Linear(self.mission_encoder_hidden_dim,hidden_dim)
                self.beta_head=nn.Linear(self.mission_encoder_hidden_dim,hidden_dim)
                self.residual_head=nn.Linear(self.mission_encoder_hidden_dim,hidden_dim)
                for head in (self.gamma_head,self.beta_head,self.residual_head):
                    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        config=dict(entity_attention_config or {})
        self.entity_attention_enabled=bool(config.get("enabled",False))
        self.entity_attention_mode=str(config.get("mode","replacement"))
        self.entity_dim=int(config.get("entity_dim",32));self.entity_attention_heads=int(config.get("attention_heads",2))
        if self.entity_attention_enabled:
            if self.entity_attention_mode not in {"replacement","residual","gated_residual","frozen_base_mean_residual","frozen_base_dual_bounded_mean_residual"}:
                raise ValueError("unsupported entity attention mode")
            if self.base_observation_dim!=52:raise ValueError("entity attention requires the fixed 52D observation layout")
            if self.context_dim:raise ValueError("entity attention cannot be combined with wave context")
            if self.recurrent_hidden_dim:raise ValueError("entity attention cannot be combined with recurrent memory")
            if hidden_dim!=256:raise ValueError("entity attention fusion hidden dimension must be 256")
            if self.entity_dim!=32 or self.entity_attention_heads!=2:raise ValueError("entity attention v1 requires entity_dim=32 and attention_heads=2")
            encode=lambda size:nn.Sequential(nn.Linear(size,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
            self.self_encoder=encode(7)
            self.ally_encoder=encode(6)
            self.enemy_encoder=encode(5)
            self.ally_attention=nn.MultiheadAttention(32,2,batch_first=True)
            self.enemy_attention=nn.MultiheadAttention(32,2,batch_first=True)
            self.entity_fusion=nn.Sequential(nn.Linear(96,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
            if self.entity_attention_mode=="replacement":
                # Preserve the V1 topology exactly: no legacy backbone, adapter,
                # or gate parameters are present in replacement checkpoints.
                del self.backbone
            elif self.entity_attention_mode in {"residual","gated_residual"}:
                self.entity_residual_adapter=nn.Linear(256,256)
                nn.init.zeros_(self.entity_residual_adapter.weight)
                nn.init.zeros_(self.entity_residual_adapter.bias)
                if self.entity_attention_mode=="gated_residual":
                    initial_gate=float(config.get("initial_gate",.05))
                    if not 0.<initial_gate<1.:
                        raise ValueError("gated_residual initial_gate must be in (0, 1)")
                    self.initial_entity_gate=initial_gate
                    self.entity_gate=nn.Linear(512,1)
                    nn.init.zeros_(self.entity_gate.weight)
                    nn.init.constant_(self.entity_gate.bias,math.log(initial_gate/(1.-initial_gate)))
            else:
                self.max_mean_correction=float(config.get("max_mean_correction",.25))
                if self.entity_attention_mode=="frozen_base_mean_residual" and not 0.<self.max_mean_correction<=1.:
                    raise ValueError("frozen_base_mean_residual max_mean_correction must be in (0, 1]")
                if self.entity_attention_mode=="frozen_base_dual_bounded_mean_residual":
                    if "max_mean_correction" in config:
                        raise ValueError("dual-bounded mode does not accept max_mean_correction")
                    self.alpha_abs=float(config.get("alpha_abs",.25))
                    self.alpha_rel=float(config.get("alpha_rel",.25))
                    if not 0.<self.alpha_abs<=1.:raise ValueError("alpha_abs must be in (0, 1]")
                    if not 0.<self.alpha_rel<=1.:raise ValueError("alpha_rel must be in (0, 1]")
                self.entity_mean_adapter=nn.Linear(256,action_dim)
                nn.init.zeros_(self.entity_mean_adapter.weight)
                nn.init.zeros_(self.entity_mean_adapter.bias)
                self.freeze_baseline_policy()
        if self.recurrent_hidden_dim:
            # Recurrent-only parameters must vary with the training seed without
            # advancing the global CPU RNG seen by the subsequently-built
            # baseline critic.  This preserves from-scratch matched
            # initialization for every parameter shared with Plain MAPPO.
            with torch.random.fork_rng(devices=[]):
                self.gru=nn.GRUCell(hidden_dim, self.recurrent_hidden_dim)
                self.mean=nn.Linear(self.recurrent_hidden_dim, action_dim)
                self.log_std=nn.Linear(self.recurrent_hidden_dim, action_dim)
        if self.wave_segmented_actor_gru:
            if self.recurrent_hidden_dim!=128:raise ValueError("wave-segmented actor requires hidden_dim=128")
            self.phase_initial_hidden=nn.Parameter(torch.zeros(3,self.recurrent_hidden_dim))
        hta=dict(hierarchical_temporal_abstraction_config or {})
        self.hierarchical_temporal_abstraction_enabled=bool(hta.get("enabled",False))
        self.num_options=int(hta.get("num_options",4))
        if self.hierarchical_temporal_abstraction_enabled:
            if self.context_dim or self.recurrent_hidden_dim or self.entity_attention_enabled or self.mission_film_enabled:
                raise ValueError("HTA V1 requires the feed-forward 52D Plain worker")
            if self.num_options!=4:raise ValueError("HTA V1 requires four options")
            # Extra heads must not advance the global stream subsequently used
            # to initialize the matched Plain tactical critic.
            with torch.random.fork_rng(devices=[]):
                self.option_mean_residuals=nn.ModuleList(
                    nn.Linear(hidden_dim,action_dim) for _ in range(self.num_options))
                for head in self.option_mean_residuals:
                    nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    @staticmethod
    def split_entities(observations):
        """Split the frozen 52D contract; alive flags are excluded from encoders."""
        if observations.shape[-1]!=52:raise ValueError("entity observation must end in 52 features")
        self_features=observations[...,0:7]
        allies=observations[...,7:28].reshape(*observations.shape[:-1],3,7)
        enemies=observations[...,28:52].reshape(*observations.shape[:-1],4,6)
        return self_features,allies[...,:6],allies[...,6],enemies[...,:5],enemies[...,5]

    @staticmethod
    def _masked_entity_attention(query,entities,alive,module,prefix):
        flat_q=query.reshape(-1,1,query.shape[-1]);flat_e=entities.reshape(-1,entities.shape[-2],entities.shape[-1])
        flat_alive=alive.reshape(-1,alive.shape[-1])>.5;any_alive=flat_alive.any(-1)
        safe_mask=~flat_alive
        if torch.any(~any_alive):
            safe_mask=safe_mask.clone();safe_mask[~any_alive,0]=False
        context,weights=module(flat_q,flat_e,flat_e,key_padding_mask=safe_mask,
                               need_weights=True,average_attn_weights=False)
        valid=flat_alive[:,None,None,:].to(weights.dtype)
        weights=weights*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        context=context*any_alive[:,None,None].to(context.dtype)
        context=context.reshape(*prefix,context.shape[-1])
        weights=weights.squeeze(-2).reshape(*prefix,weights.shape[1],weights.shape[-1])
        return context,weights

    def _entity_encode(self,observations):
        prefix=observations.shape[:-1]
        own,ally_features,ally_alive,enemy_features,enemy_alive=self.split_entities(observations)
        h_self=self.self_encoder(own);h_ally=self.ally_encoder(ally_features);h_enemy=self.enemy_encoder(enemy_features)
        c_ally,ally_weights=self._masked_entity_attention(h_self,h_ally,ally_alive,self.ally_attention,prefix)
        c_enemy,enemy_weights=self._masked_entity_attention(h_self,h_enemy,enemy_alive,self.enemy_attention,prefix)
        encoded=self.entity_fusion(torch.cat((h_self,c_ally,c_enemy),-1))
        diagnostics={"ally_attention_weights":ally_weights,"enemy_attention_weights":enemy_weights,
                     "ally_entity_alive":ally_alive,"enemy_entity_alive":enemy_alive}
        return encoded,diagnostics

    def _input(self, obs, context):
        if self.mission_film_enabled:return obs
        if not self.context_dim:return obs
        if context is None: raise ValueError("actor wave context is required")
        if context.ndim==obs.ndim-1: context=context.unsqueeze(-2).expand(*obs.shape[:-1],-1)
        return torch.cat((obs,context),-1)

    def _mission_film_encode(self,observations,context):
        if context is None:raise ValueError("mission_film actor context is required")
        if context.ndim==observations.ndim-1:context=context.unsqueeze(-2).expand(*observations.shape[:-1],-1)
        if context.shape[:-1]!=observations.shape[:-1] or context.shape[-1]!=self.context_dim:
            raise ValueError("mission_film context shape mismatch")
        h1=self.backbone[1](self.backbone[0](observations));mission=self.mission_encoder(context)
        alpha=self.mission_film_alpha
        delta_gamma=alpha*torch.tanh(self.gamma_head(mission));gamma=1.0+delta_gamma
        beta=alpha*torch.tanh(self.beta_head(mission));residual=alpha*torch.tanh(self.residual_head(mission))
        h1_film=gamma*h1+beta;pre_h2=self.backbone[2](h1_film)+residual;h2=self.backbone[3](pre_h2)
        saturation=torch.cat(((delta_gamma.abs()>.9*alpha),(beta.abs()>.9*alpha),(residual.abs()>.9*alpha)),-1).float()
        diagnostics={"mission_feature_norm":torch.linalg.vector_norm(mission,dim=-1),
            "film_delta_gamma":delta_gamma,"film_beta":beta,"film_residual":residual,"film_gamma":gamma,
            "film_hidden_base_norm":torch.linalg.vector_norm(h1,dim=-1),
            "film_hidden_modulated_norm":torch.linalg.vector_norm(h1_film,dim=-1),
            "film_saturation":saturation}
        return h2,diagnostics

    def freeze_baseline_policy(self):
        """Freeze the complete legacy policy function used by FBMR-EA."""
        for module in (self.backbone,self.mean,self.log_std):
            for parameter in module.parameters():parameter.requires_grad_(False)

    def trainable_policy_parameters(self):
        """Return exactly the actor parameters eligible for optimization."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def frozen_baseline_named_parameters(self):
        if not (self.entity_attention_enabled and self.entity_attention_mode in {"frozen_base_mean_residual","frozen_base_dual_bounded_mean_residual"}):
            return []
        return [(name,parameter) for name,parameter in self.named_parameters()
                if name.startswith(("backbone.","mean.","log_std."))]

    def distribution_step(self, observations, context=None, hidden=None, episode_mask=None, alive_mask=None,return_attention=False,option_ids=None):
        diagnostics=None
        if self.entity_attention_enabled:
            h_entity,diagnostics=self._entity_encode(observations)
            diagnostics["entity_mode"]=self.entity_attention_mode
            diagnostics["entity_feature_norm"]=torch.linalg.vector_norm(h_entity,dim=-1)
            if self.entity_attention_mode=="replacement":
                encoded=h_entity
            elif self.entity_attention_mode in {"frozen_base_mean_residual","frozen_base_dual_bounded_mean_residual"}:
                h_base=self.backbone(self._input(observations,context))
                base_mean=self.mean(h_base)
                base_log_std=self.log_std(h_base).clamp(self.log_std_min,self.log_std_max)
                raw_delta_mu=self.entity_mean_adapter(h_entity)
                if self.entity_attention_mode=="frozen_base_mean_residual":
                    # FBMR V1 compatibility: preserve the original formula exactly.
                    delta_mu=self.max_mean_correction*torch.tanh(raw_delta_mu)
                else:
                    base_std=base_log_std.exp()
                    sigma_scale=base_std.detach()
                    absolute_limit=torch.full_like(sigma_scale,self.alpha_abs)
                    relative_limit=self.alpha_rel*sigma_scale
                    dual_scale=torch.minimum(absolute_limit,relative_limit)
                    delta_v1=self.alpha_abs*torch.tanh(raw_delta_mu)
                    delta_mu=dual_scale*torch.tanh(raw_delta_mu)
                final_mean=base_mean+delta_mu
                diagnostics.update({"entity_base_feature_norm":torch.linalg.vector_norm(h_base,dim=-1),
                                    "base_mean":base_mean,"base_log_std":base_log_std,
                                    "entity_raw_delta_mu":raw_delta_mu,"entity_delta_mu":delta_mu,
                                    "entity_delta_logstd_abs_max":torch.zeros_like(base_log_std[...,0])})
                if self.entity_attention_mode=="frozen_base_dual_bounded_mean_residual":
                    diagnostics.update({"entity_dual_scale":dual_scale,
                                        "entity_relative_scale_active":relative_limit<absolute_limit,
                                        "entity_dual_bound_effective":(delta_v1-delta_mu).abs()>1e-12})
                result=(Normal(final_mean,base_log_std.exp()),hidden)
                return (*result,diagnostics) if return_attention else result
            else:
                h_base=self.backbone(self._input(observations,context))
                delta=self.entity_residual_adapter(h_entity)
                base_norm=torch.linalg.vector_norm(h_base,dim=-1)
                delta_norm=torch.linalg.vector_norm(delta,dim=-1)
                diagnostics.update({"entity_base_feature_norm":base_norm,"entity_delta_norm":delta_norm,
                                    "entity_delta_to_base_ratio":delta_norm/base_norm.clamp_min(1e-12)})
                if self.entity_attention_mode=="gated_residual":
                    gate=torch.sigmoid(self.entity_gate(torch.cat((h_base,h_entity),-1)))
                    diagnostics["entity_gate"]=gate.squeeze(-1)
                    encoded=h_base+gate*delta
                else:
                    encoded=h_base+delta
        elif self.mission_film_enabled:
            encoded,diagnostics=self._mission_film_encode(observations,context)
        else:encoded=self.backbone(self._input(observations,context))
        new_hidden=hidden
        if self.recurrent_hidden_dim:
            if hidden is None:hidden=torch.zeros(*encoded.shape[:-1],self.recurrent_hidden_dim,device=encoded.device)
            if episode_mask is not None and not self.wave_segmented_actor_gru:
                reset=episode_mask[...,None,None] if episode_mask.ndim==hidden.ndim-2 else episode_mask[...,None]
                hidden=hidden*reset
            new_hidden=self.gru(encoded.reshape(-1,encoded.shape[-1]),hidden.reshape(-1,hidden.shape[-1])).view(*encoded.shape[:-1],-1)
            if alive_mask is not None:new_hidden=new_hidden*alive_mask[...,None]
            encoded=new_hidden
        mean=self.mean(encoded)
        if self.hierarchical_temporal_abstraction_enabled:
            if option_ids is None:raise ValueError("HTA worker requires option_ids")
            option_ids=torch.as_tensor(option_ids,dtype=torch.long,device=encoded.device)
            if option_ids.shape!=encoded.shape[:-1]:raise ValueError("HTA option_ids shape mismatch")
            residuals=torch.stack([head(encoded) for head in self.option_mean_residuals],dim=-2)
            selected=torch.gather(residuals,-2,option_ids[...,None,None].expand(*option_ids.shape,1,residuals.shape[-1])).squeeze(-2)
            mean=mean+selected
            if diagnostics is None:diagnostics={}
            diagnostics.update({"hta_option_residual":selected,"hta_option_residuals":residuals})
        std=self.log_std(encoded).clamp(self.log_std_min,self.log_std_max).exp()
        result=(Normal(mean,std),new_hidden)
        return (*result,diagnostics) if return_attention else result

    def phase_hidden(self,wave_indices,alive_mask):
        """Live trainable phase initial state, shared across homogeneous agents."""
        if not self.wave_segmented_actor_gru:raise RuntimeError("actor has no phase initialization")
        waves=torch.as_tensor(wave_indices,dtype=torch.long,device=self.phase_initial_hidden.device)
        if not bool(((waves>=1)&(waves<=3)).all()):raise ValueError("phase wave must be 1..3")
        alive=torch.as_tensor(alive_mask,dtype=self.phase_initial_hidden.dtype,device=waves.device)
        return self.phase_initial_hidden[waves-1].unsqueeze(-2)*alive.unsqueeze(-1)

    def distribution(self, observations, context=None,return_attention=False,option_ids=None):
        if self.recurrent_hidden_dim: raise RuntimeError("recurrent actor requires distribution_step")
        result=self.distribution_step(observations,context,return_attention=return_attention,option_ids=option_ids)
        return (result[0],result[2]) if return_attention else result[0]


class StateMemoryMAPPOActor(ModularMAPPOActor):
    """Current 55D state has a direct path alongside a within-wave GRU memory."""
    def __init__(self,observation_dim=52,action_dim=3,hidden_dim=256,
                 log_std_min=-5.,log_std_max=2.,activation="relu"):
        if (observation_dim,action_dim,hidden_dim,activation)!=(52,3,256,"relu"):
            raise ValueError("MARC-SM requires 52D/3D, 256-256 ReLU policy")
        # Consume exactly the baseline construction stream seen by the critic.
        super().__init__(observation_dim,action_dim,hidden_dim,log_std_min,log_std_max,activation)
        self.context_dim=3;self.recurrent_hidden_dim=128
        self.state_memory_input_dim=55;self.fused_policy_input_dim=183
        with torch.random.fork_rng(devices=[]):
            self.gru=nn.GRUCell(55,128)
            self.backbone=nn.Sequential(nn.Linear(183,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())
        # Existing Gaussian heads still read the 256D policy feature.

    @staticmethod
    def wave_one_hot(wave_indices,alive_mask,device=None):
        waves=torch.as_tensor(wave_indices,device=device)
        if not bool(((waves>=1)&(waves<=3)&(waves==waves.long())).all()):
            raise ValueError("State Memory requires pre-action wave indices in 1..3")
        alive=torch.as_tensor(alive_mask,dtype=torch.float32,device=waves.device)
        if waves.shape!=alive.shape[:-1]:raise ValueError("State Memory waves/alive shape mismatch")
        return torch.nn.functional.one_hot(waves.long()-1,3).to(torch.float32).unsqueeze(-2).expand(*alive.shape,3)*alive[...,None]

    def distribution_step(self,observations,context=None,hidden=None,episode_mask=None,
                          alive_mask=None,return_attention=False,option_ids=None):
        if observations.shape[-1]!=52 or context is None or context.shape!=(*observations.shape[:-1],3):
            raise ValueError("State Memory Actor requires 52D observation and explicit 3D wave context")
        if alive_mask is None:raise ValueError("State Memory Actor requires alive mask")
        # Dead rows may have zero context; live rows must be literal one-hot.
        live=alive_mask>.5
        if not bool(((context[live]==0)|(context[live]==1)).all()) or not bool((context[live].sum(-1)==1).all()):
            raise ValueError("State Memory context must be fixed wave one-hot")
        x=torch.cat((observations,context),-1)
        if hidden is None:hidden=torch.zeros(*observations.shape[:-1],128,device=observations.device)
        if episode_mask is not None:
            reset=episode_mask[...,None,None] if episode_mask.ndim==hidden.ndim-2 else episode_mask[...,None]
            hidden=hidden*reset
        hidden=hidden*alive_mask[...,None]
        new_hidden=self.gru(x.reshape(-1,55),hidden.reshape(-1,128)).reshape(*observations.shape[:-1],128)
        new_hidden=new_hidden*alive_mask[...,None]
        encoded=self.backbone(torch.cat((x,new_hidden),-1))
        result=(Normal(self.mean(encoded),self.log_std(encoded).clamp(self.log_std_min,self.log_std_max).exp()),new_hidden)
        return (*result,{}) if return_attention else result


class HierarchicalManagerActor(nn.Module):
    """Shared low-frequency categorical actor over latent tactical options."""
    def __init__(self,observation_dim=52,hidden_dim=256,num_options=4,activation="relu"):
        super().__init__();self.observation_dim=int(observation_dim);self.hidden_dim=int(hidden_dim);self.num_options=int(num_options)
        act={"relu":nn.ReLU,"leaky_relu":nn.LeakyReLU}[activation]
        self.backbone=nn.Sequential(nn.Linear(observation_dim,hidden_dim),act(),nn.Linear(hidden_dim,hidden_dim),act())
        self.logits_head=nn.Linear(hidden_dim,num_options)
        nn.init.zeros_(self.logits_head.weight);nn.init.zeros_(self.logits_head.bias)

    def logits(self,observations):return self.logits_head(self.backbone(observations))
    def probabilities(self,observations):return torch.softmax(self.logits(observations),dim=-1)
    def distribution(self,observations):return Categorical(logits=self.logits(observations))


class ModularCentralizedCritic(CentralizedValueCritic):
    def __init__(self, observation_dim=52, hidden_dim=256, attention_heads=2,
                 activation="relu", context_dim=0, recurrent_hidden_dim=0,
                 context_injection="concat"):
        self.base_observation_dim=int(observation_dim); self.context_dim=int(context_dim)
        self.recurrent_hidden_dim=int(recurrent_hidden_dim)
        self.context_injection=str(context_injection)
        if self.context_injection not in {"concat","additive_zero"}:
            raise ValueError("unsupported critic context injection")
        if self.context_injection=="additive_zero" and not self.context_dim:
            raise ValueError("additive_zero requires a nonzero critic context dimension")
        input_dim=observation_dim if self.context_injection=="additive_zero" else observation_dim+self.context_dim
        super().__init__(input_dim,hidden_dim,attention_heads,activation)
        if self.context_injection=="additive_zero":
            # Creating an explicit zeros tensor consumes no RNG.  Consequently
            # every shared C0/C1 parameter and the post-init RNG state remain
            # bit-identical, while mission state has a learnable information-only
            # route into the first critic preactivation.
            self.mission_context_projection=nn.Parameter(torch.zeros(hidden_dim,self.context_dim))
        if self.recurrent_hidden_dim:
            act={"relu":nn.ReLU,"leaky_relu":nn.LeakyReLU}[activation]
            self.value_network=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),act())
            self.gru=nn.GRUCell(hidden_dim,self.recurrent_hidden_dim)
            self.value_head=nn.Linear(self.recurrent_hidden_dim,1)

    @property
    def output_layer(self):
        return self.value_head if self.recurrent_hidden_dim else self.value_network[-1]

    def _input(self,obs,context):
        if not self.context_dim:return obs
        if context is None:raise ValueError("critic wave context is required")
        if context.ndim==obs.ndim-1:context=context.unsqueeze(-2).expand(*obs.shape[:-1],-1)
        if self.context_injection=="additive_zero":return obs
        return torch.cat((obs,context),-1)

    def _embedding(self,observations,context):
        if self.context_injection!="additive_zero":return self.embedding(self._input(observations,context))
        if context is None:raise ValueError("critic wave context is required")
        if context.ndim==observations.ndim-1:
            context=context.unsqueeze(-2).expand(*observations.shape[:-1],-1)
        preactivation=self.embedding[0](observations)+torch.nn.functional.linear(context,self.mission_context_projection)
        value=self.embedding[1](preactivation)
        for layer in list(self.embedding)[2:]:value=layer(value)
        return value

    def forward_step(self,observations,alive_mask=None,context=None,hidden=None,episode_mask=None,return_attention=False):
        embedding=self._embedding(observations,context); batch,agents,_=embedding.shape
        def heads(x):return x.view(batch,agents,self.attention_heads,self.head_dim).transpose(1,2)
        q,k,v=heads(self.wq(embedding)),heads(self.wk(embedding)),heads(self.wv(embedding))
        logits=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        if alive_mask is None:alive_mask=torch.ones(batch,agents,device=logits.device)
        alive=alive_mask>.5; not_self=~torch.eye(agents,dtype=torch.bool,device=logits.device).view(1,1,agents,agents)
        valid=alive[:,None,None,:].expand(batch,self.attention_heads,agents,agents)&not_self
        weights=torch.softmax(logits.masked_fill(~valid,-1e9),-1)*valid.to(logits.dtype)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        weights=weights*alive[:,None,:,None]
        ctx=(weights@v).transpose(1,2).contiguous().view(batch,agents,self.hidden_dim)
        feature=self.value_network(torch.cat((embedding,ctx),-1))
        new_hidden=hidden
        if self.recurrent_hidden_dim:
            if hidden is None:hidden=torch.zeros(batch,agents,self.recurrent_hidden_dim,device=feature.device)
            if episode_mask is not None:
                reset=episode_mask[...,None,None] if episode_mask.ndim==hidden.ndim-2 else episode_mask[...,None]
                hidden=hidden*reset
            new_hidden=self.gru(feature.reshape(-1,feature.shape[-1]),hidden.reshape(-1,hidden.shape[-1])).view(batch,agents,-1)
            new_hidden=new_hidden*alive_mask[...,None]; values=self.value_head(new_hidden).squeeze(-1)
        else: values=feature.squeeze(-1)
        values=values*alive_mask
        return (values,new_hidden,weights) if return_attention else (values,new_hidden)

    def forward(self,observations,alive_mask=None,return_attention=False,context=None):
        if self.recurrent_hidden_dim:raise RuntimeError("recurrent critic requires forward_step")
        values,_,weights=self.forward_step(observations,alive_mask,context,return_attention=True)
        return (values,weights) if return_attention else values


class InterWaveStateQualityCritic(nn.Module):
    """Independent team-level future-wave completion estimator."""
    def __init__(self, observation_dim=52, hidden_dim=256, attention_heads=2, activation="relu", max_waves=3):
        super().__init__(); self.observation_dim=int(observation_dim); self.max_waves=int(max_waves)
        self.backbone=CentralizedValueCritic(observation_dim+self.max_waves+1,hidden_dim,attention_heads,activation)

    def forward(self, observations, alive_mask, credit_wave, remaining_horizon):
        if observations.ndim != 3: raise ValueError("IW critic observations must be [B,A,D]")
        wave=torch.nn.functional.one_hot((credit_wave.long()-1).clamp(0,self.max_waves-1),self.max_waves).to(observations.dtype)
        context=torch.cat((wave,remaining_horizon.reshape(-1,1).to(observations.dtype)),-1)
        augmented=torch.cat((observations,context[:,None,:].expand(-1,observations.shape[1],-1)),-1)
        logits=self.backbone(augmented,alive_mask)
        team_logit=(logits*alive_mask).sum(-1)/alive_mask.sum(-1).clamp_min(1.0)
        return torch.sigmoid(team_logit)


class InterWaveActionOutcomeCritic(nn.Module):
    """Joint state-action predictor for future Wave2/Wave3 clear events."""
    event_heads=("wave2_clear","wave3_clear")
    def __init__(self,observation_dim=52,action_dim=3,hidden_dim=256,attention_heads=2,activation="relu",max_waves=3):
        super().__init__();self.observation_dim=int(observation_dim);self.action_dim=int(action_dim);self.hidden_dim=int(hidden_dim);self.attention_heads=int(attention_heads);self.max_waves=int(max_waves)
        if self.observation_dim!=52 or self.action_dim!=3 or self.max_waves!=3:raise ValueError("CAIW critic requires 52D observations, 3D actions, and three waves")
        if hidden_dim%attention_heads:raise ValueError("hidden_dim must be divisible by attention_heads")
        act={"relu":nn.ReLU,"leaky_relu":nn.LeakyReLU}[activation]
        self.state_encoder=nn.Sequential(nn.Linear(observation_dim+max_waves+1,hidden_dim),act(),nn.Linear(hidden_dim,hidden_dim),act())
        self.action_projection=nn.Linear(action_dim,hidden_dim);nn.init.zeros_(self.action_projection.weight);nn.init.zeros_(self.action_projection.bias)
        self.wq=nn.Linear(hidden_dim,hidden_dim,bias=False);self.wk=nn.Linear(hidden_dim,hidden_dim,bias=False);self.wv=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.event_network=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),act(),nn.Linear(hidden_dim,hidden_dim),act(),nn.Linear(hidden_dim,2))
        self.head_dim=hidden_dim//attention_heads

    def forward(self,observations,actions,alive_mask,source_wave,remaining_horizon):
        if observations.ndim!=3 or actions.shape[:2]!=observations.shape[:2]:raise ValueError("CAIW critic expects [B,A,D] observations/actions")
        batch,agents,_=observations.shape;wave=torch.nn.functional.one_hot((source_wave.long()-1).clamp(0,self.max_waves-1),self.max_waves).to(observations.dtype)
        context=torch.cat((wave,remaining_horizon.reshape(-1,1).to(observations.dtype)),-1)[:,None,:].expand(-1,agents,-1)
        entity=self.state_encoder(torch.cat((observations,context),-1))+self.action_projection(actions)
        def split(x):return x.view(batch,agents,self.attention_heads,self.head_dim).transpose(1,2)
        q,k,v=split(self.wq(entity)),split(self.wk(entity)),split(self.wv(entity));logits=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        alive=alive_mask>.5;not_self=~torch.eye(agents,dtype=torch.bool,device=observations.device).view(1,1,agents,agents)
        valid=alive[:,None,None,:].expand(batch,self.attention_heads,agents,agents)&not_self
        weights=torch.softmax(logits.masked_fill(~valid,-1e9),-1)*valid.to(logits.dtype);weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12);weights*=alive[:,None,:,None]
        attended=(weights@v).transpose(1,2).contiguous().view(batch,agents,self.hidden_dim)
        per_agent=self.event_network(torch.cat((entity,attended),-1));mask=alive_mask[...,None].to(per_agent.dtype)
        return (per_agent*mask).sum(1)/mask.sum(1).clamp_min(1.0)


class BoundaryStateOutcomeCritic(nn.Module):
    """Action-free post-spawn boundary-state predictor with two event logits."""
    event_heads=("wave2_clear","wave3_clear")
    def __init__(self,observation_dim=52,hidden_dim=256,attention_heads=2,activation="relu",max_waves=3):
        super().__init__();self.observation_dim=int(observation_dim);self.hidden_dim=int(hidden_dim);self.attention_heads=int(attention_heads);self.max_waves=int(max_waves)
        if self.observation_dim!=52 or self.max_waves!=3:raise ValueError("BRSC critic requires 52D observations and three waves")
        if hidden_dim%attention_heads:raise ValueError("hidden_dim must be divisible by attention_heads")
        act={"relu":nn.ReLU,"leaky_relu":nn.LeakyReLU}[activation]
        self.state_encoder=nn.Sequential(nn.Linear(observation_dim+max_waves+1,hidden_dim),act(),nn.Linear(hidden_dim,hidden_dim),act())
        self.wq=nn.Linear(hidden_dim,hidden_dim,bias=False);self.wk=nn.Linear(hidden_dim,hidden_dim,bias=False);self.wv=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.event_network=nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim),act(),nn.Linear(hidden_dim,hidden_dim),act(),nn.Linear(hidden_dim,2))
        self.head_dim=hidden_dim//attention_heads

    def forward(self,observations,alive_mask,source_wave,remaining_horizon):
        if observations.ndim!=3:raise ValueError("BRSC critic expects [B,A,D] observations")
        batch,agents,_=observations.shape;wave=torch.nn.functional.one_hot((source_wave.long()-1).clamp(0,self.max_waves-1),self.max_waves).to(observations.dtype)
        context=torch.cat((wave,remaining_horizon.reshape(-1,1).to(observations.dtype)),-1)[:,None,:].expand(-1,agents,-1)
        entity=self.state_encoder(torch.cat((observations,context),-1))
        def split(x):return x.view(batch,agents,self.attention_heads,self.head_dim).transpose(1,2)
        q,k,v=split(self.wq(entity)),split(self.wk(entity)),split(self.wv(entity));scores=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        alive=alive_mask>.5;not_self=~torch.eye(agents,dtype=torch.bool,device=observations.device).view(1,1,agents,agents)
        valid=alive[:,None,None,:].expand(batch,self.attention_heads,agents,agents)&not_self
        weights=torch.softmax(scores.masked_fill(~valid,-1e9),-1)*valid.to(scores.dtype);weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12);weights*=alive[:,None,:,None]
        attended=(weights@v).transpose(1,2).contiguous().view(batch,agents,self.hidden_dim)
        per_agent=self.event_network(torch.cat((entity,attended),-1));mask=alive_mask[...,None].to(per_agent.dtype)
        return (per_agent*mask).sum(1)/mask.sum(1).clamp_min(1.0)


__all__=["ModularMAPPOActor","StateMemoryMAPPOActor","ModularCentralizedCritic","HierarchicalManagerActor","InterWaveStateQualityCritic","InterWaveActionOutcomeCritic","BoundaryStateOutcomeCritic"]
