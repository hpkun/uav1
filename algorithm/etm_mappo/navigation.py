"""Independent 2-D cooperative navigation, with no adversarial mechanisms/imports."""
import copy
import numpy as np


class CooperativeNavigationEnv:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.n, self.k = config['num_agents'], config['num_obstacles']
        angles = np.arange(self.n)*2*np.pi/self.n + np.pi/4
        self.offsets = .28*np.column_stack((np.cos(angles),np.sin(angles)))
        self.dt = config['dt']

    def reset(self, seed):
        self.seed = int(seed)
        # Observation permutation consumes a separate stream, so shuffling does
        # not perturb physical initialization or dynamic-obstacle trajectories.
        self.rng = np.random.default_rng(self.seed)
        self.slot_rng = np.random.default_rng(np.random.SeedSequence([self.seed,9173]))
        simple = self.config['scene'] == 'simple'
        center = np.array([-.8,0.]) if simple else self.rng.uniform([-.95,-.35],[-.6,.35])
        self.goal = np.array([.8,0.]) if simple else self.rng.uniform([.6,-.3],[1.,.3])
        self.positions = center + self.offsets
        if not simple:
            self.positions += self.rng.normal(0,.04,self.positions.shape)
        self.velocities = np.zeros((self.n,2))
        self.last_actions = np.zeros((self.n,2))
        if self.k:
            if simple:
                self.obstacles = np.column_stack((np.linspace(-.35,.45,self.k),np.where(np.arange(self.k)%2,.75,-.75)))
                self.obstacle_velocities = np.zeros((self.k,2))
            else:
                self.obstacles = self.rng.uniform([-.25,-.9],[.45,.9],(self.k,2))
                self.obstacle_velocities = self.rng.uniform(-.04,.04,(self.k,2))
        else:
            self.obstacles = np.empty((0,2));self.obstacle_velocities = np.empty((0,2))
        self.steps = 0
        self.team_return = 0.
        self.any_collision = False
        self.component_totals = {key:0. for key in self.config['reward']}
        return self.observe()

    def observe(self):
        # A robot knows its assigned geometric offset, not the true centroid of
        # potentially invisible robots. The reward may use the full formation.
        own = np.concatenate((self.positions,self.velocities,
            np.broadcast_to(self.goal,(self.n,2))-self.positions,self.offsets,self.last_actions),axis=-1).astype('f')
        pf = np.zeros((self.n,max(0,self.n-1),4),dtype='f')
        of = np.zeros((self.n,self.k,5),dtype='f')
        pi = np.full(pf.shape[:-1],-1,dtype=np.int64);oi = np.full(of.shape[:-1],-1,dtype=np.int64)
        pv = np.zeros_like(pi,dtype=bool);ov = np.zeros_like(oi,dtype=bool)
        radius = self.config['visibility_radius']
        for i in range(self.n):
            peers = np.array([j for j in range(self.n) if j != i],dtype=np.int64)
            obstacles = np.arange(self.k,dtype=np.int64)
            if self.config['shuffle_slots']:
                self.slot_rng.shuffle(peers);self.slot_rng.shuffle(obstacles)
            for slot,j in enumerate(peers):
                relative = self.positions[j]-self.positions[i]
                if np.linalg.norm(relative) <= radius:
                    pi[i,slot]=j;pv[i,slot]=True
                    pf[i,slot]=np.r_[relative,self.velocities[j]-self.velocities[i]]
            for slot,j in enumerate(obstacles):
                relative = self.obstacles[j]-self.positions[i]
                if np.linalg.norm(relative) <= radius:
                    oi[i,slot]=1000+j;ov[i,slot]=True
                    of[i,slot]=np.r_[relative,self.obstacle_velocities[j]-self.velocities[i],self.config['obstacle_radius']]
        return dict(self_features=own,peer_features=pf,obstacle_features=of,peer_ids=pi,obstacle_ids=oi,
            peer_valid=pv,obstacle_valid=ov,agent_alive=np.ones(self.n,dtype=bool))

    def step(self, actions):
        actions=np.asarray(actions,dtype=np.float32)
        if actions.shape != (self.n,2) or not np.isfinite(actions).all() or (np.abs(actions)>1).any():
            raise ValueError('navigation actions must be finite bounded [agent,2] accelerations')
        before=np.linalg.norm(self.positions-self.goal-self.offsets,axis=-1)
        self.velocities += self.dt*self.config['max_acceleration']*actions
        speed=np.linalg.norm(self.velocities,axis=-1,keepdims=True)
        self.velocities *= np.minimum(1,self.config['max_speed']/np.maximum(speed,1e-12))
        self.positions += self.dt*self.velocities
        self.obstacles += self.dt*self.obstacle_velocities
        self.obstacle_velocities[np.abs(self.obstacles)>.95] *= -1
        after=np.linalg.norm(self.positions-self.goal-self.offsets,axis=-1)
        formation=np.linalg.norm(self.positions-self.positions.mean(0)-self.offsets,axis=-1)
        pair=np.linalg.norm(self.positions[:,None]-self.positions[None,:],axis=-1)<2*self.config['robot_radius']
        np.fill_diagonal(pair,False)
        obstacle=np.linalg.norm(self.positions[:,None]-self.obstacles[None,:],axis=-1)<self.config['robot_radius']+self.config['obstacle_radius']
        collision=pair.any(-1)|obstacle.any(-1)
        weights=self.config['reward']
        components=dict(progress=weights['progress']*(before-after),formation=-weights['formation']*formation,
            collision=-weights['collision']*collision.astype(float),smoothness=-weights['smoothness']*((actions-self.last_actions)**2).sum(-1))
        reward=sum(components.values()).astype('f')
        self.last_actions=actions.copy();self.steps+=1
        self.any_collision |= bool(collision.any())
        success=bool((after<self.config['goal_tolerance']).all() and not self.any_collision)
        terminated=success
        truncated=bool(not terminated and self.steps>=self.config['max_steps'])
        self.team_return += float(reward.sum())
        for key,value in components.items():self.component_totals[key]+=float(value.sum())
        info=dict(executed_actions=actions.copy(),episode_return=self.team_return,episode_length=self.steps,
            success=success,collision_episode=self.any_collision,collision_agent_steps=int(collision.sum()),
            goal_error=float(after.mean()),formation_error=float(formation.mean()),
            component_totals=dict(self.component_totals),scene=self.config['scene'],seed=self.seed)
        return self.observe(),reward,terminated,truncated,info


def stack_observations(observations):
    return {key:np.stack([o[key] for o in observations]) for key in observations[0]}


class NavigationVectorEnv:
    """Synchronous independent simulator instances with explicit terminal frames."""
    def __init__(self, count, config, seed):
        if count<=0:raise ValueError('positive environment count required')
        self.envs=[CooperativeNavigationEnv(config) for _ in range(count)]
        self.count,self.seed=count,seed
        self.episode_indices=np.zeros(count,dtype=np.int64)
        self.used_seeds=[]
        self.current=stack_observations([self._reset(i) for i in range(count)])

    def _reset(self,i):
        seed=self.seed+int(self.episode_indices[i])*self.count+i
        self.used_seeds.append(seed)
        return self.envs[i].reset(seed)

    def step(self,actions):
        rows=[env.step(action) for env,action in zip(self.envs,actions)]
        transition=stack_observations([r[0] for r in rows])
        terminated=np.array([r[2] for r in rows],dtype=bool)
        truncated=np.array([r[3] for r in rows],dtype=bool)
        following=[r[0] for r in rows]
        for i in np.flatnonzero(terminated|truncated):
            self.episode_indices[i]+=1;following[i]=self._reset(i)
        self.current=stack_observations(following)
        return transition,np.stack([r[1] for r in rows]),terminated,truncated,[r[4] for r in rows]
