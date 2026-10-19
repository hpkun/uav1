"""On-policy step GAE and asynchronous, duration-discounted macro decisions."""
import torch


def team_gae(rewards,values,next_values,done,gamma,lam):
    advantages=torch.zeros_like(rewards);carry=torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        continuation=1-done[t]
        delta=rewards[t]+gamma*continuation*next_values[t]-values[t]
        carry=delta+gamma*lam*continuation*carry;advantages[t]=carry
    return advantages,advantages+values


def macro_advantages(rewards,values,next_values,done,alive,next_alive,decisions,gamma,lam):
    """Only real selections in this behavior rollout are trainable events.

    An unfinished decision is censored at rollout end, with a value bootstrap.
    Its runtime intention continues, but a following rollout never contributes
    rewards to the old-policy decision. No synthetic new decision is inserted.
    """
    T,E,A=alive.shape;advantages=torch.zeros_like(alive);events=[]
    for e in range(E):
        for i in range(A):
            starts=torch.nonzero(decisions[:,e,i].bool()&alive[:,e,i].bool()).flatten().tolist()
            agent_events=[]
            for start in starts:
                end=start;reward=rewards.new_zeros(());discount=1.;terminal=False;reason='rollout_bootstrap'
                while end<T:
                    reward=reward+discount*rewards[end,e];discount*=gamma
                    terminal=bool(done[end,e] or not next_alive[end,e,i])
                    end+=1
                    if terminal:
                        reason='episode_end' if bool(done[end-1,e]) else 'agent_death';break
                    if end<T and decisions[end,e,i]:reason='renew';break
                tau=end-start
                bootstrap=values[end,e] if end<T else next_values[end-1,e]
                delta=reward+(0. if terminal else discount)*bootstrap-values[start,e]
                agent_events.append(dict(environment=e,agent=i,start=start,end=end,duration=tau,
                    discounted_return=float(reward),bootstrap_discount=0. if terminal else discount,
                    reason=reason,delta=delta,terminal=terminal))
            carry=rewards.new_zeros(())
            for index in reversed(range(len(agent_events))):
                event=agent_events[index];connect=False
                if index+1<len(agent_events):
                    following=agent_events[index+1]
                    connect=not event['terminal'] and event['end']==following['start']
                carry=event['delta']+(gamma**event['duration'])*lam*carry*int(connect)
                advantages[event['start'],e,i]=carry
                event['advantage']=float(carry);event['delta']=float(event['delta'])
            events.extend(agent_events)
    return advantages,events


def normalize(values,mask,enabled=True):
    selected=values[mask.bool()]
    if not enabled or selected.numel()<2:return values
    return torch.where(mask.bool(),(values-selected.mean())/selected.std(unbiased=False).clamp_min(1e-8),torch.zeros_like(values))


def event_mean(values,mask):
    selected=values[mask.bool()]
    return selected.mean() if selected.numel() else selected.sum()


def ppo_component(logp,old_logp,advantage,mask,clip):
    valid=mask.bool()
    if not valid.any():
        zero=logp[valid].sum();return zero,zero.detach(),zero.detach()
    change=logp[valid]-old_logp[valid];ratio=change.exp();adv=advantage[valid]
    loss=-torch.minimum(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()
    kl=((ratio-1)-change).mean();fraction=((ratio-1).abs()>clip).float().mean()
    return loss,kl.detach(),fraction.detach()
