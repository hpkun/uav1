# Mission-Aware FiLM-MAPPO

## Problem

Directly concatenating the 5D mission context with the 52D tactical observation produced mixed three-seed results. The working hypothesis is that early mixing can disturb tactical representation learning even when mission state is useful.

## Design source and scope

The design uses the feature-wise affine conditioning idea from Perez et al., [*FiLM: Visual Reasoning with a General Conditioning Layer*](https://doi.org/10.1609/aaai.v32i1.11671), and the structural motivation from Leonard Bauersfeld, Elia Kaufmann, and Davide Scaramuzza, [*User-Conditioned Neural Control Policies for Mobile Robotics*](https://rpg.ifi.uzh.ch/docs/ICRA23_Bauersfeld.pdf), ICRA 2023. The latter inserts FiLM between parts of a control policy and describes an augmented FiLM route that also feeds conditioning information into the policy. This implementation is a project-specific adaptation, not a reproduction of either network or its reported experiments; no literature performance numbers are asserted here.

## Project adaptation

The environment observation remains the frozen 52D tactical vector. The canonical algorithm-side mission state is the existing 5D `mission_markov` encoding: a 3D wave one-hot vector, cumulative mission progress, and remaining horizon. It is encoded separately by a 5→32→32 MLP.

For first-layer tactical feature

\[
h_1=\operatorname{ReLU}(W_1o+b_1),\qquad m=f_{mission}(c),
\]

bounded augmented FiLM applies

\[
\Delta\gamma=\alpha\tanh(W_\gamma m+b_\gamma),\quad \gamma=1+\Delta\gamma,
\]
\[
\beta=\alpha\tanh(W_\beta m+b_\beta),\qquad r=\alpha\tanh(W_r m+b_r),
\]
\[
h_1^{film}=\gamma\odot h_1+\beta,
\]
\[
h_2=\operatorname{ReLU}(W_2h_1^{film}+b_2+r).
\]

Here `alpha=0.2`, so `gamma` lies in `[0.8,1.2]`, while `beta` and the augmented residual lie in `[-0.2,0.2]`. All three output heads are zero-initialized, giving `gamma=1`, `beta=0`, and `r=0` at initialization. The original tactical backbone and policy heads therefore define exactly the initial policy. The mission encoder remains normally initialized and all Actor parameters remain trainable.

## Controlled exclusions

The centralized Critic is unchanged and receives no mission context, isolating the Actor-conditioning question. GRU memory, entity attention, reward modification, curriculum, wave balancing, policy anchoring, PPO stabilization, warm start, and advantage priority remain disabled. Actor-only learning-rate decay remains enabled under the frozen baseline schedule.
