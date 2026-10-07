# HTA-MAPPO V1: Hierarchical Temporal-Abstraction MAPPO

## Motivation and route change

IWSC did not establish reliable future-wave state-quality credit as an actor improvement. CAIW then tested single-step action-conditioned counterfactual credit, but the available data did not identify a stable action-level causal signal. BRSC moved credit to post-spawn boundary states and redistributed it retrospectively across preceding segments; its boundary predictor did not become reliably ready across seeds, W2→W3 credit never activated, and the primary performance gate failed. These negative methods remain in the repository as historical evidence.

The separate diagnostic conclusion `STATE_QUALITY_SUPPORTED` is still useful: the quality of the state carried into later waves matters. HTA therefore changes the temporal representation rather than adding another auxiliary credit redistributor. Its scientific question is whether an explicit low-frequency latent tactical decision can expose long-horizon structure while retaining the proven high-frequency continuous controller.

AICC motivates a Manager–Worker decomposition and persistent intention, but HTA V1 does not copy AICC's GRU, intention prior, KL consistency, counterfactual machinery, or shaped reward. V1 isolates temporal abstraction.

## Policy architecture

Each Red UAV independently receives one of four learned latent options from a shared Manager. The options have no manually assigned meanings. Four options provide a small categorical abstraction without prejudging attack, cover, positioning, or evasion semantics; any semantics must be inferred after training.

The Manager observes only the original local 52D observation. It is a feed-forward 52→256→256→4 categorical actor. Its logits head is zero initialized, so the initial option distribution is exactly uniform. The centralized Manager critic is an independent standard value critic and receives no option, wave index, horizon, or outcome label.

The Worker preserves the Plain MAPPO 52D backbone, continuous 3D action `[Δψ, Δθ, Δv]`, and shared log-standard-deviation head. Four zero-initialized mean-residual heads condition only the Gaussian mean on the chosen option. Thus every option is initially exactly equivalent to Plain MAPPO. The tactical critic uses the existing additive-zero context path with a four-dimensional option one-hot; the zero projection makes its initial value exactly Plain-equivalent.

Manager and Worker share no parameter tensor, optimizer, gradient, or auxiliary loss. The only high-to-low influence is the executed latent option. Deployment therefore requires both Manager and Worker.

## Time scale and macro lifecycle

The fixed decision interval is `K=16` primitive steps. With environment `dt=0.1 s`, this is 1.6 seconds: close to the approximately 2-second controller time scale and exactly divides the 256-step training rollout. K is a fixed V1 design choice, not a claim of optimality.

A macro begins at rollout start, episode start, a periodic 16-step boundary, or immediately after a true wave transition. A wave transition is not terminal. The preceding macro closes on the transition, bootstraps from the post-spawn observation, and survivors immediately obtain new options. Individual agent death does not cut the shared environment macro; dead agents are masked and cannot bootstrap.

Training forcibly closes every open macro at a PPO rollout boundary. A nonterminal rollout truncation bootstraps from the final next state but sets the GAE trace-continuation mask to zero. No behavior-Manager transition crosses an optimizer update. The next rollout samples fresh options from the updated Manager. This artificial boundary is a training protocol only; deterministic evaluation does not truncate options every 256 steps.

## Reward and semi-MDP return

The Manager receives no new reward. For a macro of duration `d`, its per-agent reward is the discounted sum of the unchanged raw environment rewards:

`R_H = Σ(j=0..d-1) γ^j r_(t+j)`.

The Manager TD residual and GAE use the semi-MDP duration discount:

`δ_k = R_H + γ^d V_H(S_(k+1)) bootstrap_mask - V_H(S_k)`

`A_k = δ_k + γ^d λ trace_mask A_(k+1)`.

Normal periodic and wave boundaries bootstrap and continue the trace for survivors. True episode terminals neither bootstrap nor continue. Nonterminal rollout truncations bootstrap but do not continue. Worker PPO remains the original primitive-reward, primitive-GAE continuous MAPPO objective.

## Optimization, reproducibility, and diagnosis

Manager Actor PPO is standard categorical PPO; Manager critic loss uses the existing clipped-value semantics. The Manager actor follows the same delayed 3e-4→1e-4 schedule as the Worker actor; the Manager critic uses 3e-4. Manager minibatch permutations and categorical sampling have independent saved RNGs, so they do not perturb Worker Gaussian sampling or Worker PPO permutation lineage.

All added networks and heads are constructed with RNG isolation. At matched initialization, Plain-shared Worker actor and tactical critic parameters are bit-identical, residuals and option projection are zero, Manager probabilities are uniform, and Worker distribution/value outputs equal Plain MAPPO.

Logs expose Manager entropy and KL, option-use fractions, maximum option fraction, switch rate, macro durations and end causes, option-residual scale, and Manager explained variance. These diagnose option collapse but never change the loss automatically.

## Limitations

- Latent options may collapse to one category.
- Forced rollout truncation introduces a small number of artificial macro boundaries during training.
- K=16 and four options are fixed development choices rather than established optima.
- Latent options are intentionally uninterpreted until post-training behavior analysis.
