# EA-WB-MAPPO Formal Multi-Seed Protocol

EA-WB development is frozen. The final method is Entity-Aware Wave-Balanced MAPPO: Entity Attention plus Wave Balance. No algorithm, architecture, reward, observation, environment, Blue-policy, PPO-loss, Wave-Balance, Entity-Attention, or frozen hyperparameter change is allowed after formal runs begin. Poor results are not grounds for changing a seed, budget, checkpoint rule, learning rate, or method configuration. Engineering failures may be resumed or restarted only with recorded provenance.

## Experiment design

The primary 2x2 matrix is MAPPO, WB-MAPPO, EA-MAPPO, and EA-WB-MAPPO. Each uses fresh training seeds 3101, 3102, and 3103. The schedule ablation contains EA-WB Fixed LR at the same three seeds, for 15 runs total. Seed 2023 is development-only; historical seeds 2024/2025 and all earlier outputs are excluded from new formal evidence.

Every run uses `persistent_wave_v2`, 24 environments, 256 rollout steps, gamma 0.999, GAE lambda 0.95, PPO clip 0.2, entropy coefficient 0.01, value coefficient 0.5, 10 PPO epochs, minibatch size 512, and a 900,000 sampled-step budget. Critic LR remains 3e-4. The main matrix holds Actor LR at 3e-4 through step 600,000 and linearly decays it to 1e-4 at step 900,000. Fixed-LR controls hold Actor LR at 3e-4. All unrelated modules are off.

Periodic monitoring uses 20 diagnostic-only episodes at seeds 29,000,000 through 29,000,019. It must not select the primary checkpoint. `best_eval.pt` is secondary diagnostic output only. The primary endpoint checkpoint is always `latest.pt` at exactly 900,000 sampled steps.

## Holdout lock

The untouched formal holdout is the common set of 200 seeds 30,000,000 through 30,000,199. These seeds cannot be used for training, monitoring, checkpoint selection, development diagnostics, or preflight evaluation. Formal evaluation is invalid until all 12 primary matrix runs are complete, all three fixed-LR controls are complete when that ablation is reported, the checkpoint inventory is frozen, and every primary checkpoint is `latest.pt@900000`.

After all training is complete, freeze and verify the checkpoint inventory without running evaluation:

```bash
python -u tools/preflight_eawb_formal_multiseed.py --freeze-checkpoint-inventory
python -u tools/preflight_eawb_formal_multiseed.py --check-holdout-ready
```

Neither command runs environment episodes. Formal holdout evaluation remains a separate, deliberate action and is not invoked by this infrastructure.

## Endpoints and statistics

The primary endpoint is Average Waves Cleared; the key secondary endpoint is W3 clear probability. Required reporting also includes W1/W2 clear probability, return, Red/Blue loss, kill/loss ratio, boundary and ground losses, timeout rate, episode length, W1-to-W2 and W2-to-W3 conversion, Red survivors entering W2 conditioned on W1 clear, and Red survivors entering W3 conditioned on W2 clear. Existing evaluation semantics are unchanged.

The independent algorithm replication unit is the training seed (`n=3`). The 200 holdout episodes are common paired scenario trials, not 200 independent training replicates. Report mean and sample SD across three training seeds, every seed value, paired-seed deltas, and favorable-seed count. Paired scenario bootstrap may describe scenario robustness but not training-replication significance. No training-seed significance claim is required.

Factorial contrasts are:

- Entity main effect: `[(EA - MAPPO) + (EA-WB - WB)] / 2`
- WB main effect: `[(WB - MAPPO) + (EA-WB - EA)] / 2`
- EA x WB interaction: `EA-WB - EA - WB + MAPPO`

## Execution

Run preflight before any formal training. On Ubuntu, enter WSL and activate the CUDA environment:

```bash
wsl -d Ubuntu
conda activate uav
python -u tools/preflight_eawb_formal_multiseed.py
nohup bash tools/run_eawb_formal_multiseed.sh > outputs/formal_eawb_launcher.log 2>&1 &
```

The machine-readable source of truth is `experiments/ea_wb_formal_multiseed_manifest.json`. The launcher creates only the common output parent; each trainer must fresh-create its own run directory.

If WSL is unavailable, use PowerShell only after `conda activate uav`; the same CUDA preflight and launcher protocol applies. CPU training and CPU checkpoint audit are forbidden.
