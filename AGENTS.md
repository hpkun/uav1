# Repository Runtime Policy

- Run training, evaluation, and audit commands in WSL Ubuntu after `conda activate uav`.
- If WSL is unavailable, use the PowerShell/Windows `uav` Conda environment.
- CUDA is mandatory for training and checkpoint/evaluation audits. Fail explicitly when CUDA is unavailable; never fall back to CPU.
- Run repository tests and static checks from the same `uav` Conda environment.
