# Workspace collaboration

- Git is managed exclusively by the user. Do not run Git management commands,
  create commits/branches, merge, stage, reset, or delete branches unless the user
  explicitly authorizes a new Git operation. The one-off merge requested on
  2026-10-27 is complete.
- Run commands in PowerShell through the existing `uav` Conda environment.
- Preserve scientific implementations, checkpoint contracts and historical runs.
  Framework changes must reuse existing sampling and learning loops.
- The one-time Baseline Cleanup authorized on 2026-11-19 removed the old
  `runs/` and `replays/` contents. Preserve all results created after that cleanup.
- Do not launch long training/evaluation jobs without an explicit request.
- After each task, remove temporary files and directories created for that task.
  Add reusable cache directories to `.gitignore`; preserve historical runs and
  files that existed before the task.
