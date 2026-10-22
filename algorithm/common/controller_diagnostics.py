"""Read-only v3.29 step diagnostics; no optimizer or sampling changes."""
import numpy as np


def controller_metric_values(infos):
    values = {}
    for side in ('red', 'blue'):
        for info in infos:
            row = info.get('controller_diagnostics', {}).get(side)
            if not row or not row['agent_steps']:
                continue
            n = row['agent_steps']
            sums = dict(mean_nz=row['nz_sum'], nz_saturation_fraction=row['nz_saturated'],
                        mean_abs_phi=row['abs_phi_sum'], action_abs_mean=sum(row['abs_action_sum']) / 3,
                        action_saturation_fraction=sum(row['action_saturated']) / 3)
            for i, dimension in enumerate(('psi', 'theta', 'v')):
                sums[f'mean_abs_high_level_action_{dimension}'] = row['abs_action_sum'][i]
                sums[f'action_saturation_fraction_{dimension}'] = row['action_saturated'][i]
            for key, total in sums.items():
                # Existing rollout aggregation averages arrays. Replication
                # preserves live-agent weighting across steps/environments.
                values.setdefault(f'{side}_{key}', []).extend([total / n] * n)
    for key in ('live', 'range', 'range_az', 'geometry', 'eligible'):
        rows = [info['red_firing_geometry_agent_steps'][key] for info in infos
                if key in info.get('red_firing_geometry_agent_steps', {})]
        if rows:
            values[f'red_funnel_{key}_agent_steps_per_env_step'] = rows
    return {key: np.asarray(value, dtype=float) for key, value in values.items()}
