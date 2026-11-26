"""Tests for compressed raw update logs and fixed-window summary aggregation."""
from dataclasses import replace
import gzip
import json

import numpy as np
import pytest

from algorithm.maddpg_config import MADDPGConfig
from algorithm.mappo_config import MAPPOConfig
from env.interception import InterceptionConfig
from experiment import interception as exp
from experiment.cli import main as cli_main
from experiment.update_log import (
    CUMULATIVE_COUNTER_METRICS,
    DEFAULT_UPDATE_LOG_INTERVAL,
    PRIORITY_METRIC_ORDER,
    UpdateLogWriter,
    UpdateWindowAggregator,
    iter_update_log,
    read_update_log,
    read_update_summary,
)


def test_gzip_streaming_write_line_by_line_read_and_window_boundaries(tmp_path):
    writer = UpdateLogWriter(tmp_path, interval=100)
    assert not writer.closed
    for idx in range(1, 251):
        writer.record(
            step=idx * 10,
            update=idx,
            metrics=dict(
                actor_loss=-float(idx),
                critic_loss=float(idx) * 0.5,
                actor_updates=idx,
                critic_updates=idx,
                target_updates=idx,
                actor_optimizer_steps=idx * 5,
            ),
        )
    # Two full 100-update windows written before close; final 50 flushed on close.
    assert writer.summary_count == 2
    writer.close()
    assert writer.closed
    with pytest.raises(ValueError, match='closed'):
        writer.record(2510, 251, {'actor_loss': 0.0})

    raw_path = tmp_path / 'updates.jsonl.gz'
    summary_path = tmp_path / 'updates_summary.jsonl'
    assert raw_path.is_file()
    assert summary_path.is_file()
    assert not (tmp_path / 'updates.jsonl').exists()

    with gzip.open(raw_path, 'rt', encoding='utf-8') as file:
        streamed = [json.loads(line) for line in file]
    assert len(streamed) == 250
    assert streamed[0]['step'] == 10 and streamed[0]['update'] == 1
    assert streamed[-1]['step'] == 2500 and streamed[-1]['update'] == 250
    assert read_update_log(tmp_path) == streamed
    assert list(iter_update_log(raw_path)) == streamed

    summaries = read_update_summary(tmp_path)
    assert len(summaries) == 3
    assert [row['count'] for row in summaries] == [100, 100, 50]
    assert sum(row['count'] for row in summaries) == len(streamed)
    assert (summaries[0]['step_start'], summaries[0]['step_end']) == (10, 1000)
    assert (summaries[0]['update_start'], summaries[0]['update_end']) == (1, 100)
    assert (summaries[2]['step_start'], summaries[2]['step_end']) == (2010, 2500)
    assert (summaries[2]['update_start'], summaries[2]['update_end']) == (201, 250)
    assert summaries[2]['counters'] == dict(
        actor_updates=250,
        critic_updates=250,
        target_updates=250,
        actor_optimizer_steps=1250,
    )
    for counter_name in CUMULATIVE_COUNTER_METRICS:
        assert counter_name not in summaries[2]['metrics']


def test_population_statistics_and_exact_multiple_window():
    aggregator = UpdateWindowAggregator(interval=4)
    values = [2.0, 4.0, 4.0, 6.0]
    output = None
    for idx, val in enumerate(values, start=1):
        output = aggregator.add(
            step=100 + idx,
            update=idx,
            metrics=dict(
                critic_loss=val,
                custom_metric=1e9 + val,
                actor_updates=idx,
            ),
        )
    assert output is not None
    assert aggregator.flush() is None
    assert output['count'] == 4
    critic = output['metrics']['critic_loss']
    assert critic['count'] == 4
    assert critic['mean'] == pytest.approx(float(np.mean(values)))
    assert critic['std'] == pytest.approx(float(np.std(values, ddof=0)))
    assert critic['min'] == pytest.approx(2.0)
    assert critic['max'] == pytest.approx(6.0)
    # Welford keeps population std accurate even with a large constant offset.
    custom = output['metrics']['custom_metric']
    assert custom['std'] == pytest.approx(float(np.std(values, ddof=0)), rel=1e-7)
    assert output['counters'] == {'actor_updates': 4}

    # Single-sample partial window has zero population std.
    assert aggregator.add(200, 5, {'critic_loss': 3.5}) is None
    single = aggregator.flush()
    assert single['count'] == 1
    assert single['metrics']['critic_loss'] == dict(mean=3.5, std=0.0, min=3.5, max=3.5, count=1)


def test_none_nonfinite_and_cumulative_counter_handling():
    aggregator = UpdateWindowAggregator(interval=5)
    rows = [
        dict(actor_loss=None, critic_loss=1.0, td_abs=float('nan'), actor_updates=1, extra_flag=True),
        dict(actor_loss=2.0, critic_loss=float('inf'), td_abs=None, actor_updates=None),
        dict(actor_loss=float('-inf'), critic_loss=3.0, td_abs=float('inf'), actor_updates=float('nan')),
        dict(actor_loss=6.0, critic_loss=5.0, td_abs=None, actor_updates=4),
    ]
    for idx, metrics in enumerate(rows, start=1):
        assert aggregator.add(idx * 10, idx, metrics) is None
    summary = aggregator.flush()
    assert summary['count'] == 4
    # Valid actor_loss values: [2.0, 6.0] -> mean=4.0, pop std=2.0
    assert summary['metrics']['actor_loss'] == dict(mean=4.0, std=2.0, min=2.0, max=6.0, count=2)
    # Valid critic_loss values: [1.0, 3.0, 5.0] -> mean=3.0, pop std=sqrt(8/3)
    critic = summary['metrics']['critic_loss']
    assert critic['count'] == 3
    assert critic['mean'] == pytest.approx(3.0)
    assert critic['std'] == pytest.approx(float(np.std([1.0, 3.0, 5.0], ddof=0)))
    assert critic['min'] == pytest.approx(1.0)
    assert critic['max'] == pytest.approx(5.0)
    # All samples for td_abs were None/non-finite -> count=0, null stats, valid JSON.
    assert summary['metrics']['td_abs'] == dict(mean=None, std=None, min=None, max=None, count=0)
    assert 'extra_flag' not in summary['metrics']
    assert summary['counters'] == {'actor_updates': 4}
    json.dumps(summary, allow_nan=False)


def test_priority_metric_ordering_and_generic_metric_discovery():
    aggregator = UpdateWindowAggregator(interval=2)
    metrics = {
        'zeta_metric': 1.0,
        'critic_loss': 2.0,
        'alpha_metric': 3.0,
        'actor_loss': 4.0,
        'td_abs': 5.0,
        'actor_updates': 1,
    }
    aggregator.add(10, 1, metrics)
    summary = aggregator.add(20, 2, dict(metrics, actor_updates=2))
    assert list(summary['metrics'].keys()) == [
        'actor_loss',
        'critic_loss',
        'td_abs',
        'zeta_metric',
        'alpha_metric',
    ]
    assert summary['counters'] == {'actor_updates': 2}


def test_update_log_interval_cli_validation_and_preflight(tmp_path):
    default_plan = cli_main([
        'preflight',
        '--env-config', 'configs/interception_smoke.yaml',
        '--algorithm-config', 'configs/mappo_interception.yaml',
        '--algorithm', 'mappo',
        '--attack-mode', 'direct_rush',
        '--total-sampled-steps', '4',
        '--num-envs', '2',
        '--output-dir', str(tmp_path / 'plan_default'),
    ])
    assert default_plan['update_log_interval'] == DEFAULT_UPDATE_LOG_INTERVAL == 100

    custom_plan = cli_main([
        'preflight',
        '--env-config', 'configs/interception_smoke.yaml',
        '--algorithm-config', 'configs/mappo_interception.yaml',
        '--algorithm', 'mappo',
        '--attack-mode', 'direct_rush',
        '--total-sampled-steps', '4',
        '--num-envs', '2',
        '--update-log-interval', '25',
        '--output-dir', str(tmp_path / 'plan_custom'),
    ])
    assert custom_plan['update_log_interval'] == 25

    cli_out = tmp_path / 'cli_train'
    cli_main([
        'train',
        '--env-config', 'configs/interception_smoke.yaml',
        '--algorithm-config', 'configs/mappo_interception.yaml',
        '--algorithm', 'mappo',
        '--attack-mode', 'direct_rush',
        '--total-sampled-steps', '512',
        '--num-envs', '2',
        '--eval-interval', '0',
        '--log-every-steps', '512',
        '--update-log-interval', '1',
        '--output-dir', str(cli_out),
    ])
    assert json.loads((cli_out / 'settings.json').read_text(encoding='utf-8'))['update_log_interval'] == 1
    assert len(read_update_log(cli_out)) == 2
    assert [row['count'] for row in read_update_summary(cli_out)] == [1, 1]

    for bad in ('0', '-10'):
        with pytest.raises(ValueError, match='update log interval'):
            cli_main([
                'preflight',
                '--env-config', 'configs/interception_smoke.yaml',
                '--algorithm-config', 'configs/mappo_interception.yaml',
                '--algorithm', 'mappo',
                '--attack-mode', 'direct_rush',
                '--total-sampled-steps', '4',
                '--num-envs', '2',
                '--update-log-interval', bad,
                '--output-dir', str(tmp_path / f'plan_bad_{bad}'),
            ])

    for bad_value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            UpdateWindowAggregator(interval=bad_value)
        with pytest.raises(ValueError, match='positive integer'):
            exp.train(
                InterceptionConfig(episode_steps=2),
                MAPPOConfig(hidden_size=16, update_epochs=1, minibatch_size=4),
                2,
                'mappo',
                steps=4,
                num_envs=2,
                seed=7,
                device='cpu',
                attack_mode='direct_rush',
                output=tmp_path / f'bad_{type(bad_value).__name__}_{bad_value}',
                eval_interval=0,
                log_every_steps=4,
                update_log_interval=bad_value,
            )


@pytest.mark.parametrize('algorithm', ('mappo', 'maddpg'))
def test_mappo_and_maddpg_short_training_logs_and_checkpoints(tmp_path, algorithm):
    config = InterceptionConfig(episode_steps=2)
    if algorithm == 'mappo':
        parameters = MAPPOConfig(hidden_size=16, update_epochs=1, minibatch_size=4)
        rollout = 2
        steps = 20
        expected_updates = 5
    else:
        parameters = MADDPGConfig(
            hidden_dim=16,
            batch_size=2,
            buffer_capacity=16,
            learning_starts=2,
            update_interval_steps=2,
            updates_per_interval=1,
        )
        rollout = 2
        steps = 12
        expected_updates = 6

    output = tmp_path / algorithm
    summary = exp.train(
        config,
        parameters,
        rollout,
        algorithm,
        steps=steps,
        num_envs=2,
        seed=7,
        device='cpu',
        attack_mode='direct_rush',
        output=output,
        eval_interval=steps // 2,
        eval_episodes=1,
        log_every_steps=steps,
        update_log_interval=4,
    )
    assert summary['updates'] == expected_updates
    settings = json.loads((output / 'settings.json').read_text(encoding='utf-8'))
    assert settings['update_log_interval'] == 4

    assert (output / 'updates.jsonl.gz').is_file()
    assert (output / 'updates_summary.jsonl').is_file()
    assert (output / 'progress.jsonl').is_file()
    assert not (output / 'updates.jsonl').exists()

    raw_rows = read_update_log(output)
    summary_rows = read_update_summary(output)
    assert len(raw_rows) == expected_updates
    assert [row['update'] for row in raw_rows] == list(range(1, expected_updates + 1))
    assert sum(row['count'] for row in summary_rows) == expected_updates
    assert [row['count'] for row in summary_rows] == [4, expected_updates - 4]
    assert summary_rows[0]['update_start'] == 1 and summary_rows[0]['update_end'] == 4
    assert summary_rows[-1]['update_end'] == expected_updates

    # Cross-check summary stats directly against raw gzip records for the first window.
    first_window = raw_rows[:4]
    for metric_name, stat in summary_rows[0]['metrics'].items():
        vals = [row[metric_name] for row in first_window if row.get(metric_name) is not None]
        assert stat['count'] == len(vals)
        if vals:
            assert stat['mean'] == pytest.approx(float(np.mean(vals)))
            assert stat['std'] == pytest.approx(float(np.std(vals, ddof=0)))
            assert stat['min'] == pytest.approx(float(np.min(vals)))
            assert stat['max'] == pytest.approx(float(np.max(vals)))

    if algorithm == 'maddpg':
        assert list(summary_rows[0]['metrics'].keys())[:len(PRIORITY_METRIC_ORDER)] == list(PRIORITY_METRIC_ORDER)
        assert summary_rows[-1]['counters']['actor_updates'] == raw_rows[-1]['actor_updates']
        assert summary_rows[-1]['counters']['critic_updates'] == raw_rows[-1]['critic_updates']
        assert summary_rows[-1]['counters']['target_updates'] == raw_rows[-1]['target_updates']
        assert summary_rows[-1]['counters']['actor_optimizer_steps'] == raw_rows[-1]['actor_optimizer_steps']
        for counter_name in CUMULATIVE_COUNTER_METRICS:
            assert counter_name not in summary_rows[0]['metrics']
    else:
        assert {'actor_loss', 'critic_loss', 'entropy', 'approx_kl', 'explained_variance'} <= summary_rows[0]['metrics'].keys()

    # Existing checkpoint save, load and evaluation still work unchanged.
    restored, policy, name, _, _ = exp.load_run(output, 'Best', 'cpu')
    assert name == algorithm
    exp.load_run(output, 'Last', 'cpu')
    exp.load_run(output, f'step_{steps // 2}', 'cpu')
    eval_rows = exp.evaluate_policy(restored, policy, 'direct_rush', 1, 900001)
    assert len(eval_rows) == 1


def test_exception_flushes_partial_summary_and_closes_gzip_cleanly(tmp_path, monkeypatch):
    config = InterceptionConfig(episode_steps=2)
    parameters = MAPPOConfig(hidden_size=16, update_epochs=1, minibatch_size=4)
    real_evaluate = exp.evaluate_policy
    calls = 0

    def fail_after_first_eval(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated crash')
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(exp, 'evaluate_policy', fail_after_first_eval)
    output = tmp_path / 'crashed'
    with pytest.raises(RuntimeError, match='simulated crash'):
        exp.train(
            config,
            parameters,
            2,
            'mappo',
            steps=8,
            num_envs=2,
            seed=7,
            device='cpu',
            attack_mode='direct_rush',
            output=output,
            eval_interval=4,
            eval_episodes=1,
            log_every_steps=8,
            update_log_interval=100,
        )

    assert (output / 'updates.jsonl.gz').is_file()
    assert (output / 'updates_summary.jsonl').is_file()
    assert not (output / 'updates.jsonl').exists()
    raw_rows = read_update_log(output)
    summary_rows = read_update_summary(output)
    assert len(raw_rows) == 2
    assert len(summary_rows) == 1
    assert summary_rows[0]['count'] == 2
    assert summary_rows[0]['update_start'] == 1 and summary_rows[0]['update_end'] == 2


def test_read_update_log_supports_legacy_uncompressed_and_compressed(tmp_path):
    legacy_dir = tmp_path / 'legacy'
    legacy_dir.mkdir()
    (legacy_dir / 'updates.jsonl').write_text(
        json.dumps({'step': 10, 'update': 1, 'critic_loss': 1.25}) + '\n',
        encoding='utf-8',
    )
    assert read_update_log(legacy_dir) == [{'step': 10, 'update': 1, 'critic_loss': 1.25}]
    assert read_update_log(legacy_dir / 'updates.jsonl') == [{'step': 10, 'update': 1, 'critic_loss': 1.25}]

    new_dir = tmp_path / 'compressed'
    new_dir.mkdir()
    with UpdateLogWriter(new_dir, interval=10) as writer:
        writer.record(20, 1, {'critic_loss': 0.75})
    assert read_update_log(new_dir) == [{'step': 20, 'update': 1, 'critic_loss': 0.75}]
    assert read_update_log(new_dir / 'updates.jsonl.gz') == [{'step': 20, 'update': 1, 'critic_loss': 0.75}]
