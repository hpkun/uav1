"""Streaming gzip raw update logger and fixed-window summary aggregator."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np

RAW_UPDATE_LOG_FILENAME = 'updates.jsonl.gz'
LEGACY_UPDATE_LOG_FILENAME = 'updates.jsonl'
SUMMARY_UPDATE_LOG_FILENAME = 'updates_summary.jsonl'
DEFAULT_UPDATE_LOG_INTERVAL = 100

PRIORITY_METRIC_ORDER = (
    'actor_loss',
    'critic_loss',
    'td_abs',
    'q_mean',
    'q_std',
    'target_q_mean',
    'actor_grad_norm',
    'critic_grad_norm',
    'actor_clip_rate',
    'critic_clip_rate',
    'active_critic_count',
)

CUMULATIVE_COUNTER_METRICS = frozenset({
    'actor_updates',
    'critic_updates',
    'target_updates',
    'actor_optimizer_steps',
})


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _validate_positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _is_numeric(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


class _RunningMetricStats:
    """Online population statistics (Welford) for a single metric over one window."""

    __slots__ = ('count', 'mean', 'm2', 'minimum', 'maximum')

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value

    def summarize(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return dict(mean=None, std=None, min=None, max=None, count=0)
        variance = max(0.0, self.m2 / self.count)
        return dict(
            mean=float(self.mean),
            std=float(math.sqrt(variance)),
            min=float(self.minimum),
            max=float(self.maximum),
            count=self.count,
        )


class UpdateWindowAggregator:
    """Aggregate streaming policy-update metrics into fixed-size summary windows."""

    def __init__(
        self,
        interval: int = DEFAULT_UPDATE_LOG_INTERVAL,
        *,
        counter_keys=CUMULATIVE_COUNTER_METRICS,
        priority_metrics=PRIORITY_METRIC_ORDER,
    ):
        self.interval = _validate_positive_int(interval, 'update_log_interval')
        self.counter_keys = frozenset(counter_keys)
        self.priority_metrics = tuple(priority_metrics)
        self._reset()

    def _reset(self) -> None:
        self.step_start = None
        self.step_end = None
        self.update_start = None
        self.update_end = None
        self.count = 0
        self._stats: dict[str, _RunningMetricStats] = {}
        self._counters: dict[str, int | float] = {}

    def add(self, step: int, update: int, metrics: dict) -> dict | None:
        step_int = int(step)
        update_int = int(update)
        if self.count == 0:
            self.step_start = step_int
            self.update_start = update_int
        self.step_end = step_int
        self.update_end = update_int
        self.count += 1

        for key, value in metrics.items():
            if key in ('step', 'update'):
                continue
            if key in self.counter_keys:
                if value is not None and _is_numeric(value):
                    numeric = float(value)
                    if math.isfinite(numeric):
                        if isinstance(value, (int, np.integer)):
                            self._counters[key] = int(value)
                        else:
                            self._counters[key] = numeric
                continue
            if value is None:
                self._stats.setdefault(key, _RunningMetricStats())
                continue
            if _is_numeric(value):
                stat = self._stats.setdefault(key, _RunningMetricStats())
                numeric = float(value)
                if math.isfinite(numeric):
                    stat.add(numeric)

        if self.count >= self.interval:
            return self.flush()
        return None

    def flush(self) -> dict | None:
        if self.count == 0:
            return None
        ordered_keys = [key for key in self.priority_metrics if key in self._stats]
        priority_set = set(ordered_keys)
        ordered_keys.extend(key for key in self._stats if key not in priority_set)
        summary = dict(
            step_start=self.step_start,
            step_end=self.step_end,
            update_start=self.update_start,
            update_end=self.update_end,
            count=self.count,
            metrics={key: self._stats[key].summarize() for key in ordered_keys},
            counters=dict(self._counters),
        )
        self._reset()
        return summary


class UpdateLogWriter:
    """Stream raw updates into updates.jsonl.gz and window summaries into updates_summary.jsonl."""

    def __init__(
        self,
        root: str | Path,
        *,
        interval: int = DEFAULT_UPDATE_LOG_INTERVAL,
        raw_flush_interval: int | None = None,
    ):
        self.root = Path(root)
        self.interval = _validate_positive_int(interval, 'update_log_interval')
        if raw_flush_interval is None:
            self.raw_flush_interval = min(self.interval, 100)
        else:
            self.raw_flush_interval = _validate_positive_int(raw_flush_interval, 'raw_flush_interval')
        self.raw_path = self.root / RAW_UPDATE_LOG_FILENAME
        self.summary_path = self.root / SUMMARY_UPDATE_LOG_FILENAME
        self._aggregator = UpdateWindowAggregator(self.interval)
        self.raw_count = 0
        self.summary_count = 0
        self._closed = False
        self._raw_file = gzip.open(self.raw_path, 'xt', encoding='utf-8')
        try:
            self._summary_file = self.summary_path.open('x', encoding='utf-8')
        except BaseException:
            self._raw_file.close()
            raise

    @property
    def closed(self) -> bool:
        return self._closed and self._raw_file.closed and self._summary_file.closed

    def _write_summary(self, summary: dict) -> None:
        self._summary_file.write(json.dumps(summary, allow_nan=False, default=_plain) + '\n')
        self._summary_file.flush()
        self.summary_count += 1

    def record(self, step: int, update: int, metrics: dict) -> dict | None:
        if self._closed:
            raise ValueError('Cannot write to a closed UpdateLogWriter')
        row = dict(step=int(step), update=int(update), **metrics)
        self._raw_file.write(json.dumps(row, allow_nan=False, default=_plain) + '\n')
        self.raw_count += 1
        summary = self._aggregator.add(int(step), int(update), metrics)
        if summary is not None:
            self._write_summary(summary)
            self._raw_file.flush()
        elif self.raw_count % self.raw_flush_interval == 0:
            self._raw_file.flush()
        return summary

    def _flush_open(self) -> dict | None:
        summary = self._aggregator.flush()
        if summary is not None:
            self._write_summary(summary)
        else:
            self._summary_file.flush()
        self._raw_file.flush()
        return summary

    def flush(self) -> dict | None:
        if self._closed:
            return None
        return self._flush_open()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error = None
        try:
            self._flush_open()
        except BaseException as error:
            first_error = error
        for handle in (self._summary_file, self._raw_file):
            try:
                if not handle.closed:
                    handle.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def resolve_update_log_path(run_or_path: str | Path) -> Path:
    """Resolve either a run directory or a direct raw update log path (.jsonl.gz or legacy .jsonl)."""
    target = Path(run_or_path)
    if target.is_dir():
        compressed = target / RAW_UPDATE_LOG_FILENAME
        if compressed.is_file():
            return compressed
        legacy = target / LEGACY_UPDATE_LOG_FILENAME
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(compressed)
    return target


def iter_update_log(run_or_path: str | Path):
    """Yield parsed rows from updates.jsonl.gz or legacy updates.jsonl."""
    path = resolve_update_log_path(run_or_path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def read_update_log(run_or_path: str | Path) -> list[dict]:
    """Read all raw update rows from either compressed (.jsonl.gz) or legacy (.jsonl) logs."""
    return list(iter_update_log(run_or_path))


def read_update_summary(run_or_path: str | Path) -> list[dict]:
    """Read all aggregated summary rows from updates_summary.jsonl."""
    target = Path(run_or_path)
    path = target / SUMMARY_UPDATE_LOG_FILENAME if target.is_dir() else target
    with path.open('rt', encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]
