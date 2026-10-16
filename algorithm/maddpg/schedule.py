"""Update credit counts only new environment transitions strictly after warmup."""


class SampledTransitionSchedule:
    def __init__(self, learning_starts, sampled_steps_per_update):
        if any(type(v) is not int or v <= 0 for v in (learning_starts, sampled_steps_per_update)):
            raise ValueError('positive integer warmup and update interval required')
        self.learning_starts = learning_starts
        self.interval = sampled_steps_per_update
        self.credit = 0
        self.event_count = 0

    def add(self, previous_steps, sampled_steps):
        if sampled_steps < previous_steps:
            raise ValueError('sampled steps must increase')
        self.credit += max(0, sampled_steps-max(previous_steps, self.learning_starts))

    def consume(self):
        if self.credit < self.interval:
            raise RuntimeError('insufficient sampled transition update credit')
        self.credit -= self.interval
        self.event_count += 1
