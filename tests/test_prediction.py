"""Geometric estimates and bounded-acceleration reference rollout."""

import math

import numpy as np
import pytest

from assignment.prediction import (constant_velocity_eit, forecast_direct_attacker,
                                   predict_encounter)


def test_constant_velocity_eit_handles_head_on_stationary_and_unreachable():
    assert constant_velocity_eit([0, 0], [0, 10], [0, -5], 5) == pytest.approx(1)
    assert constant_velocity_eit([0, 0], [0, 10], [0, 0], 5) == pytest.approx(2)
    assert math.isinf(constant_velocity_eit([0, 0], [0, 10], [0, 6], 5))
    assert constant_velocity_eit([0, 0], [0, 0], [0, 6], 5) == 0


def test_forecast_respects_speed_and_acceleration_with_environment_step_order():
    forecast = forecast_direct_attacker([0, 80], [0, -2], [0, -80], dt=.1,
                                        speed_max=5, acceleration_max=2,
                                        breach_radius=15, max_steps=500)
    np.testing.assert_allclose(forecast.positions[1], [0, 79.8])
    assert np.linalg.norm(forecast.velocities[1]) == pytest.approx(2.2)
    assert np.linalg.norm(forecast.velocities, axis=1).max() <= 5 + 1e-12
    assert forecast.breach_time_s is not None
    assert 29 < forecast.breach_time_s < 31


def test_reference_prediction_reports_capture_before_breach_and_fallback():
    forecast = forecast_direct_attacker([0, 20], [0, -2], [0, 0], dt=.1,
                                        speed_max=5, acceleration_max=2,
                                        breach_radius=2, max_steps=100)
    meeting = predict_encounter([0, -10], [0, 2], forecast,
                                defender_speed_max=5, defender_acceleration_max=2,
                                capture_radius=5)
    assert meeting.reference_capture_s is not None
    assert meeting.reference_capture_s <= meeting.forecast_breach_s
    assert meeting.closest_distance_m <= 5
    assert meeting.geometric_eit_s is not None
    away = predict_encounter([0, -100], [0, -5], forecast,
                             defender_speed_max=5, defender_acceleration_max=2,
                             capture_radius=5)
    assert away.reference_capture_s is None
    assert away.closest_distance_m > 5
