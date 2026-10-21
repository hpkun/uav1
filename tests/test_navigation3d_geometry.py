"""Independent vector references for navigation-only NED geometry."""
import itertools
import numpy as np
import pytest
from env.navigation3d.geometry import direction, body_basis, waypoint_geometry


@pytest.mark.parametrize('pitch,heading,elevation,bearing', list(itertools.product(
    [-np.pi/3, 0, np.pi/3], [-np.pi, -.7, 0, np.pi-1e-12], [-.8, 0, .8], [-3., 0., 2.])))
def test_geometry_against_independent_dot(pitch, heading, elevation, bearing):
    p = np.array([123., -456., -789.])
    displacement = 1500*np.array([np.cos(elevation)*np.cos(bearing),
        np.cos(elevation)*np.sin(bearing), -np.sin(elevation)])
    g = waypoint_geometry(p, pitch, heading, p+displacement)
    reference_forward = np.array([np.cos(pitch)*np.cos(heading), np.cos(pitch)*np.sin(heading), -np.sin(pitch)])
    expected = np.arccos(np.clip(reference_forward.dot(displacement)/np.linalg.norm(displacement), -1, 1))
    assert g.distance == pytest.approx(1500)
    assert g.angle == pytest.approx(expected, abs=3e-8)
    assert g.height_difference == pytest.approx(-displacement[2])
    assert np.cos(g.angle) == pytest.approx(np.cos(g.body_elevation)*np.cos(g.body_azimuth))
    basis = body_basis(pitch, heading)
    assert basis@basis.T == pytest.approx(np.eye(3), abs=1e-12)
    body = basis @ displacement
    assert basis.T@body == pytest.approx(displacement, abs=1e-9)


@pytest.mark.parametrize('angle', [0, 15, 30, 30.001, 60, 90, 180])
def test_horizontal_angle(angle):
    g = waypoint_geometry([0, 0, 0], 0, 0, 100*direction(0, np.deg2rad(angle)))
    assert np.rad2deg(g.angle) == pytest.approx(angle, abs=1e-9)


def test_pitch_up_waypoint_below_and_reverse():
    for pitch, elevation in [(50, -20), (-50, 20)]:
        g = waypoint_geometry([0, 0, 0], np.deg2rad(pitch), 0, 1000*direction(np.deg2rad(elevation), 0))
        assert np.rad2deg(g.angle) == pytest.approx(70)
        assert np.rad2deg(g.body_elevation) == pytest.approx(elevation-pitch)


@pytest.mark.parametrize('pitch', [-1., 0., 1.])
def test_direction_follows_climb_and_descent(pitch):
    v = direction(pitch, .7)
    assert np.linalg.norm(v) == pytest.approx(1)
    assert np.sign(-v[2]) == np.sign(pitch)
    assert waypoint_geometry([0, 0, 0], pitch, .7, v*100).angle == pytest.approx(0., abs=1e-15)


def test_heading_wrap_and_vertical_waypoint():
    g = waypoint_geometry([0, 0, 0], 0, np.deg2rad(179), direction(0, np.deg2rad(-179))*100)
    assert np.rad2deg(g.heading_error) == pytest.approx(2)
    for height in [-100, 100]:
        g = waypoint_geometry([0, 0, 0], 0, 2., [0, 0, -height])
        assert g.angle == pytest.approx(np.pi/2)
        assert g.heading_error == 0


@pytest.mark.parametrize('distance', [0., 1e-300, 1e-15, 1., 1e150])
def test_tiny_coincident_and_large_coordinates(distance):
    g = waypoint_geometry([0, 0, 0], 0, 0, [distance, 0, 0])
    assert np.isfinite(list(g.__dict__.values())).all()
    assert g.distance == distance and g.angle == 0


@pytest.mark.parametrize('bad', [[np.nan, 0, 0], [np.inf, 0, 0], [1, 2]])
def test_nonfinite_or_invalid_input_rejected(bad):
    with pytest.raises(ValueError):
        waypoint_geometry(bad, 0, 0, [0, 0, 0])
