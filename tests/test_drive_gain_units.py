# MIT License
#
# Copyright (c) 2023-2025 omni-mcp
# Copyright (c) 2026 whats2000
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Drive gains must reach the caller in the same units as positions and limits.

``UsdPhysicsDriveAPI`` stores an angular drive's stiffness and damping per
*degree* and the USD-to-PhysX parse multiplies them by 180/pi; a linear drive's
gains are per stage unit and pass through. Measured on the Isaac Sim 6.0 Franka
Panda with ``metersPerUnit=1.0``:

    joint                     USD          PhysX runtime        ratio
    panda_joint1  (revolute)  400 / 80     22918.31 / 4583.66   180/pi
    panda_finger_joint1 (pris) 400 / 80      400.00 /   80.00   1

So one authored ``400`` meant 22918 N*m/rad on the arm and 400 N/m on the
gripper, in a payload whose every position and limit was already radians and
metres, with nothing to say which convention a given number followed. This is
the joint-limit split one level down, in the field an agent reaches for when a
drive is not tracking.
"""

from __future__ import annotations

import ast
import math
import os

from isaac_sim_mcp_extension.adapters.units import (
    PER_METER,
    PER_RADIAN,
    denormalize_gain,
    drive_type_for,
    gain_units,
    normalize_gain,
)

ADAPTERS = os.path.join(
    os.path.dirname(__file__), "..", "isaac.sim.mcp_extension", "isaac_sim_mcp_extension", "adapters"
)

# Straight off the wire: the USD value, then what the PhysX articulation reports.
FR3_ANGULAR_USD = 400.0
FR3_ANGULAR_RUNTIME = 22918.3125
FR3_LINEAR_USD = 400.0


class _close:
    """`pytest.approx` for floats, without it.

    conftest stubs numpy, and approx reaches into `numpy.isscalar` on every
    comparison — a name the stub does not carry, so the assertion dies with an
    AttributeError that says nothing about the value being tested.
    """

    def __init__(self, expected, rel=1e-6):
        self.expected = float(expected)
        self.rel = rel

    def __eq__(self, other):
        return math.isclose(float(other), self.expected, rel_tol=self.rel)

    def __repr__(self):
        return f"{self.expected} (+-{self.rel:g} relative)"


def test_angular_gain_converts_per_degree_to_per_radian():
    """The measured ratio, not one derived from the formula under test."""
    assert normalize_gain(FR3_ANGULAR_USD, "angular") == _close(FR3_ANGULAR_RUNTIME)
    assert gain_units("angular") == PER_RADIAN


def test_linear_gain_is_left_alone():
    """The gripper case: a linear drive's gains are per stage unit already."""
    assert normalize_gain(FR3_LINEAR_USD, "linear") == FR3_LINEAR_USD
    assert denormalize_gain(FR3_LINEAR_USD, "linear") == FR3_LINEAR_USD
    assert gain_units("linear") == PER_METER


def test_gain_round_trips():
    for drive_type in ("angular", "linear"):
        assert denormalize_gain(normalize_gain(400.0, drive_type), drive_type) == _close(400.0)
        assert normalize_gain(denormalize_gain(400.0, drive_type), drive_type) == _close(400.0)


def test_denormalize_is_the_inverse_not_a_repeat_of_normalize():
    """Both directions applied once must not compound — 400 -> 22918 -> 400."""
    assert denormalize_gain(FR3_ANGULAR_RUNTIME, "angular") == _close(FR3_ANGULAR_USD)


def test_the_measured_ratio_is_the_one_units_applies():
    """Guards the constant against a direction flip, which round-tripping hides."""
    assert normalize_gain(1.0, "angular") == _close(180.0 / math.pi)
    assert denormalize_gain(180.0 / math.pi, "angular") == _close(1.0)


def test_unauthored_gain_stays_none_rather_than_zero():
    """0.0 would read as a deliberately disabled drive and trip the zero-gain warning."""
    assert normalize_gain(None, "angular") is None
    assert normalize_gain("nonsense", "angular") is None
    assert denormalize_gain(None, "angular") is None


def test_infinite_gain_is_preserved():
    assert normalize_gain(float("inf"), "angular") == float("inf")
    assert denormalize_gain(float("inf"), "angular") == float("inf")


def test_gain_units_key_on_the_drive_not_the_joint():
    """A prismatic joint carrying an angular drive still stores per-degree gains.

    gain_units takes the DriveAPI instance name; limit_units takes the joint
    type. The two normally agree, but nothing in USD enforces it, and keying
    gains on the joint type would skip the conversion on exactly the malformed
    asset that needs it. get_joint_config already probes for the drive rather
    than inferring it, so it reports such a joint as `angular` today.
    """
    assert gain_units("angular") == PER_RADIAN
    assert gain_units("linear") == PER_METER
    assert drive_type_for("prismatic") == "linear"
    assert drive_type_for("revolute") == "angular"


def _source(filename):
    with open(os.path.join(ADAPTERS, filename)) as f:
        return f.read()


def _func_src(filename, name):
    text = _source(filename)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{name} not found in {filename}")


def test_neither_adapter_returns_a_raw_usd_gain():
    """Both adapters must go through units.py, the way both already do for limits.

    get_joint_config is the only surface reporting gains; when it passed the raw
    value through, the number it gave and the number PhysX ran differed by 57.3x
    on every revolute joint.
    """
    for filename in ("v5.py", "v6.py"):
        body = _func_src(filename, "get_joint_config")
        assert "GetStiffnessAttr" in body, f"{filename}:get_joint_config no longer reads a gain"
        assert "normalize_gain" in body, f"{filename}:get_joint_config returns a raw USD gain"
        assert "gain_units" in body, f"{filename}:get_joint_config reports a gain with no unit"


def test_the_conversion_lives_only_in_units():
    """One place, so a third caller cannot reintroduce the split."""
    for filename in ("v5.py", "v6.py", "base.py"):
        assert "180.0 / math.pi" not in _source(filename), f"{filename} hardcodes the degree/radian gain conversion"
