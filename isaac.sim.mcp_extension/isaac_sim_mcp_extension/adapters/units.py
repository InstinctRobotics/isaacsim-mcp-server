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

"""Joint-limit and drive-gain units.

USD stores a revolute joint's limits in **degrees**, while every joint position
this API reads or writes is in **radians** — ``set_joint_positions``,
``get_joint_positions`` and the drive targets the adapters convert explicitly.
Passing the USD value straight through therefore puts two different units in one
payload: measured on a Franka FR3, joint 1 reported ``limits=[-157.2, 157.2]``
next to ``actual_position=0.5``, where the real limit is ±2.7437 rad. An agent
clamping a target to those limits commands 25 revolutions.

Prismatic limits are already in stage units (metres) and must be left alone —
converting them turns a 0.04 m gripper stroke into 0.0007. That asymmetry is the
whole reason this lives in one place: the naive fix is a one-line sweep that
silently breaks every gripper.

Drive **gains** carry the same split, one level down. ``UsdPhysicsDriveAPI``
defines an angular drive's stiffness and damping per *degree* of error, and the
USD-to-PhysX parse multiplies them by 180/pi; a linear drive's gains are per
stage unit and pass through untouched. Measured on the Isaac Sim 6.0 Franka
Panda, stage ``metersPerUnit=1.0``:

    joint                    USD          PhysX runtime        ratio
    panda_joint1  (revolute) 400 / 80     22918.31 / 4583.66   180/pi
    finger_joint1 (prismatic) 400 / 80      400.00 /   80.00   1

So the same authored ``400`` means 22918 N·m/rad on the arm and 400 N/m on the
gripper. An agent that sizes a gain from the radian-valued positions and limits
this API gives it, then writes that number back unconverted, is off by 57.3x on
every revolute joint — the limits bug again, in the one field an agent reaches
for when a drive is not tracking.

``max_force`` is not converted: it is a torque on an angular drive and a force on
a linear one, and neither carries an angle.

The gain helpers key on the **drive instance name** ("angular"/"linear") while
the limit helpers key on the **joint type** ("revolute"/"prismatic"). That is not
an oversight: the two normally agree, but nothing in USD enforces it, and it is
the drive that decides how its own gains are stored. A prismatic joint carrying a
hand-authored angular drive — which ``get_joint_config`` already reports as
``drive_type: angular``, because it probes rather than infers — has per-degree
gains and metre limits at the same time.

Both adapters and every reporting surface (``get_joint_config``,
``get_robot_info``, ``set_drive_params``) go through here, so a third caller
cannot reintroduce the split.
"""

from __future__ import annotations

import math
from typing import Any, Optional

RADIANS = "radians"
METERS = "meters"
PER_RADIAN = "per_radian"
PER_METER = "per_meter"

# An angular drive gain is authored per degree of error and consumed per radian.
# Spelled out rather than reached through math.degrees(), which reads as "this
# value is an angle" — it is not; it is a gain whose denominator is an angle, so
# the conversion runs the opposite way from the one the name suggests.
_DEGREES_PER_RADIAN = 180.0 / math.pi


def limit_units(joint_type: Optional[str]) -> str:
    """Unit that this joint's limits are reported in."""
    return METERS if (joint_type or "").lower() == "prismatic" else RADIANS


def normalize_limit(value: Any, joint_type: Optional[str]) -> Optional[float]:
    """Convert one raw USD limit into the unit the rest of the API speaks.

    Revolute limits are converted degrees -> radians; prismatic limits are
    returned unchanged because USD already stores them in stage units. ``None``
    (an unauthored limit) stays ``None`` rather than becoming 0.0, which would
    read as a joint pinned shut.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isinf(number) or math.isnan(number):
        # An unlimited (continuous) revolute joint authors +-inf; converting is
        # meaningless but harmless, so keep the sentinel intact for the caller.
        return number
    if limit_units(joint_type) == METERS:
        return number
    return math.radians(number)


LINEAR = "linear"
ANGULAR = "angular"


def drive_type_for(joint_type: Optional[str]) -> str:
    """The DriveAPI instance name a joint of this type would normally carry.

    Only for joints that have no drive yet — read the instance off the prim when
    one exists, because an asset is free to disagree with this.
    """
    return LINEAR if (joint_type or "").lower() == "prismatic" else ANGULAR


def gain_units(drive_type: Optional[str]) -> str:
    """Unit that this drive's gains are reported and accepted in.

    Keyed on the drive instance name, not the joint type — see the module
    docstring.
    """
    return PER_METER if (drive_type or "").lower() == LINEAR else PER_RADIAN


def _as_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def normalize_gain(value: Any, drive_type: Optional[str]) -> Optional[float]:
    """Convert one raw USD drive gain into the unit the rest of the API speaks.

    Angular gains are converted per-degree -> per-radian; linear gains are
    returned unchanged. ``None`` (an unauthored gain, or a joint carrying no
    ``DriveAPI`` at all) stays ``None`` rather than becoming 0.0, which would
    read as a deliberately disabled drive and trip the zero-gain warning.
    """
    number = _as_number(value)
    if number is None:
        return None
    if math.isinf(number) or math.isnan(number):
        # PhysX spells "no limit" as inf on maxForce and it can appear here too;
        # scaling is meaningless but harmless, so keep the sentinel intact.
        return number
    if gain_units(drive_type) == PER_METER:
        return number
    return number * _DEGREES_PER_RADIAN


def denormalize_gain(value: Any, drive_type: Optional[str]) -> Optional[float]:
    """Convert a caller's drive gain into the raw value USD stores.

    The inverse of :func:`normalize_gain`, so a value read back through that
    function round-trips. Writers must go through here: authoring a per-radian
    gain straight onto ``drive:angular:physics:stiffness`` makes the drive 57.3x
    stiffer than asked, which on a 400-stiffness arm is the difference between
    tracking and a joint that fights its own limits.
    """
    number = _as_number(value)
    if number is None:
        return None
    if math.isinf(number) or math.isnan(number):
        return number
    if gain_units(drive_type) == PER_METER:
        return number
    return number / _DEGREES_PER_RADIAN
