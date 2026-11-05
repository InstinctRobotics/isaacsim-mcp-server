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

"""Robot creation and control command handlers."""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Sequence

from ..adapters.base import IsaacAdapterBase
from .objects import prim_missing

# Hardcoded fallback — used only if live discovery fails.
# Keys are lowercase robot names, asset_path is relative to the assets root.
# These files are still published under /Isaac/Robots/ on 6.x.
FALLBACK_ROBOT_LIBRARY: Dict[str, Dict[str, str]] = {
    "frankapanda": {
        "asset_path": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "description": "FrankaRobotics FrankaPanda",
        "manufacturer": "FrankaRobotics",
    },
    "jetbot": {
        "asset_path": "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
        "description": "NVIDIA Jetbot",
        "manufacturer": "NVIDIA",
    },
    "carter_v1": {
        "asset_path": "/Isaac/Robots/NVIDIA/Carter/carter_v1.usd",
        "description": "NVIDIA Carter",
        "manufacturer": "NVIDIA",
    },
    "novacarter": {
        "asset_path": "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
        "description": "NVIDIA NovaCarter",
        "manufacturer": "NVIDIA",
    },
    "g1": {"asset_path": "/Isaac/Robots/Unitree/G1/g1.usd", "description": "Unitree G1", "manufacturer": "Unitree"},
    "go1": {"asset_path": "/Isaac/Robots/Unitree/Go1/go1.usd", "description": "Unitree Go1", "manufacturer": "Unitree"},
    "spot": {
        "asset_path": "/Isaac/Robots/BostonDynamics/spot/spot.usd",
        "description": "BostonDynamics spot",
        "manufacturer": "BostonDynamics",
    },
}

# Cached discovered robots — populated on first call to list_robots.
_discovered_robots: Optional[Dict[str, Dict[str, str]]] = None


def _get_robot_library(adapter: IsaacAdapterBase) -> Dict[str, Dict[str, str]]:
    """Return the robot library, discovering from the asset server on first call.

    Falls back to FALLBACK_ROBOT_LIBRARY if discovery fails.
    """
    global _discovered_robots
    if _discovered_robots is not None:
        return _discovered_robots

    try:
        robots = adapter.discover_robots()
        if robots:
            _discovered_robots = robots
            print(f"Discovered {len(robots)} robots from asset server")
            return _discovered_robots
    except Exception as e:
        print(f"Robot discovery failed, using fallback: {e}")

    _discovered_robots = FALLBACK_ROBOT_LIBRARY
    return _discovered_robots


def _find_robot(adapter: IsaacAdapterBase, query: str) -> Optional[Dict[str, Any]]:
    """Find a robot by name. Tries exact key match, then partial match on key/description/manufacturer."""
    library = _get_robot_library(adapter)
    q = query.lower().strip()

    # Exact key match
    if q in library:
        return {"key": q, **library[q]}

    # Partial match on key, description, manufacturer
    matches = []
    for key, info in library.items():
        searchable = f"{key} {info.get('description', '')} {info.get('manufacturer', '')}".lower()
        if q in searchable:
            matches.append({"key": key, **info})

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Return closest match (shortest key that contains the query)
        matches.sort(key=lambda m: len(m["key"]))
        return matches[0]

    return None


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["robots.create"] = lambda **p: create(adapter, **p)
    registry["robots.list"] = lambda **p: list_robots(adapter, **p)
    registry["robots.refresh"] = lambda **p: refresh_robots(adapter, **p)
    registry["robots.get_info"] = lambda **p: get_info(adapter, **p)
    registry["robots.set_joints"] = lambda **p: set_joints(adapter, **p)
    registry["robots.get_joints"] = lambda **p: get_joints(adapter, **p)


def engine_drive_warning(adapter: IsaacAdapterBase) -> Optional[str]:
    """Warn when the active engine will not track commanded joint targets.

    Newton's drives do not converge (issue #21): the MuJoCo actuator gain cannot
    be integrated at the configured timestep, and the GPU path runs `mjw_model`
    rather than the `mj_model` copy reachable from here, so it is not fixable in
    this repository. Measured on 6.0.1: the FR3 sails past a -0.4 target to
    -0.736 by 1200 steps and keeps going, and `fr3_joint6` comes to rest outside
    its own lower limit, while PhysX settles at -0.398 within 150 and holds.

    An unfixable behaviour still has to be signalled at the point of use, or the
    caller watches an articulation diverge with nothing to say that is expected
    — the same reason create_camera warns about a camera it cannot delete.
    """
    if adapter.engine != adapter.ENGINE_NEWTON:
        return None
    return (
        "Joint drives do not converge on the Newton engine — commanded targets are overshot and "
        "the joint keeps going, and joint limits are not enforced either. Scene setup, stepping "
        "and inspection are fine here; run motion work on the PhysX engine (isaac-sim.sh) instead."
    )


def create(
    adapter: IsaacAdapterBase,
    robot_type: str = "franka",
    position: Optional[Sequence[float]] = None,
    name: Optional[str] = None,
    prim_path: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        match = _find_robot(adapter, robot_type)
        if not match:
            library = _get_robot_library(adapter)
            available = list(library.keys())[:20]
            return {
                "status": "error",
                "message": f"Robot '{robot_type}' not found. Try robots.list to see available robots. Some options: {available}",
            }

        assets_root = adapter.get_assets_root_path()
        asset_path = assets_root + match["asset_path"]
        if prim_path is None:
            prim_name = name or match["key"].capitalize()
            prim_path = f"/{prim_name}"
        adapter.add_reference_to_stage(asset_path, prim_path)
        if position:
            # set_prim_transform works on both V5 (omni.isaac.core XFormPrim)
            # and V6 (experimental Articulation) — the experimental XformPrim
            # only exposes the batched set_world_poses(), not the singular form.
            adapter.set_prim_transform(prim_path, position=position)
        result = {
            "status": "success",
            "message": f"Created {match['description']} robot",
            "prim_path": prim_path,
            "robot_key": match["key"],
        }
        joint_read_problem = None
        try:
            info = adapter.get_robot_joint_info(prim_path)
            result["joint_names"] = info.get("joint_names", [])
            result["num_dof"] = info.get("num_dof", 0)
            # V6 does not raise for an asset that failed to resolve — it falls
            # back to a USD walk and answers 0 DOF — so a robot that is not
            # there reads as a successful create, with the tool still promising
            # joint_names and num_dof.
            if not result["num_dof"]:
                joint_read_problem = (
                    f"{prim_path} was created but reports 0 joints. The robot asset most likely did "
                    "not resolve — check get_isaac_logs, confirm the asset server is reachable, and "
                    "verify with get_robot_info before commanding joints."
                )
        except Exception as e:
            # Previously printed to Kit's log and swallowed, so the response
            # promised joint_names and num_dof and carried neither.
            print(f"create_robot: get_robot_joint_info failed for {prim_path}: {e}")
            traceback.print_exc()
            joint_read_problem = (
                f"Could not read joints from {prim_path}: {e}. The prim exists but its articulation "
                "could not be inspected, so joint_names and num_dof are missing from this response."
            )
        # Check for broken drive configs (zero stiffness + zero damping)
        try:
            joint_config = adapter.get_joint_config(prim_path)
            warnings = list(joint_config.get("warnings", []))
            if joint_read_problem:
                warnings.append(joint_read_problem)
            engine_warning = engine_drive_warning(adapter)
            if engine_warning:
                warnings.append(engine_warning)
            if warnings:
                result["warnings"] = warnings
        except Exception as e:
            print(f"create_robot: get_joint_config failed for {prim_path}: {e}")
            traceback.print_exc()
            if joint_read_problem:
                result.setdefault("warnings", []).append(joint_read_problem)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_robots(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    library = _get_robot_library(adapter)
    return {"status": "success", "robot_count": len(library), "robots": library}


def refresh_robots(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    """Force re-scan the asset server for available robots."""
    global _discovered_robots
    _discovered_robots = None
    library = _get_robot_library(adapter)
    return {
        "status": "success",
        "message": f"Refreshed robot library, found {len(library)} robots",
        "robot_count": len(library),
    }


def get_info(adapter: IsaacAdapterBase, prim_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not prim_path:
            return {"status": "error", "message": "prim_path is required"}
        if prim_missing(adapter, prim_path):
            return {"status": "error", "message": f"Prim not found: {prim_path}"}
        info = adapter.get_robot_joint_info(prim_path)
        return {"status": "success", **info}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def set_joints(
    adapter: IsaacAdapterBase,
    prim_path: Optional[str] = None,
    joint_positions: Optional[Sequence[float]] = None,
    joint_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    try:
        if not prim_path or joint_positions is None:
            return {"status": "error", "message": "prim_path and joint_positions are required"}
        if prim_missing(adapter, prim_path):
            return {"status": "error", "message": f"Prim not found: {prim_path}"}

        # Neither the adapters nor the physics API complain about a
        # wrong-length array or an out-of-range index — the call reported
        # success and the robot did not move, or moved the wrong joint.
        dof = len(adapter.get_joint_positions(prim_path) or [])
        if joint_indices is not None:
            if len(joint_positions) != len(joint_indices):
                return {
                    "status": "error",
                    "message": (
                        f"joint_positions has {len(joint_positions)} value(s) but joint_indices has "
                        f"{len(joint_indices)}; they must match."
                    ),
                }
            if dof and any((i < 0 or i >= dof) for i in joint_indices):
                return {
                    "status": "error",
                    "message": f"joint_indices out of range for {prim_path}: valid indices are 0..{dof - 1}.",
                }
        elif dof and len(joint_positions) != dof:
            return {
                "status": "error",
                "message": (
                    f"{prim_path} has {dof} DOF but {len(joint_positions)} position(s) were given. "
                    "Pass one value per joint, or use joint_indices to address a subset."
                ),
            }
        adapter.set_joint_positions(prim_path, joint_positions, joint_indices)
        # Mirror what get_joints does for reads. A drive-target write is not a
        # failure, but it is not a command the solver has seen either, and a
        # flat success for both is what let "the robot did not move" look like
        # a physics problem rather than a write that never landed.
        source = adapter.joint_command_source
        result = {"status": "success", "message": f"Set joint positions on {prim_path}", "command_source": source}
        if source != adapter.JOINT_COMMAND_ARTICULATION:
            result["warning"] = (
                "The live articulation did not take this command — the values were written to USD "
                "drive targets instead, which move nothing until physics is initialized again. "
                "Step the simulation and read the joints back to confirm, and check get_isaac_logs."
            )
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_joints(adapter: IsaacAdapterBase, prim_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not prim_path:
            return {"status": "error", "message": "prim_path is required"}
        # A missing prim answers with an empty list rather than raising, which
        # used to be reported as a successful read of a robot with no joints.
        if prim_missing(adapter, prim_path):
            return {"status": "error", "message": f"Prim not found: {prim_path}"}
        positions = adapter.get_joint_positions(prim_path)
        source = adapter.joint_position_source
        result = {"status": "success", "joint_positions": positions, "position_source": source}
        if source != adapter.JOINT_SOURCE_PHYSICS:
            result["warning"] = (
                "These are authored drive targets, not simulated positions — the physics view "
                "could not serve this read, so the values echo the last set_joint_positions call. "
                "A robot that looks perfectly converged here may not have moved at all. Check "
                "get_isaac_logs, and confirm the timeline has been stepped."
            )
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
