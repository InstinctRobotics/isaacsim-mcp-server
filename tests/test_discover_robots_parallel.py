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

"""discover_robots walks the asset server concurrently."""

import ast
import os

import pytest


def _adapter_src(name):
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "isaac.sim.mcp_extension",
        "isaac_sim_mcp_extension",
        "adapters",
        name,
    )


V5 = _adapter_src("v5.py")
V6 = _adapter_src("v6.py")

# Both adapters walk the same asset server and must behave identically. V6
# originally shipped a plain nested-loop walk without either guard: on 6.0.1 it
# took 45.12s (blocking kit's main loop) and listed ".thumbs" as a robot.
ADAPTERS = [("v5", V5, "IsaacAdapterV5"), ("v6", V6, "IsaacAdapterV6")]
ADAPTER_IDS = [a[0] for a in ADAPTERS]


def _discover_src(path=V5):
    with open(path) as f:
        text = f.read()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "discover_robots":
            return ast.get_source_segment(text, node)
    raise AssertionError("discover_robots not found")


@pytest.mark.parametrize("path", [V5, V6], ids=["v5", "v6"])
def test_walk_is_concurrent(path):
    """Sequential listings cost ~28s (5.1) / ~45s (6.0.1) on a cold cache and
    block kit's main loop for the whole time. The calls are latency bound, so
    they must overlap."""
    src = _discover_src(path)
    assert "ThreadPoolExecutor" in src


@pytest.mark.parametrize("path", [V5, V6], ids=["v5", "v6"])
def test_walk_falls_back_to_sequential(path):
    """If threads are unavailable the walk must still work, just slower."""
    src = _discover_src(path)
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert code.count("_list_dir(p) for p in paths") >= 2, "needs a sequential fallback path"


@pytest.mark.parametrize("path", [V5, V6], ids=["v5", "v6"])
def test_ordering_preserved_for_key_preference(path):
    """Results are zipped back against the input order, because the 'shorter
    filename wins' rule depends on deterministic iteration order."""
    src = _discover_src(path)
    assert "zip(pairs, model_files)" in src
    assert "zip(mfr_names, mfr_models)" in src


# ── Functional tests against a fake asset server ────────────────────────────


class _Entry:
    def __init__(self, name):
        self.relative_path = name


def _fake_client(tree):
    """Minimal omni.client stand-in over a {path: [names]} dict."""
    import types

    class _Result:
        OK = "OK"

    mod = types.SimpleNamespace(Result=_Result)

    def _list(path):
        key = path.rstrip("/") + "/"
        if key not in tree:
            return ("MISSING", [])
        return (_Result.OK, [_Entry(n) for n in tree[key]])

    mod.list = _list
    return mod


def _run_discovery(monkeypatch, tree, cls_name="IsaacAdapterV5"):
    import importlib
    import sys
    import types

    fake = _fake_client(tree)
    monkeypatch.setitem(sys.modules, "omni.client", fake)
    # `import omni.client` reads the attribute off the parent package, so the
    # stub has to be attached there as well as in sys.modules.
    monkeypatch.setattr(sys.modules["omni"], "client", fake, raising=False)

    native = types.SimpleNamespace(get_assets_root_path=lambda: "ROOT")
    monkeypatch.setitem(sys.modules, "isaacsim.storage.native", native)
    if "isaacsim" in sys.modules:
        storage = getattr(sys.modules["isaacsim"], "storage", None)
        if storage is not None:
            monkeypatch.setattr(storage, "native", native, raising=False)

    module = "v5" if cls_name.endswith("V5") else "v6"
    mod = importlib.import_module(f"isaac_sim_mcp_extension.adapters.{module}")
    return getattr(mod, cls_name)().discover_robots()


@pytest.mark.parametrize("cls_name", ["IsaacAdapterV5", "IsaacAdapterV6"], ids=ADAPTER_IDS)
def test_hidden_thumbnail_directories_are_not_robots(cls_name):
    """Every manufacturer ships a .thumbs folder of <model>.thumb.usd previews.

    Unfiltered they registered as a robot literally named ".thumbs" whose
    asset_path pointed at a thumbnail — observed on Isaac Sim 5.1 and again on
    6.0.1, where discovery returned 208 entries of which one was ".thumbs".
    """
    tree = {
        "ROOT/Isaac/Robots/": ["ANYbotics/", "Idealworks/"],
        "ROOT/Isaac/Robots/ANYbotics/": [".thumbs/", ".cache/", "anymal_c/"],
        "ROOT/Isaac/Robots/ANYbotics/.thumbs/": ["anymal_c.thumb.usd"],
        # A hidden directory holding a plain .usd — covered by the hidden-dir
        # rule rather than the .thumb.usd rule, so both guards are exercised.
        "ROOT/Isaac/Robots/ANYbotics/.cache/": ["scratch.usd"],
        "ROOT/Isaac/Robots/ANYbotics/anymal_c/": ["anymal_c.usd"],
        "ROOT/Isaac/Robots/Idealworks/": [".thumbs/", "iwhub/"],
        "ROOT/Isaac/Robots/Idealworks/.thumbs/": ["iw_hub.thumb.usd"],
        "ROOT/Isaac/Robots/Idealworks/iwhub/": ["iw_hub.usd"],
    }
    mp = pytest.MonkeyPatch()
    try:
        robots = _run_discovery(mp, tree, cls_name)
    finally:
        mp.undo()

    assert ".thumbs" not in robots
    assert not [k for k in robots if k.startswith(".")]
    assert not [r for r in robots.values() if r["asset_path"].endswith(".thumb.usd")]
    assert set(robots) == {"anymal_c", "iwhub"}


@pytest.mark.parametrize("cls_name", ["IsaacAdapterV5", "IsaacAdapterV6"], ids=ADAPTER_IDS)
def test_colliding_model_names_keep_a_consistent_record(cls_name):
    """Two vendors can ship the same directory name.

    The "shorter filename wins" rule used to overwrite asset_path only, leaving
    a record that described one manufacturer while pointing at another's asset.
    """
    tree = {
        "ROOT/Isaac/Robots/": ["AlphaCorp/", "BetaCorp/"],
        "ROOT/Isaac/Robots/AlphaCorp/": ["shared/"],
        "ROOT/Isaac/Robots/AlphaCorp/shared/": ["a_very_long_name.usd"],
        "ROOT/Isaac/Robots/BetaCorp/": ["shared/"],
        "ROOT/Isaac/Robots/BetaCorp/shared/": ["b.usd"],
    }
    mp = pytest.MonkeyPatch()
    try:
        robots = _run_discovery(mp, tree, cls_name)
    finally:
        mp.undo()

    rec = robots["shared"]
    assert f"/Isaac/Robots/{rec['manufacturer']}/" in rec["asset_path"], rec
    assert rec["manufacturer"] == "BetaCorp"  # shorter filename won
    assert rec["description"] == "BetaCorp shared"


def test_v6_prefers_robots_multiphysics_and_one_extra_directory():
    """6.1 moved shipped references to Robots_Multiphysics.

    Several 6.2 catalog entries are one directory deeper than the legacy
    layout and use .usda (FrankaPanda/franka/franka.usda). A file in that
    directory whose name does not match the directory is not a robot.
    When the multiphysics tree lists, the legacy tree is not scanned.
    """
    tree = {
        "ROOT/Isaac/Robots_Multiphysics/": ["FrankaRobotics/", "NVIDIA/", "Unitree/"],
        "ROOT/Isaac/Robots_Multiphysics/FrankaRobotics/": ["FrankaPanda/", ".thumbs/"],
        "ROOT/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/": ["franka/", "notes.txt"],
        "ROOT/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/": [
            "franka.usda",
            "config.usd",
            "franka.thumb.usd",
        ],
        "ROOT/Isaac/Robots_Multiphysics/NVIDIA/": ["Jetbot/", "Carter/"],
        "ROOT/Isaac/Robots_Multiphysics/NVIDIA/Jetbot/": ["jetbot.usda"],
        "ROOT/Isaac/Robots_Multiphysics/NVIDIA/Carter/": ["carter_v1/"],
        "ROOT/Isaac/Robots_Multiphysics/NVIDIA/Carter/carter_v1/": ["carter_v1.usda"],
        "ROOT/Isaac/Robots_Multiphysics/Unitree/": ["G1/"],
        "ROOT/Isaac/Robots_Multiphysics/Unitree/G1/": ["g1.usda"],
        # Present, and must be ignored once the multiphysics listing succeeds.
        "ROOT/Isaac/Robots/": ["LegacyCo/"],
        "ROOT/Isaac/Robots/LegacyCo/": ["oldbot/"],
        "ROOT/Isaac/Robots/LegacyCo/oldbot/": ["oldbot.usd"],
    }
    mp = pytest.MonkeyPatch()
    try:
        robots = _run_discovery(mp, tree, "IsaacAdapterV6")
    finally:
        mp.undo()

    assert set(robots) == {"frankapanda", "jetbot", "carter", "g1"}
    assert robots["frankapanda"]["asset_path"] == (
        "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
    )
    assert robots["carter"]["asset_path"].endswith("/NVIDIA/Carter/carter_v1/carter_v1.usda")
    assert robots["jetbot"]["asset_path"].endswith("/NVIDIA/Jetbot/jetbot.usda")
    assert robots["g1"]["asset_path"].endswith("/Unitree/G1/g1.usda")
    assert "oldbot" not in robots


def test_fallback_library_follows_the_adapter():
    """Discovery failure still has to name a file the runtime actually ships."""
    from isaac_sim_mcp_extension.handlers.robots import _fallback_library

    class IsaacAdapterV5:
        pass

    class IsaacAdapterV6:
        pass

    legacy = _fallback_library(IsaacAdapterV5())
    current = _fallback_library(IsaacAdapterV6())
    assert legacy["frankapanda"]["asset_path"] == "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    assert current["frankapanda"]["asset_path"] == (
        "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
    )
    assert current["carter_v1"]["asset_path"].endswith("/carter_v1/carter_v1.usda")
    assert current["g1"]["asset_path"].endswith("/Unitree/G1/g1.usda")
    assert current["go1"]["asset_path"].endswith("/Unitree/Go1/go1.usda")
    assert current["spot"]["asset_path"].endswith("/BostonDynamics/spot/spot.usda")
    assert current["jetbot"]["asset_path"].endswith("/NVIDIA/Jetbot/jetbot.usda")
    # No multiphysics entry in the 6.2.0 wheeled catalog.
    assert current["novacarter"]["asset_path"] == "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
