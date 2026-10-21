"""Independent checks against the pinned, unedited algorithm source."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import gymnasium as gym
import numpy as np
import pytest
import torch
import trimesh

from space_robotics_nav.config import ROOT
from space_robotics_nav.env import REWARDS
from space_robotics_nav.learning import (
    create_agent,
    load_config,
    policy_action,
    restore_state,
    save_state,
)
from space_robotics_nav.raycast import Raycaster
from space_robotics_nav.vendor import observations, rewards, terminations

SOURCE = ROOT / "third_party/rlroverlab"


def original_functions(name):
    path = SOURCE / f"rover_envs/envs/navigation/mdp/{name}.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ] + [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace


def context():
    g = torch.Generator().manual_seed(219)
    command = torch.randn(32, 4, generator=g) * 4
    command[:4, :2] = torch.tensor([[0, 0], [0.179, 0], [0.18, 0], [12.01, 0]])
    command[:4, 3] = torch.tensor([0, 0.099, 0, 0.1])
    force = torch.randn(32, 4, 2, 3, generator=g)
    force[0] = 0
    return NS(
        num_envs=32,
        device="cpu",
        max_episode_length=750,
        episode_length_buf=torch.arange(32),
        action_manager=NS(
            action=torch.randn(32, 2, generator=g),
            prev_action=torch.randn(32, 2, generator=g),
        ),
        command_manager=NS(get_command=lambda _: command),
        scene=NS(
            terrain=NS(target_distance=9),
            sensors={"contact_sensor": NS(data=NS(force_matrix_w=NS(torch=force)))},
        ),
    )


def test_snapshot_and_model_architecture():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        assert hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() == digest
    upstream = ast.parse(
        (SOURCE / "rover_envs/envs/navigation/learning/skrl/models.py").read_text()
    )
    ported = ast.parse((ROOT / "space_robotics_nav/vendor/models.py").read_text())
    classes = lambda tree: [
        ast.dump(n) for n in tree.body if isinstance(n, ast.ClassDef)
    ]
    assert classes(upstream) == classes(ported)
    assert load_config(ROOT / "configs/navigation/ppo.yaml") == __import__(
        "yaml"
    ).safe_load(
        (
            SOURCE / "rover_envs/envs/navigation/learning/skrl/configs/rover_ppo.yaml"
        ).read_text()
    )
    # Only algorithm source is vendored: no robot models, USD, terrain meshes or textures.
    assert not any(
        p.suffix.lower() in {".usd", ".usdc", ".obj", ".stl", ".glb", ".png", ".jpg"}
        for p in SOURCE.rglob("*")
        if p.is_file()
    )


@pytest.mark.parametrize(
    "name,fn,params",
    [("rewards", fn.__name__, params) for fn, _, params in REWARDS.values()]
    + [
        (
            "terminations",
            "is_success",
            {"command_name": "target_pose", "threshold": 0.18},
        ),
        ("terminations", "far_from_target", {"command_name": "target_pose"}),
        (
            "terminations",
            "collision_with_obstacles",
            {"sensor_cfg": NS(name="contact_sensor"), "threshold": 1.0},
        ),
        (
            "observations",
            "distance_to_target_euclidean",
            {"command_name": "target_pose"},
        ),
        (
            "observations",
            "angle_to_target_observation",
            {"command_name": "target_pose"},
        ),
        ("observations", "angle_diff", {"command_name": "target_pose"}),
    ],
)
def test_mdp_numerical_parity(name, fn, params):
    modules = {
        "rewards": rewards,
        "terminations": terminations,
        "observations": observations,
    }
    env = context()
    actual = getattr(modules[name], fn)(env, **params)
    expected = original_functions(name)[fn](env, **params)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_success_and_reward_scale():
    env = context()
    assert terminations.is_success(env, "target_pose", 0.18)[:4].tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert rewards.distance_to_target_reward(env, "target_pose")[
        0
    ] * 5 * 0.2 == pytest.approx(1 / 750)
    env.scene.sensors["contact_sensor"].data.force_matrix_w = None
    with pytest.raises(RuntimeError, match="unavailable"):
        rewards.collision_penalty(env, NS(name="contact_sensor"), 1)


def test_height_scan_data_format():
    e = NS(
        scene=NS(
            sensors={
                "height_scanner": NS(
                    data=NS(
                        pos_w=NS(torch=torch.tensor([[0.0, 0.0, 10.5]])),
                        ray_hits_w=NS(torch=torch.zeros(1, 10201, 3)),
                    )
                )
            }
        )
    )
    scan = observations.height_scan_rover(e, NS(name="height_scanner"))
    torch.testing.assert_close(scan, torch.full((1, 10201), 10.5 - 0.26878))


def test_mesh_ray_coordinates_and_concurrency():
    from concurrent.futures import ThreadPoolExecutor

    m = trimesh.Trimesh(
        vertices=[[-5, -5, 2], [5, -5, 2], [5, 5, 2], [-5, 5, 2]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    rays = Raycaster(m)
    xy = np.array([[0, 0], [1, 1], [-1, 2]])
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: rays.heights(xy, 10), range(12)))
    for result in results:
        np.testing.assert_allclose(result, 2, atol=1e-6)
    np.testing.assert_allclose(
        rays.distances(np.array([[0, 0, 10]]), [0, 0, -1]), 8, atol=1e-6
    )


def test_checkpoint_preserves_policy_optimizer_and_rng(tmp_path):
    torch.set_num_threads(2)
    env = NS(
        num_envs=2,
        device=torch.device("cpu"),
        observation_space=gym.spaces.Box(-np.inf, np.inf, (10206,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
    )
    experiment = load_config()
    experiment["agent"]["mini_batches"] = 2
    agent = create_agent(env, experiment, directory=tmp_path)
    agent.init()
    # Populate real Adam state through a nonzero policy gradient.
    obs = torch.randn(2, 10206)
    mean, _ = agent.policy.compute({"observations": obs}, role="policy")
    agent.optimizer.zero_grad()
    mean.square().mean().backward()
    agent.optimizer.step()
    agent.policy.eval()
    before = policy_action(agent, obs).clone()
    path = tmp_path / "model.pt"
    save_state(agent, path, {"marker": "native"})
    expected_random = torch.rand(4)
    for param in agent.policy.parameters():
        param.data.zero_()
    assert restore_state(agent, path)["marker"] == "native"
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
    torch.testing.assert_close(policy_action(agent, obs), before, rtol=0, atol=0)
    assert agent.optimizer.state_dict()["state"]


def test_paired_evaluation_gate():
    from space_robotics_nav.evaluate import compare

    def report(values):
        return {
            "episodes": [
                {"seed": i, "return": v, "length": 10} for i, v in enumerate(values)
            ],
            "metrics": {"success": 0, "collision": 0},
        }

    base = report([1, 2, 3, 4])
    assert not compare(base, base)["return_improved"]
    assert compare(base, report([2, 3, 4, 5]))["step_return_improved"]
    assert not compare(base, report([0, 1, 2, 3]))["return_improved"]


def test_original_sampling_algorithms_are_preserved():
    source = ast.parse(
        (
            SOURCE / "rover_envs/envs/navigation/utils/terrains/terrain_utils.py"
        ).read_text()
    )
    port = ast.parse((ROOT / "space_robotics_nav/vendor/terrain.py").read_text())
    original = {n.name: n for n in source.body if isinstance(n, ast.ClassDef)}
    adapted = {n.name: n for n in port.body if isinstance(n, ast.ClassDef)}
    assert ast.dump(original["HeightmapManager"]) == ast.dump(
        adapted["HeightmapManager"]
    )
    source_methods = {
        n.name: ast.dump(n)
        for n in original["TerrainManager"].body
        if isinstance(n, ast.FunctionDef)
    }
    for node in adapted["TerrainOps"].body:
        if isinstance(node, ast.FunctionDef):
            assert ast.dump(node) == source_methods[node.name]
