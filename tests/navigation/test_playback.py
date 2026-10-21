from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from space_robotics_nav.playback import (
    NoValidTargetError,
    advance_goal,
    playback_seed,
    sample_next_target,
)


def terrain():
    return NS(
        target_distance=9,
        check_if_target_is_valid=lambda ids, xy, device: (ids[:0], 0),
        _heightmap_manager=NS(get_height_at=lambda xy: torch.zeros(len(xy))),
    )


def test_ten_distinct_targets_and_fixed_seed_replay():
    t = terrain()
    position = np.array([2.0, 3.0, 1.0])
    targets = []
    for index in range(10):
        target, heading = sample_next_target(
            t, position, np.random.default_rng(100 + index), targets
        )
        repeated, repeated_heading = sample_next_target(
            t, position, np.random.default_rng(100 + index), targets
        )
        np.testing.assert_equal(target, repeated)
        assert heading == repeated_heading and -np.pi <= heading <= np.pi
        assert np.linalg.norm(target[:2] - position[:2]) == pytest.approx(9)
        assert all(
            np.linalg.norm(target[:2] - previous[:2]) >= 1 for previous in targets
        )
        targets.append(target)
        position = target  # The next target is relative to the actual next origin.


def test_sampler_rejects_invalid_targets_and_limits_attempts():
    t = terrain()
    t.check_if_target_is_valid = lambda ids, xy, device: (ids, int(xy[0, 0] > 0))
    target, _ = sample_next_target(t, [0, 0, 0], np.random.default_rng(1))
    assert target[0] <= 0
    t.check_if_target_is_valid = lambda *args: (torch.tensor([0]), 1)
    with pytest.raises(NoValidTargetError):
        sample_next_target(t, [0, 0, 0], np.random.default_rng(1))


def test_random_run_seed_and_explicit_replay(monkeypatch):
    values = iter([1000, 2000])
    monkeypatch.setattr(
        "space_robotics_nav.playback.secrets.randbelow", lambda _: next(values)
    )
    assert playback_seed() == 1000
    assert playback_seed() == 2000
    assert playback_seed(30002) == 30002
    with pytest.raises(ValueError):
        playback_seed(-1)


@pytest.mark.parametrize(
    "collision,bounds,reset_between,expected",
    [
        (False, False, False, "continue_from_current_pose"),
        (True, False, False, "reset_after_failure"),
        (False, True, False, "reset_after_failure"),
        (False, False, True, "reset_requested"),
    ],
)
def test_success_timeout_continue_and_failure_recovery(
    collision, bounds, reset_between, expected
):
    env = NS(
        reset=Mock(return_value=("reset", {})),
        next_goal=Mock(return_value=("next", {})),
        target=np.array([9.0, 0, 0]),
        terrain=terrain(),
    )
    obs, mode = advance_goal(
        env,
        {"collision": collision, "out_of_bounds": bounds},
        seed=123,
        reset_between=reset_between,
    )
    assert mode == expected
    if expected == "continue_from_current_pose":
        assert obs == "next" and env.reset.call_count == 0
    else:
        assert obs == "reset" and env.next_goal.call_count == 0


def test_no_local_goal_recovers_to_native_spawn():
    env = NS(
        reset=Mock(return_value=("reset", {})),
        next_goal=Mock(side_effect=NoValidTargetError()),
        target=np.array([9.0, 0, 0]),
        terrain=terrain(),
    )
    obs, mode = advance_goal(
        env, {"collision": False, "out_of_bounds": False}, seed=123
    )
    assert obs == "reset" and mode == "reset_no_valid_local_goal"


def test_tipped_rover_recovers_before_next_goal():
    env = NS(
        reset=Mock(return_value=("reset", {})),
        next_goal=Mock(),
        state={"rpy": [0, 1.15, 0]},
        target=np.array([9.0, 0, 0]),
        terrain=terrain(),
    )
    obs, mode = advance_goal(
        env, {"collision": False, "out_of_bounds": False}, seed=123
    )
    assert obs == "reset" and mode == "reset_tipped_rover"
    env.next_goal.assert_not_called()
