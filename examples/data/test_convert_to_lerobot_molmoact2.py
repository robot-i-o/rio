# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""
Resampling and feature-spec checks for the MolmoAct2 LeRobot converter.
"""

import numpy as np
import pytest
from examples.data.convert_to_lerobot_molmoact2 import (
    MOTOR_NAMES,
    build_features,
    find_trajectories,
    resample_indices,
)

STATE_DIM = 14

pytestmark = pytest.mark.unit


def test_resample_50hz_to_30hz():
    # The real case: 27.2 s of 50 Hz recording onto the 30 fps grid the checkpoint expects
    t = np.arange(1359, dtype=np.float64) / 50.0
    keep = resample_indices(t, 30.0)

    assert len(keep) == 815
    assert np.all(np.diff(keep) > 0), "indices must be strictly increasing"
    # Every kept frame is within half a source period of its grid point
    grid = np.arange(len(keep), dtype=np.float64) / 30.0
    assert np.abs(t[keep] - grid).max() < 1.0 / 50.0


def test_resample_never_invents_frames():
    # Asking above the source rate is what makes robodm's own resampler allocate hundreds
    # of GiB here; this one deduplicates instead
    t = np.arange(100, dtype=np.float64) / 30.0
    assert len(resample_indices(t, 50.0)) <= len(t)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_resample_degenerate_episodes(n):
    assert len(resample_indices(np.arange(n, dtype=np.float64) / 50.0, 30.0)) <= n


def test_trajectory_discovery_merges_colliding_names(tmp_path):
    # Recording sessions reuse filenames, so episode index has to come from position
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        for name in ("traj_0002.vla", "traj_0003.vla"):
            (tmp_path / sub / name).touch()

    found = find_trajectories([tmp_path / "a", tmp_path / "b"])
    assert [f"{p.parent.name}/{p.name}" for p in found] == [
        "a/traj_0002.vla",
        "a/traj_0003.vla",
        "b/traj_0002.vla",
        "b/traj_0003.vla",
    ]


def test_feature_spec_matches_the_norm_tag():
    features = build_features(480, 640)
    assert features["action"]["names"] == features["observation.state"]["names"] == MOTOR_NAMES
    assert features["action"]["shape"] == (STATE_DIM,)
    assert all(features[k]["dtype"] == "video" for k in features if k.startswith("observation.images."))
