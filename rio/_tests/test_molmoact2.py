# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tier 1: MolmoAct2 observation handling, plus a GPU check against the model card.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
transformers = pytest.importorskip("transformers")

from rio.policies.molmoact2 import MolmoAct2  # noqa: E402

REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
CAMERA_KEYS = ["top", "left", "right"]
STATE_DIM = 14
ACTION_DIM = 14
HORIZON = 30

# The model card's sample: episode 0, frame 0 of ai2-cortex/31122025-tablebuss-04.
SAMPLE_TASK = "Place cups and plate in dishwasher rack, dispose of food waste, and organize remaining items."
# fmt: off
SAMPLE_STATE = np.array([
    -0.06656748056411743, 0.014686808921396732, 0.016594186425209045, -0.08602273464202881,
    -0.014686808921396732, 0.13904783129692078, 0.9922363758087158, 0.19512474536895752,
    0.010872052982449532, 0.010872052982449532, -0.06771191209554672, -0.07305257022380829,
    -0.08945601433515549, 0.9888537526130676,
], dtype=np.float32)
# fmt: on


@pytest.fixture
def policy():
    """Wrapper with the checkpoint's metadata filled in, but no weights loaded"""
    policy = MolmoAct2(camera_keys=CAMERA_KEYS, norm_tag=NORM_TAG)
    policy.state_dim = STATE_DIM
    policy.set_instruction("pick up the plushie")
    return policy


@pytest.fixture
def make_dummy_observation():
    """Create dummy observation matching the checkpoint schema"""

    def _make(**overrides):
        obs = {key: np.zeros((16, 24, 3), dtype=np.uint8) for key in CAMERA_KEYS}
        obs["proprio"] = np.zeros(STATE_DIM, dtype=np.float32)
        obs.update(overrides)
        return obs

    return _make


@pytest.mark.unit
def test_discrete_mode_requires_a_tokenizer():
    # The action tokenizer is a separate repo, so discrete mode cannot be satisfied by the
    # checkpoint alone
    with pytest.raises(ValueError, match="action_tokenizer_path"):
        MolmoAct2(inference_action_mode="discrete")


@pytest.mark.unit
def test_invalid_inference_mode():
    with pytest.raises(ValueError, match="continuous"):
        MolmoAct2(inference_action_mode="flow")


@pytest.mark.unit
def test_convert_image_uint8(policy):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    np.testing.assert_array_equal(policy._convert_image(frame), frame)


@pytest.mark.unit
def test_convert_image_float_in_byte_range(policy):
    # The policy node declares camera frames float32, so a uint8 camera arrives already
    # widened and must not be rescaled again
    frame = np.array([[[0.0, 127.0, 255.0], [10.0, 20.0, 30.0]]], dtype=np.float32)
    np.testing.assert_array_equal(policy._convert_image(frame), frame.astype(np.uint8))


@pytest.mark.unit
def test_convert_image_normalized_float(policy):
    frame = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    assert policy._convert_image(frame).tolist() == [[[0, 127, 255]]]


@pytest.mark.unit
def test_convert_image_pil(policy):
    assert policy._convert_image(Image.new("RGB", (4, 4), (1, 2, 3))).shape == (4, 4, 3)


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(8, 12), (8, 12, 1), (8, 12, 4)])
def test_convert_image_rejects_non_rgb(policy, shape):
    with pytest.raises(ValueError, match="RGB frame"):
        policy._convert_image(np.zeros(shape, dtype=np.uint8))


@pytest.mark.unit
def test_views_follow_camera_keys(policy):
    # The checkpoint reads its views positionally, so a station whose cameras are declared
    # in another order must still work
    marks = {"top": 10, "left": 20, "right": 30}
    obs = {key: np.full((4, 4, 3), value, dtype=np.uint8) for key, value in reversed(list(marks.items()))}
    obs["proprio"] = np.zeros(STATE_DIM, dtype=np.float32)

    images, _ = policy._process_observation(obs)
    assert [int(image[0, 0, 0]) for image in images] == [marks[key] for key in CAMERA_KEYS]


@pytest.mark.unit
def test_missing_observation_keys(policy, make_dummy_observation):
    obs = make_dummy_observation()
    del obs["left"]
    del obs["proprio"]
    with pytest.raises(KeyError, match="left"):
        policy._process_observation(obs)


@pytest.mark.unit
def test_state_of_the_wrong_width(policy, make_dummy_observation):
    with pytest.raises(ValueError, match=f"state of {STATE_DIM}"):
        policy._process_observation(make_dummy_observation(proprio=np.zeros(7, dtype=np.float32)))


@pytest.mark.unit
def test_obs_transforms_run_first():
    policy = MolmoAct2(
        camera_keys=CAMERA_KEYS,
        obs_transforms=lambda obs, prompt=None: {
            **{key: obs[f"camera_{i + 1}"] for i, key in enumerate(CAMERA_KEYS)},
            "proprio": obs["joints"],
        },
    )
    policy.state_dim = STATE_DIM
    policy.set_instruction("pick up the plushie")

    obs = {f"camera_{i + 1}": np.full((4, 4, 3), i, dtype=np.uint8) for i in range(3)}
    obs["joints"] = np.zeros(STATE_DIM, dtype=np.float32)

    images, state = policy._process_observation(obs)
    assert [int(image[0, 0, 0]) for image in images] == [0, 1, 2]
    assert state.shape == (STATE_DIM,)


@pytest.mark.unit
def test_inference_before_construct_policy(policy, make_dummy_observation):
    with pytest.raises(AssertionError, match="construct_policy"):
        policy.inference(make_dummy_observation())


@pytest.mark.gpu
def test_inference_matches_the_model_card():
    """Load the real checkpoint and check the wrapper against the model card's own call.

    Both paths are seeded identically, so the flow solver draws the same noise and the two
    action chunks have to agree. That pins down the argument marshalling, the camera order
    and the tensor handed back to the policy node.
    """
    if tuple(int(part) for part in transformers.__version__.split(".")[:2]) < (5, 3):
        pytest.skip(f"MolmoAct2 checkpoints need transformers >= 5.3, found {transformers.__version__}")
    hf_hub_download = pytest.importorskip("huggingface_hub").hf_hub_download

    images = [
        Image.open(hf_hub_download(REPO_ID, f"assets/sample_{view}_rgb.png")).convert("RGB")
        for view in ("top", "left", "right")
    ]

    policy = MolmoAct2(
        policy_path=REPO_ID,
        norm_tag=NORM_TAG,
        camera_keys=CAMERA_KEYS,
        dtype="bfloat16",
        enable_cuda_graph=True,
        warm_start_iters=1,
        dummy_obs={
            **{key: np.zeros((480, 640, 3), dtype=np.uint8) for key in CAMERA_KEYS},
            "proprio": np.zeros(STATE_DIM, dtype=np.float32),
        },
    )
    policy.construct_policy()

    # construct_policy reads these off norm_stats.json rather than trusting the config
    assert (policy.state_dim, policy.action_dim, policy.chunk_size) == (STATE_DIM, ACTION_DIM, HORIZON)

    policy.set_instruction(SAMPLE_TASK)
    # float32 in 0-255, as the policy node's shared-memory schema delivers frames
    obs = {key: np.asarray(image, dtype=np.float32) for key, image in zip(CAMERA_KEYS, images, strict=True)}
    obs["proprio"] = SAMPLE_STATE

    torch.manual_seed(0)
    actions = policy.inference(obs)

    assert isinstance(actions, np.ndarray)
    assert actions.shape == (HORIZON, ACTION_DIM)
    assert np.isfinite(actions).all()

    # The model card's call, verbatim, on the same seed
    torch.manual_seed(0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = policy.model.predict_action(
            processor=policy.processor,
            images=images,
            task=SAMPLE_TASK,
            state=SAMPLE_STATE,
            norm_tag=NORM_TAG,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=10,
            normalize_language=True,
            enable_cuda_graph=True,
        ).actions
    np.testing.assert_allclose(actions, expected.squeeze(0).float().cpu().numpy(), rtol=0, atol=1e-3)

    # Absolute joint targets in robot scale: the first step sits near the state the model
    # was given, and the whole chunk inside the range the checkpoint normalizes over
    stats = policy.model._get_robot_stats().get_metadata(NORM_TAG)["action_stats"]
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    assert ((actions >= low - 0.1) & (actions <= high + 0.1)).all()
    assert np.abs(actions[0] - SAMPLE_STATE).max() < 0.5

    # Both conditioning paths have to actually reach the model. A wrapper that dropped the
    # instruction, or handed the same frame to all three slots, would still match the call
    # above -- it would just stop responding to the inputs the policy node feeds it
    policy.set_instruction("stop and do nothing")
    assert np.abs(policy.inference(obs) - actions).max() > 0.05

    policy.set_instruction(SAMPLE_TASK)
    blank = {**{key: np.zeros_like(obs[key]) for key in CAMERA_KEYS}, "proprio": SAMPLE_STATE}
    assert np.abs(policy.inference(blank) - actions).max() > 0.05
