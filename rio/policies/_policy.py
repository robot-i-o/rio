# SPDX-FileCopyrightText: 2026 RIO Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol


class Policy(Protocol):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Policy __init__ not implemented yet")

    def construct_policy(self):
        raise NotImplementedError("Policy construct_policy not implemented yet")

    @staticmethod
    def create_obs(env):
        raise NotImplementedError("Policy create_obs not implemented yet")

    def set_instruction(self, instruction):
        raise NotImplementedError("Policy set_instruction not implemented yet")

    def _process_observation(self, observation):
        raise NotImplementedError("Policy _process_observation not implemented yet")

    def inference(self, observation, current_plan=None):
        raise NotImplementedError("Policy inference not implemented yet")

    def get_action(self, observation, current_plan=None):
        return self.inference(observation, current_plan=current_plan)
