# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


class QuadCopterAction(ActionTerm):
    r"""Quadcopter action term.

    This action term applies the desired thrust and moment to the robot.
    """

    cfg: QuadCopterActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: QuadCopterActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.cfg = cfg
        self._robot: Articulation = env.scene[cfg.asset_name]
        self._actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.actions = torch.zeros_like(self._actions)

        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)        

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self.cfg.num_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._actions[:] = actions
        self.actions = self._actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self.actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self.actions[:, 1:]


    def apply_actions(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)


@configclass
class QuadCopterActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`QuadCopterAction` for more details.
    """

    class_type: type[ActionTerm] = QuadCopterAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """ Number of actions  """
    num_actions: int = 4
    """ Thrust to weight ratio  (copied from /home/anaveen/Documents/research_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/quadcopter/quadcopter_env.py)"""
    thrust_to_weight: float = 1.9
    """ Moment scaling ratio (copied from /home/anaveen/Documents/research_ws/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/quadcopter/quadcopter_env.py)"""
    moment_scale: float = 0.01
    """ Gravity Vector (copied from /home/anaveen/Documents/research_ws/IsaacLab/source/extensions/omni.isaac.lab/omni/isaac/lab/sim/simulation_context.py)"""
    gravity: tuple = (0.0, 0.0, -9.81)
