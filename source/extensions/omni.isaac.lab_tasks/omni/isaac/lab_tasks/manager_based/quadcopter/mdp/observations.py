from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import Camera, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

"""
Root state.
"""

def desired_pos_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    desired_pos_w = env.command_manager.get_command("desired_drone_pose")
    desired_pos_b, _ = subtract_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], desired_pos_w
    )
    return desired_pos_b