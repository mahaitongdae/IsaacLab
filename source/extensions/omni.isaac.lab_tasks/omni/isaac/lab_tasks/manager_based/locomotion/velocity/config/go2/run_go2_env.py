# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.utils import configclass
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
from omni.isaac.lab.utils.math import euler_xyz_from_quat

# class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # override rewards
#         self.rewards.flat_orientation_l2.weight = -2.5
#         self.rewards.feet_air_time.weight = 0.25

#         # change terrain to flat
#         self.scene.terrain.terrain_type = "plane"
#         self.scene.terrain.terrain_generator = None
#         # no height scan
#         self.scene.height_scanner = None
#         self.observations.policy.height_scan = None
#         # no terrain curriculum
#         self.curriculum.terrain_levels = None

class myEnvConfig(UnitreeGo2FlatEnvCfg):
    
    def __post_init__(self):
        
        super().__post_init__()
        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel = None
        self.observations.policy.projected_gravity = None #ObsTerm(func=mdp.root_quat_w)
        self.observations.policy.velocity_commands = None
        self.observations.policy.joint_pos = None #ObsTerm(func=mdp.joint_pos)
        self.observations.policy.joint_vel = None
        # self.observations.policy.actions = None
        self.actions.joint_pos.scale = 0.25


def main():
    """Main function."""
    # create environment configuration
    env_cfg = myEnvConfig()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.zeros_like(env.action_manager.action)
            # joint_efforts[:, 10] = torch.ones((2,))
            # joint_efforts[:, 11] = torch.ones((2,))
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # print current orientation of pole
            # print(f"[Env 0 step {count % 300}]: Obs: ",*[ s / torch.pi * 180 for s in  euler_xyz_from_quat(obs["policy"]) ])
            print(f"[Env 0 step {count % 300}]: Obs: ",obs["policy"])
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
