import argparse

from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG


cfg = UNITREE_GO2_CFG.copy()
robot = Articulation(cfg=cfg)

a = 1