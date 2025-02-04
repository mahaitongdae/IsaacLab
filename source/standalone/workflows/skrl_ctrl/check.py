# from omni.isaac.lab_tasks.direct.quadcopter.quadcopter_env import QuadcopterTrajectoryTrainingActiveBOTaskEnvCfg
from omni.isaac.lab.app import AppLauncher
import argparse
from utils.utils import load_isaaclab_env

parser = argparse.ArgumentParser(description="Run the eval script with customizable parameters.")
# Add arguments
parser.add_argument("--env_version", type=str, default="legtrain-active-bo")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse arguments
args, _ = parser.parse_known_args()

task_version = args.env_version
print(task_version)

task_name = f"Isaac-Quadcopter-{task_version}-Trajectory-Direct-v0"
env = load_isaaclab_env(task_name = task_name, num_envs=32, cli_args=['--headless'])

print(env.cfg)