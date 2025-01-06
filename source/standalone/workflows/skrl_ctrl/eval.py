import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from utils.utils import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer, StepTrainer
from skrl.utils import set_seed

from networks.actor import DiagGaussianActor, StochasticActor
from networks.critic import Critic, SACCritic
from networks.feature import Phi, Mu, Theta

from agents.ctrlsac_agent import CTRLSACAgent
from omni.isaac.lab.utils.dict import print_dict
import os
import gymnasium as gym
import numpy as np

import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the eval script with customizable parameters.")
# Add arguments
parser.add_argument("--experiment", type=str, default="OOD", choices=["legeval", "legood", "OOD", "legtrain"], help="Specify the task name (default: OOD).")
parser.add_argument("--agent_type", type=str, default="CTRLSAC", choices=["CTRLSAC", "SAC"], help="Specify the agent type (default: CTRLSAC).")
parser.add_argument("--ckpt", type=str, default="best_agent", help="Specify the checkpoint name (default: best_agent).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse arguments
args, _ = parser.parse_known_args()
task = args.experiment
agent_type = args.agent_type
ckpt = args.ckpt

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


cli_args = ["--video"]
# load and wrap the Isaac Gym environment
experiments = {
    "legeval": [50000, False],
    "legood": [500, True],
    "OOD": [500, True],
    "legtrain": [3000, True]
}



experiment_length = experiments[task][0]
record_video = experiments[task][1] 
output_dir = f"runs/experiments/{task}/{agent_type}-{ckpt}"

video_kwargs = {
    "video_folder": os.path.join(output_dir, "videos"),
    "step_trigger": lambda step: step % 10000== 0,
    "video_length": experiment_length,
    "disable_logger": True,
}
print("[INFO] Recording videos during training.")
print_dict(video_kwargs, nesting=4)
env = load_isaaclab_env(task_name=f"Isaac-Quadcopter-{task}-Trajectory-Direct-v0", num_envs=1, cli_args=cli_args)

if record_video: env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = wrap_env(env)


device = env.device


# instantiate a memory as experience replay
sacmemory = RandomMemory(memory_size=experiment_length, num_envs=env.num_envs, device=device)
ctrlmemory = RandomMemory(memory_size=experiment_length, num_envs=env.num_envs, device=device)

# define hidden dimension
actor_hidden_dim = 256
actor_hidden_depth = 3

# define feature dimension 
feature_dim = 512
feature_hidden_dim = 1024

# define task feature dimensions
cdim = 512

# state dimensions
task_state_dim = 60
drone_state_dim = 13
multitask=True

# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
sacmodels = {}
sacmodels["policy"] = StochasticActor(observation_space = env.observation_space,
                                     action_space = env.action_space, 
                                     hidden_dim = actor_hidden_dim, 
                                     hidden_depth = actor_hidden_depth,
                                     log_std_bounds = [-5., 2.], 
                                     device = device)

sacmodels["critic_1"] = SACCritic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            device = device)

sacmodels["critic_2"] = SACCritic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            device = device)

sacmodels["target_critic_1"] = SACCritic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   device = device)

sacmodels["target_critic_2"] = SACCritic(observation_space = env.observation_space,
                                action_space = env.action_space, 
                                feature_dim = feature_dim, 
                                device = device)

ctrlmodels = {}
ctrlmodels["policy"] = StochasticActor(observation_space = env.observation_space,
                                     action_space = env.action_space, 
                                     hidden_dim = actor_hidden_dim, 
                                     hidden_depth = actor_hidden_depth,
                                     log_std_bounds = [-5., 2.], 
                                     device = device)

ctrlmodels["critic_1"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            task_state_dim = task_state_dim,
                            cdim = cdim,
                            multitask = multitask,
                            device = device)

ctrlmodels["critic_2"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            task_state_dim = task_state_dim,
                            cdim = cdim,
                            multitask = multitask,
                            device = device)

ctrlmodels["target_critic_1"] = Critic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   task_state_dim = task_state_dim,
                                   cdim = cdim,
                                   multitask = multitask,
                                   device = device)

ctrlmodels["target_critic_2"] = Critic(observation_space = env.observation_space,
                                action_space = env.action_space, 
                                feature_dim = feature_dim, 
                                task_state_dim = task_state_dim,
                                cdim = cdim,
                                multitask = multitask,
                                device = device)


ctrlmodels["phi"] = Phi(observation_space = env.observation_space, 
				    action_space = env.action_space, 
				    feature_dim = feature_dim, 
				    hidden_dim = feature_hidden_dim,
                    drone_state_dim = drone_state_dim,
                    multitask = multitask,
                    device = device
                )

ctrlmodels["frozen_phi"] = Phi(observation_space = env.observation_space, 
    				       action_space = env.action_space, 
	    			       feature_dim = feature_dim, 
		    	           hidden_dim = feature_hidden_dim,
                           drone_state_dim = drone_state_dim,
                           multitask = multitask,
                           device = device
                        )

ctrlmodels["theta"] = Theta(
    		        observation_space = env.observation_space,
		            action_space = env.action_space, 
		            feature_dim = feature_dim, 
                    device = device
                )

ctrlmodels["mu"] = Mu(
                observation_space = env.observation_space, 
                action_space = env.action_space, 
                feature_dim = feature_dim, 
                hidden_dim = feature_hidden_dim,
                drone_state_dim = drone_state_dim,
                multitask = multitask,
                device = device
            )



# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 1024
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-4
cfg["critic_learning_rate"] = 1e-4
cfg["weight_decay"] = 0
cfg["feature_learning_rate"] = 5e-5
cfg["grad_norm_clip"] = 1.0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 1e-4
cfg["initial_entropy_value"] = 1.0
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1e6
cfg["experiment"]["write_interval"] = 0
cfg["experiment"]["checkpoint_interval"] = 0
cfg['use_feature_target'] = False
cfg['extra_feature_steps'] = 1
cfg['target_update_period'] = 1
cfg['eval'] = True
cfg['alpha'] = None




sacagent = SAC(
            models=sacmodels,
            memory=sacmemory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )

ctrlagent = CTRLSACAgent(
            models=ctrlmodels,
            memory=ctrlmemory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )


ctrlagent.load(f"/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0/CTRL-SAC/True/25-01-05_20-26-49-392901_CTRLSACAgent/checkpoints/{ckpt}.pt")
sacagent.load(f"/home/naliseas-workstation/Documents/anaveen/IsaacLab/runs/torch/Isaac-Quadcopter-legtrain-Trajectory-Direct-v0/SAC/25-01-06_10-11-53-625502_SAC/checkpoints/{ckpt}.pt")

agents = {
            "SAC": sacagent, 
            "CTRLSAC": ctrlagent
          }

cfg_trainer = {"timesteps": experiment_length, "headless": True}


agent = agents[agent_type]
env.eval_mode()
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
trainer.eval()

os.makedirs(output_dir, exist_ok=True)
torch.save(env.results, f"{output_dir}/results.pth")


