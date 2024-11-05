import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer, ParallelTrainer
from skrl.utils import set_seed

from sac.actor import DiagGaussianActor
from sac.critic import Critic, TestCritic
from sac.feature import Phi, Mu, Theta

from ctrlsac_agent import CTRLSACAgent

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

# load and wrap the Isaac Gym environment
env = load_isaaclab_env(task_name="Isaac-Quadcopter-Direct-v0", num_envs=1)
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

# define hidden dimension
actor_hidden_dim = 256
actor_hidden_depth = 2

# define feature dimension 
feature_dim = 2048
feature_hidden_dim = 1024

# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = DiagGaussianActor(observation_space = env.observation_space,
                                     action_space = env.action_space, 
                                     hidden_dim = actor_hidden_dim, 
                                     hidden_depth = actor_hidden_depth,
                                     log_std_bounds = [-5., 2.], 
                                     device = device)

models["critic_1"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            hidden_dim = feature_hidden_dim, 
                            device = device)

models["critic_2"] = Critic(observation_space = env.observation_space,
                            action_space = env.action_space, 
                            feature_dim = feature_dim, 
                            hidden_dim = feature_hidden_dim, 
                            device = device)

models["target_critic_1"] = Critic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   hidden_dim = feature_hidden_dim, 
                                   device = device)

models["target_critic_2"] = Critic(observation_space = env.observation_space,
                                   action_space = env.action_space, 
                                   feature_dim = feature_dim, 
                                   hidden_dim = feature_hidden_dim, 
                                   device = device)



models["phi"] = Phi(observation_space = env.observation_space, 
				    action_space = env.action_space, 
				    feature_dim = feature_dim, 
				    hidden_dim = feature_hidden_dim,
                    device = device
                )

models["frozen_phi"] = Phi(observation_space = env.observation_space, 
    				       action_space = env.action_space, 
	    			       feature_dim = feature_dim, 
		    	           hidden_dim = feature_hidden_dim,
                           device = device
                        )

models["theta"] = Theta(
    		        observation_space = env.observation_space,
		            action_space = env.action_space, 
		            feature_dim = feature_dim, 
                    device = device
                )

models["mu"] = Mu(
                observation_space = env.observation_space, 
                action_space = env.action_space, 
                feature_dim = feature_dim, 
                hidden_dim = feature_hidden_dim,
                device = device
            )

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 256
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["weight_decay"] = 0
cfg["feature_learning_rate"] = 5e-4
cfg["random_timesteps"] = 80
cfg["learning_starts"] = 80
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0
# cfg["state_preprocessor"] = RunningStandardScaler
# cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 80
cfg["experiment"]["checkpoint_interval"] = 800
cfg["experiment"]["directory"] = "runs/torch/Quadcopter"
cfg['use_feature_target'] = True
cfg['extra_feature_steps'] = 3



agent = CTRLSACAgent(
            models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )
# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 160000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()

