from torch import nn
import torch.nn.functional as F
from skrl.models.torch import Model, DeterministicMixin
import torch


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, feature_dim, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        self.feature_dim = feature_dim

        self.net1 = nn.Sequential(nn.Linear(self.feature_dim, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))
        
        self.net2 = nn.Sequential(nn.Linear(self.feature_dim, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):

        q1 = self.net1(inputs['z_phi'])
        q2 = self.net2(inputs['z_phi'])
        return (q1, q2), {}


class TestCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}
