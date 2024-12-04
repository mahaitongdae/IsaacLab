from torch import nn
import torch.nn.functional as F
from skrl.models.torch import Model, DeterministicMixin
import torch


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, feature_dim, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        self.feature_dim = feature_dim
        
        self.net = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):

        q = self.net(inputs['z_phi'])
        return q, {}

