import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_of_nodes):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_of_nodes)
        self.layer2 = nn.Linear(n_of_nodes, n_of_nodes)
        self.layer3 = nn.Linear(n_of_nodes, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)