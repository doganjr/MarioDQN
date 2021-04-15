import torch.nn as nn
import torch.nn.functional as F
import torch

class NetworkDQN(nn.Module):
    def __init__(self, fs, input_dim, fc1, fc2, n_actions):
        super(NetworkDQN, self).__init__()
        self.conv1 = nn.Conv2d(fs, 64, 8, 4)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 48, 3, 1)

        self.lin1 = nn.Linear(256*3+2, fc1)
        self.lin2 = nn.Linear(fc1+2, fc2)
        self.actl = nn.Linear(fc2+2, n_actions)

    def forward(self, observation):
        jr = observation[:, 0, 0:2, 0]

        observation = (observation - 127) / 255

        x = F.relu(self.conv1(observation))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, 1)

        x = torch.cat([x, jr], dim=1)
        x = F.relu(self.lin1(x))

        x = torch.cat([x, jr], dim=1)
        x = F.relu(self.lin2(x))

        x = torch.cat([x, jr], dim=1)
        action_val = self.actl(x)

        return action_val
