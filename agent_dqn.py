import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model_dqn import NetworkDQN
from memory_agent import Memory
import os

class Agent():
    def __init__(self, gamma, eps, lr, tau, batch_size, max_mem, obs_dim, n_actions, frame_stack):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.obs_dim = obs_dim
        self.max_mem = max_mem
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.tau = tau
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.memory = Memory(max_mem, obs_dim, frame_stack)

        self.pri_network = NetworkDQN(fs=frame_stack, input_dim=self.obs_dim, fc1=400, fc2=300, n_actions=n_actions).to(self.device)
        self.target_network = NetworkDQN(fs=frame_stack, input_dim=self.obs_dim, fc1=400, fc2=300, n_actions=n_actions).to(self.device)

        for target_param, param in zip(self.target_network.parameters(), self.pri_network.parameters()):
            target_param.data.copy_(param.data)

        self.optimizer = optim.Adam(self.pri_network.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def choose_action(self, observation, random_flag):
        actions = self.pri_network.forward(torch.tensor(observation).unsqueeze(0).to(self.device))
        if random_flag:
            if np.random.random() > self.eps:
                action = int(torch.argmax(actions))
            else:
                action = np.random.choice(self.n_actions)
        else:
            if np.random.random() > 0.025:
                action = int(torch.argmax(actions))
            else:
                action = np.random.choice(self.n_actions)
        return action, actions

    def update(self):
        if self.memory.mem_total < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, batch_index = self.memory.sample_batch(self.batch_size)

        q_curr = self.pri_network.forward(state_batch).gather(1, action_batch.unsqueeze(1).long())
        q_next = self.target_network.forward(next_state_batch).detach().max(1)[0].unsqueeze(1)

        q_target = reward_batch.unsqueeze(-1) + self.gamma * (1 - terminal_batch.unsqueeze(-1)) * q_next

        self.optimizer.zero_grad()
        loss = self.loss(q_target, q_curr)
        loss.backward()
        self.optimizer.step()

        for target_param, param in zip(self.target_network.parameters(), self.pri_network.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_agent(self, trainstep, id):
        folder_name = f"saved_agents"
        if os.path.isdir(folder_name) == False:
            os.mkdir(folder_name)

        folder_name = f"saved_agents/{id}/"
        if os.path.isdir(folder_name) == False:
            os.mkdir(folder_name)

        f = open(folder_name+"_hyperparams.txt", "w")
        f.write(f"BS: {self.batch_size}, Mem: {self.max_mem}, gamma: {self.gamma}, LR: {self.lr}, EPS: {self.eps}")

        torch.save(self.pri_network.state_dict(), folder_name+f"/{trainstep}_net.pth")

    def load_agent(self, trainstep, id):
        folder_name = f"saved_agents/{id}/{trainstep}_net.pth"
        self.pri_network.load_state_dict(torch.load(folder_name,  map_location=self.device))
        self.target_network.load_state_dict(torch.load(folder_name,map_location=self.device))