import torch
import numpy as np
class Memory():
    def __init__(self, max_mem, obs_dim, frame_stack):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mem_index = 0
        self.mem_total = 0
        self.max_mem = max_mem

        self.state_memory = torch.zeros((self.max_mem, frame_stack, *obs_dim), dtype=torch.uint8)
        self.action_memory = torch.zeros(self.max_mem, dtype= torch.int32)
        self.reward_memory = torch.zeros(self.max_mem, dtype=torch.float32)
        self.next_state_memory = torch.zeros((self.max_mem, frame_stack, *obs_dim), dtype=torch.uint8)
        self.terminal_memory = torch.zeros(self.max_mem, dtype=torch.int32)

        self.x_loc = torch.zeros(self.max_mem, dtype= torch.float32)

    def store_transition(self, state, action, reward, next_state, done, trans_x):
        index = self.mem_index % self.max_mem
        self.state_memory[index] = torch.tensor(state)
        self.action_memory[index] = torch.tensor(action)
        self.reward_memory[index] = torch.tensor(reward)
        self.next_state_memory[index] = torch.tensor(next_state)
        self.terminal_memory[index] = torch.tensor(int(done))

        self.x_loc[index] = torch.tensor(trans_x)

        self.mem_total += 1
        self.mem_index += 1


    def sample_batch(self, batch_size):
        idx1 = np.random.choice(np.minimum(self.mem_total, self.max_mem), batch_size, replace=True)
        idx2 = np.random.choice(np.minimum(self.mem_total, self.max_mem), batch_size, replace=True)

        if np.random.random() > 0.5:
            batch_index = idx1
        else:
            m1, m2 = torch.mean(self.x_loc[idx1]), torch.mean(self.x_loc[idx2])
            batch_index = float(m1 > m2) * idx1 + float(m2 >= m1) * idx2

        batch_index = np.random.choice(np.minimum(self.mem_total, self.max_mem), batch_size, replace=True)

        state_batch = self.state_memory[batch_index].to(self.device)
        action_batch = self.action_memory[batch_index].to(self.device)
        reward_batch = self.reward_memory[batch_index].to(self.device)
        next_state_batch = self.next_state_memory[batch_index].to(self.device)
        terminal_batch = self.terminal_memory[batch_index].to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, batch_index
