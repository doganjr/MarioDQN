import gym
import numpy as np
import collections
from agent_dqn import Agent
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def rgb_to_gry(obs, dsr):
    x = np.asarray(np.around(obs[..., :3] @ [0.299, 0.587, 0.114] ), dtype=np.int8)
    return x[::dsr, ::dsr]

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
frame_stack_len = 4
downsampling_ratio = 2
obs_dims = tuple([int(x/downsampling_ratio) for x in list(env.observation_space.shape[:2])])

gamma = 0.99
agent = Agent(gamma=gamma, eps=0.05, lr=0.0003, tau=0.0001, batch_size=64, max_mem=150*1000, obs_dim=obs_dims, frame_stack=frame_stack_len, n_actions=env.action_space.n)
frame_stack = collections.deque(maxlen=frame_stack_len)
x_stack = collections.deque(maxlen=90)
save_freq = 100
step_size = 6
exp_id = 0
training_steps = 0
while training_steps<=5e5:
    ep_rew = 0
    obs = env.reset()
    obs[0, 0, :], obs[2, 0, :] = 0, 0
    [frame_stack.append(rgb_to_gry(obs, downsampling_ratio)) for ii in range(frame_stack_len)]
    [x_stack.append(np.random.randint(60, 80)) for ii in range(len(x_stack))]
    stacked_obs = np.asarray(frame_stack, dtype=np.uint8)
    done = False
    last_action, max_x = 0, 40
    while not done:
        training_steps += 1
        all_frame_rew = 0
        for step in range(step_size):
            frame_rew = 0
            action, action_val = agent.choose_action(stacked_obs, True)
            for f in range(frame_stack_len):
                if done:
                    pass
                else:
                    env.render()
                    obs, reward, done, info = env.step(action)
                    # Jump Including Action
                    obs[0, 0, :] = float((last_action == 2) or (last_action == 5) or (last_action == 4))
                    # Right Including Action
                    obs[2, 0, :] = float((last_action >=1) and (last_action <= 4))
                    reward += float(reward < 0) * 2 * reward
                    frame_rew += reward / 2
                    frame_stack.append(rgb_to_gry(obs, downsampling_ratio))
            if info['flag_get'] == True:
                frame_rew += 600
            if info['x_pos'] > max_x:
                max_x = info['x_pos']

            last_action = action
            x_stack.append(max_x)
            ep_rew += frame_rew
            all_frame_rew += frame_rew * (gamma ** (step_size - step - 1))
            next_stacked_obs = np.asarray(frame_stack, dtype=np.uint8)
            agent.memory.store_transition(stacked_obs, action, all_frame_rew, next_stacked_obs, done, float(info['x_pos']))
            if done:
                agent.memory.store_transition(stacked_obs, action, all_frame_rew, next_stacked_obs, done, float(info['x_pos']))
            agent.update()
            stacked_obs = next_stacked_obs

        if len(set(x_stack)) == 1:
            done = True
        if training_steps % save_freq == 0:
            agent.save_agent(training_steps, exp_id)
            print(f"AGENT SAVED @{training_steps}")


