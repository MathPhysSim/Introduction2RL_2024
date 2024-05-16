import os
import random
from collections import deque
from datetime import datetime

import gym
import gym.wrappers
import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx("float64")

lr = 0.005
batch_size = 256
gamma = 0.95
eps = 1
eps_decay = 0.995
eps_min = 0.01
logdir = 'logs'

logdir = os.path.join(logdir, "Classical_DQN.py", 'Car_pole', datetime.now().strftime("%Y%m%d-%H%M%S"))
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)

# this class builds up the ReplayBuffer
# the function sample chooses one experience out of the replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


# this class builds up the DQN algorithm. Please fill in the missing parts of nn_model and get_action
class DQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = eps
        self.model = self.nn_model()

    def nn_model(self):
        '''here comes your neural network'''
        pass

    def predict(self, state):
        return self.model(state)

    def get_action(self, state):
        '''this function should choose the next action based on the state with an epsilon-greedy strategy'''
        return action

    def train(self, states, targets):
        history = self.model.fit(states, targets, epochs=1, verbose=2)
        loss = history.history['loss'][0]
        return loss

# this class builds up the learning agent. Please fill in the missing parts of update_target and replay_experience
class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = DQN(self.state_dim, self.action_dim)
        self.target_model = DQN(self.state_dim, self.action_dim)
        self.update_target()
        self.buffer = ReplayBuffer()

    def update_target(self):
        '''here you update your target'''
        pass

    def replay_experience(self):
        '''update using target network to stabilize'''
        pass

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            done, episode_reward = False, 0
            state = self.env.reset()
            while not done:
                action = DQN.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.store(state, action, reward * 0.01, next_state, done)
                episode_reward += reward
                state = next_state
            if self.buffer.size() >= batch_size:
                self.replay_experience()
            self.update_target()
            print(f"Episoden#{ep} Gesammter Reward:{episode_reward}")


class SimWrapper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        self.render()
        return self.env.step(action)


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # env = SimWrapper(env)
    agent = Agent(env)
    agent.train(max_episodes=2000)
