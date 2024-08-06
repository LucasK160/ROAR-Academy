## This is course material for Introduction to Modern Artificial Intelligence
## Example code: cartpole_dqn.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020-2024. Intelligent Racing Inc. Not permitted for commercial use

# Please make sure to install openAI gym module
# pip install gym==0.17.3
# pip install pyglet==1.5.29

import random
import gym
import os
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

EPISODES = 100
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class PIDAgent:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def act(self, state):
        # The state vector for CartPole-v1 is [cart position, cart velocity, pole angle, pole velocity at tip]
        pole_angle = state[2]
        error = pole_angle
        self.integral += error
        derivative = error - self.previous_error

        # PID control law
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        # Convert the control signal to action: 0 (left) or 1 (right)
        action = 0 if control < 0 else 1
        return action

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # agent.load("./save/cartpole-dqn.h5")
    kp =1.0
    ki=0.0
    kd=0.1
    agent = PIDAgent(kp,ki,kd)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                print(f"episode: {e}/{EPISODES}, score: {time}")
                break
    env.close()
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
