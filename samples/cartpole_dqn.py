## This is course material for Introduction to Modern Artificial Intelligence
## Example code: cartpole_p_control.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020-2024. Intelligent Racing Inc. Not permitted for commercial use

# Please make sure to install openAI gym module
# pip install gym==0.17.3
# pip install pyglet==1.5.29

import gym
import numpy as np

EPISODES = 100

def get_action(state, kp):
    angle = state[2]
    error = -angle  # Ideal pole angle is zero
    control_signal = kp * error
    # CartPole action: 0 -> push cart to the left, 1 -> push cart to the right
    return 1 if control_signal > 0 else 0

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # Proportional gain
    kp = 1.0

    scores = []

    for e in range(EPISODES):
        state = env.reset()
        score = 0
        for time in range(500):
            env.render()
            action = get_action(state, kp)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                print(f"Episode: {e+1}/{EPISODES}, Score: {score}")
                scores.append(score)
                break
    env.close()

    # Print summary statistics
    print(f"Average score over {EPISODES} episodes: {np.mean(scores)}")
    print(f"Highest score: {np.max(scores)}")
    print(f"Lowest score: {np.min(scores)}")
