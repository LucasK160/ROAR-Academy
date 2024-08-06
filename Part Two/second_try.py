if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize PID constants
    Kp, Ki, Kd = 1.0, 0.0, 0.0  # Adjust these constants as needed
    integral = 0.0
    previous_error = 0.0

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # Calculate PID control signal
            error = state[0][2]  # The pole angle is the third observation
            integral += error
            derivative = error - previous_error
            pid_control = Kp * error + Ki * integral + Kd * derivative

            # Determine the action based on the PID control signal
            action = 0 if pid_control < 0 else 1

            next_state, reward, done, _ = env.step(action)
            env.render()
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            previous_error = error  # Update the previous error

            if done:
                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, time))
                break
