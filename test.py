def test_model(env, model, episodes=10):
    for episode in range(1, episodes + 1):
        observation = env.reset()
        done = False
        total_reward = 0

        while True:
            env.render()
            action, _states = model.predict(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                break

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()
