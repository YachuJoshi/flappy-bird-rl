import time
from env import FlappyBirdEnv
from stable_baselines3 import PPO
from utils.callback import TrainAndLoggingCallback

# LOG_DIR = "./logs"
# CHECKPOINT_DIR = "./models"
# callback = TrainAndLoggingCallback(check_freq=50000, save_path=CHECKPOINT_DIR)

env = FlappyBirdEnv(render_mode="human")
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
# model.learn(total_timesteps=400000, callback=callback)

episodes = 10
model = PPO.load("./models/flappy_bird_ppo_400000")

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
    time.sleep(1)

env.close()
