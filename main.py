from env import env
from train import train
from test import test_model
from stable_baselines3 import PPO


agent = PPO
# train(env=env, agent=PPO, total_steps=2000000, frequency=50000)
model = agent.load("./models/flappy_bird_1800000")
test_model(env=env, model=model, episodes=20)
