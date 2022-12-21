from env import FlappyBirdEnv
from train import train
from test import test_model
from stable_baselines3 import PPO


env = FlappyBirdEnv()  # Training
# env = FlappyBirdEnv(render_mode="human")  # Testing

agent = PPO

# Training
train(env=env, agent=PPO, total_steps=2000000, frequency=50000)

# Testing
# model = agent.load("./models/flappy_bird_1800000")
# test_model(env=env, model=model, episodes=20)
