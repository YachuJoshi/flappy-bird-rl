from stable_baselines3 import A2C, PPO, DQN
from utils.callback import TrainAndLoggingCallback

LOG_DIR = "./logs"
CHECKPOINT_DIR = "./models"


def train(env, agent, total_steps, frequency):
    callback = TrainAndLoggingCallback(check_freq=frequency, save_path=CHECKPOINT_DIR)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=0.0001,
    )
    model.learn(total_timesteps=total_steps, callback=callback)
