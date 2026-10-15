from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

NUM_ENVS = 4
ENV_RAND_SEED = 21

MODELS_DIR = "models"
CHECKPOINT_MODELS_SUBDIR = "checkpoints"
FINAL_MODEL_NAME = "final"

LOGS_DIR = "logs"
EVAL_LOGS_SUBDIR = "evaluations"

LLV3_GYM_NAME = "LunarLander-v3"

MODEL_RAND_SEED = 101
MODEL_N_STEPS = 1024
MODEL_BATCH_SIZE = 64
MODEL_LEARNING_RATE = 3e-4
MODEL_TOTAL_TIMESTEPS = 500_000

CHECKPOINT_SAVE_FREQ = 50_000

EVAL_FREQ = 25_000
NUM_EVAL_EPISODES = 10


def train_main():
    Path(MODELS_DIR).mkdir(exist_ok=True)
    Path(LOGS_DIR).mkdir(exist_ok=True)

    train_envs = make_vec_env(LLV3_GYM_NAME, n_envs=NUM_ENVS, seed=ENV_RAND_SEED)

    eval_env = Monitor(gym.make(LLV3_GYM_NAME))
    eval_env.reset(seed=ENV_RAND_SEED * 2)

    model = PPO(
        "MlpPolicy",
        train_envs,
        device="cpu",
        seed=MODEL_RAND_SEED,
        verbose=1,
        n_steps=MODEL_N_STEPS,
        batch_size=MODEL_BATCH_SIZE,
        learning_rate=MODEL_LEARNING_RATE,
    )

    checkpoints = CheckpointCallback(
        save_freq=CHECKPOINT_SAVE_FREQ // NUM_ENVS,
        save_path=MODELS_DIR + "/" + CHECKPOINT_MODELS_SUBDIR,
    )

    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOGS_DIR + "/" + EVAL_LOGS_SUBDIR,
        eval_freq=EVAL_FREQ // NUM_ENVS,
        n_eval_episodes=NUM_EVAL_EPISODES,
        deterministic=True,
    )

    try:
        model.learn(
            total_timesteps=MODEL_TOTAL_TIMESTEPS, callback=[checkpoints, evaluation]
        )
        model.save(MODELS_DIR + "/" + FINAL_MODEL_NAME)
    finally:
        train_envs.close()
        eval_env.close()


if __name__ == "__main__":
    train_main()
