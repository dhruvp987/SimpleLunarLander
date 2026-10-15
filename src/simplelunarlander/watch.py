import argparse

import gymnasium as gym
from stable_baselines3 import PPO

RAND_SEED = 21
NUM_EPISODES = 5

LLV3_GYM_NAME = "LunarLander-v3"


def watch_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="Path to a saved PPO model")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu") if args.model else None

    env = gym.make(LLV3_GYM_NAME, render_mode="human")

    try:
        for episode in range(NUM_EPISODES):
            obs, _ = env.reset(seed=RAND_SEED + episode)
            env.action_space.seed(RAND_SEED + episode)

            total_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)

                (
                    obs,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = env.step(action)
                total_reward += reward

            print(f"Flight {episode + 1}: Total reward is {total_reward:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    watch_main()
