# SimpleLunarLander 🚀

A simple Reinforcement Learning demo using Gymnasium's LunarLander-v3 and the Proximal Policy Optimization algorithm

## Overview

### What is Reinforcement Learning?

Reinforcement Learning (RL) is an area of Machine Learning (ML) where an agent learns to make decisions by interacting with an environment.

The agent learns which actions lead to better outcomes by receiving rewards or penalties based on the results of its actions.

### The Breakdown

Gymnasium's LunarLander-v3 is an environment where the goal is to safely land a vehicle between two yellow goalposts.

Proximal Policy Optimization (PPO) is an algorithm that helps an agent learn to make better decisions by using the rewards and penalties
collected through repeated interactions with an environment.

In the demo, PPO is used to train an agent to safely land a vehicle in LunarLander-v3.

### The Demo

The first part of the demo shows an agent taking random actions, typically resulting in crashes.

The second part shows an agent trained through repeated interactions with the environment, leading
to safer, more precise landings.

## Tech Stack

- Gymnasium: for the LunarLander-v3 environment
- Stable Baselines3: for the PPO algorithm
- Python: the programming language
- uv: the package manager

## Usage

- src/simplelunarlander/watch.py
  - **Run an agent on LunarLander-v3**
  - Run without any command line arguments to use an agent that takes random actions
  - Run with a model file passed as the first argument to use the agent from the model file

- src/simplelunarlander/train.py
  - **Train a series of agents**
  - Run without any command line arguments
  - Each agent is saved as a model in a .zip file