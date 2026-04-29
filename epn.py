import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import SyncConfig

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, ReLU, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Model, Sequential, layers

# Note: You will need to wrap your Environment into a standard gymnasium.Env
from data_generation import Environment

# --- Custom Environment Wrapper ---
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from data_generation import Environment
from google.colab import drive
drive.mount('/content/drive')

# Create a specific folder for your RL experiments
CHECKPOINT_PATH = "/content/ray_results/epn_experiment"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


class ESWMGymEnv(gym.Env):
    def __init__(self, env_config=None):
        self.env = Environment(side_length=4, add_wall=True, hidden=5, possible_states=37)
        self.action_space = Discrete(6)

        self.max_steps = 200
        self.current_step = 0

        self.seq_length = 201
        # 3 dimensions: (source_index, action_index, end_index)
        self.observation_space = Box(low=0, high=40, shape=(self.seq_length, 3), dtype=np.float32)

    def _get_obs(self):
        """Formats the memory bank and the current goal into the 3-dim structure."""
        obs = np.zeros((self.seq_length, 3), dtype=np.float32)

        # Fill the memory bank transitions
        for i, (s1_val, action, s2_val) in enumerate(self.env.bank):
            if i >= self.seq_length - 1:
                break

            # NOTE: Adding +1 to states so '0' is strictly used for padding masking!
            obs[i, 0] = s1_val + 1
            obs[i, 1] = action + 1  # Optional: shift actions too if mask_zero applies to action_embed
            obs[i, 2] = s2_val + 1

        current_loc = self.env._agent_location
        target_loc = self.env._target_location

        # Append the target condition using observations
        obs[-1, 0] = self.env.observations[current_loc] + 1
        obs[-1, 1] = 7
        obs[-1, 2] = self.env.observations[target_loc] + 1

        return obs

    def reset(self, *, seed=None, options=None):
      super().reset(seed=seed)
      self.current_step = 0
      self.env.reset(seed=seed)

      # Start with the MST to ensure connectivity
      mst_bank = self.env.generate_memory_bank().tolist()

      # Fill remaining slots with random transitions from the observable edges
      # to reach the 200-transition "dense" bank used in the paper
      while len(mst_bank) < 200:
          extra_edge = random.choice(self.env.obs_edges)
          t = self.env.get_transition(extra_edge)
          # Ensure it's not a wall transition
          if t[0] not in self.env.wall and t[2] not in self.env.wall:
              formatted_t = np.hstack([self.env.observations[t[0]], [t[1]], self.env.observations[t[2]]])
              mst_bank.append(formatted_t)

      self.env.bank = mst_bank

      # Standard reset logic...
      valid_locations = [loc for loc in self.env.locations if loc not in self.env.wall]
      self.env._agent_location, self.env._target_location = random.sample(valid_locations, 2)
      return self._get_obs(), {}

    def step(self, action):
      self.current_step += 1
      self.env.move(action)
      obs = self._get_obs()

      # Calculate current distance for reward shaping
      curr_coords = self.env.coords(self.env._agent_location)
      target_coords = self.env.coords(self.env._target_location)
      dist = np.linalg.norm(curr_coords - target_coords)

      if self.env._agent_location == self.env._target_location:
          reward = 10.0  # Increased success reward
          terminated = True
      else:
          # Dense reward: step penalty + distance penalty
          reward = -0.01 - (dist * 0.01)
          terminated = False

      truncated = self.current_step >= self.max_steps
      info = {"success": terminated, "dist": dist}
      return obs, reward, terminated, truncated, info

def env_creator(env_config):
    return ESWMGymEnv(env_config)

class RandomWall(Model):
    def __init__(self):
        super().__init__()
        # Embedding sizes matched to PyTorch RandomWall (39, 8, 39 to 1024)
        self.source_embed = layers.Embedding(39, 256, mask_zero=True)
        self.action_embed = layers.Embedding(8, 256, mask_zero=True)
        self.end_embed = layers.Embedding(39, 256, mask_zero=True)

    def call(self, x):
        # Cast observations to integer indices
        x = tf.cast(x, tf.int32)

        start = self.source_embed(x[:, :, 0])
        action = self.action_embed(x[:, :, 1])
        end = self.end_embed(x[:, :, 2])

        return tf.reduce_mean(tf.stack([start, action, end], axis=0), axis=0)

class EPN(Model):
    # Added num_outputs to the constructor
    def __init__(self, sa_iterations, input_dim, num_outputs):
        super().__init__()
        self.attention_iterations = sa_iterations
        self.norm = LayerNormalization()
        self.mha = MultiHeadAttention(num_heads=1, key_dim=64)
        self.add = layers.Add()

        self.add_state = layers.Concatenate(axis=1)
        self.planner = Sequential([
            Dense(input_dim, activation="relu"), # ReLU AFTER the first projection
            Dense(input_dim)
        ])

        self.pool = layers.Maximum()
        self.policy_net = Sequential([
            Dense(64, activation="relu"), Dense(64, activation="relu")
        ])

        # Output exactly the number of actions RLlib expects
        self.logits = Dense(num_outputs)
        self.baseline = Dense(1)

    def call(self, x, state):
      # x: (batch, seq-1, dim), state: (batch, 1, dim)
      b = []
      for i in range(self.attention_iterations):
          # Concatenate memory and goal BEFORE attention
          combined = self.add_state([x, state])

          # Self-attention over both memory AND goal
          att = self.norm(combined)
          att = self.mha(att, att)
          combined = self.add([combined, att])

          # Pass through the planner MLP
          xc = self.planner(combined)

          # Split back for the next iteration
          x = xc[:, :-1, :]
          state = xc[:, -1:, :]
          b.append(xc)

      # Element-wise max over the iteration history
      x = self.pool(b)
      x = tf.reduce_max(x, axis=1) # Global pooling

      x = self.policy_net(x)
      return self.logits(x), self.baseline(x)


class RLlibEPNModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        sa_iters = model_config.get("custom_model_config", {}).get("sa_iterations", 6)
        input_dim = model_config.get("custom_model_config", {}).get("input_dim", 1024)
        self.embed = RandomWall()

        # Pass num_outputs to the EPN initialization
        self.epn = EPN(sa_iters, input_dim, num_outputs)
        self._value_out = None

        # Force Keras variable initialization (your previous fix)
        dummy_obs = tf.zeros((1, obs_space.shape[0], obs_space.shape[1]), dtype=tf.float32)
        dummy_seq_lens = tf.ones([1], dtype=tf.int32)
        self.forward({"obs": dummy_obs}, [], dummy_seq_lens)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        em = self.embed(obs)
        epn_state = tf.expand_dims(em[:, -1, :], axis=1)
        logits, baseline = self.epn(em[:, :-1, :], epn_state)
        self._value_out = tf.reshape(baseline, [-1])
        return logits, state

    def value_function(self):
        return self._value_out

if __name__ == "__main__":
    # Initialize Ray locally inside Colab
    ray.init(ignore_reinit_error=True)

    # 1. Register Environment and Model
    register_env("eswm_env", env_creator)
    ModelCatalog.register_custom_model("epn_rllib_model", RLlibEPNModel)

    # 2. Define IMPALA Configuration (MODIFIED FOR COLAB)
    config = (
        ImpalaConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .environment("eswm_env")
        .framework("tf2", eager_tracing=True)
        .resources(
            num_gpus=1
        )
        .env_runners(
            num_env_runners=10,
            num_cpus_per_env_runner=1,
            num_envs_per_env_runner=4,
            rollout_fragment_length=400
        )
        .training(
            train_batch_size=4000,
            lr=0.0001,
            entropy_coeff=0.01,
            model={
                "custom_model": "epn_rllib_model",
                "custom_model_config": {
                    "sa_iterations": 4,
                    "input_dim": 256
                }
            }
        )
    )

    print("\nStarting IMPALA Training on Colab...\n")

    # 4. Start Training (Azure SyncConfig REMOVED for Colab)
    tune.run(
        "IMPALA",
        config=config.to_dict(),
        name="IMPALA_2026-04-27_21-19-57",
        resume="AUTO",
        stop={"training_iteration": 100000},
        checkpoint_freq=500,
        keep_checkpoints_num=5,
        checkpoint_at_end=True,
        storage_path=CHECKPOINT_PATH,
        callbacks=[
            MLflowLoggerCallback(
                experiment_name="epn-impala-colab",
                save_artifact=False
            )
        ]
    )