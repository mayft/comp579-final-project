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

from data_generation import Environment

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from data_generation import Environment
from google.colab import drive
drive.mount('')

CHECKPOINT_PATH = "" # Set to blank for security
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


class ESWMGymEnv(gym.Env):
    def __init__(self, env_config=None):
        self.env = Environment(side_length=4, add_wall=True, hidden=5, possible_states=37)
        self.action_space = Discrete(6)

        self.max_steps = 200
        self.current_step = 0

        self.seq_length = 201
        self.observation_space = Box(low=0, high=40, shape=(self.seq_length, 3), dtype=np.float32)

    def _get_obs(self):
        """Formats the memory bank and the current goal into the 3-dim structure."""
        obs = np.zeros((self.seq_length, 3), dtype=np.float32)

        # Fill the memory bank transitions
        for i, (s1_val, action, s2_val) in enumerate(self.env.bank):
            if i >= self.seq_length - 1:
                break

            obs[i, 0] = s1_val + 1
            obs[i, 1] = action + 1  
            obs[i, 2] = s2_val + 1

        current_loc = self.env._agent_location
        target_loc = self.env._target_location

        obs[-1, 0] = self.env.observations[current_loc] + 1
        obs[-1, 1] = 7
        obs[-1, 2] = self.env.observations[target_loc] + 1

        return obs

    def reset(self, *, seed=None, options=None):
      super().reset(seed=seed)
      self.current_step = 0
      self.env.reset(seed=seed)

      mst_bank = self.env.generate_memory_bank().tolist()

      while len(mst_bank) < 200:
          extra_edge = random.choice(self.env.obs_edges)
          t = self.env.get_transition(extra_edge)
          if t[0] not in self.env.wall and t[2] not in self.env.wall:
              formatted_t = np.hstack([self.env.observations[t[0]], [t[1]], self.env.observations[t[2]]])
              mst_bank.append(formatted_t)

      self.env.bank = mst_bank

      valid_locations = [loc for loc in self.env.locations if loc not in self.env.wall]
      self.env._agent_location, self.env._target_location = random.sample(valid_locations, 2)
      return self._get_obs(), {}

    def step(self, action):
      self.current_step += 1
      self.env.move(action)
      obs = self._get_obs()

      curr_coords = self.env.coords(self.env._agent_location)
      target_coords = self.env.coords(self.env._target_location)
      dist = np.linalg.norm(curr_coords - target_coords)

      if self.env._agent_location == self.env._target_location:
          reward = 10.0 
          terminated = True
      else:
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
        self.source_embed = layers.Embedding(39, 256, mask_zero=True)
        self.action_embed = layers.Embedding(8, 256, mask_zero=True)
        self.end_embed = layers.Embedding(39, 256, mask_zero=True)

    def call(self, x):
        x = tf.cast(x, tf.int32)

        start = self.source_embed(x[:, :, 0])
        action = self.action_embed(x[:, :, 1])
        end = self.end_embed(x[:, :, 2])

        return tf.reduce_mean(tf.stack([start, action, end], axis=0), axis=0)

class EPN(Model):
    def __init__(self, sa_iterations, input_dim, num_outputs):
        super().__init__()
        self.attention_iterations = sa_iterations
        self.norm = LayerNormalization()
        self.mha = MultiHeadAttention(num_heads=1, key_dim=64)
        self.add = layers.Add()

        self.add_state = layers.Concatenate(axis=1)
        self.planner = Sequential([
            Dense(input_dim, activation="relu"),
            Dense(input_dim)
        ])

        self.pool = layers.Maximum()
        self.policy_net = Sequential([
            Dense(64, activation="relu"), Dense(64, activation="relu")
        ])

        self.logits = Dense(num_outputs)
        self.baseline = Dense(1)

    def call(self, x, state):
      b = []
      for i in range(self.attention_iterations):
          combined = self.add_state([x, state])

          att = self.norm(combined)
          att = self.mha(att, att)
          combined = self.add([combined, att])

          xc = self.planner(combined)

          x = xc[:, :-1, :]
          state = xc[:, -1:, :]
          b.append(xc)

      x = self.pool(b)
      x = tf.reduce_max(x, axis=1)

      x = self.policy_net(x)
      return self.logits(x), self.baseline(x)


class RLlibEPNModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        sa_iters = model_config.get("custom_model_config", {}).get("sa_iterations", 6)
        input_dim = model_config.get("custom_model_config", {}).get("input_dim", 1024)
        self.embed = RandomWall()

        self.epn = EPN(sa_iters, input_dim, num_outputs)
        self._value_out = None

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
    ray.init(ignore_reinit_error=True)

    register_env("eswm_env", env_creator)
    ModelCatalog.register_custom_model("epn_rllib_model", RLlibEPNModel)

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

    tune.run(
        "IMPALA",
        config=config.to_dict(),
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