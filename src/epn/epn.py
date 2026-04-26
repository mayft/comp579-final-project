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

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from data_generation import Environment

class ESWMGymEnv(gym.Env):
    def __init__(self, env_config=None):
        self.env = Environment(side_length=3, add_wall=True, hidden=0, possible_states=64)
        
        self.action_space = Discrete(6)
        
        self.max_steps = 200
        self.current_step = 0
        
        self.seq_length = len(self.env.locations) + 1 
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.seq_length, 13), dtype=np.float32)

    def _get_obs(self):
        """Formats the memory bank and the current goal into the 13-dim structure."""
        obs = np.zeros((self.seq_length, 13), dtype=np.float32)
        
        # 1. Fill the memory bank transitions
        for i, (s1_val, action, s2_val) in enumerate(self.env.bank):
            if i >= self.seq_length - 1:
                break
            
            s1_loc = next((k for k, v in self.env.observations.items() if v == s1_val), None)
            s2_loc = next((k for k, v in self.env.observations.items() if v == s2_val), None)
            
            if s1_loc is not None and s2_loc is not None:
                obs[i, 0:6] = self.env.states[s1_loc]
                obs[i, 6] = action
                obs[i, 7:13] = self.env.states[s2_loc]
                
        current_loc = self.env._agent_location
        target_loc = self.env._target_location
        
        obs[-1, 0:6] = self.env.states[current_loc]
        obs[-1, 6] = 0 # Dummy action since this row represents the goal condition
        obs[-1, 7:13] = self.env.states[target_loc]
        
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Generate a new random environment layout and memory bank
        self.env.rng = np.random.default_rng(seed=seed)
        self.env.generate_memory_bank(1)
        
        # Pick a random start and target location that are NOT in the wall
        valid_locations = [loc for loc in self.env.locations if loc not in self.env.wall]
        
        if len(valid_locations) >= 2:
            self.env._agent_location, self.env._target_location = random.sample(valid_locations, 2)
        else:
            # Fallback if the wall completely fills the map (highly unlikely in normal generation)
            self.env._agent_location = self.env.locations[0]
            self.env._target_location = self.env.locations[-1]
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        
        intended_state = self.env.states[self.env._agent_location][action]
        
        if intended_state and intended_state not in self.env.wall:
            self.env._agent_location = intended_state
            
        obs = self._get_obs()
        
        if self.env._agent_location == self.env._target_location:
            reward = 1.0
            terminated = True
        else:
            reward = -0.01
            terminated = False
            
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            
        info = {
            "success": terminated,
            "path_length": self.current_step
        }
            
        return obs, reward, terminated, truncated, info
    
def env_creator(env_config):
    return ESWMGymEnv(env_config)

class OpenWorld(Model):
    def __init__(self):
        super().__init__()
        self.source_embed = layers.Embedding(2, 128)
        self.end_embed = layers.Embedding(2, 128)
        self.action_embed = layers.Embedding(6, 768)

    def call(self, x):
        raw_start = x[:, :, 0:6]
        raw_action = x[:, :, 6]
        raw_end = x[:, :, 7:13]

        start_idx = tf.cast(raw_start > 0, tf.int32)
        end_idx = tf.cast(raw_end > 0, tf.int32)
        action_idx = tf.cast(raw_action, tf.int32)

        start = tf.concat([self.source_embed(start_idx[:, :, i]) for i in range(6)], axis=-1)
        action = self.action_embed(action_idx)
        end = tf.concat([self.end_embed(end_idx[:, :, i]) for i in range(6)], axis=-1)
        
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
            ReLU(),
            Dense(64),
            Dense(64),
            Dense(input_dim)
        ])

        self.pool = layers.Maximum()
        self.policy_net = Sequential([
            Dense(64), Dense(64), ReLU()
        ])

        self.logits = Dense(num_outputs) 
        self.baseline = Dense(1) 

    def call(self, x, state):
        b = []
        for i in range(self.attention_iterations):
            att = self.norm(x)
            att = self.mha(att, att)
            x = self.add([x, att])
            
            xc = self.add_state([x, state])
            
            xc = self.planner(xc)
            
            x = xc[:, :-1, :]
            state = xc[:, -1:, :]
            
            b.append(xc)

        x = self.pool(b)
        
        x = tf.reduce_max(x, axis=1)
        
        x = self.policy_net(x)
        logits = self.logits(x)
        baseline = self.baseline(x)
        return logits, baseline


class RLlibEPNModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        sa_iters = model_config.get("custom_model_config", {}).get("sa_iterations", 6)
        input_dim = model_config.get("custom_model_config", {}).get("input_dim", 768)
        
        self.embed = OpenWorld()
        
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
            num_gpus=1,             
            num_cpus_per_worker=1 
        )
        .env_runners(
            num_env_runners=60,  
            rollout_fragment_length=200 
        )
        .training(
            train_batch_size=12000,
            lr=0.0001,
            model={
                "custom_model": "epn_rllib_model",
                "custom_model_config": {
                    "sa_iterations": 4, 
                    "input_dim": 768
                }
            }
        )
    )

    AZURE_URI = "az://ray-checkpoints/epn-massive-run"

    tune.run(
        "IMPALA",
        config=config.to_dict(),
        stop={"timesteps_total": 285000000},
        checkpoint_freq=100,
        
        storage_path=AZURE_URI, 
        
        sync_config=SyncConfig(
            sync_period=300,      # Sync to Azure every 5 minutes (standard)
            sync_artifacts=True   # Ensures MLflow artifacts also move to the cloud
        ),
        
        callbacks=[
            MLflowLoggerCallback(experiment_name="epn-massive-azure")
        ]
    )