import os
import csv
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from data_generation import Environment

# Force Pygame to run headlessly just in case env.render() is called
os.environ["SDL_VIDEODRIVER"] = "dummy"

class RandomWall(nn.Module):
    def __init__(self):
        super(RandomWall, self).__init__()
        self.source_embed = nn.Embedding(39, 1024, padding_idx=0)
        self.action_embed = nn.Embedding(8, 1024, padding_idx=0)
        self.end_embed = nn.Embedding(39, 1024, padding_idx=0)

    def forward(self, x, mask=None):
        x = x + 2
        start = self.source_embed(x[:, :, 0])
        action = self.action_embed(x[:, :, 1])
        end = self.end_embed(x[:, :, 2])
        return torch.mean(torch.stack([start, action, end], dim=0), dim=0)

class ESWM(nn.Module):
    def __init__(self, embedder, num_layers=2, input_dim=1024, num_heads=8, feedforward=2048, dropout=0.1, state_dim=38, num_actions=7):
        super(ESWM, self).__init__()
        self.embedding = embedder
        self.transformer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=feedforward, dropout=dropout, batch_first=True)
        self.model = nn.TransformerEncoder(encoder_layer=self.transformer, num_layers=num_layers)
        self.source = nn.Linear(input_dim, state_dim)
        self.action = nn.Linear(input_dim, num_actions)
        self.end = nn.Linear(input_dim, state_dim)

    def forward(self, x, padding_mask, transition_mask=None):
        src = self.embedding(x)
        src = self.model(src, src_key_padding_mask=padding_mask)
        query = src[:, -1]
        return self.source(query), self.action(query), self.end(query)

class BaselineA2CAgent(nn.Module):
    def __init__(self, num_states=39, num_actions=6):
        super(BaselineA2CAgent, self).__init__()
        self.embed_s_prev = nn.Embedding(num_states, 128)
        self.embed_a_prev = nn.Embedding(num_actions + 1, 128)
        self.embed_s_curr = nn.Embedding(num_states, 128)
        self.embed_s_goal = nn.Embedding(num_states, 128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.actor_head = nn.Linear(256, num_actions)
        self.critic_head = nn.Linear(256, 1)
        
    def forward(self, history_sequence):
        e_s_prev = self.embed_s_prev(history_sequence[:, :, 0])
        e_a_prev = self.embed_a_prev(history_sequence[:, :, 1])
        e_s_curr = self.embed_s_curr(history_sequence[:, :, 2])
        e_s_goal = self.embed_s_goal(history_sequence[:, :, 3])
        merged_obs = (e_s_prev + e_a_prev + e_s_curr + e_s_goal) / 4.0
        lstm_out, _ = self.lstm(merged_obs)
        final_hidden = lstm_out[:, -1, :] 
        return self.actor_head(final_hidden), self.critic_head(final_hidden)

class ESWMDynaQAgent:
    def __init__(self, env, num_states, num_actions, eswm_model, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=5):
        self.env = env
        self.Q = {}
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n_planning
        self.eswm = eswm_model
        self.memory_bank = []
        self.observed_states = []
        self.observed_actions = {}
        
    def get_q(self, state, target_loc):
        state_key = (state, target_loc)
        if state_key not in self.Q: 
            self.Q[state_key] = np.zeros(self.num_actions)
        return self.Q[state_key]
        
    def select_action(self, state, target_loc):
        if np.random.rand() < self.epsilon: 
            return np.random.choice(self.num_actions)
        return np.argmax(self.get_q(state, target_loc))
        
    def step(self, state, action, reward, next_state, target_loc, device='cpu'):
        q_next = self.get_q(next_state, target_loc)
        td_target = reward + self.gamma * np.max(q_next)
        
        # Update goal-conditioned Q-value
        self.get_q(state, target_loc)[action] += self.alpha * (td_target - self.get_q(state, target_loc)[action])
        
        transition = [state, action, next_state]
        if transition not in self.memory_bank:
            self.memory_bank.append(transition)
            if state not in self.observed_states:
                self.observed_states.append(state)
                self.observed_actions[state] = []
            if action not in self.observed_actions[state]:
                self.observed_actions[state].append(action)
                
        if len(self.observed_states) > 0 and self.n > 0:
            self._plan(target_loc, device)
            
    def _plan(self, target_loc, device):
        self.eswm.eval()
        for _ in range(self.n):
            sim_s = np.random.choice(self.observed_states)
            sim_a = np.random.choice(self.observed_actions[sim_s])
            
            sim_s_prime = self._query_eswm(sim_s, sim_a, device)
            sim_reward = 1.0 if sim_s_prime == target_loc else -0.01
            
            q_sim_next = self.get_q(sim_s_prime, target_loc)
            sim_td_target = sim_reward + self.gamma * np.max(q_sim_next)
            
            # Update goal-conditioned Q-value
            self.get_q(sim_s, target_loc)[sim_a] += self.alpha * (sim_td_target - self.get_q(sim_s, target_loc)[sim_a])

    def _query_eswm(self, start_state, action, device):
        mapped_memory = [[self.env.observations[s1], a, self.env.observations[s2]] for s1, a, s2 in self.memory_bank]
        query = [self.env.observations[start_state], action, -1] 
        
        seq = mapped_memory + [query]
        x = torch.tensor([seq], dtype=torch.long, device=device)
        padding_mask = torch.zeros((1, len(seq)), dtype=torch.bool, device=device)
        
        with torch.no_grad():
            _, _, out_end = self.eswm(x, padding_mask)
            
        predicted_obs = torch.argmax(out_end[0, -1]).item()
        obs_to_loc = {obs: loc for loc, obs in self.env.observations.items()}
        return obs_to_loc.get(predicted_obs, start_state)

def run_sample_efficiency_test(env, eswm_model, episodes=100, max_steps=100, device='cpu'):
    n_values = [0, 5, 20]
    all_results = {}
    
    for n in n_values:
        print(f"Training ESWM-Dyna-Q with n={n}...")
        agent = ESWMDynaQAgent(env, env.num_states, 6, eswm_model, n_planning=n)
        steps_history = []
        for ep in range(episodes):
            valid_locs = [l for l in env.locations if l not in env.wall]
            start, target = np.random.choice(valid_locs, size=2, replace=False)
            env._agent_location = start
            env._target_location = target
            state, target = env._agent_location, env._target_location
            steps = 0
            while state != target and steps < max_steps:
                action = agent.select_action(state, target)
                env.move(action)
                next_state = env._agent_location
                reward = 1.0 if next_state == target else -0.01
                agent.step(state, action, reward, next_state, target, device)
                state = next_state
                steps += 1
            steps_history.append(steps)
        all_results[f'n={n}'] = steps_history
        
    plt.figure(figsize=(10, 5))
    with open('sample_efficiency_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'n=0_Steps', 'n=5_Steps', 'n=20_Steps'])
        for i in range(episodes):
            writer.writerow([i+1, all_results['n=0'][i], all_results['n=5'][i], all_results['n=20'][i]])
            
    for n, data in all_results.items():
        smoothed = [np.mean(data[max(0, i-10):i+1]) for i in range(len(data))]
        plt.plot(smoothed, label=f'Dyna-Q ({n})')
        
    plt.title('Sample Efficiency: Impact of ESWM Planning Steps')
    plt.xlabel('Episodes')
    plt.ylabel('Steps to Goal (Smoothed)')
    plt.legend()
    plt.grid(True)
    plt.savefig('sample_efficiency_graph.png', dpi=300)
    plt.close()

def run_adaptability_test(env, baseline_model, eswm_model, trials=100, device='cpu'):
    
    original_wall = list(env.wall)
    obstacle_counts = [1, 2, 3]
    baseline_final_results = []
    eswm_final_results = []
    
    loc_to_idx = {loc: idx for idx, loc in enumerate(env.locations)}

    for num_new in obstacle_counts:
        env.wall = list(original_wall)
        valid_locs = [l for l in env.locations if l not in env.wall]
        new_obstacles = np.random.choice(valid_locs, size=num_new, replace=False)
        env.wall.extend(new_obstacles)
        print(f"Testing {num_new} new obstacles: {new_obstacles}")

        eswm_agent = ESWMDynaQAgent(env, env.num_states, 6, eswm_model, n_planning=100)

        for _ in range(150):
            safe_locs = [l for l in env.locations if l not in env.wall]
            start, goal = np.random.choice(safe_locs, size=2, replace=False)
            env._agent_location = start
            state = start
            for _ in range(50):
                action = eswm_agent.select_action(state, goal) 
                env.move(action)
                next_state = env._agent_location
                reward = 1.0 if next_state == goal else -0.01
                eswm_agent.step(state, action, reward, next_state, goal, device)
                if next_state == goal: break
                state = next_state

        b_success, e_success = [], []
        for _ in range(trials):
            safe_locs = [l for l in env.locations if l not in env.wall]
            start, goal = np.random.choice(safe_locs, size=2, replace=False)
            
            env._agent_location, env._target_location = start, goal
            b_state, b_prev_a = start, 6
            history = []
            success = 0
            for _ in range(50):
                history.append([loc_to_idx.get(b_state,0), b_prev_a, loc_to_idx.get(b_state,0), loc_to_idx.get(goal,0)])
                hist_tensor = torch.tensor([history], dtype=torch.long).to(device)
                logits, _ = baseline_model(hist_tensor)
                action = torch.argmax(logits).item()
                env.move(action)
                if env._agent_location == goal:
                    success = 1; break
                b_prev_a, b_state = action, env._agent_location
            b_success.append(success)
            
            env._agent_location, env._target_location = start, goal
            e_state = start
            success = 0
            for _ in range(50):
                action = eswm_agent.select_action(e_state, goal)
                env.move(action)
                if env._agent_location == goal:
                    success = 1; break
                e_state = env._agent_location
            e_success.append(success)

        baseline_final_results.append(np.mean(b_success))
        eswm_final_results.append(np.mean(e_success))

    plt.figure(figsize=(8, 6))
    plt.plot(obstacle_counts, baseline_final_results, marker='s', label='Baseline (A2C)', color='#1f77b4', linewidth=2)
    plt.plot(obstacle_counts, eswm_final_results, marker='o', label='ESWM (Dyna-Q)', color='#ff7f0e', linewidth=2)
    plt.title('Adaptability', fontsize=14)
    plt.xlabel('Number of Unexpected Obstacles Added', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(obstacle_counts)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Pipeline on: {device}")

    env = Environment(side_length=4, add_wall=True, possible_states=37)
    
    baseline_model = BaselineA2CAgent(num_states=len(env.locations), num_actions=6).to(device)
    if os.path.exists('baseline_a2c.pth'):
        baseline_model.load_state_dict(torch.load('baseline_a2c.pth', map_location=device))
        baseline_model.eval()

    eswm_params = {'embedder': RandomWall(), 'input_dim': 1024, 'state_dim': 38, 'num_actions': 7}
    eswm_model = ESWM(num_layers=10, **eswm_params).to(device)
    
    if os.path.exists('ESWM-T10-R-159999.pth'):
        checkpoint = torch.load('ESWM-T10-R-159999.pth', map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        eswm_model.load_state_dict(state_dict)
        eswm_model.eval()

    
    dyna_q_agent = ESWMDynaQAgent(env, env.num_states, 6, eswm_model, n_planning=10)
    run_adaptability_test(env, baseline_model, eswm_model, device=device)

    try:
        from google.colab import files
        files.download('sample_efficiency_graph.png')
        files.download('sample_efficiency_data.csv')
        files.download('adaptability_graph.png')
        files.download('adaptability_data.csv')
    except ImportError:
        print("\nTests complete. Results saved locally.")