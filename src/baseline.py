import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from data_generation import Environment

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

        s_prev = history_sequence[:, :, 0]
        a_prev = history_sequence[:, :, 1]
        s_curr = history_sequence[:, :, 2]
        s_goal = history_sequence[:, :, 3]
        
        e_s_prev = self.embed_s_prev(s_prev)
        e_a_prev = self.embed_a_prev(a_prev)
        e_s_curr = self.embed_s_curr(s_curr)
        e_s_goal = self.embed_s_goal(s_goal)
        merged_obs = (e_s_prev + e_a_prev + e_s_curr + e_s_goal) / 4.0
        
        lstm_out, _ = self.lstm(merged_obs)
        
        final_hidden = lstm_out[:, -1, :] 
        
        action_logits = self.actor_head(final_hidden)
        state_value = self.critic_head(final_hidden)
        
        return action_logits, state_value

def train_baseline_a2c():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = Environment(side_length=4, add_wall=True, hidden=0, possible_states=37)
    env.reset()
    
    loc_to_idx = {loc: idx for idx, loc in enumerate(env.locations)}
    
    num_actions = 6
    dummy_action = num_actions
    
    model = BaselineA2CAgent(num_states=len(env.locations), num_actions=num_actions).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    episodes = 5000 
    gamma = 0.99  
    max_steps = 100
    
    episode_rewards = []
    success_rates = []
    recent_successes = []


    for ep in range(episodes):
        valid_locations = [loc for loc in env.locations if loc not in env.wall]
        env._agent_location = np.random.choice(valid_locations)
        env._target_location = np.random.choice(valid_locations)
        
        s_curr = env._agent_location
        s_goal = env._target_location
        
        history = []
        s_prev = s_curr
        a_prev = dummy_action
        
        log_probs = []
        values = []
        rewards = []
        
        ep_reward = 0
        
        for step in range(max_steps):
            history.append([loc_to_idx[s_prev], a_prev, loc_to_idx[s_curr], loc_to_idx[s_goal]])
            
            history_tensor = torch.tensor([history], dtype=torch.long).to(device)
            
            logits, value = model(history_tensor)
            
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            env.move(action.item())
            s_next = env._agent_location
            
            terminated = (s_next == s_goal)
            reward = 1.0 if terminated else -0.01
            
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            ep_reward += reward
            
            if terminated:
                break
                
            s_prev = s_curr
            a_prev = action.item()
            s_curr = s_next
            
        episode_rewards.append(ep_reward)
        recent_successes.append(1 if terminated else 0)
        
        if len(recent_successes) > 100:
            recent_successes.pop(0)
        success_rates.append(sum(recent_successes) / len(recent_successes))
        
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.cat(values).squeeze(-1)
        log_probs = torch.cat(log_probs)
        
        advantages = returns - values
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (ep + 1) % 500 == 0:
            print(f"Episode {ep + 1}/{episodes} | Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f} | Success Rate: {success_rates[-1]:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(success_rates, color='orange', label='Baseline A2C Success Rate')
    plt.title('Baseline A2C Navigation Training (Single Environment)')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (Rolling Window 100)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model

if __name__ == "__main__":
    trained_baseline = train_baseline_a2c()