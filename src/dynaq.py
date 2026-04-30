import numpy as np
import torch
import matplotlib.pyplot as plt

# Import the environment and models from your provided scripts
from data_generation import Environment 
from eswm_model import ESWM, RandomWall 

class ESWMDynaQAgent:
    def __init__(self, num_states, num_actions, eswm_model, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=5):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n_planning
        self.eswm = eswm_model
        
        self.memory_bank = []
        self.observed_states = []
        self.observed_actions = {}
        
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.Q.shape[1])
        return np.argmax(self.Q[state])
        
    def step(self, state, action, reward, next_state, target_loc, device='cpu'):
        
        best_next_a = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_a]
        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])
        
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
        """Hallucinates experience by querying the ESWM."""
        self.eswm.eval()
        for _ in range(self.n):
            sim_s = np.random.choice(self.observed_states)
            sim_a = np.random.choice(self.observed_actions[sim_s])
            sim_s_prime = self._query_eswm(sim_s, sim_a, device)
            
            sim_reward = 1 if sim_s_prime == target_loc else -0.01
            
            best_sim_next_a = np.argmax(self.Q[sim_s_prime])
            sim_td_target = sim_reward + self.gamma * self.Q[sim_s_prime][best_sim_next_a]
            self.Q[sim_s, sim_a] += self.alpha * (sim_td_target - self.Q[sim_s, sim_a])

    def _query_eswm(self, start_state, action, device):
        """Formats the memory bank and query for the PyTorch ESWM model."""
        query = [start_state, action, 0] 
        
        seq = self.memory_bank + [query]
        x = torch.tensor([seq], dtype=torch.long, device=device)
        
        padding_mask = torch.zeros((1, len(seq)), dtype=torch.bool, device=device)
        
        with torch.no_grad():
            _, _, out_end = self.eswm(x, padding_mask)
        
        predicted_end_state = torch.argmax(out_end[0, -1]).item()
        return predicted_end_state

def train_agent(episodes=100, n_planning=10, max_steps=150):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = Environment(side_length=4, add_wall=True, hidden=0, possible_states=37, query_all=True)
    
    eswm_params = {'embedder': RandomWall(), 'input_dim': 1024, 'state_dim': 38, 'num_actions': 7}
    eswm_model = ESWM(num_layers=2, **eswm_params).to(device)
    
    agent = ESWMDynaQAgent(num_states=env.num_states, num_actions=6, eswm_model=eswm_model, n_planning=n_planning)
    steps_per_episode = []

    for ep in range(episodes):
        obs, _ = env.reset()
        state = obs['agent']
        target_loc = obs['target']
        
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs['agent']
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, target_loc, device)
            
            state = next_state
            steps += 1
            
        steps_per_episode.append(steps)
        if ep % 10 == 0:
            print(f"Episode {ep}/{episodes} - Steps to goal: {steps}")
        
    plt.figure(figsize=(12, 6))
    plt.plot(steps_per_episode, label=f'ESWM-Dyna-Q (n={n_planning})', color='#4A90E2', linewidth=2)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Steps to Goal', fontsize=12)
    plt.title('Dyna-Q Training Convergence via ESWM Planning', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_agent(episodes=150, n_planning=10)