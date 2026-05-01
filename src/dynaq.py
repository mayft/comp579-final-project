import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Import the environment and models from your provided scripts
from data_generation import Environment 
from eswm import ESWM, RandomWall ,OpenWorld

class ESWMDynaQAgent:
    def __init__(self, num_states, num_actions, eswm_model, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning=5,seed=123):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n_planning
        self.eswm = eswm_model
        
        self.memory_bank = []
        self.observed_states = []
        self.observed_actions = {}
        self.rng = np.random.default_rng(seed=seed)
        
    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.Q.shape[1])
        return np.argmax(self.Q[state])
        
    def step(self, state, action, reward, next_state, target_loc, device='cpu'):
        
        best_next_a = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_a]
        self.Q[state, action] += self.alpha * (td_target - self.Q[state, action])
        
        transition = ([state, action, next_state])
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
        '''for _ in range(self.n):
            sim_s = self.rng.choice(self.observed_states)
            sim_a = self.rng.choice(self.observed_actions[sim_s])
            sim_s_prime = self._query_eswm(sim_s, sim_a, device)
            
            sim_reward = 1 if sim_s_prime == target_loc else -0.01
            
            best_sim_next_a = np.argmax(self.Q[sim_s_prime])
            sim_td_target = sim_reward + self.gamma * self.Q[sim_s_prime][best_sim_next_a]
            self.Q[sim_s, sim_a] += self.alpha * (sim_td_target - self.Q[sim_s, sim_a])'''

        sim_s = self.rng.choice(self.observed_states,size=n)
        sim_a = [self.rng.choice(self.observed_actions[s]) for s in sim_s]
        sim_s_prime = self._query_eswm(sim_s,sim_a,device)
        sim_reward = np.where(sim_s_prime==target_loc,1,-0.1)
        for i in range(self.n):
            best_sim_next_a = np.argmax(self.Q[sim_s_prime[i]])
            sim_td_target = sim_reward[i] + self.gamma * self.Q[sim_s_prime[i]][best_sim_next_a]
            self.Q[sim_s[i], sim_a[i]] += self.alpha * (sim_td_target - self.Q[sim_s[i], sim_a[i]])

        


    def _query_eswm(self, start_state, action, device):
        """Formats the memory bank and query for the PyTorch ESWM model."""
        m =torch.tensor(self.memory_bank,dtype=torch.long,device=device)
        query = torch.tensor(np.column_stack([start_state, action, [-1]*len(action)]),dtype=torch.long,device=device).unsqueeze(1)
        x = torch.cat([m.repeat([query.shape[0],1,1]),query],dim=1)   
    
        #seq = self.memory_bank + [query]
        #x = torch.tensor([seq], dtype=torch.long, device=device)
        
        #padding_mask = torch.zeros((1, len(seq)), dtype=torch.bool, device=device)
        
        with torch.no_grad():
            _, _, out_end = self.eswm(x)
        predicted_end_state = torch.argmax(out_end,dim=1).cpu()
        return predicted_end_state

def train_agent(episodes=100, n_planning=10, max_steps=100,s=123):
    device = 'mps' if torch.mps.is_available() else 'cpu'
    env = Environment(side_length=4, add_wall=True, hidden=0, possible_states=37, query_all=True,seed=s)
    env.set_wall([3,103,203,303,402,401])
    #env = Environment()
    eswm_params = {'embedder': RandomWall(), 'input_dim': 1024, 'state_dim': 38, 'num_actions': 7,}
    #eswm_params = {'embedder': OpenWorld()}
    eswm_model = ESWM(num_layers=10, **eswm_params).to(device)
    model_weights= torch.load('tests/ESWM-T10-R-159999.pth',weights_only=True,map_location=device)
    eswm_model.load_state_dict(model_weights['model_state_dict'])
    eswm_model.to(device)
    
    agent = ESWMDynaQAgent(num_states=env.num_states, num_actions=6, eswm_model=eswm_model, n_planning=n_planning,seed=s)
    steps_per_episode = []
    reward_per_episode = []

    for ep in tqdm(range(episodes)):

        obs, _ = env.reset(new_env=False)
        env._target_location=102
        env._agent_location=206
        obs=env._get_obs()
        state = obs['agent']
        target_loc = obs['target']
        done = False
        steps = 0
        total_reward=0
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward+=reward
            next_state = next_obs['agent']
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, target_loc, device)
            state = next_state
            steps += 1
        reward_per_episode.append(total_reward)    
        steps_per_episode.append(steps)
        if ep % 10 == 0:
            print(f"Episode {ep}/{episodes} - Steps to goal: {steps}, reward: {total_reward} {obs['agent']}->{obs['target']}")    
    '''plt.figure(figsize=(12, 6))
    plt.plot(steps_per_episode, label=f'ESWM-Dyna-Q (n={n_planning})', color='#4A90E2', linewidth=2)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Steps to Goal', fontsize=12)
    plt.title('Dyna-Q Training Convergence via ESWM Planning', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()'''
    return steps_per_episode

if __name__ == "__main__":
    steps=pd.DataFrame()
    #steps.append(train_agent(episodes=100, n_planning=0,max_steps=20,s=123))
    #steps.append(train_agent(episodes=100, n_planning=1,max_steps=200,s=123))
    #steps.append(train_agent(episodes=100, n_planning=5,max_steps=200,s=123))
    #sns.lineplot(steps)
    #plt.savefig('dyna.png',bbox_inches='tight',dpi=500)
    for n in [1]:
        for i in range(1):
            steps[f'{123+i}',n]=(train_agent(episodes=100, n_planning=n,max_steps=200,s=123+i))
    steps = steps.melt(var_name='vars',value_name='steps',ignore_index=False)
    steps['seed'],steps['n'] = zip(*steps['vars'])
    #steps.to_csv('dynaq_data')
    sns.lineplot(steps,y='steps',x=steps.index,hue='n')
    #plt.savefig('dyna.png',bbox_inches='tight',dpi=500)
