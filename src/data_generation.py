import numpy as np
import math
#import gymnasium as gym
#from gymnasium.core import ObsType, ActType, RenderFrame
#from gymnasium.spaces import Discrete, Graph
from enum import Enum
import torch
from joblib import Parallel, delayed

'''
hexgrid coordinate system
odd row:    even row:
     * *    * *
   * 0 *    * 0 *
     * *    * * 

env:
    arch:
        physical grid (states and transitions)
        allowable actions
        hyperparameters
    instance:
        state mapping
        action mapping
        
        reward function
        terminal states
        current state
        *goal state
        
        *hidden,unseen,seen partition
        *filter function
    
    run:
        reset -> new env
        step
        initial
        
    sample:
        new env - state, action, wall, filter
        mst
        generate memory bank
        generate query
        generate sample

training:
    eswm:
        base
        output layers
        embedding
        
        padding mask
        query mask
        output mask
        input shaping - binary states
        loss fn
        
    epn:
        impala
        env wrapper
        epn agent
        
    Dyna:
        eswm model - modified??
        Dyna model
        train loop
                 
modelling:
    display:
         grid image fn
         pygame update state for video?
    record:
        loss
        accuracies
    experiments:
        navigation
        adaptability
        planning 
    
'''

class Actions(Enum):
    NE = 0
    E = 1
    SE = 2
    SW = 3
    W = 4
    NW = 5

class Environment():
    def __init__(self, side_length=3,add_wall=False,seed=123, hidden=0,possible_states=64,render=None,query_all=False):
        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.len = side_length
        self.hidden = hidden
        self.has_walls = add_wall
        self.render_mode = render
        self.num_states = 0
        #self.action_map={0:'NE',1:'E',2:'SE',3:'SW',4:'W',5:'NW'}
        #self.actions = {v:k for k,v in self.action_map.items()}
        self.states = {}
        self.locations = []
        self.edges = []

        self.wall = []
        self.obs_edges = []
        self.hidden_edges = []
        self.seen = []
        self.unseen = []
        self.observations = {}


        self.possible_states = range(possible_states) if possible_states<60 else [i for i in range(possible_states) if i % 3]
        self.test_states = [] if possible_states<60 else [i for i in range(possible_states) if not i % 3]
        self.query_probs = [0.68,0.15,0.17] if query_all else [1,0,0]
        self.make_grid()

        self.window=None
        self.window_size=600
        self.clock=None
        self.size=(2*side_length-1)*1.6
        self._agent_location=side_length*100
        self._target_location=100
        self.step_limit = None
        self.step_count = 0
        self.reset()

        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.action_space =None#Discrete(6)
        self.observation_space = None#Discrete(self.num_states)

    def make_grid(self):
        width = self.len
        offset = self.len
        if self.len % 2:
            offset -= 0.5
        for i in range(2 * self.len - 1):
            for j in range(math.floor(offset - width / 2), math.floor(offset + width / 2)):
                loc = j * 100 + i
                self.locations.append(loc)
                self.states[loc] = np.zeros(6)
                idx = (j - 1) * 100 + i
                if idx in self.states.keys():
                    self.states[loc][4] = idx
                    self.states[idx][1] = loc

                if j != math.floor(offset - width / 2):
                    self.edges.append(((j - 1) * 100 + i, loc))

                if i != 0:
                    corner = j - 1 + (i % 2)
                    if j != math.floor(offset - width / 2) or i >= self.len:
                        self.edges.append((corner * 100 + i - 1, loc))
                        idx = corner * 100 + i - 1
                        if idx in self.states.keys():
                            self.states[loc][5] = idx
                            self.states[idx][2] = loc
                    if j + 1 < math.floor(offset + width / 2) or i >= self.len:
                        self.edges.append(((corner + 1) * 100 + i - 1, loc))
                        idx = (corner + 1) * 100 + i - 1
                        if idx in self.states.keys():
                            self.states[loc][0] = idx
                            self.states[idx][3] = loc

            if i >= self.len - 1:
                width -= 1
            else:
                width += 1
        self.num_states = len(self.locations)

    def build_wall(self,test):
        start = self.rng.choice(self.locations)
        wall = [start]
        #direction 1, direction 2, length 1, length 2
        d = self.rng.integers(1,[6,3,6,4])
        d[1]  = (d[0]+2+self.rng.integers(3))%6
        s = start
        for i in range(d[2]):
            s = self.states[s][d[0]]
            if s in self.locations:
                wall.append(s)
            else:
                break
        if test:
            s = start
            for i in range(d[3]):
                s = self.states[s][d[1]]
                if s in self.locations:
                    wall.append(s)
                else:
                    break
        self.wall=wall
        return wall

    def set_wall(self,wall_locations):
        self.wall = wall_locations

    def filter_observable(self,num_hidden):
        self.obs_edges = self.edges.copy()
        
        if num_hidden == 0: return self.obs_edges,None
        #observable = list(filter(lambda x: x not in self.wall,self.locations))
        #observable = list(set(self.locations) - set(self.wall))
        num_hidden=self.rng.integers(num_hidden)+1
        hidden_locations = self.rng.choice(self.locations,num_hidden,replace=False)
        hidden_edges = [(a,b) for (a,b) in obs_edges if (a in hidden_locations)!= (b in hidden_locations)]
        obs_edges = list(set(obs_edges) - set(hidden_edges))
        self.obs_edges=obs_edges
        self.hidden_edges = hidden_edges
        return obs_edges,hidden_edges
        #ignore = []
        #for (a,b) in self.obs_edges:
        #    if a in hidden_locations or b in hidden_locations:
        #        self.hidden_edges.append((a,b))
        #        if a in hidden_locations and b in hidden_locations:
        #            ignore.append((a,b))
        #self.obs_edges = list(filter(lambda x: x not in self.hidden_edges,self.obs_edges))
        #self.hidden_edges = list(filter(lambda x: x not in ignore, self.hidden_edges))

    def get_transition(self,e):

        actions = []
        a,b= e
        a1 = a % 100
        a2 = a // 100
        b1 = b % 100
        b2 = b //100
        #a = self.observations[a]
        #b = self.observations[b]
        if a1 == b1:
            if a2 < b2:
                actions = [1,4]#[(a,1,b),(b,4,a)]
            else:
                actions = [4,1]#[(a, 4, b), (b, 1, a)]
        else:
            if a1 % 2 ==0:
                b2+=1

            if a1 < b1:
                if a2 == b2:
                    actions= [3,0]#[(a,3,b),(b,0,a)]
                else:
                    actions = [2,5]#[(a, 2, b), (b, 5, a)]
            else:
                if a2 == b2:
                    actions= [5,2]#[(a,5,b),(b,2,a)]
                else:
                    actions = [0,3]#[(a, 0, b), (b, 3, a)]

        if self.rng.integers(2):
            return [b,actions[1],a]
        else:
            return [a, actions[0], b]

    def sample_env(self,test):
        if test:
            s= self.rng.permutation(self.test_states)
        else:
            s = self.rng.permutation(self.possible_states)
        observations = dict(zip(self.locations, s))
        self.observations=observations
        return observations

    def mst(self,obs_edges):
        V = self.num_states
        weights = sorted(self.rng.random(len(obs_edges)))
        edges = zip(weights,self.rng.permutation(obs_edges).tolist())
        parent = list(range(V))
        rank = [1]*V
        cost = []
        count = 0

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        for w, (a,b) in edges:
            x = self.locations.index(a)
            y = self.locations.index(b)
            # Make sure that there is no cycle
            if find(x) != find(y):
                #union
                s1 = find(x)
                s2 = find(y)
                if s1 != s2:
                    if rank[s1] < rank[s2]:
                        parent[s1] = s2
                    elif rank[s1] > rank[s2]:
                        parent[s2] = s1
                    else:
                        parent[s2] = s1
                        rank[s1] += 1
                cost.append((a,b))
                count += 1
                if count == V - 1:
                    break
        return cost

    def new_environment(self,test_states=False):
        obs,hidden = self.filter_observable(self.hidden)
        if self.has_walls:
            states= self.sample_env(test=False)
            wall = self.build_wall(test_states)
        else:
            states=self.sample_env(test_states)
            wall=[]
        self.wall = wall
        return obs,hidden,states,wall

    def generate_memory_bank(self,obs_edges,observations,wall):
        paths = self.mst(obs_edges)
        seen = paths

        trans = []
        for e in paths:
            t= self.get_transition(e)

            if t[0] not in wall:
                if t[2] in wall:
                    t[2] = t[0]
                trans.append(np.hstack([observations[t[0]], [t[1]], observations[t[2]]]))

        unseen = list(set(obs_edges)-set(seen))
        trans = self.rng.permutation(trans)
        self.seen = seen
        self.unseen= unseen
        return trans,seen,unseen

    def get_query(self,transitions,observations):
        '''if transition_set == 'unseen':
            transitions = self.unseen
        elif transition_set == 'seen':
            transitions = self.seen
        else:
            transitions = self.hidden_edges'''
        q = self.rng.choice(transitions, axis=0)

        while q[0] in self.wall and q[1] in self.wall:
            q = self.rng.choice(transitions, axis=0)

        q = self.get_transition(q)

        if q[0] in self.wall:
            q[0] = q[2]
            q[1] = (q[1] + 3) % 6
        elif q[2] in self.wall:
            q[2] = q[0]

        q = np.hstack([observations[q[0]], [q[1]], observations[q[2]]])
        return q
            
    def sample_environments(self,n_samples,kind='unseen',test=False,jobs=2):
        banks = []
        for i in range(n_samples):
            #banks = Parallel(n_jobs=jobs)(delayed(self.fn)(kind,test) for _ in range(n_samples))
            obs,hidden,states,wall=self.new_environment(test)
            bank,seen,unseen = self.generate_memory_bank(obs,states,wall)
            if kind == 'unseen':
                transitions =unseen
            elif kind == 'seen':
                transitions = seen
            else:
                transitions = hidden
            query = self.get_query(transitions,states)
            bank = np.concatenate([bank, [query]], axis=0)
            banks.append(bank)
        return (banks)

    def move(self,action):
        new_state = self.states[self._agent_location][action]
        if new_state and new_state not in self.wall :
            self._agent_location=new_state

    def render(self):
        import pygame
        query = None
        if self.render_mode:
            running = True
            while running:
                self._render_frame(query)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            return
                        if event.key == pygame.K_n:
                            self.reset()
                            self.generate_memory_bank()
                        if event.key == pygame.K_w:
                            self.move(5)
                        if event.key == pygame.K_e:
                            self.move(0)
                        if event.key == pygame.K_d:
                            self.move(1)
                        if event.key == pygame.K_x:
                            self.move(2)
                        if event.key == pygame.K_z:
                            self.move(3)
                        if event.key == pygame.K_a:
                            self.move(4)
                        if event.key == pygame.K_3:
                            query = self.get_query('unsolvable')
                            print(query)
                        if event.key == pygame.K_1:
                            query = self.get_query('seen')
                            print(query)
                        if event.key == pygame.K_2:
                            query = self.get_query('unseen')
                            print(query)

    def coords(self,state,hexes=False):
        coords= np.array([state//100,state%100],dtype=float)
        if coords[1]%2 and self.len%2:
            coords[0]+=0.5
        elif not coords[1]%2 and not self.len%2:
            coords[0]-=0.5

        coords[1]*=1.5
        coords[0]*=1.5
        if not hexes:
            coords+=1
            coords[0]+=0.25
            return coords
        return coords+0.5
    
    def hex_points(self, state,side_len):
        x,y= self.coords(state,True)*side_len
        x-=0.5
        y-=0.5
        width= side_len*1.5
        points = [(x,y),
                  (x+width/2,y-side_len/2),
                  (x+width,y),
                  (x+width,y+side_len),
                  (x+width/2,y+1.5*side_len),
                  (x,y+side_len),
                  ]
        return points

    def _render_frame(self,query):
        import pygame
        if self.window is None and self.render_mode == "human":
            pygame.init()
            #pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size // self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        # Convert [row, col] to pygame (x, y) by reversing the coordinates
        for loc in self.locations:
            square = self.coords(loc)-0.5
            color = [0,120,0]
            if (loc//100)%2:
                color[1] +=50
            if loc%2:

                color[1]-=20

            '''pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    square * pix_square_size,
                    (pix_square_size, pix_square_size),
                ),)'''
            pygame.draw.polygon(
                canvas,
                color,
                points=self.hex_points(loc,pix_square_size)
            )
            

        for state in self.wall:
            pygame.draw.polygon(
                canvas,
                (50,50,50),
                points=self.hex_points(state,pix_square_size)
            )
        pygame.draw.polygon(
                canvas,
                (200,0,0),
                points=self.hex_points(self._target_location,pix_square_size)
            )
        '''for s, edges in self.states.items():
            start = self.coords(s)*pix_square_size
            for e in edges:
                if e:#(s,e) in self.seen or (e,s) in self.seen:
                    end = self.coords(e)*pix_square_size
                    pygame.draw.line(
                        canvas,
                        0,
                        start,
                        end,
                        width=3,
                    )'''

        for (s1,s2) in self.seen:
            if s1 not in self.wall or s2 not in self.wall:
                #a = next((k for k, v in self.observations.items() if v == s1), None)
                #b = next((k for k, v in self.observations.items() if v == s2), None)
                a=s1
                b=s2
                color=(0,0,255)
                pygame.draw.line(
                    canvas,
                    color,
                    self.coords(a)*pix_square_size,
                    self.coords(b)*pix_square_size,
                    width=3,
                )
        '''for (s1,s2) in self.unseen:
            if s1 not in self.wall or s2 not in self.wall:
                a = s1#next((k for k, v in self.observations.items() if v == s1), None)
                b = s2#next((k for k, v in self.observations.items() if v == s2), None)
                color=(100,100,0)
                pygame.draw.line(
                    canvas,
                    color,
                    self.coords(a)*pix_square_size,
                    self.coords(b)*pix_square_size,
                    width=3,
                )'''
        for (s1,s2) in self.hidden_edges:
            if s1 not in self.wall or s2 not in self.wall:
                a = s1#next((k for k, v in self.observations.items() if v == s1), None)
                b = s2#next((k for k, v in self.observations.items() if v == s2), None)
                color=(100,100,100)
                pygame.draw.line(
                    canvas,
                    color,
                    self.coords(a)*pix_square_size,
                    self.coords(b)*pix_square_size,
                    width=3,
                )
        for (a,b) in [(205,305)]:
            color=(200,0,0)
            pygame.draw.line(
                canvas,
                color,
                self.coords(a)*pix_square_size,
                self.coords(b)*pix_square_size,
                width=3,
            )
        for (a,b) in [(206,205)]:
            color=(0,0,200)
            pygame.draw.line(
                canvas,
                color,
                self.coords(a)*pix_square_size,
                self.coords(b)*pix_square_size,
                width=3,
            )
        
        if query is not None:
            a = next((k for k, v in self.observations.items() if v == query[0]), None)
            b = next((k for k, v in self.observations.items() if v == query[2]), None)

            pygame.draw.line(
                canvas,
                (255,0,0),
                self.coords(a) * pix_square_size,
                self.coords(b) * pix_square_size,
                width=3,
            )
        # Now we draw the agent
        c=(self.coords(self._agent_location)+0.5)
        c[0]+=0.25
        pygame.draw.circle(
            canvas,
            (100, 0, 50),
            self.coords(self._agent_location) * pix_square_size,
            pix_square_size *0.6,
        )

        running = True
        if self.render_mode == "human":

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            #pygame.event.pump()



            pygame.display.update()


            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

        '''else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )'''

    def _get_obs(self):
        return {'agent':self.observations[self._agent_location],'target':self.observations[self._target_location]}

    def reset(self,seed=None,options=None,new_env=True):
        #super().reset(seed=seed)
        self.step_count=0
        if new_env:
            self.new_environment()

        # when implementing ensure states are different and allowed
        self._agent_location = self.rng.choice(self.locations)
        while self._agent_location in self.wall or self._target_location == self._agent_location:
            self._agent_location = self.rng.choice(self.locations)
        if new_env:
            self._target_location = self.rng.choice(self.locations)
            while self._target_location in self.wall or self._target_location == self._agent_location:
                self._target_location = self.rng.choice(self.locations)
        observation = self._get_obs()
        info = None

        return observation,info

    def step(self, action):
        self.move(action)
        self.step_count +=1
        terminated = (self._agent_location==self._target_location)
        truncated = (self.step_count >= self.step_limit) if self.step_limit is not None else False
        observation = self._get_obs()
        info = None
        reward = 1 if terminated else -0.01
        return observation,reward,terminated,truncated,info

if __name__ == '__main__':
    env = Environment(side_length=4,render='human',add_wall=True,hidden=0,possible_states=37,query_all=True,seed=1)

    #env = Environment(side_length=4,possible_states=37,render='human')
    #env = Environment(render='human')
    #env.reset()
    env.wall = [3,103,203,303,402]
    env._target_location =102
    env._agent_location = 206 
    #env.generate_memory_bank(env.obs_edges,env.observations,env.wall)
    env.render()




