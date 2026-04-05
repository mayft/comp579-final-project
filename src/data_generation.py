import numpy as np
import math
#from tqdm import tqdm
#import torch


'''
hexgrid coordinate system
odd row:    even row:
     * *    * *
   * 0 *    * 0 *
     * *    * * 
     
    \|\|\|
    -0-0-0-
    |/|/|/
   -0-0-0-
    |\|\|
'''

'''
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
class Environment:
    def __init__(self, side_length=3,add_wall=False,seed=123, hidden=0,possible_states=64,render=False):
        self.rng = np.random.default_rng(seed=seed)
        self.len = side_length
        self.action_map={0:'NE',1:'E',2:'SE',3:'SW',4:'W',5:'NW'}
        self.actions = {v:k for k,v in self.action_map.items()}
        self.states = {}
        self.locations = []
        self.edges = []
        self.hidden = hidden
        self.walls= add_wall
        self.wall = []
        self.obs_edges = []
        self.hidden_edges = []
        self.obs_locations = []
        self.hidden_locations = []
        self.observations = []
        self.unseen_prob =1#0.68 if add_wall else 1
        self.possible_states = [i for i in range(possible_states) if i % 3]
        self.test_states = [i for i in range(possible_states) if not i % 3]
        self.render_mode = render
        self.directions = [0,1,2]

        self.window=None
        self.window_size=500
        self.clock=None
        self.size=2*side_length-1
        self._agent_location=side_length*100
        self._target_location=(np.array([1,1]))
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

        width = side_length
        offset = side_length
        if side_length%2:
            offset-=0.5
        for i in range(2*side_length-1):
            for j in range(math.floor(offset-width/2), math.floor(offset+width/2)):
                loc = j*100+i
                self.locations.append(loc)
                self.states[loc] = np.zeros(6)
                idx = (j-1)*100+i
                if idx in self.states.keys():
                    self.states[loc][4] = idx
                    self.states[idx][1] = loc

                if j != math.floor(offset-width/2):
                    self.edges.append(((j-1)*100+i,loc))

                if i !=0:
                    corner = j-1+(i%2)
                    if j!=math.floor(offset-width/2) or i >= side_length:
                        self.edges.append((corner*100+i-1,loc))
                        idx = corner*100+i-1
                        if idx in self.states.keys():
                            self.states[loc][5] = idx
                            self.states[idx][2] = loc
                    if j+1 < math.floor(offset+width/2) or i >= side_length:
                        self.edges.append(((corner+1)*100+i-1,loc))
                        idx=(corner+1)*100+i-1
                        if idx in self.states.keys():
                            self.states[loc][0] = idx
                            self.states[idx][3] = loc

            if i >= side_length-1:
                width-=1
            else:
                width+=1

    def build_wall(self):
        start = self.rng.choice(self.locations)
        self.wall = [start]
        d = self.rng.integers(1,[6,3,4,4])
        d[1]  = (d[0]+2+self.rng.integers(3))%6
        s = start
        for i in range(d[2]):
            s = self.states[s][d[0]]
            if s in self.locations:
                self.wall.append(s)
            else:
                break
        s = start
        for i in range(d[3]):
            s = self.states[s][d[1]]
            if s in self.locations:
                self.wall.append(s)
            else:
                break

    def filter_observable(self,num_hidden):
        self.obs_edges = self.edges.copy()
        self.obs_locations = self.locations.copy()
        self.hidden_locations = []
        self.hidden_edges = []
        if num_hidden == 0:
            return
        num_hidden=self.rng.integers(num_hidden)+1
        for i in range(num_hidden):
            v = self.obs_locations.pop()
            self.hidden_locations.append(v)
        for (a,b) in self.obs_edges:
            if a in self.hidden_locations or b in self.hidden_locations:
                self.obs_edges.remove((a,b))
                if a not in self.hidden_locations or b not in self.hidden_locations:
                    self.hidden_edges.append((a,b))

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
        self.observations = dict(zip(self.locations, s))

    def mst(self):
        V = len(self.locations)
        weights = sorted(self.rng.random(len(self.obs_edges)))
        edges = zip(weights,self.rng.permutation(self.obs_edges).tolist())
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

    def generate_memory_bank(self,n_samples,test=False):
        banks = []
        mask = np.zeros([n_samples,len(self.locations)])
        for i in range(n_samples):
            self.filter_observable(self.hidden)
            edges = self.mst()
            self.memory = edges

            unseen = list(set(self.obs_edges)-set(edges))

            self.sample_env(test)
            if self.walls:
                self.build_wall()
            #trans = [self.get_transition(e) for e in edges]
            trans = []
            for j,e in enumerate(edges):
                t= self.get_transition(e)

                if t[0] not in self.wall:
                    if t[2] in self.wall:
                        t[2] = t[0]
                    trans.append(np.hstack([self.observations[t[0]], [t[1]], self.observations[t[2]]]))
                else:
                    mask[i,j]=1
                    trans.append([0]*3)
            order = self.rng.permutation(len(trans))
            trans = np.array(trans)[order]
            mask[i,:-1]= mask[i,:-1][order]

            pick = self.rng.random()
            if pick < self.unseen_prob:
                #q = self.rng.choice(unseen, axis=0)
                t_set = unseen
            elif pick < 0.83:
                #q = self.rng.choice(edges, axis=0)
                t_set = edges
            else:
                t_set = self.hidden_edges

            q = self.rng.choice(t_set, axis=0)

            while q[0] in self.wall and q[1] in self.wall:
                q = self.rng.choice(t_set, axis=0)

            q = self.get_transition(q)

            if q[0] in self.wall:
                q[0] = q[2]
                q[1] = (q[1]+3)%6
            elif q[2] in self.wall:
                q[2] = q[0]
            q = np.hstack([self.observations[q[0]], [q[1]], self.observations[q[2]]])

            trans = np.concat([trans,[q]],axis=0)
            banks.append(trans)
            self.bank=trans
        return np.array(banks),mask==1

    def move(self,action):
        new_state = self.states[self._agent_location][action]
        if new_state:
            self._agent_location=new_state

    def render(self):
        import pygame
        if self.render_mode:
            running = True
            while running:
                self._render_frame()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            return None
                        if event.key == pygame.K_n:
                            self.generate_memory_bank(1)
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

    def coords(self,state):
        coords= np.array([state//100,state%100],dtype=float)
        if coords[1]%2 and self.len%2:
            coords[0]+=0.5
        elif not coords[1]%2 and not self.len%2:
            coords[0]-=0.5
        return coords+0.5
    def _render_frame(self):
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
                self.window_size // (self.size)
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

            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    square * pix_square_size,
                    (pix_square_size, pix_square_size),
                ),
            )

        for state in self.wall:
            pygame.draw.rect(
                canvas,
                (50,50,50),
                pygame.Rect(
                    (self.coords(state)-0.5) * pix_square_size,
                    (pix_square_size, pix_square_size),
                ),
            )

        for s, edges in self.states.items():
            start = self.coords(s)*pix_square_size
            for e in edges:
                if (s,e) in self.memory or (e,s) in self.memory:
                    end = self.coords(e)*pix_square_size
                    pygame.draw.line(
                        canvas,
                        0,
                        start,
                        end,
                        width=3,
                    )

        for i, (s1,e,s2) in enumerate(self.bank):

            a = next((k for k, v in self.observations.items() if v == s1), None)
            b = next((k for k, v in self.observations.items() if v == s2), None)
            color=(0,0,255)
            if i == len(self.bank)-1:
                color = (255,0,0)
            if a==b:
                b= self.states[a][e]
                color = (100,100,100)
            pygame.draw.line(
                canvas,
                color,
                self.coords(a)*pix_square_size,
                self.coords(b)*pix_square_size,
                width=3,
            )

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (100, 0, 50),
            self.coords(self._agent_location) * pix_square_size,
            pix_square_size / 3,
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

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


if __name__ == '__main__':
    env = Environment(side_length=4,render='human',add_wall=True,hidden=5,possible_states=37)
    env.generate_memory_bank(1)
    env.render()




