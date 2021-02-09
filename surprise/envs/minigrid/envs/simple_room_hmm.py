#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

from gym import spaces

from operator import add

import matplotlib.pyplot as plt
import pdb

class SimpleEnemyEnvHMM(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, max_steps=500, agent_pos=None, grid_size=8, goal_pos=None, num_obstacles=1, seed=1327, agent_view_size=5):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.door_color = COLOR_NAMES[1]
        self.n_obstacles = num_obstacles
        self._size=grid_size
        self._viz_size = agent_view_size
        super().__init__(grid_size=self._size, max_steps=max_steps, seed=seed, agent_view_size=self._viz_size)
        self.action_space = gym.spaces.Discrete(4)
        
        # self.observation_space = gym.spaces.Box(low=np.zeros((self._size-2)*(self._size-2)*2+(len(DIR_TO_VEC))), 
        #                                         high=np.ones((self._size-2)*(self._size-2)*2+(len(DIR_TO_VEC))))
        self.observation_space = gym.spaces.Box(low=np.zeros(((self._size)*(2))+(self._viz_size*self._viz_size)), 
                                                high=np.ones(((self._size)*(2))+(self._viz_size*self._viz_size)))
        
        ### State = x loc for agent y loc for agent, x loc for enemey and y loc for enemy.
        self._hmm = np.zeros((self._size, self._size, self._size, self._size))

    def valid_pos(self, pos):
        
        return pos[0] >= 0 and pos[0] <= self._size-1 and pos[1] >= 0 and pos[1] <= self._size-1
    
    def entropy(self, state, ps):
        obs = state
        # ForkedPdb().set_trace()
        thetas = ps

        # For numerical stability, clip probs to not be 0 or 1
        thresh = 1e-5
        thetas = np.clip(thetas, thresh, 1 - thresh)

        # Bernoulli log prob
        probs = obs*thetas + (1-obs)*(1-thetas)
        logprob = np.sum(np.log(probs))
        return logprob
    
    def entropyState(self, state):
        from scipy.stats import entropy
        # For numerical stability, clip probs to not be 0 or 1
        p = np.zeros((self._size, self._size)) + 0.5
        # e = entropy(p, state)
        e = entropy(np.array(state).flatten())
        # Bernoulli log prob
        
        return e
    
     
    def construct_transition_matrix(self, action, obs_pos=None):
        import copy
        # state_ = np.zeros((self._size, self._size, self._size, self._size))
        state_ = np.zeros((self._size, self._size))
        state_agent = np.zeros((self._size, self._size))
        agent_pos_ = copy.deepcopy(self.agent_pos)
        axis = action // 2
        direction = (action % 2) * 2 - 1
        agent_pos_[axis] += direction
        if (self.valid_pos(agent_pos_)):
            state_agent[agent_pos_[0], agent_pos_[1]] = 1
        
        # state_[self.agent_pos[0],self.agent_pos[1]] = 0
        if (obs_pos is None):
            obs_pos = self.obstacles[0].cur_pos
        ### Going to literally need to handle some corner cases here.
        if (obs_pos[0]-1 < 0
            and obs_pos[1]-1 < 0): ## Bottom left corner
            state_[obs_pos[0]+1,obs_pos[1]] = 0.5
            state_[obs_pos[0],obs_pos[1]+1] = 0.5
        elif (obs_pos[0] >= self._size-1
            and obs_pos[1] >=  self._size-1): ## Top Right corner
            state_[obs_pos[0]-1,obs_pos[1]] = 0.5
            state_[obs_pos[0],obs_pos[1]-1] = 0.5
        elif (obs_pos[0]-1 < 0
            and obs_pos[1] >=  self._size-1): ## Top Left corner
            state_[obs_pos[0]+1,obs_pos[1]  ] = 0.5
            state_[obs_pos[0]  ,obs_pos[1]-1] = 0.5
        elif (obs_pos[0] >= self._size-1
            and obs_pos[1]-1 < 0): ## Bottom Right corner
            state_[obs_pos[0]-1,obs_pos[1]] = 0.5
            state_[obs_pos[0],obs_pos[1]+1] = 0.5
        elif (obs_pos[0] >= self._size-1): ## Right wall
            state_[obs_pos[0]-1,obs_pos[1]] = 1/3
            state_[obs_pos[0],obs_pos[1]+1] = 1/3
            state_[obs_pos[0],obs_pos[1]-1] = 1/3
        elif (obs_pos[0] <= 0): ## Left wall
            state_[obs_pos[0]+1,obs_pos[1]] = 1/3
            state_[obs_pos[0],obs_pos[1]+1] = 1/3
            state_[obs_pos[0],obs_pos[1]-1] = 1/3
        elif (obs_pos[1] >= self._size-1): ## Top wall
            state_[obs_pos[0]+1,obs_pos[1]] = 1/3
            state_[obs_pos[0]-1,obs_pos[1]] = 1/3
            state_[obs_pos[0],obs_pos[1]-1] = 1/3
        elif (obs_pos[1] <= 0 ): ## Bottom wall
            state_[obs_pos[0]+1,obs_pos[1]] = 1/3
            state_[obs_pos[0]-1,obs_pos[1]] = 1/3
            state_[obs_pos[0],obs_pos[1]+1] = 1/3
        else:## Somewhere in the middle of the map
            state_[obs_pos[0]+1,obs_pos[1]] = 0.25
            state_[obs_pos[0]-1,obs_pos[1]] = 0.25
            state_[obs_pos[0],obs_pos[1]+1] = 0.25
            state_[obs_pos[0],obs_pos[1]-1] = 0.25
            
            
        
        
        # agent_pos[self.agent_pos[0],self.agent_pos[1]] = 1
        ### It is a rather sparse matrix
        assert np.isclose(np.sum(state_),1.0), "np.sum(state_)" + str(np.sum(state_))
        return (state_, state_agent)
        
        
    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        topX = self.agent_pos[0] + (self._viz_size//2)
        botX = self.agent_pos[0] - (self._viz_size//2)
        topY = self.agent_pos[1] + (self._viz_size//2)
        botY = self.agent_pos[1] - (self._viz_size//2)
       

        return (topX, topY, botX, botY)
    
    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        # self.grid.horz_wall(0, 0)
        # self.grid.horz_wall(0, height - 1)
        # self.grid.vert_wall(0, 0)
        # self.grid.vert_wall(width - 1, 0)

        # self.grid.vert_wall(width // 2, 0)
        # self.grid.set(width // 2, height // 2, Door(self.door_color))
        # for i in [1,2,3]:
        #     self.grid.set(width // 2, height // 2 - i, Door(self.door_color))
        #     self.grid.set(width // 2, height // 2 + i, Door(self.door_color))

        self.place_agent(size=(width, height))
        self.agent_dir = 0
        '''
        if self._goal_default_pos is not None:
            goal = Goal()
            self.grid.set(*self._goal_default_pos, goal)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())
        '''

        self.mission = "Follow the Enemy"

        # Place obstacles
        self.obstacles = []
        
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], 
                           size=(width // 2, height), 
                           max_tries=100)
        old_pos = self.obstacles[0].cur_pos
        self.obstacles[0].cur_pos = [self._size//2,self._size//2]
        self.grid.set(*old_pos, None)
        self._direction = 1

    def convert_obs(self, obs):
        # Get rid of color channel (1), flatten, binarize
        # We encode where there is an obstacle (inc. closed doors) (7x7)
        # And where the doors are (7x7)
        obs = obs['image']
        obstacles = np.zeros((7,7))
        doors = np.zeros((7,7))

        # 2 is wall, 4 is door, 6 is ball
        for idx in [6]:
            obstacles += (obs[:,:,0] == idx).astype(int)

        # Don't show unlocked doors:
        # Channel 2 is state, 0 -> unlocked, 1 -> locked
        """
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                if obs[i,j,0] == 4 and obs[i,j,2] == 1:
                    obstacles[i,j] = 1
                if obs[i,j,0] == 4:
                    doors[i,j] = 1
        """
        [self.agent_pos]
        old_pos = self.obstacles[0].cur_pos
        obs_ = np.hstack((obstacles)).flatten()
        # see_enemy = np.sum(obs_) > 0.5
        
        agent_pos = np.zeros((self._size,self._size))
        agent_pos[self.agent_pos[0],self.agent_pos[1]] = 1 
        
        enemy_pos = np.zeros((self._viz_size,self._viz_size))
        if self.seesEnemey():
            # print ("old_pos: ", old_pos)
            enemy_pos[old_pos[0]-self.agent_pos[0] + (self._viz_size//2),old_pos[1]-self.agent_pos[1]+(self._viz_size//2)] = 1 
        else:
            ### Otherwise some kind of random noisy belief
            for i in range(1,8):
                for j in range(1,8):
                    if (not self.in_view(i,j)):
                        enemy_pos[i,j] = np.random.randint(2)
        obs_ = np.concatenate((agent_pos.flatten(), enemy_pos.flatten()))
        # obs_ = np.concatenate((obs_, agent_dir))
        # print ("see_enemy : ", see_enemy , obs_)
        return obs_
    
    def getTrueState(self):
        obs_pos = self.obstacles[0].cur_pos
        state_ = np.zeros((self._size, self._size))
        state_[obs_pos[0], obs_pos[1]] = 1
        return state_
        

    def step(self, action):
        self.agent_dir = 0
        info = {}
        # obs, reward, done, info = MiniGridEnv.step(self, action)
        old_pos = self.obstacles[0].cur_pos
        prev_state = (old_pos[0]*self._size) + old_pos[1]
        self.move(action)
        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            ### Add a bit more randomness to the movements.
            if np.random.rand() < 0.1:
                continue
            top = tuple(map(add, old_pos, (-1, -1)))
            # print ("old_pos: ", old_pos, "direction: ", self._direction)
            try:
                self.place_obj(self.obstacles[i_obst], top=top, size=(3,3), max_tries=100)
                self.grid.set(*old_pos, None)
            except:
                pass
        
        old_pos = self.obstacles[0].cur_pos
        new_state = (old_pos[0]*self._size) + old_pos[1]
        self._forward.append(self._forward[-1])
        check_ = np.sum(self._forward[-1])
        b = self.belief(obs_pos=old_pos, agent_pos=self.agent_pos)
        b_ = (b).flatten()
        
        if (self.seesEnemey()):
            ### Beleif if perfect in this case
            self._forward[-1] = b_
        else:
            for i in range(self._size* self._size):
                for j in range(self._size * self._size):
                    ind = (i*self._size) + j
                    ### transition prob for old states (ind) to new state
                    trans_prob = self._transitionMatrix[j,i]
                    ### Compute belief for next state
                    # check_ = np.sum(b_)
                    # new_state_prob = np.array([ self._forward[-1][k] * trans_prob[k] for k in range(self._size*self._size)]) * b_[ind]
                    # new_state_prob = np.array([ self._forward[-1][k] * trans_prob[k] for k in range(self._size*self._size)])
                    new_state_prob =  self._forward[-1][j] * trans_prob
                    check_ = np.sum(new_state_prob)
                    self._forward[-1][i] = self._forward[-1][i] + new_state_prob
            ### Should this be 1?
        check_ = np.sum(self._forward[-1])
        self._forward[-1] = self._forward[-1] / check_ ### Normalize for error
        # reward_ = self.computeReward()
        assert np.isclose(np.sum(self._forward[-1]),1.0, rtol=1e-01), "np.sum(self._forward[-1])" + str(np.sum(self._forward[-1]))
        # reward_ = - self.entropyState(self._forward[-1])
        reward_ = self.entropy(self.getTrueState().flatten(), self._forward[-1])
        # print ("reward: ", reward_)
        return self.get_obs(), reward_, False, info
        # return obs, reward, done, info
    
    def computeReward(self):
        """
            Computes the entropy over the belief on the true state.
        """
        if (self.seesEnemey()):
            old_pos = self.obstacles[0].cur_pos
            squares_ = np.zeros((self._size,self._size))
            squares_[old_pos[0], old_pos[1]] = 1
            entropy = self.entropyState(squares_)
            reward = -entropy 
        else: 
            ### Compute squares agent can't see
            squares_ = np.zeros((self._size,self._size))
            for i in range(self._size):
                for j in range(self._size):
                    if (self.seesEnemey(old_pos=[i,j])):
                        squares_[i,j] = 0
                    else:
                        squares_[i,j] = 1
            count = np.where(squares_ == 1)
            prob = 1/len(count[0])
            squares_ = squares_ * prob
            entropy = self.entropyState(squares_)
            reward = -entropy
        
        return reward
        
    def seesEnemey(self, obs_pos=None, agent_pos=None, dist=None):
        if (obs_pos is None):
            obs_pos = self.obstacles[0].cur_pos
        if (agent_pos is None):
            agent_pos = self.agent_pos
        if dist is None:
            dist = self._viz_size
        see_enemy = (np.abs(obs_pos[0] - agent_pos[0]) <= (dist//2))
        see_enemy = see_enemy and (np.abs(obs_pos[1] - agent_pos[1]) <= (dist//2))
        return see_enemy
        
    def move(self, direction):
        axis = direction // 2
        direction = (direction % 2) * 2 - 1

        self.agent_pos[axis] += direction

        # Undo if there is a collision
        collision = np.max(self.agent_pos) >= (self._size) or np.min(self.agent_pos) < 0
        if collision:
            self.agent_pos[axis] -= direction

    def get_obs(self):
        # Get rid of color channel (1), flatten, binarize
        # We encode where there is an obstacle (inc. closed doors) (7x7)
        # And where the doors are (7x7)
        
        old_pos = self.obstacles[0].cur_pos
        # see_enemy = np.sum(obs_) > 0.5
        
        agent_pos = np.zeros((self._size,self._size))
        agent_pos[self.agent_pos[0],self.agent_pos[1]] = 1 
        
        enemy_pos = np.zeros((self._viz_size,self._viz_size))
        if self.seesEnemey():
            # print ("old_pos: ", old_pos)
            enemy_pos[self.agent_pos[0] - old_pos[0] + self._viz_size//2, self.agent_pos[1] - old_pos[1] + self._viz_size//2] = 1 
            
        ### Separate agent pos into two vectors to reduce size
        obs_ = np.concatenate((np.max(agent_pos, axis=0), np.max(agent_pos, axis=1), enemy_pos.flatten()))
        # obs_ = np.concatenate((obs_, agent_dir))
        # print ("see_enemy : ", see_enemy , obs_)
        return obs_
    
    def reset(self):
        #obs = super(MiniGridEnv, self).reset()
        obs = MiniGridEnv.reset(self)
        obs = self.get_obs()
        # return self.convert_obs(obs)
        ### This is just going to be defined over the particle for now
        states_num = self._size * self._size
        self._transitionMatrix = np.zeros((states_num, states_num))
        for i in range(self._size): ### For each row
            for j in range(self._size):
                (T, T_agent) = self.construct_transition_matrix(2, obs_pos=[i,j])
                self._transitionMatrix[(i*self._size) + j] = T.flatten()
        initialStateProb = np.ones((self._size * self._size)) * 1/states_num
        assert (np.isclose(np.sum(initialStateProb),1.0))
        old_pos = self.obstacles[0].cur_pos
        
        b = self.belief(obs_pos=old_pos, agent_pos=self.agent_pos).flatten()
        # self._forward = [initialStateProb * b]
        b = b * 0 
        b[(old_pos[0] * self._size) + old_pos[1]] = 1
        self._forward = [b]
        assert np.isclose(np.sum(self._forward[-1]),1.0), "np.sum(self._forward[-1])" + str(np.sum(self._forward[-1]))
        return obs
    
    def belief(self, obs_pos, agent_pos):
        # print ("see_enemy2: ", see_enemy)
        if (self.seesEnemey(obs_pos=obs_pos, agent_pos=agent_pos)):
            old_pos = self.obstacles[0].cur_pos
            squares_ = np.zeros((self._size,self._size))
            squares_[old_pos[0], old_pos[1]] = 1
        else: 
            ### Compute squares agent can't see
            squares_ = np.zeros((self._size,self._size))
            for i in range(self._size):
                for j in range(self._size):
                    if (self.seesEnemey(obs_pos=[i,j], agent_pos=agent_pos)):
                        squares_[i,j] = 0
                    else:
                        squares_[i,j] = 1
                        
            count = np.where(squares_ == 1)
            prob = 1/len(count[0])
            squares_ = squares_ * prob
        
        assert (np.isclose(np.sum(squares_),1.0))
        return squares_
        

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        
        top_left = self.agent_pos - (self.agent_view_size//2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # For each cell in the visibility mask
        render_transition_prob = False
        if (render_transition_prob):
            (T, T_agent) = self.construct_transition_matrix(3)
            for viz_j in range(0, self._size):
                for viz_i in range(0, self._size):
                    # If this cell is not visible, don't highlight it
                    if (T[viz_j,viz_i] > 0):
                        highlight_mask[viz_j, viz_i] = True
                        
                    if (T_agent[viz_j,viz_i] > 0):
                        highlight_mask[viz_j, viz_i] = True
        else:
            for vis_j in range(0, self.agent_view_size):
                for vis_i in range(0, self.agent_view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue
    
                    # Compute the world coordinates of this cell
                    abs_i = top_left[0] + vis_i
                    abs_j = top_left[1] + vis_j
    
                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue
    
                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True
        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img

        # obs = MiniGridEnv.render(self, mode='rgb_array')
        # plt.imshow(obs)
        # plt.savefig('rollouts/minigrid/{0:04d}.png'.format(self.step_count))
        # plt.clf()

'''
register(
    id='Simple-Enemy-v0',
    entry_point='gym_minigrid.envs:SimpleEnemyEnv'
)
'''
