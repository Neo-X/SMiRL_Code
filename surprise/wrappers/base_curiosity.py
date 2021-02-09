import gym

class BaseCuriosityWrapper(gym.Env):
    def __init__(self, env, forward_model, inverse_model, featurizer):
        '''
        params
        ======
        env (gym.Env) : environment to wrap

        forward_model (torch module) : predicts \phi(s') from \phi(s), a

        inverse_model (torch module) : predicts a from \phi(s') and \phi(s)

        featurizer (torch module) : featurizes states: \phi

        '''

        self.env = env
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        self.featurizer = featurizer

        # Gym spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        # Take Action
        obs, rew, done, info = self.env.step(action)


        # Get wrapper outputs
        rew = self.buffer.logprob(self.encode_obs(obs))
        # Add observation to buffer
        self.buffer.add(self.encode_obs(obs))
        obs = self.get_obs(obs)
        done = self.get_done(done)

        return obs, rew, done, info

    def get_obs(self, obs):
        '''
        Augment observation, perhaps with generative model params
        '''
        return obs

    def get_done(self, env_done):
        '''
        figure out if we're done

        params
        ======
        env_done (bool) : done bool from the wrapped env, doesn't 
            necessarily need to be used
        '''
        return env_done

    def reset(self):
        '''
        Reset the wrapped env and the buffer
        '''
        self.env.reset()
        self.buffer.reset()

    def render(self):
        self.env.render()

    def encode_obs(self, obs):
        '''
        Used to encode the observation before putting on the buffer
        '''
        return obs.copy()
