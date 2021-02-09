import numpy as np

class BaseBuffer():
    '''
    Abstract buffer class
    '''

    def __init__(self):
        pass

    def add(self, obs):
        '''
        Add an observation to the buffer
        '''
        raise NotImplementedError

    def get_params(self):
        '''
        Get the sufficient statistics for the buffer
        '''
        raise NotImplementedError

    def logprob(self, obs):
        '''
        Return the logprob of an observation
        '''
        raise NotImplementedError

    def reset(self):
        '''
        Reset the buffer, clear its contents
        '''
        raise NotImplementedError

class BernoulliBuffer(BaseBuffer):
    def __init__(self, obs_dim):
        super().__init__()
        self.buffer = np.zeros(obs_dim) 
        self.buffer_size = 1
        self.obs_dim = obs_dim
        
    def add(self, obs):
        self.buffer += obs
        self.buffer_size += 1

    def get_params(self):
        theta = np.array(self.buffer) / self.buffer_size
        thresh = 1e-4
        theta = np.clip(theta, a_min=thresh, a_max=(1-thresh))
        return theta

    def logprob(self, obs):
        obs = obs.copy()
        # ForkedPdb().set_trace()
        thetas = self.get_params()

        # For numerical stability, clip probs to not be 0 or 1
        thresh = 1e-5
        thetas = np.clip(thetas, thresh, 1 - thresh)

        # Bernoulli log prob
        probs = obs*thetas + (1-obs)*(1-thetas)
        logprob = np.sum(np.log(probs))
        return logprob

    def reset(self):
        self.buffer = np.zeros(self.obs_dim)
        self.buffer_size = 1
        
    def entropy(self):
        thetas = self.get_params()
#         print ("thetas: ", thetas)
        thresh = 1e-4
        thetas = np.clip(thetas, a_min=thresh, a_max=(1-thresh))
        return np.sum(-thetas * np.log(thetas) - (1-thetas) * np.log(1-thetas))

class GaussianBuffer(BaseBuffer):
    def __init__(self, obs_dim):
        super().__init__()
        self.buffer = np.zeros((1,obs_dim))
        self.buffer_size = 1
        self.obs_dim = obs_dim
        self.add(np.ones((1,obs_dim)))
        self.add(-np.ones((1,obs_dim)))
        
    def add(self, obs):
        self.buffer = np.concatenate((self.buffer,obs.flatten()[np.newaxis,:]), axis=0)
        self.buffer_size += 1

    def get_params(self):
        #import pdb; pdb.set_trace()
        means = np.mean(self.buffer, axis=0)
        stds = np.std(self.buffer, axis=0)
        params = np.concatenate([means, stds])
        return params

    def logprob(self, obs):
        obs = obs.copy()
        means = np.mean(self.buffer, axis=0)
        stds = np.std(self.buffer, axis=0)
        
        # For numerical stability, clip stds to not be 0
        thresh = 1e-5
        stds = np.clip(stds, thresh, None)
#         print ("stds, means: ", np.mean(stds), np.mean(means))
#         import pdb; pdb.set_trace()
        # Gaussian log prob
        logprob = -0.5*np.sum(np.log(2*np.pi*stds)) - np.sum(np.square(obs-means)/(2*np.square(stds)))
        return logprob

    def reset(self):
        self.buffer = np.zeros((1, self.obs_dim))
        self.buffer_size = 1
        
class GaussianBufferIncremental(BaseBuffer):
    def __init__(self, obs_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.reset()
        
    def inserts(self):
        return self.buffer_size
    
    def add(self, state):
        if (self.inserts() == 0):
             self._state_mean = state
             self._state_var = np.ones_like(state)
        else:
            x_mean_old = self._state_mean
            self._state_mean = self._state_mean + ((state - self._state_mean)/self.inserts())
            
        if (self.inserts() == 2):
            self._state_var = (state - ((self._prev_state[0]+state)/2.0)**2)/2.0
        elif (self.inserts() > 2):
            self._state_var = (((self.inserts()-2)*self._state_var) + ((self.inserts()-1)*(x_mean_old - self._state_mean)**2) + ((state - self._state_mean)**2))
            self._state_var = (self._state_var/float(self.inserts()-1))
            
        self._prev_state = state
        self._state_var = np.fabs(self._state_var)
#         print ("self._state_var: ", np.sqrt(self._state_var))
        self.buffer_size += 1

    def get_params(self):
        #import pdb; pdb.set_trace()
        means = self._state_mean
        stds = np.sqrt(self._state_var)
        params = np.concatenate([means, stds])
        return params

    def logprob(self, obs):
        obs = obs.copy()
        means = self._state_mean
        stds = np.sqrt(self._state_var)
        
        # For numerical stability, clip stds to not be 0
        thresh = 1e-3
#         print ("thresh: ", thresh)
        stds = np.clip(stds, thresh, None)
#         print ("stds, means: ", np.mean(stds), np.mean(means))
#         import pdb; pdb.set_trace()
        # Gaussian log prob
        logprob = -0.5*np.sum(np.log(2*np.pi*stds)) - np.sum(np.square(obs-means)/(2*np.square(stds)))
        return logprob

    def reset(self):
        self._state_mean =  np.zeros(self.obs_dim)
        self._state_var = np.ones(self.obs_dim)
        self.buffer_size = 0
        
    def entropy(self):
#         stds = np.sqrt(self._state_var)
        # For numerical stability, clip stds to not be 0
        thresh = 1e-3
        var = np.clip(self._state_var, thresh, None)
        return -0.5*np.sum(np.log(2*np.pi*np.e*var))
        
        
        
        
class GaussianCircularBuffer(BaseBuffer):
    def __init__(self, obs_dim, size):
        super().__init__()
        self.buffer_max_size=size
        self.obs_dim = obs_dim
        ### Where to insert the next item
        self.reset()
        
    def add(self, obs):
        self.buffer[self.buffer_pointer] = obs
        self.buffer_size += 1
        self.buffer_pointer = self.buffer_pointer + 1
        if self.buffer_pointer >= self.buffer_max_size:
            self.buffer_pointer = 0
        

    def get_params(self):
        #import pdb; pdb.set_trace()
        means = np.mean(self.buffer, axis=0)
        stds = np.std(self.buffer, axis=0)
        params = np.concatenate([means, stds])
        return params

    def logprob(self, obs):
        obs = obs.copy()
        ### Make sure to not include extra buffer samples in the beginning
        means = np.mean(self.buffer[:self.buffer_size], axis=0)
        stds = np.std(self.buffer[:self.buffer_size], axis=0)
        
        # For numerical stability, clip stds to not be 0
        thresh = 1e-5
        stds = np.clip(stds, thresh, None)
#         print ("self._state_var: ", stds)
#         print ("stds, means: ", np.mean(stds), np.mean(means))
#         import pdb; pdb.set_trace()
        # Gaussian log prob
        logprob = -0.5*np.sum(np.log(2*np.pi*stds)) - np.sum(np.square(obs-means)/(2*np.square(stds)))
        return logprob

    def reset(self):
        self.buffer = np.zeros((self.buffer_max_size, self.obs_dim))
        self.buffer_size = 0
        self.buffer_pointer = 0
        self.add(np.ones((1,self.obs_dim)))
        self.add(-np.ones((1,self.obs_dim)))
