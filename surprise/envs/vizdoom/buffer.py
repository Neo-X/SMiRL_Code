import numpy as np
import torch
import pdb

class Buffer():
    def __init__(self, size=10000, device=0):
        self.buffer = []
        self.size = size
        self.device = device
        self._loc = 0
    
    def add(self, data):
        # Delete if too much data
        if len(self.buffer) >= self.size:
            if (self._loc >= len(self.buffer)):
                self._loc = 0
#             print ("self._loc: ", self._loc)
            self.buffer[self._loc] = data
        else:
        # Append data
            self.buffer.append(data)
        self._loc = self._loc + 1

    def sample(self, n):
        n = min(n, len(self.buffer))
        random_indices = np.random.choice(range(len(self.buffer)), n, replace=False)
        samples = [self.buffer[i] for i in random_indices]

        state = np.array([x[0] for x in samples])
        action = np.array([x[1] for x in samples])
        state_prime = np.array([x[2] for x in samples])

        state = torch.tensor(state).to(self.device).float()
        action = torch.tensor(action).to(self.device).float()
        state_prime = torch.tensor(state_prime).to(self.device).float()

        return state, action, state_prime

    def __len__(self):
        return len(self.buffer)

class SimpleBuffer(Buffer):
    def __init__(self, size=100000, device=0):
        super().__init__(size=size, device=device)
    
    def sample(self, n):
        n = min(n, len(self.buffer))
        random_indices = np.random.choice(range(len(self.buffer)), n, replace=False)
        samples = [self.buffer[i] for i in random_indices]
        return map(list, zip(*samples))

class VAEBuffer(Buffer):
    def __init__(self, size=100000, device=0, dtype="float32"):
        super().__init__(size=size, device=device)
        self._dtype=dtype
        
    def add(self, data):
        # Delete if too much data
        if self._dtype == "uint8":
            data = data * 255
        super().add(np.array(data, dtype=self._dtype))
    

    def sample(self, n):
        n = min(n, len(self.buffer))
        random_indices = np.random.choice(range(len(self.buffer)), n, replace=False)
#         print ("random_indices:", random_indices)
        samples = [self.buffer[i] for i in random_indices]
        samples2 = samples
        if self._dtype == "uint8":
#             print ("normalizing vae batch: ", samples)
            samples = np.array(samples) / 255
        return torch.tensor(samples).to(self.device).float(), torch.tensor(samples2).to(self.device).float()

