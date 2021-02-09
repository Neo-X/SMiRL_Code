import os
from vizdoomEnv import TakeCoverEnv
import numpy as np
from multiprocessing import Pool
import pdb

datapath = './data'

def collect_trajectory(save_idx):
    savepath = os.path.join(datapath, '{:03d}'.format(save_idx))
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    env = TakeCoverEnv()
    env.reset()
    done = False
    t = 0

    while not done:
        _, _, done, _ = env.step(np.random.randint(2))
        lowres_obs = env.render_lowres().flatten()
        np.save(os.path.join(savepath, '{:03d}.npy'.format(t)), lowres_obs)
        t += 1

p = Pool(12)
indices = list(range(100))

p.map(collect_trajectory, indices)

#collect_trajectory(0)
