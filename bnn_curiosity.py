# -*- coding: utf-8 -*-

import tensorflow as tf

#import tensorflow_probability as tfp

import numpy as np

import threading
import queue
import matplotlib.pyplot as plt
# for future maybe, alternatively
# import mlagents
# import gym
import obstacle_tower_env

# import ppo
from PPO import PPO

# import curiosity 
from Curiosity import Curiosity

# import Worker
from Worker import Worker

### starting out by writing ppo
## good code example here: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/discrete_DPPO.py
## because its short and doesnt do anything extra
## Next step would be to implement curiosity. But this should be the minimal baseline because thats already implemented in the Unity MLagents toolkit...<

OBS_DIM = 84*84*3
ACTION_DIM = 54 # ALSO QUESTIONABLE IF THATS A GOOD IDEA -> LOOK AT WORK HOW TO DEAL WITH LARGE ACTION SPACE...

EPISODE_MAX = 1000
MIN_BATCH_SIZE = 64

NUMBER_OF_WORKERS = 4

#### Curiosity stuff:
STATE_LATENT_SHAPE = 64

### MAIN, in the actual implementation uncomment the next line and re-indent everything
# if __name__=='__main__':
tf.logging.set_verbosity(tf.logging.INFO)



UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
UPDATE_EVENT.clear()
ROLLING_EVENT.set()
COORD = tf.train.Coordinator()
QUEUE = queue.Queue()

GLOBAL_PPO = PPO(STATE_LATENT_SHAPE,OBS_DIM,ACTION_DIM,UPDATE_EVENT,ROLLING_EVENT,COORD,QUEUE)
GLOBAL_CURIOSITY = GLOBAL_PPO.curiosity
workers = [Worker(i,UPDATE_EVENT,ROLLING_EVENT,COORD,QUEUE,GLOBAL_CURIOSITY,GLOBAL_PPO) for i in range(NUMBER_OF_WORKERS)]

threads = []
for worker in workers:
  thread = threading.Thread(target=worker.work,args=())
  thread.start()
  threads.append(thread)
ppo_update_thread = threading.Thread(target=GLOBAL_PPO.update)
ppo_update_thread.start()
threads.append(ppo_update_thread)

COORD.join(threads)
