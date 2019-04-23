# -*- coding: utf-8 -*-

import tensorflow as tf

#import tensorflow_probability as tfp

import numpy as np

import threading
import queue
import matplotlib.pyplot as plt

import gym

# import ppo
from PPO import PPO

# import curiosity 
from Curiosity import Curiosity

# import Worker
from Worker import Worker

### starting out by writing ppo
## good code example here: 44444
## because its short and doesnt do anything extra
## Next step would be to implement curiosity. But this should be the minimal baseline because thats already implemented in the Unity MLagents toolkit...<


GAME_NAME = 'Seaquest-v0'
env = gym.make(GAME_NAME)


# TODO have an option to downscale this maybe?
print(env.observation_space.shape)
OBS_DIM = env.observation_space.shape

if(isinstance(env.action_space,gym.spaces.Discrete)):
  ACTION_DIM = env.action_space.n
 


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

GLOBAL_PPO = PPO(STATE_LATENT_SHAPE,OBS_DIM,ACTION_DIM,UPDATE_EVENT,ROLLING_EVENT,COORD,QUEUE,
                  EPISODE_MAX=EPISODE_MAX)
GLOBAL_CURIOSITY = GLOBAL_PPO.curiosity
workers = [Worker(i,UPDATE_EVENT,ROLLING_EVENT,COORD,QUEUE,GLOBAL_CURIOSITY,GLOBAL_PPO,GAME_NAME,
                  EPISODE_MAX=EPISODE_MAX
                  ) for i in range(NUMBER_OF_WORKERS)]

threads = []
for worker in workers:
  thread = threading.Thread(target=worker.work,args=())
  thread.start()
  threads.append(thread)
ppo_update_thread = threading.Thread(target=GLOBAL_PPO.update)
threads.append(ppo_update_thread)
threads[-1].start()

COORD.join(threads,stop_grace_period_secs=10)

print('Running a test')


done = True
for t in range(1000):
  if(done):
    state = env.reset()
    state = np.expand_dims(state.flatten(),axis=0)
  env.render()
  action = GLOBAL_PPO.get_action(state)
  state,_,done,_ = env.step(action)
  state = np.expand_dims(state.flatten(),axis=0)



