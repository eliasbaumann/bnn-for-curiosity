# -*- coding: utf-8 -*-

import tensorflow as tf

#import tensorflow_probability as tfp

import numpy as np

import threading
import queue
import matplotlib.pyplot as plt
from PIL import Image
import gym

# import ppo
from PPO import PPO

# import curiosity
from Curiosity import Curiosity

# import Worker
from Worker import Worker

from utils import get_env_mean_std,make_env

tf.logging.set_verbosity(tf.logging.INFO)
GAME_NAME = 'Seaquest-v0'
env = make_env(GAME_NAME)

OBS_DIM = env.observation_space.shape

if(isinstance(env.action_space, gym.spaces.Discrete)):
    ACTION_DIM = env.action_space.n

# hidsize = 512
tf.reset_default_graph()
EPISODE_MAX = 1000
MIN_BATCH_SIZE = 128

NUMBER_OF_WORKERS = 12

# Curiosity stuff:
STATE_LATENT_SHAPE = 512

# if __name__=='__main__':
OBS_MEAN,OBS_STD = get_env_mean_std(GAME_NAME, n_steps=10)

tf.logging.set_verbosity(tf.logging.INFO)

UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
UPDATE_EVENT.clear()
ROLLING_EVENT.set()
COORD = tf.train.Coordinator()
QUEUE = queue.Queue()



GLOBAL_PPO = PPO(STATE_LATENT_SHAPE, OBS_DIM, ACTION_DIM, UPDATE_EVENT, ROLLING_EVENT, COORD, QUEUE,OBS_MEAN,OBS_STD,
                 EPISODE_MAX=EPISODE_MAX)
GLOBAL_CURIOSITY = GLOBAL_PPO.curiosity
workers = [Worker(i, UPDATE_EVENT, ROLLING_EVENT, COORD, QUEUE, GLOBAL_CURIOSITY, GLOBAL_PPO, GAME_NAME,
                  EPISODE_MAX=EPISODE_MAX
                  ) for i in range(NUMBER_OF_WORKERS)]

threads = []
for worker in workers:
    thread = threading.Thread(target=worker.work, args=())
    thread.start()
    threads.append(thread)
ppo_update_thread = threading.Thread(target=GLOBAL_PPO.update)
threads.append(ppo_update_thread)
threads[-1].start()

COORD.join(threads, stop_grace_period_secs=10)

print('saving test to file')


done = False
state = env.reset()
state = np.expand_dims(state, axis=0)
frames = []
while(not done):
    frames.append(Image.fromarray(env.render(mode='rgb_array')))
    action = GLOBAL_PPO.get_action(state)
    state, _, done, _ = env.step(action)
    state = np.expand_dims(state, axis=0)

with open('C:/Users/Elias/OneDrive/Winfo Studium/SS19/Masterarbeit/gifs/'+GAME_NAME+'.gif', 'wb') as f:  # change the path if necessary
    im = Image.new('RGB', frames[0].size)
    im.save(f, save_all=True, append_images=frames)
