# Worker for multiple agents at the same time

import tensorflow as tf
import numpy as np

import PPO
import obstacle_tower_env


GLOBAL_RUNNING_REWARD = []



class Worker(object):
  def __init__(self,wid,UPDATE_EVENT,ROLLING_EVENT,COORD,QUEUE,GLOBAL_CURIOSITY,GLOBAL_PPO,EPISODE_MAX=1000,MIN_BATCH_SIZE=64,GAMMA=.9,PATH='C:/Users/Elex/Downloads/obstacle-tower-challenge/ObstacleTower/obstacletower'):
    self.wid = wid
    self.env = obstacle_tower_env.ObstacleTowerEnv(PATH,retro=True,worker_id=wid)
    
    self.UPDATE_EVENT = UPDATE_EVENT
    self.ROLLING_EVENT = ROLLING_EVENT
    self.COORD = COORD
    self.QUEUE = QUEUE

    self.cur = GLOBAL_CURIOSITY
    self.ppo = GLOBAL_PPO
    
    self.EPISODE_MAX = EPISODE_MAX
    self.MIN_BATCH_SIZE = MIN_BATCH_SIZE
    self.GAMMA = GAMMA

  
  def vectorize_state(self,state):
    return None
  
  def work(self):
    while not self.COORD.should_stop():
      state = self.env.reset()
      state = np.expand_dims(state.flatten(), axis=0)
      episode_reward = 0
      done = False
      buffer_state,buffer_state_,buffer_action,buffer_reward = [],[],[],[]
      while not done: 
        
        if not self.ROLLING_EVENT.is_set():
          self.ROLLING_EVENT.wait()
          buffer_state,buffer_state_,buffer_action,buffer_reward = [],[],[],[]
        
        # Add multiple runs until batch size
        action = self.ppo.get_action(state)
        state_,reward,done,_ = self.env.step(action)
        
        curiosity = self.cur.get_reward(state,state_,action)
        reward += curiosity
        
        state_ = np.expand_dims(state_.flatten(),axis=0)
        
        buffer_state.append(state)
        buffer_state_.append(state_)
        buffer_action.append(action)
        buffer_reward.append(reward) #he normalized this, probably smart to do when we add curiosity as well...
        state = state_
        episode_reward += reward
        PPO.alterGlobalUpdateCounter(1)#GLOBAL_UPDATE_COUNTER += 1
        
        # If enough state,action,reward triples are collected:
        if PPO.GLOBAL_UPDATE_COUNTER >= self.MIN_BATCH_SIZE or done:
          state_value = self.ppo.get_value(state_)
          discounted_reward = []
          for r in buffer_reward: #[::-1] he adds that, but need to check if needed
            state_value = r + self.GAMMA * state_value
            discounted_reward.append(state_value)
          discounted_reward.reverse()
          
          bs,bs_, ba, br = np.vstack(buffer_state),np.vstack(buffer_state_),np.vstack(buffer_action),np.vstack(discounted_reward) #[:,np.newaxis] ??
          buffer_state,buffer_state_,buffer_action,buffer_reward = [],[],[],[]
          self.QUEUE.put(np.hstack((bs,bs_,ba,br)))
          
          self.ROLLING_EVENT.clear()
          self.UPDATE_EVENT.set()
          
          if PPO.GLOBAL_EPISODE >= self.EPISODE_MAX:
            self.COORD.request_stop()
            break
      
      
      if len(GLOBAL_RUNNING_REWARD) == 0: 
        GLOBAL_RUNNING_REWARD.append(episode_reward)
      else:
        GLOBAL_RUNNING_REWARD.append(GLOBAL_RUNNING_REWARD[-1]*0.9+episode_reward*0.1)
      PPO.alterGlobalEpisode(1) # GLOBAL_EPISODE += 1
      print('{0:.1f}%'.format(PPO.GLOBAL_EPISODE/self.EPISODE_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % episode_reward,)