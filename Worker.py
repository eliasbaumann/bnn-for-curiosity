# Worker for multiple agents at the same time

import tensorflow as tf
import numpy as np

import PPO
import gym

from utils import WarpFrame

GLOBAL_RUNNING_REWARD = []
EPISODE_LENGTH = 2000


class Worker(object):
  def __init__(self,wid,UPDATE_EVENT,ROLLING_EVENT,COORD,QUEUE,GLOBAL_CURIOSITY,GLOBAL_PPO,GAME_NAME,EPISODE_MAX=1000,MIN_BATCH_SIZE=64,GAMMA=.9):
    self.wid = wid
    self.env = gym.make(GAME_NAME)
    if(len(self.env.observation_space.shape)>2):
      self.env = WarpFrame(self.env,84,84,True)
    
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
    global GLOBAL_RUNNING_REWARD
    while not self.COORD.should_stop():
      state = self.env.reset()
      state = np.expand_dims(state,axis=0)
      # state = np.expand_dims(state.flatten(), axis=0)
      
      episode_reward = 0
      reward_list = []
      done = False
      buffer_state,buffer_state_,buffer_action,buffer_reward = [],[],[],[]
      for t in range(EPISODE_LENGTH): 
        
        if not self.ROLLING_EVENT.is_set():
          self.ROLLING_EVENT.wait()
          buffer_state,buffer_state_,buffer_action,buffer_reward = [],[],[],[]

        # Add multiple runs until batch size
        action = self.ppo.get_action(state)
        state_,reward,done,_ = self.env.step(action)
        # if done: reward = -10 # -> should not exist in version which compares to pathak et al.
        state_ = np.expand_dims(state_,axis=0)
        # state_ = np.expand_dims(state_.flatten(),axis=0)
        
        buffer_state.append(state.reshape((1,-1)))
        buffer_state_.append(state_.reshape((1,-1)))
        buffer_action.append(action)
        if(self.cur.uncertainty==True):
          curiosity = self.cur.get_reward(state,state_,action)
          buffer_reward.append(curiosity) 
        else:
          curiosity = None
          buffer_reward.append(reward-1) 
        
        state = state_
        episode_reward+=reward
        PPO.alterGlobalUpdateCounter(1)#GLOBAL_UPDATE_COUNTER += 1
        
        # If enough state,action,reward triples are collected:
        if t == EPISODE_LENGTH-1 or PPO.GLOBAL_UPDATE_COUNTER >= self.MIN_BATCH_SIZE: #or done:
          # if done:
          #   state_value = 0
          # else:
          #   state_value = self.ppo.get_value(state_)
          
          state_value = self.ppo.get_value(state_)

          discounted_reward = []
          for r in buffer_reward: #[::-1] he adds that, but need to check if needed
            state_value = r + self.GAMMA * state_value
            discounted_reward.append(state_value)
          discounted_reward.reverse()
          
          bs,bs_, ba, br = np.vstack(buffer_state),np.vstack(buffer_state_),np.vstack(buffer_action),np.vstack(discounted_reward) #[:,np.newaxis] ??
          buffer_state,buffer_state_,buffer_action,buffer_reward = [],[],[],[]
          
          self.QUEUE.put(np.hstack((bs,bs_,ba,br)))
          
          if PPO.GLOBAL_UPDATE_COUNTER>=self.MIN_BATCH_SIZE:
            self.ROLLING_EVENT.clear()
            self.UPDATE_EVENT.set()
          
          if PPO.GLOBAL_EPISODE >= self.EPISODE_MAX:
            print(self.wid,': requested stop')
            self.ROLLING_EVENT.set()
            self.COORD.request_stop()
            self.env.close()
            break
          if done:
            reward_list.append(episode_reward)
            episode_reward = 0
            break
        if done:
          reward_list.append(episode_reward)
          episode_reward = 0
          state = self.env.reset()
          state = np.expand_dims(state,axis=0)
      
      if len(GLOBAL_RUNNING_REWARD) == 0: 
        GLOBAL_RUNNING_REWARD.append(np.mean(episode_reward))
      else:
        GLOBAL_RUNNING_REWARD.append(GLOBAL_RUNNING_REWARD[-1]*0.9+np.mean(episode_reward)*0.1)
      PPO.alterGlobalEpisode(1) # GLOBAL_EPISODE += 1
      print('{0:.1f}%'.format(PPO.GLOBAL_EPISODE/self.EPISODE_MAX*100), '|W%i' % self.wid,  '|Mean_Ep_r: %.2f' % np.mean(episode_reward), 'Cur_r %.10f' % curiosity)
      
    print(self.wid,': stopped')