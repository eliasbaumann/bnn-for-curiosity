import tensorflow as tf
import numpy as np

from Curiosity import Curiosity

GLOBAL_UPDATE_COUNTER = 0
GLOBAL_EPISODE = 0

def alterGlobalUpdateCounter(x):
    global GLOBAL_UPDATE_COUNTER
    GLOBAL_UPDATE_COUNTER += x

def alterGlobalEpisode(x):
  global GLOBAL_EPISODE
  GLOBAL_EPISODE += x

def resetGlobalUpdateCounter():
  global GLOBAL_UPDATE_COUNTER
  GLOBAL_UPDATE_COUNTER = 0

class PPO(object):

  def _actor_network(self,name,trainable):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
      act = tf.layers.dense(self.inp,200,tf.nn.relu,trainable=trainable)
      
      
      # mu = 2* tf.layers.dense(act,ACTION_DIM,tf.nn.tanh,trainable=trainable)
      action_probs = tf.layers.dense(act,self.ACTION_DIM,tf.nn.softmax,trainable=trainable)
      # norm_dist = tf.distributions.Normal(loc=mu,scale=sigma)
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)
    return action_probs,params
  
  def __init__(self,STATE_LATENT_SHAPE,OBS_DIM,ACTION_DIM,UPDATE_EVENT,ROLLING_EVENT,COORD,QUEUE,CRITIC_LR=.0001,ACTOR_LR = .0002,EPSILON=.2,GAMMA=.9,EPISODE_MAX = 1000,UPDATE_STEP=10):
    self.sess = tf.Session()
    self.OBS_DIM = OBS_DIM
    self.ACTION_DIM = ACTION_DIM
    self.CRITIC_LR = CRITIC_LR
    self.ACTOR_LR = ACTOR_LR
    self.EPSILON = EPSILON
    self.GAMMA = GAMMA
    self.EPISODE_MAX = EPISODE_MAX
    self.UPDATE_STEP = UPDATE_STEP
    self.STATE_LATENT_SHAPE = STATE_LATENT_SHAPE

    self.UPDATE_EVENT = UPDATE_EVENT
    self.ROLLING_EVENT = ROLLING_EVENT
    self.COORD = COORD
    self.QUEUE = QUEUE
    
    self.inp = tf.placeholder(tf.float32,[None,self.OBS_DIM],name='state')
    
    # ppo is based on actor critic 
    
    # Critic
    crit = tf.layers.dense(self.inp,100,tf.nn.relu)
    self.value = tf.layers.dense(crit,1)
    self.dc_reward = tf.placeholder(tf.float32,[None,1],name='discounted_r')
    self.advantage = self.dc_reward - self.value
    self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
    self.critic_train_opt = tf.train.AdamOptimizer(self.CRITIC_LR).minimize(self.critic_loss)
    
    
    # Actor 
    self.policy, theta_p = self._actor_network('policy',trainable=True)
    
    old_policy, old_theta_p = self._actor_network('old_policy',trainable=False)
    
    self.update_old_policy_op =[oldp.assign(p) for p,oldp in zip(theta_p,old_theta_p)]
    
    self.action = tf.placeholder(tf.int32, [None,],name='action')
    self.full_adv = tf.placeholder(tf.float32, [None,1],name='advantage')
    
    a_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
    pi_prob = tf.gather_nd(params=self.policy, indices=a_indices)   
    oldpi_prob = tf.gather_nd(params=old_policy, indices=a_indices) 
    ratio = pi_prob/oldpi_prob
    surr = ratio * self.full_adv
    
    
    self.actor_loss = -tf.reduce_mean(tf.minimum(surr,tf.clip_by_value(ratio,1.-self.EPSILON,1.+self.EPSILON)*self.full_adv))
    
    
    
    self.actor_train_opt = tf.train.AdamOptimizer(self.ACTOR_LR).minimize(self.actor_loss)
    
    self.curiosity = Curiosity(self.sess,self.STATE_LATENT_SHAPE,self.OBS_DIM,self.ACTION_DIM,self.UPDATE_STEP)
    
    self.sess.run(tf.global_variables_initializer())
    
    
  def get_action(self,state):
    action_probs = self.sess.run(self.policy, {self.inp:state})
    action = np.random.choice(range(action_probs.shape[1]),p=action_probs.ravel())
    return action

  def get_value(self,state):
    return self.sess.run(self.value,{self.inp:state})[0,0] #TODO same as with get_action, need to figure out what comes out of here...

  def update(self):
    while not self.COORD.should_stop():
      if GLOBAL_EPISODE < self.EPISODE_MAX:
        self.UPDATE_EVENT.wait()
        self.sess.run(self.update_old_policy_op) #TODO something similar for curiosity, maybe i have to include it in PPO
        data = [self.QUEUE.get() for _ in range(self.QUEUE.qsize())]
        data = np.vstack(data)
        state,state_,action,reward = data[:, :self.OBS_DIM],data[:,self.OBS_DIM:2*self.OBS_DIM], data[:, 2*self.OBS_DIM: 2*self.OBS_DIM + 1].ravel(), data[:, -1:]
        advantage = self.sess.run(self.advantage, {self.inp: state,self.dc_reward: reward})
        # update actor and critic in a update loop
        [self.sess.run(self.actor_train_opt, {self.inp: state, self.action: action, self.full_adv: advantage}) for _ in range(self.UPDATE_STEP)]
        [self.sess.run(self.critic_train_opt, {self.inp: state, self.dc_reward: reward}) for _ in range(self.UPDATE_STEP)]
        
        self.curiosity.update(state,state_,action)
        
        self.UPDATE_EVENT.clear()        # updating finished
        resetGlobalUpdateCounter() # GLOBAL_UPDATE_COUNTER = 0   # reset counter
        self.ROLLING_EVENT.set()         # set roll-out available

