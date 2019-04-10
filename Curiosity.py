import tensorflow as tf
#import tensorflow_probability as tfp
import numpy as np

class Curiosity(object):
  def __init__(self,sess,STATE_LATENT_SHAPE,OBS_DIM,ACTION_DIM,UPDATE_STEP,INV_LR=.0001,FOR_LR=.0001,ETA=1,uncertainty=False):
    
    self.sess = sess
    self.uncertainty = uncertainty

    self.STATE_LATENT_SHAPE = STATE_LATENT_SHAPE
    self.OBS_DIM = OBS_DIM
    self.ACTION_DIM = ACTION_DIM

    self.INV_LR = INV_LR
    self.FOR_LR = FOR_LR
    self.ETA = ETA
    self.UPDATE_STEP = UPDATE_STEP

    self.inp_st = tf.placeholder(tf.float32,[None,self.OBS_DIM],name='S_t_input')
    self.inp_st_ = tf.placeholder(tf.float32,[None,self.OBS_DIM],name='S_t_1_input')
    self.inp_at = tf.placeholder(tf.float32,[None,self.ACTION_DIM],name = 'A_t_input')

    if(self.uncertainty):
      
      i_model = self.bnn_inverse_model('BNN_inverse')
      f_model = self.bnn_forward_model('BNN_forward')
      
    else:
      i_model = self.inverse_model('ICM_inverse')
      f_model = self.forward_model('ICM_forward')
      self.inverse_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels =self.inp_at, logits = self.a_hat))# THIS PROBABLY WONT WORK... need different loss
      self.forward_loss = .5 * tf.reduce_mean(tf.square(tf.subtract(self.phi_hat_st_,self.phi_st)))
      self.curiosity = tf.divide(ETA,2.0) * tf.reduce_mean(tf.square(tf.subtract(self.phi_hat_st_,self.phi_st)))

      
    self.inv_opt = tf.train.AdamOptimizer(INV_LR).minimize(self.inverse_loss) #somewhere this has to happen...
    self.for_opt = tf.train.AdamOptimizer(FOR_LR).minimize(self.forward_loss)
    
    
    
    
                         
    
  def inverse_model(self,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
      # TODO could be done using a CNN feature extractor instead
      features = tf.layers.dense(tf.concat([self.inp_st,self.inp_st_],axis=1),200,tf.nn.relu)


      self.phi_st = tf.layers.dense(features,self.STATE_LATENT_SHAPE,tf.nn.relu) # Activation here is TBD 
      self.phi_st_ = tf.layers.dense(features,self.STATE_LATENT_SHAPE,tf.nn.relu) 

      inv1 = tf.layers.dense(tf.concat([self.phi_st,self.phi_st_],axis=1),200,tf.nn.relu)
      self.a_hat = tf.layers.dense(inv1,self.ACTION_DIM,tf.nn.softmax)
    
    
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)
    return params
    
  def forward_model(self,name):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
      f1 = tf.layers.dense(tf.concat([self.inp_at,self.phi_st],axis=1),200,tf.nn.leaky_relu)
      self.phi_hat_st_ = tf.layers.dense(f1,self.STATE_LATENT_SHAPE,tf.nn.relu)
    
    
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)
    return params
    
  def bnn_forward_model(self,name):
    # might need input size etc.
    #with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
      #bf1 = tfp.layers.DenseFlipout(200,activation=tf.nn.relu)(tf.concat([self.inp_at,self.phi_st],axis=0))
      #self.phi_hat_st_ =  tfp.layers.DenseFlipout(STATE_LATENT_SHAPE,tf.nn.relu)
      
    return None#bnn_forward
  
  def bnn_inverse_model(self,name):
#     with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
#       bnn_inverse = tf.keras.Sequential([
#           #tfp.layers.DenseFlipout(200,activation=tf.nn.relu),
#           #tfp.layers.DenseFlipout(STATE_LATENT_SHAPE,tf.nn.relu)
#       ])
    return None#bnn_inverse
      
      
      
  def reshape_data(self,s_t,s_t_,a_t):
    
    batch_size = s_t.shape[0]

    s_t = s_t.reshape((batch_size,-1))
    s_t_ = s_t_.reshape((batch_size,-1))
    
    
    a_t_new = np.reshape(np.repeat(np.repeat(0.0001,self.ACTION_DIM),s_t.shape[0]),(batch_size,-1))
    
    a_t_new[np.arange(a_t_new.shape[0]),np.int8(a_t)]=.9999
    
    return s_t,s_t_,a_t_new
                         
  
  def get_reward(self,s_t,s_t_,a_t):
    
    s_t,s_t_,a_t = self.reshape_data(s_t,s_t_,a_t)
    if(self.uncertainty):
      return None
    else:
      return self.sess.run(self.curiosity, {self.inp_st:s_t,self.inp_st_:s_t_,self.inp_at:a_t})
  
  def update(self,state,state_,action):
    
    state,state_,action = self.reshape_data(state,state_,action)
    [self.sess.run(self.inv_opt, {self.inp_st: state,self.inp_st_:state_,self.inp_at:action}) for _ in range(self.UPDATE_STEP)]
    [self.sess.run(self.for_opt, {self.inp_st: state,self.inp_st_:state_,self.inp_at:action}) for _ in range(self.UPDATE_STEP)]
        
    
  
