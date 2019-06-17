
import tensorflow as tf
from baselines.common.distributions import make_pdtype
from utils import flatten_two_dims,unflatten_first_dim,fc,activ,small_convnet

class CnnPolicy(object):
    def __init__(self,ob_space,ac_space,hidsize,ob_mean,ob_std,feat_dim,layernormalize,nl,scope='policy'):
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        self.hidsize = hidsize
        self.feat_dim = feat_dim
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space)
            self.placeholder_observation = tf.placeholder(dtype=tf.int32,shape=(None,None) + ob_space.shape,name='observation')
            self.placeholder_action = self.ac_pdtype.sample_placeholder([None,None],name='action')
            self.pd = self.vpred = None 
            self.scope = scope
            pdparamsize = self.ac_pdtype.param_shape()[0]

            shape = tf.shape(self.placeholder_observation)
            x = flatten_two_dims(self.placeholder_observation)
            self.flat_features = self.get_features(x,reuse=False)
            self.features = unflatten_first_dim(self.flat_features,shape)

            with tf.variable_scope(scope,reuse=False):
                x = fc(self.flat_features,units=hidsize,activation=activ)
                x = fc(x,units = hidsize,activation=activ)
                pdparam = fc(x,name='pd',units = pdparamsize, activation=None)
                value_pred = fc(x,name='value_func_output',units=1,activation=None)
            pdparam = unflatten_first_dim(pdparam, shape)
            self.vpred = unflatten_first_dim(value_pred,shape)[:,:,0]
            self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)
            self.a_samp = pd.sample()
            self.entropy = pd.entropy()
            self.nlp_samp = pd.neglogp(self.a_samp)
    
    def get_features(self,x,reuse):
        if(x.get_shape().ndims==5):
            shape=tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope+'_features',reuse=reuse):
            x = (tf.cast(x,tf.float32)-self.ob_mean) / self.ob_std
            x = small_convnet(x,nl=self.nl,feat_dim=self.feat_dim,last_nl=None,layernormalize=self.layernormalize)
        if(x.get_shape().ndims==5):
            x = unflatten_first_dim(x,shape)
        return x

    def get_ac_value_nlp(self,ob): #action_value_neglogprob
        a,vpred,nlp = tf.get_default_session().run([self.a_samp,self.vpred,self.nlp_samp],feed_dict={self.placeholder_observation: ob[:,None]})
        return a[:,0],vpred[:,0],nlp[:,0]

