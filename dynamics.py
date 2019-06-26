import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import bernoulli

from utils import small_convnet, flatten_two_dims, unflatten_first_dim,unflatten_tiled
from ConcreteDropout import concrete_dropout

class Dynamics(object):
    ''' mode can be set to :
        - 'none'
        - 'flipout'
        - 'dropout'
        - 'bootstrapped'
         '''
    def __init__(self, auxiliary_task, feat_dim=None,mode='none', scope='dynamics'):
        
        self.n_chunks = 12
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        self.obs = self.auxiliary_task.obs
        self.last_ob = self.auxiliary_task.last_ob
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        self.features = tf.stop_gradient(self.auxiliary_task.features)
        self.out_features = self.auxiliary_task.next_features

        self.bootstrapped = False
        self.flipout = False
        self.dropout = False

        if mode == 'flipout':
            self.flipout = True
            print('-----------------------------------')
            print('using flipout uncertainty as reward')
            print('-----------------------------------')
        elif mode == 'dropout':
            self.dropout = True
            self.drop_iters = 20
            print('--------------------------------------------')
            print('using concrete dropout uncertainty as reward')
            print('--------------------------------------------')
        elif mode == 'bootstrapped':
            self.bootstrapped = True
            print('----------------------------------------')
            print('using bootstrapped uncertainty as reward')
            print('----------------------------------------')
        else:
            print('----------------------------------------')
            print('           Running baseline             ')
            print('----------------------------------------')

        with tf.variable_scope(self.scope + "_loss"):
            self.loss = self.get_loss()

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=nl, layernormalize=False)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)

        def add_ac(x):
            return tf.concat([x, ac], axis=-1)
        

        def vstack(x,n):
            return tf.tile(tf.expand_dims(x,0),[n,1,1])

        def stacked_add_ac(x):
            return tf.concat([x,vstack(ac,50)],axis=-1)

        if self.flipout:
            with tf.variable_scope(self.scope):
                x = flatten_two_dims(self.features)
                x = add_ac(x)
                x = vstack(x,50)
                x = tfp.layers.DenseFlipout(self.hidsize,activation=tf.nn.leaky_relu)(x)

                def residual(x):
                    res = tfp.layers.DenseFlipout(self.hidsize,activation=tf.nn.leaky_relu)(stacked_add_ac(x))
                    res = tfp.layers.DenseFlipout(self.hidsize,activation=None)(stacked_add_ac(res))
                    return x+res
                
                for _ in range(2):
                    x = residual(x)
                
                n_out_features = self.out_features.get_shape()[-1].value
                x = tfp.layers.DenseFlipout(n_out_features,activation=None)(stacked_add_ac(x))
                x = unflatten_tiled(x,sh)
            
            feats = tf.tile(tf.expand_dims(tf.stop_gradient(self.out_features),0),[50,1,1,1])
            res_x = tf.reduce_mean(x-feats,axis=-1)
            _,var = tf.nn.moments(res_x,axes=[0])
            res = tf.sqrt(var)

        elif self.bootstrapped:
            self.n_heads=20

            def mask_gradients(x,mask):
                mask_h = tf.abs(mask-1)
                return tf.stop_gradient(mask_h*x)+mask*x

            def residual(x):
                res = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)
                res = tf.layers.dense(add_ac(res), self.hidsize, activation=None)
                return x + res
            
            self.heads = []
            for i in range(self.n_heads):
                with tf.variable_scope(self.scope+'head_'+str(i)):
                    x = flatten_two_dims(self.features)
                    x = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)
                    for _ in range(2):
                        x = residual(x)
                    n_out_features = self.out_features.get_shape()[-1].value
                    x = tf.layers.dense(add_ac(x),n_out_features,activation=None)
                    x = unflatten_first_dim(x,sh)
                    x = tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, -1)
                    self.heads.append(x)
            with tf.variable_scope(self.scope):
                self.mask_placeholder = tf.placeholder(tf.float32, shape=(None, self.n_heads, 1),name='mask')
                self.heads = tf.stack(self.heads,axis=1)
                self.heads = mask_gradients(self.heads,self.mask_placeholder)

            with tf.variable_scope('heads_moments'):
                _,var = tf.nn.moments(self.heads,axes=1)
            
            res = tf.sqrt(var)
        elif self.dropout:
            with tf.variable_scope(self.scope):
                self.is_training = tf.placeholder(tf.bool,shape=[])
                self.dropout_params = {'init_min': 0.1, 'init_max': 0.1,
                      'weight_regularizer': 1e-6, 'dropout_regularizer': 1e-5,
                      'training': self.is_training}
                def residual(x):
                    res,reg = concrete_dropout(x,**self.dropout_params)
                    res = tf.layers.dense(add_ac(res),self.hidsize,activation=tf.nn.leaky_relu,kernel_regularizer=reg,bias_regularizer=reg)
                    res,reg = concrete_dropout(res,**self.dropout_params)
                    res = tf.layers.dense(add_ac(res),self.hidsize,activation=None,kernel_regularizer=reg,bias_regularizer=reg)
                    return x + res
                
                x = flatten_two_dims(self.features)
                x = add_ac(x)
                
                x,reg = concrete_dropout(x,**self.dropout_params)
                x = tf.layers.dense(add_ac(x),self.hidsize,activation=tf.nn.leaky_relu,kernel_regularizer=reg,bias_regularizer=reg)

                for _ in range(2):
                    x = residual(x)
                
                n_out_features = self.out_features.get_shape()[-1].value
                x,reg = concrete_dropout(x,**self.dropout_params)
                x = tf.layers.dense(add_ac(x),n_out_features,activation=None,kernel_regularizer=reg,bias_regularizer=reg)
                x = unflatten_first_dim(x,sh)
            res = tf.reduce_mean((x - tf.stop_gradient(self.out_features))**2,-1)

        else:
            with tf.variable_scope(self.scope):
                x = flatten_two_dims(self.features)
                x = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)

                def residual(x):
                    res = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)
                    res = tf.layers.dense(add_ac(res), self.hidsize, activation=None)
                    return x + res

                for _ in range(4):
                    x = residual(x)
                n_out_features = self.out_features.get_shape()[-1].value
                x = tf.layers.dense(add_ac(x), n_out_features, activation=None)
                x = unflatten_first_dim(x, sh)
            res = tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, -1)
            
        return res

    def calculate_loss(self, ob, last_ob, acs):
        
        n = ob.shape[0]
        chunk_size = n // self.n_chunks
        assert n % self.n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        if self.bootstrapped:
            mask_rv = bernoulli(.5)
            mask = [mask_rv.rvs(size=(chunk_size,self.n_heads,1)).astype('float32') for _ in range(self.n_chunks)]
            return np.concatenate([tf.get_default_session().run(self.loss,
                                            {self.obs: ob[sli(i)],self.last_ob: last_ob[sli(i)],
                                            self.ac: acs[sli(i)],self.mask_placeholder:mask[i]}) for i in range(self.n_chunks)],0),mask

        if self.dropout:
            return np.concatenate([tf.get_default_session().run(self.loss,
                                             {self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                              self.ac: acs[sli(i)],self.is_training: False}) for i in range(self.n_chunks)], 0),None
        # if self.flipout:
        #     return np.concatenate([tf.get_default_session().run([self.loss,self.addit_loss],
        #                                      {self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
        #                                       self.ac: acs[sli(i)]}) for i in range(self.n_chunks)], 0),None
        else:
            return np.concatenate([tf.get_default_session().run(self.loss,
                                             {self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                              self.ac: acs[sli(i)]}) for i in range(self.n_chunks)], 0),None

