import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from utils import small_convnet,flatten_2d

class Curiosity(object):
    def __init__(self, sess, STATE_LATENT_SHAPE, OBS_DIM, ACTION_DIM, UPDATE_STEP,OBS_MEAN,OBS_STD,PPO_input,feature_dims=256, INV_LR=.0001, FOR_LR=.0001, ETA=1, uncertainty=True,MIN_BATCH_SIZE=64,HID_SIZE = 256):

        self.sess = sess
        self.uncertainty = uncertainty
        self.scope = 'feature_extract'

        self.STATE_LATENT_SHAPE = STATE_LATENT_SHAPE
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM

        self.OBS_MEAN = OBS_MEAN
        self.OBS_STD = OBS_STD

        self.INV_LR = INV_LR
        self.FOR_LR = FOR_LR
        self.ETA = ETA
        self.UPDATE_STEP = UPDATE_STEP
        self.MIN_BATCH_SIZE = MIN_BATCH_SIZE
        self.HID_SIZE = HID_SIZE

        self.inp = PPO_input
        self.inp_1 = tf.placeholder(
            tf.float32, (None,)+self.OBS_DIM, name='S_t_1_input')
        self.inp_at = tf.placeholder(
            tf.float32, [None, self.ACTION_DIM], name='A_t_input')
        
        self.feature_dims = feature_dims

        if(len(self.OBS_DIM) > 2):
            self.cnn = self.get_features(self.inp,reuse=False)
            self.cnn_1 = self.get_features(self.inp_1,reuse=True)
        
        if(self.uncertainty):
            self.i_model,self.i_losses = self.bnn_inverse_model('BNN_inverse')
            self.f_model,self.f_losses = self.bnn_forward_model('BNN_forward')
        else:
            self.i_model = self.inverse_model('ICM_inverse')
            self.f_model = self.forward_model('ICM_forward')

        self.inverse_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            onehot_labels=self.inp_at, logits=self.a_hat))
        self.forward_loss = .5 * \
            tf.reduce_mean(tf.square(tf.subtract(
                self.phi_hat_st_, self.phi_st)))
        self.curiosity = tf.divide(
            ETA, 2.0) * tf.reduce_mean(tf.square(tf.subtract(self.phi_hat_st_, self.phi_st)), axis=1)

        if(self.uncertainty):
            kl_i = np.sum(self.i_losses)/self.MIN_BATCH_SIZE #TODO: what exactly to we need to use as divisor?
            kl_f = np.sum(self.f_losses)/self.MIN_BATCH_SIZE
            self.inverse_loss+=kl_i
            self.forward_loss+=kl_f

        self.inv_opt = tf.train.AdamOptimizer(
            INV_LR).minimize(self.inverse_loss)
        self.for_opt = tf.train.AdamOptimizer(
            FOR_LR).minimize(self.forward_loss)

    def inverse_model(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if(len(self.OBS_DIM) > 2):
                features = tf.layers.dense(
                    tf.concat([self.cnn, self.cnn_1], axis=1), 200, tf.nn.leaky_relu)
            else:
                features = tf.layers.dense(
                    tf.concat([self.inp, self.inp_1], axis=1), 200, tf.nn.leaky_relu)

            self.phi_st = tf.layers.dense(
                features, self.STATE_LATENT_SHAPE, tf.nn.leaky_relu)
            self.phi_st_ = tf.layers.dense(
                features, self.STATE_LATENT_SHAPE, tf.nn.leaky_relu)
            inv1 = tf.layers.dense(
                tf.concat([self.phi_st, self.phi_st_], axis=1), 200, tf.nn.leaky_relu)
            self.a_hat = tf.layers.dense(inv1, self.ACTION_DIM, tf.nn.softmax)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return params

    def forward_model(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            def add_ac(x):
                return tf.concat([self.inp_at,x],axis=1)

            x = tf.layers.dense(
                add_ac(self.phi_st), self.HID_SIZE, tf.nn.leaky_relu)
            
            def residual(x):
                res = tf.layers.dense(add_ac(x),self.HID_SIZE,activation=tf.nn.leaky_relu)
                res = tf.layers.dense(add_ac(res),self.HID_SIZE,activation=None)
                return x + res

            for _ in range(4):
                x = residual(x)

            self.phi_hat_st_ = tf.layers.dense(
                x, self.STATE_LATENT_SHAPE, tf.nn.leaky_relu)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return params
    
    # I am pretty sure the complexity of this network is too high, maybe reduce if by only using flipout in the end?
    def bnn_forward_model(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            def add_ac(x):
                return tf.concat([self.inp_at,x],axis=1)
            
            def residual(x):
                res = tfp.layers.DenseFlipout(self.HID_SIZE,activation=tf.nn.leaky_relu)(add_ac(x))
                res = tfp.layers.DenseFlipout(self.HID_SIZE,activation=None)(add_ac(res))
                return x+res

            x = tfp.layers.DenseFlipout(
                self.HID_SIZE, tf.nn.leaky_relu)(add_ac(self.phi_st))
            # for _ in range(2):
            x = residual(x)
            
            flipout_layer = tfp.layers.DenseFlipout(
                self.STATE_LATENT_SHAPE, tf.nn.leaky_relu)
            self.phi_hat_st_ = flipout_layer(x)
            losses = flipout_layer.losses

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return params,losses

    def bnn_inverse_model(self, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if(len(self.OBS_DIM) > 2):
                features = tfp.layers.DenseFlipout(self.HID_SIZE, tf.nn.leaky_relu)(tf.concat([self.cnn, self.cnn_1], axis=1))
            else:
                features = tfp.layers.DenseFlipout(self.HID_SIZE, tf.nn.leaky_relu)(tf.concat([self.inp, self.inp_1], axis=1))

            self.phi_st = tfp.layers.DenseFlipout(
                self.STATE_LATENT_SHAPE, tf.nn.leaky_relu)(features)
            self.phi_st_ = tfp.layers.DenseFlipout(
                self.STATE_LATENT_SHAPE, tf.nn.leaky_relu)(features)
            inv1 = tfp.layers.DenseFlipout(
                self.HID_SIZE, tf.nn.leaky_relu)(tf.concat([self.phi_st, self.phi_st_], axis=1))
            flipout_layer = tfp.layers.DenseFlipout(self.ACTION_DIM, tf.nn.softmax)
            self.a_hat = flipout_layer(inv1)
            losses = flipout_layer.losses
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        
        return params,losses

    def reshape_data(self, s_t, s_t_, a_t, from_pixels):
        batch_size = s_t.shape[0]
        if(not from_pixels):
            s_t = s_t.reshape((batch_size, -1))
            s_t_ = s_t_.reshape((batch_size, -1))

        a_t_new = np.reshape(
            np.repeat(np.repeat(0., self.ACTION_DIM), s_t.shape[0]), (batch_size, -1))

        a_t_new[np.arange(a_t_new.shape[0]), np.int8(a_t)] = 1.

        return s_t, s_t_, a_t_new

    def repeat_data(self, s_t, s_t_, a_t, times):
        s_t = np.repeat(s_t, times, 0)
        s_t_ = np.repeat(s_t_, times, 0)
        a_t = np.repeat(a_t, times, 0)
        return s_t, s_t_, a_t

    def get_reward(self, s_t, s_t_, a_t):

        s_t, s_t_, a_t = self.reshape_data(
            s_t, s_t_, a_t, len(self.OBS_DIM) > 2)

        if(self.uncertainty):
            s_t, s_t_, a_t = self.repeat_data(s_t, s_t_, a_t, 50)

            curs = self.sess.run(
                self.curiosity, {self.inp: s_t, self.inp_1: s_t_, self.inp_at: a_t})
            
            # tbd if var or stdev, probably does not make any difference, except for how the value is scaled
            # var = np.var(curs)
            stdev = np.std(curs)
            return stdev
        else:
            return self.sess.run(self.curiosity, {self.inp: s_t, self.inp_1: s_t_, self.inp_at: a_t})
    
    def get_features(self,x,reuse):
        with tf.variable_scope(self.scope,reuse=reuse):
            x = (tf.cast(x,tf.float32) - self.OBS_MEAN) / self.OBS_STD
            x = small_convnet(x, nl=tf.nn.leaky_relu, feat_dim=self.feature_dims, last_nl=None,layernormalize=True)
        return x

    def update(self, state, state_, action):
        state, state_, action = self.reshape_data(
            state, state_, action, len(self.OBS_DIM) > 2)
        [self.sess.run(self.inv_opt, {self.inp: state, self.inp_1: state_,
                                      self.inp_at: action}) for _ in range(self.UPDATE_STEP)]
        [self.sess.run(self.for_opt, {self.inp: state, self.inp_1: state_,
                                      self.inp_at: action}) for _ in range(self.UPDATE_STEP)]
