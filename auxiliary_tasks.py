import tensorflow as tf
from utils import flatten_two_dims,unflatten_first_dim,small_convnet,activ,fc

class InverseDynamics(object):
    def __init__(self,policy,feat_dim=None,layernormalize=None,scope='feature_extractor'):
        self.scope = scope
        self.feat_dim = feat_dim
        self.layernormalize = layernormalize
        self.policy = policy
        self.hidsize = policy.hidsize
        self.ob_space = policy.ob_space
        self.ac_space = policy.ac_space
        self.obs = self.policy.placeholder_observation
        self.ob_mean = self.policy.ob_mean
        self.ob_std = self.policy.ob_std

        with tf.variable_scope(scope):
            self.last_ob = tf.placeholder(dtype=tf.int32,
                                          shape=(None, 1) + self.ob_space.shape, name='last_ob')
            self.next_ob = tf.concat([self.obs[:, 1:], self.last_ob], 1)
            self.features = self.get_features(self.obs, reuse=False)
            self.last_features = self.get_features(self.last_ob, reuse=True)
        self.next_features = tf.concat([self.features[:, 1:], self.last_features], 1)
        self.ac = self.policy.placeholder_action
        self.loss = self.get_loss()

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x
    
    def get_loss(self):
        with tf.variable_scope(self.scope):
            x = tf.concat([self.features, self.next_features], 2)
            sh = tf.shape(x)
            x = flatten_two_dims(x)
            x = fc(x, units=self.policy.hidsize, activation=activ)
            x = fc(x, units=self.ac_space.n, activation=None)
            param = unflatten_first_dim(x, sh)
            idfpd = self.policy.ac_pdtype.pdfromflat(param)
            return idfpd.neglogp(self.ac)
