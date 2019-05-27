# All code here is yoinked from somewhere else...

import numpy as np
import tensorflow as tf
from functools import partial
import gym
from gym import spaces
from wrappers import WarpFrame,NoopResetEnv,MaxAndSkipEnv,FrameStack

def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)
    return _initializer


def layernorm(x):
    m, v = tf.nn.moments(x, -1, keep_dims=True)
    return (x - m) / (tf.sqrt(v) + 1e-8)


fc = partial(tf.keras.layers.Dense, kernel_initializer=normc_initializer(1.))

# last_nl =tf.nn.leaky_relu


def small_convnet(x, nl, feat_dim, last_nl, layernormalize, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    x = bn(tf.keras.layers.Conv2D(filters=32,
                            kernel_size=8, strides=(4, 4), activation=nl)(x))
    x = bn(tf.keras.layers.Conv2D(filters=64,
                            kernel_size=4, strides=(2, 2), activation=nl)(x))
    x = bn(tf.keras.layers.Conv2D(filters=64,
                            kernel_size=3, strides=(1, 1), activation=nl)(x))
    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = bn(fc(units=feat_dim, activation=last_nl)(x))
    # if last_nl is not None:
    #     x = last_nl(x)
    if layernormalize:
        x = layernorm(x)
    return x

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class TfRunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=(), scope=''):
        sess = tf.get_default_session()

        self._new_mean = tf.placeholder(shape=shape, dtype=tf.float64)
        self._new_var = tf.placeholder(shape=shape, dtype=tf.float64)
        self._new_count = tf.placeholder(shape=(), dtype=tf.float64)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self._mean = tf.get_variable('mean',  initializer=np.zeros(
                shape, 'float64'),      dtype=tf.float64)
            self._var = tf.get_variable('std',   initializer=np.ones(
                shape, 'float64'),       dtype=tf.float64)
            self._count = tf.get_variable('count', initializer=np.full(
                (), epsilon, 'float64'), dtype=tf.float64)

        self.update_ops = tf.group([
            self._var.assign(self._new_var),
            self._mean.assign(self._new_mean),
            self._count.assign(self._new_count)
        ])

        sess.run(tf.variables_initializer(
            [self._mean, self._var, self._count]))
        self.sess = sess
        self._set_mean_var_count()

    def _set_mean_var_count(self):
        self.mean, self.var, self.count = self.sess.run(
            [self._mean, self._var, self._count])

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_mean, new_var, new_count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

        self.sess.run(self.update_ops, feed_dict={
            self._new_mean: new_mean,
            self._new_var: new_var,
            self._new_count: new_count
        })

        self._set_mean_var_count()


def make_env(GAME_NAME):
    env = gym.make(GAME_NAME)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if(len(env.observation_space.shape) > 2):
        env = WarpFrame(env, 84, 84, True)    
    env = FrameStack(env, 4)
    return env    
    
def get_env_mean_std(GAME_NAME, n_steps=10000):
    env = make_env(GAME_NAME)
    states = []
    state = env.reset()
    states.append(np.asarray(state))
    for _ in range(n_steps):
        action = env.action_space.sample()
        state, _, done, _ = env.step(action)
        if done:
            state = env.reset()
        states.append(np.asarray(state))
    mean,std = np.mean(states,axis=0).astype(np.float32),np.std(states,axis=0).mean().astype(np.float32)
    return mean,std

def flatten_2d(x):
    return tf.reshape(x, [-1] + x.get_shape().as_list()[1:])
