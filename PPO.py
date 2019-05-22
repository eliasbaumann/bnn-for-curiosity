import tensorflow as tf
import numpy as np
import os
FILE_PATH = os.path.dirname(os.path.abspath(__file__))


from Curiosity import Curiosity

from utils import small_convnet, RunningMeanStd, flatten_2d

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

    def _actor_network(self, name, trainable):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if(len(self.OBS_DIM) > 2):
                act = tf.layers.dense(
                    self.cnn, 200, tf.nn.leaky_relu, trainable=trainable)
            else:
                act = tf.layers.dense(
                    self.inp, 200, tf.nn.leaky_relu, trainable=trainable)
            # mu = 2* tf.layers.dense(act,ACTION_DIM,tf.nn.tanh,trainable=trainable)
            action_probs = tf.layers.dense(
                act, self.ACTION_DIM, tf.nn.softmax, trainable=trainable)
            # norm_dist = tf.distributions.Normal(loc=mu,scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return action_probs, params

    def __init__(self, STATE_LATENT_SHAPE, OBS_DIM, ACTION_DIM, UPDATE_EVENT, ROLLING_EVENT, COORD, QUEUE,OBS_MEAN,OBS_STD,feature_dims=256, CRITIC_LR=.0001, ACTOR_LR=.0002, EPSILON=.2, EPISODE_MAX=1000,MIN_BATCH_SIZE=64, UPDATE_STEP=10):
        self.sess = tf.Session()
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM
        self.CRITIC_LR = CRITIC_LR
        self.ACTOR_LR = ACTOR_LR
        self.EPSILON = EPSILON
        self.EPISODE_MAX = EPISODE_MAX
        self.UPDATE_STEP = UPDATE_STEP
        self.STATE_LATENT_SHAPE = STATE_LATENT_SHAPE
        self.MIN_BATCH_SIZE = MIN_BATCH_SIZE

        self.UPDATE_EVENT = UPDATE_EVENT
        self.ROLLING_EVENT = ROLLING_EVENT
        self.COORD = COORD
        self.QUEUE = QUEUE

        self.OBS_MEAN =OBS_MEAN
        self.OBS_STD = OBS_STD
        self.feature_dims = feature_dims

        self.inp = tf.placeholder(
            tf.float32, (None,)+self.OBS_DIM, name='state')
        if(len(OBS_DIM) > 2):
            self.inp = tf.div_no_nan(tf.subtract(tf.to_float(self.inp),self.OBS_MEAN),self.OBS_STD)
            # self.inp = flatten_2d(self.inp)
            self.cnn = small_convnet(x=self.inp, nl=tf.nn.leaky_relu,
                                     feat_dim=self.feature_dims, last_nl=tf.nn.leaky_relu,
                                     layernormalize=False)
            crit = tf.layers.dense(
                self.cnn, 200, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0., .1))
        else:
            crit = tf.layers.dense(
                self.inp, 200, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0., .1))
        # Critic
        self.value = tf.layers.dense(crit, 1)
        self.dc_reward = tf.placeholder(
            tf.float32, [None, 1], name='discounted_reward')
        self.advantage = self.dc_reward - self.value
        self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
        self.critic_train_opt = tf.train.AdamOptimizer(
            self.CRITIC_LR).minimize(self.critic_loss)

        # Actor
        self.policy, theta_p = self._actor_network('policy', trainable=True)
        old_policy, old_theta_p = self._actor_network(
            'old_policy', trainable=False)
        self.update_old_policy_op = [oldp.assign(
            p) for p, oldp in zip(theta_p, old_theta_p)]

        self.action = tf.placeholder(tf.int32, [None, ], name='action')
        self.full_adv = tf.placeholder(tf.float32, [None, 1], name='advantage')

        a_indices = tf.stack(
            [tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
        pi_prob = tf.gather_nd(params=self.policy, indices=a_indices)
        oldpi_prob = tf.gather_nd(params=old_policy, indices=a_indices)

        surr = (pi_prob/oldpi_prob) * self.full_adv

        self.actor_loss = -tf.reduce_mean(tf.minimum(surr, tf.clip_by_value(
            pi_prob/oldpi_prob, 1.-self.EPSILON, 1.+self.EPSILON)*self.full_adv))

        self.actor_train_opt = tf.train.AdamOptimizer(
            self.ACTOR_LR).minimize(self.actor_loss)

        self.curiosity = Curiosity(
            self.sess, self.STATE_LATENT_SHAPE, self.OBS_DIM, self.ACTION_DIM, self.UPDATE_STEP,
            self.OBS_MEAN,self.OBS_STD,self.inp,MIN_BATCH_SIZE=self.MIN_BATCH_SIZE,uncertainty=True)

        self.r_rew_tracker = RunningMeanStd()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        # self.saver.save(self.sess,FILE_PATH+'\model')
        # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

    def get_action(self, state):
        action_probs = self.sess.run(self.policy, {self.inp: state})
        if(np.isnan(action_probs).any()):
            action = np.random.choice(range(action_probs.shape[1]))
        else:
            action = np.random.choice(
                range(action_probs.shape[1]), p=action_probs.ravel())
        return action

    def get_value(self, state):
        return self.sess.run(self.value, {self.inp: state})[0, 0]

    def norm_adv(self, adv):
        return (adv-np.mean(adv))/np.std(adv)

    def update(self):
        while not self.COORD.should_stop():
            if GLOBAL_EPISODE < self.EPISODE_MAX:
                self.UPDATE_EVENT.wait()
                
                self.sess.run(self.update_old_policy_op)
                data = []
                for _ in range(self.QUEUE.qsize()):
                    data.append(self.QUEUE.get())
                    self.QUEUE.task_done()

                data = np.vstack(data)
                interv = np.prod(self.OBS_DIM)
                state, state_, action, reward = data[:, :interv], data[:, interv:2 *
                                                                       interv], data[:, 2*interv: 2*interv + 1].ravel(), data[:, -1:]
                state = state.reshape((data.shape[0],)+self.OBS_DIM)
                state_ = state_.reshape((data.shape[0],)+self.OBS_DIM)

                self.r_rew_tracker.update_from_moments(
                    np.mean(reward), np.var(reward), reward.shape[0])  # mean,var,count
                self.norm_rew = reward/np.sqrt(self.r_rew_tracker.var)

                advantage = self.sess.run(
                    self.advantage, {self.inp: state, self.dc_reward: self.norm_rew})

                normalized_adv = self.norm_adv(advantage)

                [self.sess.run([self.actor_train_opt, self.actor_loss], {
                                        self.inp: state, self.action: action, self.full_adv: normalized_adv}) for _ in range(self.UPDATE_STEP)]
                [self.sess.run([self.critic_train_opt, self.critic_loss], {
                                        self.inp: state, self.dc_reward: reward}) for _ in range(self.UPDATE_STEP)]

                self.curiosity.update(state, state_, action)

                self.UPDATE_EVENT.clear()        # updating finished
                resetGlobalUpdateCounter()     # reset counter
                self.ROLLING_EVENT.set()         # set roll-out available
