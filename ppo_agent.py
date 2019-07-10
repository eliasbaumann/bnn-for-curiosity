import tensorflow as tf
from mpi4py import MPI
import numpy as np
from utils import MpiAdamOptimizer,bcast_tf_vars_from_root,get_mean_and_std
import time

from vec_env import ShmemVecEnv as VecEnv
from rollouts import Rollout

from baselines.common import explained_variance
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.mpi_moments import mpi_moments


class PpoOptimizer(object):
    def __init__(self,scope,ob_space,ac_space,policy,use_news,gamma,lam,nepochs,nminibatches,lr,cliprange,nsteps_per_seg,nsegs_per_env,ent_coeff,normrew,normadv,ext_coeff,int_coeff,dynamics):
        self.dynamics = dynamics
        with tf.variable_scope(scope):

            self.bootstrapped = self.dynamics.bootstrapped
            self.flipout = self.dynamics.flipout

            self.use_recorder = True
            self.n_updates = 0
            self.scope = scope
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.policy = policy
            self.nepochs = nepochs
            self.lr = lr
            self.cliprange = cliprange
            self.nsteps_per_seg = nsteps_per_seg
            self.nsegs_per_env = nsegs_per_env
            self.nminibatches = nminibatches
            self.gamma = gamma
            self.lam = lam
            self.normrew = normrew
            self.normadv = normadv
            self.use_news = use_news
            self.ext_coeff = ext_coeff
            self.int_coeff = int_coeff
            self.ent_coeff = ent_coeff

            self.placeholder_advantage = tf.placeholder(tf.float32, [None,None])
            self.placeholder_ret = tf.placeholder(tf.float32, [None,None])
            self.placeholder_rews = tf.placeholder(tf.float32, [None,None])
            self.placeholder_oldnlp = tf.placeholder(tf.float32, [None,None])
            self.placeholder_oldvpred = tf.placeholder(tf.float32, [None,None])
            self.placeholder_lr = tf.placeholder(tf.float32, [])
            self.placeholder_cliprange = tf.placeholder(tf.float32, [])

            # if self.flipout:
            #     self.placeholder_dyn_mean = tf.placeholder(tf.float32, [None,None])
            #     self.dyn_mean = tf.reduce_max(self.placeholder_dyn_mean)

            neglogpa = self.policy.pd.neglogp(self.policy.placeholder_action)
            entropy = tf.reduce_mean(self.policy.pd.entropy())
            vpred = self.policy.vpred

            c1 = .5

            vf_loss = c1* tf.reduce_mean(tf.square(vpred - self.placeholder_ret))
            ratio = tf.exp(self.placeholder_oldnlp - neglogpa)
            negadv = -self.placeholder_advantage

            polgrad_losses1 = negadv*ratio
            polgrad_losses2 = negadv * tf.clip_by_value(ratio,1.0-self.placeholder_cliprange,1.0+self.placeholder_cliprange)
            polgrad_loss_surr = tf.maximum(polgrad_losses1,polgrad_losses2)
            polgrad_loss = tf.reduce_mean(polgrad_loss_surr)
            entropy_loss = (-self.ent_coeff) * entropy
            
            approxkl = .5 * tf.reduce_mean(tf.square(neglogpa - self.placeholder_oldnlp))
            clipfrac = tf.reduce_mean(tf.to_float(tf.abs(polgrad_losses2 - polgrad_loss_surr) > 1e-6))
            if self.dynamics.dropout:
                regloss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.total_loss = polgrad_loss + entropy_loss + vf_loss + regloss #TODO i tried with negative for fun
                self.to_report = {'tot': self.total_loss, 'pg': polgrad_loss, 'vf': vf_loss, 'ent': entropy, 
                'approxkl': approxkl, 'clipfrac': clipfrac,'regloss':regloss}
                self.dropout_rates = tf.get_collection('DROPOUT_RATES')
            else:
                self.total_loss = polgrad_loss + entropy_loss + vf_loss
                self.to_report = {'tot': self.total_loss, 'pg': polgrad_loss, 'vf': vf_loss, 'ent': entropy, 
                'approxkl': approxkl, 'clipfrac': clipfrac}
            
            
            


        
    def start_interaction(self,env_fns,dynamics,nlump=2):
        self.loss_names, self._losses = zip(*list(self.to_report.items()))

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if MPI.COMM_WORLD.Get_size() > 1:
            trainer = MpiAdamOptimizer(learning_rate=self.placeholder_lr, comm=MPI.COMM_WORLD)
        else:
            trainer = tf.train.AdamOptimizer(learning_rate=self.placeholder_lr)
        gradsandvars = trainer.compute_gradients(self.total_loss, params)
        self._train = trainer.apply_gradients(gradsandvars)

        if MPI.COMM_WORLD.Get_rank() == 0:
            tf.get_default_session().run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        bcast_tf_vars_from_root(tf.get_default_session(), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        self.all_visited_rooms = []
        self.all_scores = []
        self.nenvs = nenvs = len(env_fns)
        self.nlump = nlump
        self.lump_stride = nenvs // self.nlump
        self.envs = [
            VecEnv(env_fns[l * self.lump_stride: (l + 1) * self.lump_stride], spaces=[self.ob_space, self.ac_space]) for
            l in range(self.nlump)]

        self.rollout = Rollout(ob_space=self.ob_space, ac_space=self.ac_space, nenvs=nenvs,
                            nsteps_per_seg=self.nsteps_per_seg,
                            nsegs_per_env=self.nsegs_per_env, nlumps=self.nlump,
                            envs=self.envs,
                            policy=self.policy,
                            int_rew_coeff=self.int_coeff,
                            ext_rew_coeff=self.ext_coeff,
                            record_rollouts=self.use_recorder,
                            dynamics=dynamics)

        self.buf_advs = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_rets = np.zeros((nenvs, self.rollout.nsteps), np.float32)

        if self.normrew:
            self.rff = RewardForwardFilter(self.gamma)
            self.rff_rms = RunningMeanStd()
            if self.dynamics.dropout:
                self.rff2 = RewardForwardFilter(self.gamma)
                self.rff_rms2 = RunningMeanStd()

        self.step_count = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

    def stop_interaction(self):
        for env in self.envs:
            env.close()

        
    def calculate_advantages(self, rews, use_news, gamma, lam):
        nsteps = self.rollout.nsteps
        lastgaelam = 0
        for t in range(nsteps - 1, -1, -1):  # nsteps-2 ... 0
            nextnew = self.rollout.buf_news[:, t + 1] if t + 1 < nsteps else self.rollout.buf_new_last
            if not use_news:
                nextnew = 0
            nextvals = self.rollout.buf_vpreds[:, t + 1] if t + 1 < nsteps else self.rollout.buf_vpred_last
            nextnotnew = 1 - nextnew
            delta = rews[:, t] + gamma * nextvals * nextnotnew - self.rollout.buf_vpreds[:, t]
            self.buf_advs[:, t] = lastgaelam = delta + gamma * lam * nextnotnew * lastgaelam
        self.buf_rets[:] = self.buf_advs + self.rollout.buf_vpreds

    def update(self):
        if self.normrew:
            rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])
            rffs_mean, rffs_std, rffs_count = mpi_moments(rffs.ravel())
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = self.rollout.buf_rews / np.sqrt(self.rff_rms.var)
            if self.dynamics.dropout:
                rffs2 = np.array([self.rff2.update(rew) for rew in self.rollout.buf_rews_mean.T])
                rffs2_mean, rffs2_std, rffs2_count = mpi_moments(rffs2.ravel())
                self.rff_rms2.update_from_moments(rffs2_mean, rffs2_std ** 2, rffs2_count)
                rews_m = self.rollout.buf_rews_mean / np.sqrt(self.rff_rms2.var)
                rews = rews_m+rews

        else:
            rews = np.copy(self.rollout.buf_rews)
        self.calculate_advantages(rews=rews, use_news=self.use_news, gamma=self.gamma, lam=self.lam)

        info = dict(
            advmean=self.buf_advs.mean(),
            advstd=self.buf_advs.std(),
            retmean=self.buf_rets.mean(),
            retstd=self.buf_rets.std(),
            vpredmean=self.rollout.buf_vpreds.mean(),
            vpredstd=self.rollout.buf_vpreds.std(),
            ev=explained_variance(self.rollout.buf_vpreds.ravel(), self.buf_rets.ravel()),
            rew_mean=np.mean(self.rollout.buf_rews),
            recent_best_ext_ret=self.rollout.current_max
        )
        if self.rollout.best_ext_ret is not None:
            info['best_ext_ret'] = self.rollout.best_ext_ret
        
        # if self.flipout:
        #     info['dyn_mean'] = np.mean(self.rollout.buf_dyn_rew)
        # normalize advantages
        if self.normadv:
            m, s = get_mean_and_std(self.buf_advs)
            self.buf_advs = (self.buf_advs - m) / (s + 1e-7)
        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        envsperbatch = max(1, envsperbatch)
        envinds = np.arange(self.nenvs * self.nsegs_per_env)

        def resh(x):
            if self.nsegs_per_env == 1:
                return x
            sh = x.shape
            return x.reshape((sh[0] * self.nsegs_per_env, self.nsteps_per_seg) + sh[2:])

        ph_buf = [
            (self.policy.placeholder_action, resh(self.rollout.buf_acs)),
            (self.placeholder_rews, resh(self.rollout.buf_rews)),
            (self.placeholder_oldvpred, resh(self.rollout.buf_vpreds)),
            (self.placeholder_oldnlp, resh(self.rollout.buf_nlps)),
            (self.policy.placeholder_observation, resh(self.rollout.buf_obs)),
            (self.placeholder_ret, resh(self.buf_rets)),
            (self.placeholder_advantage, resh(self.buf_advs)),
        ]
        ph_buf.extend([
            (self.dynamics.last_ob,
             self.rollout.buf_obs_last.reshape([self.nenvs * self.nsegs_per_env, 1, *self.ob_space.shape]))
        ])
        # if self.flipout:
        #     ph_buf.extend([(self.placeholder_dyn_mean, resh(self.buf_n_dyn_rew))])

        if self.bootstrapped:
            ph_buf.extend([(self.dynamics.mask_placeholder,self.rollout.buf_mask.reshape(-1,self.dynamics.n_heads,1))])
        mblossvals = []

        for _ in range(self.nepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                fd = {ph: buf[mbenvinds] for (ph, buf) in ph_buf}
                fd.update({self.placeholder_lr: self.lr, self.placeholder_cliprange: self.cliprange})
                if self.dynamics.dropout:
                    fd.update({self.dynamics.is_training: True})
                mblossvals.append(tf.get_default_session().run(self._losses + (self._train,), fd)[:-1])

        mblossvals = [mblossvals[0]]
        info.update(zip(['opt_' + ln for ln in self.loss_names], np.mean([mblossvals[0]], axis=0)))
        info["rank"] = MPI.COMM_WORLD.Get_rank()
        self.n_updates += 1
        info["n_updates"] = self.n_updates
        info.update({dn: (np.mean(dvs) if len(dvs) > 0 else 0) for (dn, dvs) in self.rollout.statlists.items()})
        info.update(self.rollout.stats)
        if "states_visited" in info:
            info.pop("states_visited")
        tnow = time.time()
        info["ups"] = 1. / (tnow - self.t_last_update)
        info["total_secs"] = tnow - self.t_start
        info['tps'] = MPI.COMM_WORLD.Get_size() * self.rollout.nsteps * self.nenvs / (tnow - self.t_last_update)
        self.t_last_update = tnow

        return info

    def step(self):
        self.rollout.collect_rollout()
        update_info = self.update()
        return {'update': update_info}

    def get_var_values(self):
        return self.policy.get_var_values()

    def set_var_values(self, vv):
        self.policy.set_var_values(vv)
        

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
