import gym
import tensorflow as tf
from mpi4py import MPI

from cnn_policy import CnnPolicy

from wrappers import MaxAndSkipEnv,ProcessFrame84,ExtraTimeLimit,AddRandomStateToInfo
from baselines.common.atari_wrappers import NoopResetEnv,FrameStack
from functools import partial
from auxiliary_tasks import InverseDynamics
from dynamics import Dynamics
from ppo_agent import PpoOptimizer

import os

from baselines import logger
from baselines.common import set_global_seeds
from baselines.bench import Monitor

from utils import setup_mpi_gpus,setup_tensorflow_session,random_agent_ob_mean_std
from gym.utils.seeding import hash_seed

class Trainer(object):
    def __init__(self,make_env,num_timesteps,envs_per_process):
        self.make_env = make_env
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()

        self.policy = CnnPolicy(scope='cnn_pol',ob_space = self.ob_space,ac_space=self.ac_space,hidsize=512,feat_dim=512,ob_mean=self.ob_mean,ob_std = self.ob_std,layernormalize=False,nl=tf.nn.leaky_relu)
        self.feature_extractor = InverseDynamics(policy=self.policy,
                                                 feat_dim=512,
                                                 layernormalize = 0) # DEFAULT is 0 dont know hwat it does
        self.dynamics = Dynamics(auxiliary_task = self.feature_extractor,uncertainty=True,bootstrapped=False,feat_dim=512)
        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space = self.ob_space,
            ac_space = self.ac_space,
            policy = self.policy,
            use_news = 0,
            gamma = .99,
            lam = .95,
            nepochs = 3,
            nminibatches = 6,
            lr = 1e-4,
            cliprange = .1,
            nsteps_per_seg = 128, 
            nsegs_per_env = 1,
            ent_coeff = .001,
            normrew = 1,
            normadv = 1,
            ext_coeff = 0.,
            int_coeff= 1.,
            dynamics = self.dynamics
        )
        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss']
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        env = self.make_env(0)
        self.ob_space, self.ac_space = env.observation_space,env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [partial(self.make_env,i) for i in range(self.envs_per_process)]
    
    def train(self):
        self.agent.start_interaction(self.envs,nlump=1,dynamics = self.dynamics)
        while True:
            info = self.agent.step()
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break
        self.agent.stop_interaction()

def get_experiment_environment(**args):
    process_seed = 1234+1000*MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed,max_bytes=4)
    set_global_seeds(1234)
    setup_mpi_gpus()

    
    logger_context = logger.scoped_configure(dir=None,
                                             format_strs=['stdout', 'log',
                                                          'csv','tensorboard'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])

    tf_context = setup_tensorflow_session()
    return logger_context,tf_context



def start_experiment(**args):
    make_env = partial(make_env_all_params,args=args)
    logger.set_level(logger.DEBUG)
    
    trainer = Trainer(make_env=make_env,num_timesteps=int(1e8),envs_per_process=N_THREADS)#TODO 
    log,tf_sess = get_experiment_environment(**args)
    with log, tf_sess:
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        trainer.train()

def make_env_all_params(rank,args):
    env = gym.make(GAME_NAME)
    env = NoopResetEnv(env,noop_max=NOOP_MAX)
    env = MaxAndSkipEnv(env,skip=4)
    env = ProcessFrame84(env,crop=False)
    env = FrameStack(env,4)
    # env = ExtraTimeLimit(env,10000)
    env = AddRandomStateToInfo(env)
    env = Monitor(env, os.path.join('C:/Users/Elias/OneDrive/Winfo Studium/SS19/Masterarbeit/logs', '%.2i' % rank))
    return env

if __name__ == '__main__':
    N_THREADS = 24
    GAME_NAME = 'Riverraid-v0'
    NOOP_MAX = 30
    start_experiment()