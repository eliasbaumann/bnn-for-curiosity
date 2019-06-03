import multiprocessing
import tensorflow as tf
import numpy as np
import random
import os
import platform
from mpi4py import MPI
from baselines.common.tf_util import normc_initializer
from functools import partial

from tensorflow.python.client import device_lib

def setup_tensorflow_session():
    num_cpu = int(multiprocessing.cpu_count())
    
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu
    )
    return tf.Session(config=tf_config)

def guess_gpus():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_divices = os.environ['CUDA_VISIBLE_DEVICES']
        cuda_visible_divices = cuda_visible_divices.split(',')
        return [int(n) for n in cuda_visible_divices]
    nvidia_dir = '/proc/driver/nvidia/gpus/'
    if os.path.exists(nvidia_dir):
        n_gpus = len(os.listdir(nvidia_dir))
        return list(range(n_gpus))
    
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    n = get_available_gpus()
    if len(n)!=0:
        return list(range(len(n)))
    raise Exception('no gpu found')

def setup_mpi_gpus():
    available_gpus = guess_gpus()
    node_id = platform.node()
    nodes_ordered_by_rank = MPI.COMM_WORLD.allgather(node_id)
    processes_outranked_on_this_node = [n for n in nodes_ordered_by_rank[:MPI.COMM_WORLD.Get_rank()] if n == node_id]
    local_rank = len(processes_outranked_on_this_node)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[local_rank])

def random_agent_ob_mean_std(env,nsteps=10000):
    ob = np.asarray(env.reset())
    if MPI.COMM_WORLD.Get_rank() == 0:
        obs = [ob]
        for _ in range(nsteps):
            ac = env.action_space.sample()
            ob,_,done,_ = env.step(ac)
            if done:
                ob = env.reset()
            obs.append(np.asarray(ob))
        mean = np.mean(obs,0).astype(np.float32)
        std = np.std(obs,0).mean().astype(np.float32)
    else:
        mean = np.empty(shape=ob.shape, dtype=np.float32)
        std = np.empty(shape=(), dtype=np.float32)
    MPI.COMM_WORLD.Bcast(mean, root=0)
    MPI.COMM_WORLD.Bcast(std, root=0)
    return mean, std

def flatten_two_dims(x):
    return tf.reshape(x, [-1] + x.get_shape().as_list()[2:])
    
def unflatten_first_dim(x, sh):
    return tf.reshape(x, [sh[0], sh[1]] + x.get_shape().as_list()[1:])

def unflatten_tiled(x,sh):
    return tf.reshape(x, [x.get_shape().as_list()[0],sh[0],sh[1]]+x.get_shape().as_list()[2:])

def layernorm(x):
    m, v = tf.nn.moments(x, -1, keep_dims=True)
    return (x - m) / (tf.sqrt(v) + 1e-8)

def small_convnet(x, nl, feat_dim, last_nl, layernormalize, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    x = bn(tf.layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=nl))
    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = bn(fc(x, units=feat_dim, activation=None))
    if last_nl is not None:
        x = last_nl(x)
    if layernormalize:
        x = layernorm(x)
    return x

class MpiAdamOptimizer(tf.train.AdamOptimizer):
    """Adam optimizer that averages gradients across mpi processes."""

    def __init__(self, comm, **kwargs):
        self.comm = comm
        tf.train.AdamOptimizer.__init__(self, **kwargs)

    def compute_gradients(self, loss, var_list, **kwargs):
        grads_and_vars = tf.train.AdamOptimizer.compute_gradients(self, loss, var_list, **kwargs)
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g, v in grads_and_vars], axis=0)
        shapes = [v.shape.as_list() for g, v in grads_and_vars]
        sizes = [int(np.prod(s)) for s in shapes]

        _task_id, num_tasks = self.comm.Get_rank(), self.comm.Get_size()
        buf = np.zeros(sum(sizes), np.float32)

        def _collect_grads(flat_grad):
            self.comm.Allreduce(flat_grad, buf, op=MPI.SUM)
            np.divide(buf, float(num_tasks), out=buf)
            return buf

        avg_flat_grad = tf.py_func(_collect_grads, [flat_grad], tf.float32)
        avg_flat_grad.set_shape(flat_grad.shape)
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = [(tf.reshape(g, v.shape), v)
                              for g, (_, v) in zip(avg_grads, grads_and_vars)]

        return avg_grads_and_vars
def bcast_tf_vars_from_root(sess, vars):
    """
    Send the root node's parameters to every worker.

    Arguments:
      sess: the TensorFlow session.
      vars: all parameter variables including optimizer's
    """
    rank = MPI.COMM_WORLD.Get_rank()
    for var in vars:
        if rank == 0:
            MPI.COMM_WORLD.bcast(sess.run(var))
        else:
            sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))

def get_mean_and_std(array):
    comm = MPI.COMM_WORLD
    num_tasks = comm.Get_size()
    local_mean = np.array(np.mean(array))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    mean = sum_of_means / num_tasks

    n_array = array - mean
    sqs = n_array ** 2
    local_mean = np.array(np.mean(sqs))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    var = sum_of_means / num_tasks
    std = var ** 0.5
    return mean, std

fc = partial(tf.layers.dense, kernel_initializer=normc_initializer(1.))
activ = tf.nn.relu
