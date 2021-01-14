#!/usr/bin/env python3
"""Example script to run RL2 in any ML1 env."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import metaworld
import os

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer

# yapf: enable


@click.command()
@click.option('--gpu', default=0)
@click.option('--seed', default=0)
@click.option('--meta_batch_size', default=10)
@click.option('--n_epochs', default=200)
@click.option('--episode_per_task', default=10)
@click.option('--name', default='reach-v1')
@click.option('--prefix', default='rl2_ppo')
@wrap_experiment
def rl2_ppo_metaworld_ml1(ctxt=None, 
                          gpu=0, 
                          seed=0, 
                          meta_batch_size=10, 
                          n_epochs=10,
                          episode_per_task=10,
                          name='reach-v1', 
                          prefix='rl2_ppo'):
    """Train RL2 PPO with ML1 environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        gpu (int)
        seed (int): Used to seed the random number generator to produce
            determinism.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.
        name (str)
        prefix (str)

    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) 
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    set_seed(seed)
    ml1 = metaworld.ML1(name)

    task_sampler = MetaWorldTaskSampler(ml1, 'train',
                                        lambda env, _: RL2Env(env))
    env = task_sampler.sample(1)[0]()
    test_task_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                       env=MetaWorldSetTaskEnv(ml1, 'test'),
                                       wrapper=lambda env, _: RL2Env(env))
    env_spec = env.spec

    with TFTrainer(snapshot_config=ctxt) as trainer:
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=env_spec,
                                   state_include_action=False)

        meta_evaluator = MetaEvaluator(test_task_sampler=test_task_sampler, 
                                       n_test_tasks=10)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        envs = task_sampler.sample(meta_batch_size)
        sampler = LocalSampler(
            agents=policy,
            envs=envs,
            max_episode_length=env_spec.max_episode_length,
            is_tf_worker=True,
            n_workers=meta_batch_size,
            worker_class=RL2Worker,
            worker_args=dict(n_episodes_per_trial=episode_per_task))

        algo = RL2PPO(meta_batch_size=meta_batch_size,
                      task_sampler=task_sampler,
                      env_spec=env_spec,
                      policy=policy,
                      baseline=baseline,
                      sampler=sampler,
                      discount=0.99,
                      gae_lambda=0.95,
                      lr_clip_range=0.2,
                      optimizer_args=dict(batch_size=32,
                                          max_optimization_epochs=10),
                      stop_entropy_gradient=True,
                      entropy_method='max',
                      policy_ent_coeff=0.02,
                      center_adv=False,
                      meta_evaluator=meta_evaluator,
                      episodes_per_trial=episode_per_task)

        trainer.setup(algo, envs)

        trainer.train(n_epochs=n_epochs,
                      batch_size=episode_per_task *
                      env_spec.max_episode_length * meta_batch_size)


rl2_ppo_metaworld_ml1()
