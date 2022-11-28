from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys
import os

from config import Config

import tensorflow as tf
import numpy as np
import random

from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.agents import ReinforceAgent
from tf_agents.trajectories.time_step import StepType

import pickle

from rlEnvironment import RLEnvironment
from policyNetwork import KGActionDistNet

tf.compat.v1.enable_v2_behavior()
train_step_counter = tf.compat.v2.Variable(0)

config = Config.parse_from_file("")

# get parameters from config file
observation_spec_shape_x = 1001
observation_spec_shape_y = 769
action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=999, name="action")
observation_spec = array_spec.ArraySpec(shape=(config.observation_spec_shape_x, config.observation_spec_shape_y), dtype=np.float32, name="observation")


# initialize the environment
kgEnv = RLEnvironment(observation_spec=observation_spec,
                      action_spec=action_spec)

train_env = tf_py_environment.TFPyEnvironment(kgEnv)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=config.learning_rate)

# initialize the policy network
actor_network = KGActionDistNet(
    config.seed,
    train_env.observation_spec(),
    train_env.action_spec()
)

# initialize the agent
rfAgent = ReinforceAgent(
    train_env.time_step_spec(),
    tensor_spec.from_spec(action_spec),
    actor_network,
    optimizer,
    entropy_regularization=config.entropy_const,
    train_step_counter=train_step_counter
)
rfAgent.initialize()

# use a sampling policy
collect_policy = rfAgent.collect_policy


# buffer to store collected experience
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=collect_policy.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=5000
)


def collect_episodes_with_rollouts(environment, policy, num_episodes, num_rollouts):
    episode_counter = 0
    episode_return = 0.0

    while episode_counter < num_episodes:
        environment.set_is_rollout(False)
        # we are moving only one step each time
        time_step = environment.reset()

        if environment.is_final_observation():
            avg_return = episode_return / ((episode_counter+1)*num_rollouts)
            print("final obs")
            return avg_return

        # get back an action given your current state
        action_step = policy.action(time_step, seed=config.seed)

        # do next step on the environment
        next_time_step = environment.step(action_step.action)

        # collect the reward
        episode_return += next_time_step.reward
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # store collected experience
        replay_buffer.add_batch(traj)

        # get action distribution from policy network
        distribution = actor_network.get_distribution()

        # sample numrollout-1 additional actions
        selectedActions = tf.nest.map_structure(
            lambda d: d.sample((num_rollouts-1), seed=config.seed),
            distribution
        )
        print("selected Actions: ", selectedActions)

        environment.set_is_rollout(True)

        # get from environment potential new state when alternative action is chosen
        for selAction in selectedActions:
            new_policy_step = action_step.replace(action=selAction)
            next_time_step = environment.step(selAction)
            episode_return += next_time_step.reward
            traj = trajectory.from_transition(time_step, new_policy_step, next_time_step)
            # store additional experience
            replay_buffer.add_batch(traj)

        episode_counter += 1
    # calculate average reward
    avg_return = episode_return / (num_episodes*num_rollouts)
    return avg_return

# create checkpoint for weights of the policy network
checkpoint = tf.train.Checkpoint(actor_net=actor_network)

if config.pretrained:
    checkpoint.restore(config.pretrained_path)

# main training loop
for j in range(config.num_epochs):
    kgEnv.reset_env()
    i = -1
    while True:
        i += 1
        # collect experience with additional rollouts
        average_return = collect_episodes_with_rollouts(kgEnv, collect_policy, config.num_episodes, config.num_rollouts)
        experience = replay_buffer.gather_all()
        if i==0:
            print("weights: ", actor_network.trainable_weights, flush=True)
        # calculate loss
        train_loss = rfAgent.train(experience)
        if i % 100 == 0:
            print("iteration: ", i, flush=True)
            print("loss: ", train_loss.loss, flush=True)
            print("avg return: ", average_return, flush=True)
        replay_buffer.clear()

        if kgEnv.is_final_observation():
            break
    # save checkpoints for each epoch
    checkpoint.save(config.checkpoint_path + "-seed-"+str(config.seed) + "/ckpt")

print("trained weights: ", actor_network.trainable_weights, flush=True)

