from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import sys

# Set seed values everywhere to make results reproducible.
with open(sys.argv[1], 'r') as config_file:
    config = json.load(config_file)
# set a seed value
seed_value = config["seed"]
# Set 'PYTHONHASHSEED' environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)
# set 'python' built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# set 'numpy' pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# set 'tensorflow' pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.agents import ReinforceAgent

import pickle

from rlEnvironment import RLEnvironment
from policyNetwork import KGActionDistNet

tf.compat.v1.enable_v2_behavior()
train_step_counter = tf.compat.v2.Variable(0)

# get parameters from config file
entropy_const = config["entropy_const"]
learning_rate = config["learning_rate"]
num_episodes = config["num_episodes"]
num_rollouts = config["num_rollouts"]
num_rollout_steps = config["num_rollout_steps"]
num_epochs = config["num_epochs"]
discount = config["discount"]

# wheter a pretrained model should be used
if "pretrained" in config.keys():
    pretrained = config["pretrained"]
else:
    pretrained = False

# wheter negative reward should be used instead reward=0
if "negative_reward" in config.keys():
    alt_reward = config["negative_reward"]
else:
    alt_reward = True

# get parameters from config file
observation_spec_shape_x = config["observation_spec_shape_x"]
observation_spec_shape_y = config["observation_spec_shape_y"]
action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=999, name="action")
observation_spec = array_spec.ArraySpec(shape=(observation_spec_shape_x, observation_spec_shape_y), dtype=np.float32, name="observation")

# list with all question ids
with open(config["question_list"], "r") as questions_file:
    question_list = json.load(questions_file)

# the startpoints we have determined upfront for each question in the trainset
with open(config["starts_per_question"], "r") as start_file:
    starts_per_question = json.load(start_file)

# paths per startpoint
with open(config["paths"], "r") as path_file:
    paths = json.load(path_file)

# all answers in trainset
with open(config["answers"], "r") as answer_file:
    answers = json.load(answer_file)

# BERT embedded questions
with open(config["encoded_questions"], "rb") as q_file:
    encoded_questions = pickle.load(q_file)

# BERT embedded actions
with open(config["encoded_actions"], "rb") as a_file:
    encoded_actions = pickle.load(a_file)

# number of actions available per startpint
with open(config["action_nbrs"], "r") as nbr_file:
    action_nbrs = json.load(nbr_file)

# indices of question and startpoint ids
with open(config["q_start_indices"], "r") as qfile:
    q_start_indices = json.load(qfile)

#whether negative reward should be used instead reward = 0
if "negative_reward" in config.keys():
  alt_reward = config["negative_reward"]
else:
  alt_reward = True

# initialize the environment
kgEnv = RLEnvironment(observation_spec=observation_spec,
                      action_spec=action_spec,
                      all_questions=encoded_questions,
                      all_actions=encoded_actions,
                      question_ids=question_list,
                      starts_per_question=starts_per_question,
                      action_nbrs=action_nbrs,
                      all_answers=answers,
                      paths=paths,
                      q_start_indices=q_start_indices,
                      alt_reward=alt_reward,
                      discount=discount,
                      num_rollouts=num_rollouts,
                      num_rollout_steps=num_rollout_steps)

train_env = tf_py_environment.TFPyEnvironment(kgEnv)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

# initialize the policy network
actor_network = KGActionDistNet(
    seed_value,
    train_env.observation_spec(),
    train_env.action_spec()
)

# initialize the agent
rfAgent = ReinforceAgent(
    train_env.time_step_spec(),
    tensor_spec.from_spec(action_spec),
    actor_network,
    optimizer,
    entropy_regularization=config["entropy_const"],
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


def collect_episodes_with_rollouts(environment, policy, num_episodes, num_rollouts, num_rollout_steps):
    episode_return = 0.0

    for episode_counter in range(num_episodes):

        next_time_steps_dict = dict({rollout_step: [] for rollout_step in range(num_rollout_steps+1)})

        # we are moving only one step each time
        initial_time_step = environment.reset()

        if environment.is_final_observation():
            avg_return = episode_return / ((episode_counter+1)*num_rollouts*num_rollouts)
            print("final obs")
            return avg_return

        next_time_steps_dict[0].append(initial_time_step)

        for rollout_step in range(1, num_rollout_steps+1):

            environment.set_current_rollout_step(rollout_step)

            for prev_time_step in next_time_steps_dict[rollout_step-1]:
                # get back an action given your current state
                action_step = policy.action(prev_time_step, seed=seed_value)
                # do next step on the environment
                next_time_step = environment.step(action_step.action)
                # collect the reward
                episode_return += next_time_step.reward
                traj = trajectory.from_transition(prev_time_step, action_step, next_time_step)
                # store collected experience
                replay_buffer.add_batch(traj)

                if not next_time_step.is_last():
                    next_time_steps_dict[rollout_step].append(next_time_step)

                # get action distribution from policy network
                distribution = actor_network.get_distribution()

                # sample numrollout-1 additional actions
                selectedActions = tf.nest.map_structure(
                     lambda d: d.sample((num_rollouts-1), seed=seed_value),
                     distribution
                )
                #print("selected Actions: ", selectedActions)

                environment.set_is_rollout(True)

                # get from environment potential new state when alternative action is chosen
                for selAction in selectedActions:
                    new_policy_step = action_step._replace(action=selAction)
                    next_time_step = environment.step(selAction)
                    episode_return += next_time_step.reward
                    traj = trajectory.from_transition(prev_time_step, new_policy_step, next_time_step)
                    # store additional experience
                    replay_buffer.add_batch(traj)

                    if not next_time_step.is_last():
                        next_time_steps_dict[rollout_step].append(next_time_step)

                environment.set_is_rollout(False)
        environment.next_question_counter()
    # calculate average reward
    avg_return = episode_return / (num_episodes*num_rollouts*num_rollout_steps)
    return avg_return

# create checkpoint for weights of the policy network
checkpoint = tf.train.Checkpoint(actor_net=actor_network)

if pretrained:
    checkpoint.restore(config["pretrained_path"])

# main training loop
for j in range(num_epochs):
    kgEnv.reset_env()
    i = -1
    while True:
        i += 1
        # collect experience with additional rollouts
        average_return = collect_episodes_with_rollouts(kgEnv, collect_policy, num_episodes, num_rollouts, num_rollout_steps)
        experience = replay_buffer.gather_all()
        # calculate loss
        train_loss = rfAgent.train(experience)
        print("iteration: {0}/{1}/{2}, loss: {3}, avg return: {4}".format(i, j, num_epochs, train_loss.loss, average_return), flush=True)
        replay_buffer.clear()

        if kgEnv.is_final_observation():
            break
    # save checkpoints for each epoch
    checkpoint.save(config["checkpoint_path"] + "-seed-"+str(seed_value) + "/ckpt")

print("trained weights: ", actor_network.trainable_weights, flush=True)

