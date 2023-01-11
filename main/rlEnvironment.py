from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


"""KGQA Environment"""
class RLEnvironment(py_environment.PyEnvironment):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 all_questions,
                 question_ids,
                 starts_per_question,
                 q_start_indices,
                 all_actions,
                 action_nbrs,
                 all_answers,
                 paths,
                 alt_reward,
                 discount,
                 num_rollouts,
                 num_rollout_steps):
        """
        :param observation_spec: observeration specification
        :param action_spec: action specification
        :param all_questions: question encodings
        :param question_ids: list with all question ids
        :param starts_per_question: context entities per startpoint
        :param q_start_indices: indices for qid and context entity number
        :param all_actions: action encodings
        :param action_nbrs: number of action (= number of paths per context entity)
        :param all_answers: gold label answer
        :param paths: KG paths from context entities
        """

        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self.question_ids = question_ids
        self.all_answers = all_answers
        self.q_start_indices = q_start_indices
        self.all_questions = all_questions
        self.all_actions = all_actions
        self.number_of_actions = action_nbrs

        self.starts_per_question = starts_per_question
        self.question_counter = 0

        self.num_rollouts = num_rollouts
        self.num_rollout_steps = num_rollout_steps

        self.paths = paths
        self._is_final_observation = False
        self._is_right_answer_found = False
        self._batch_size = 1
        self.current_rollout_step = 0

        self.alt_reward = alt_reward
        self.discount = discount

        self.curr_startpoint_id = ""

        self.reasoning_paths = defaultdict(list)

        super(RLEnvironment, self).__init__()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    @property
    def batched(self):
        return True

    @property
    def batch_size(self):
        return self._batch_size

    def _empty_observation(self):
        return tf.nest.map_structure(lambda x: np.zeros(x.shape, x.dtype), self.observation_spec())

    def setTopActions(self, actions):
        self.topActions = actions

    def _get_observation(self):
        """Returns an observation"""
        # we want to go over each question, and for each question over each possible starting point

        if self.question_counter == len(self.q_start_indices):
            print("end of training samples: empty observation returned.")
            self._is_final_observation = True
            return self._empty_observation()

        # get next training ids for question and startpoints
        q_counter, start_counter = self.q_start_indices[self.question_counter]
        self.curr_question_id = self.question_ids[q_counter]

        # get pre-computed bert embeddings for the question
        encoded_question = self.all_questions[self.curr_question_id]

        if self.current_rollout_step == 1:
            self.orig_startpoint_id = self.starts_per_question[self.curr_question_id][start_counter]
            self.curr_startpoint_id = self.orig_startpoint_id

        # get action embeddings
        #encoded_actions = self.all_actions[self.curr_startpoint_id]
        #action_nbr = self.number_of_actions[self.curr_startpoint_id]
        encoded_actions = self.all_actions[self.orig_startpoint_id]
        action_nbr = self.number_of_actions[self.orig_startpoint_id]

        mask = tf.ones(action_nbr)
        zeros = tf.zeros((1001-action_nbr))
        mask = tf.keras.layers.concatenate([mask, zeros], axis=0)
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, -1) # [1,1001,1]

        # put them together as next observation for the policy network
        observation = tf.keras.layers.concatenate([encoded_question, encoded_actions], axis=0) # [1001, 768]
        observation = tf.expand_dims(observation, 0) # [1, 1001, 768]
        observation = tf.keras.layers.concatenate([observation, mask], axis=2) #[1, 1001, 769]
        tf.dtypes.cast(observation, tf.float32)

        return observation


    def _reset(self):
        obs = self._get_observation()
        if self._is_final_observation:
            print("final obs inside reset")
            return ts.termination(self._empty_observation(), [0.0])
        return ts.restart(obs, batch_size=self._batch_size)

    def is_final_observation(self):
        return self._is_final_observation

    def set_is_rollout(self, rollout):
        self.is_rollout = rollout

    # reset the environment to its initial state
    def reset_env(self):
        self._is_final_observation = False  # reset the flag of the final observation
        self.is_rollout = False
        self._is_episode_ended = False
        self.question_counter = 0
        self.current_rollout_step = 1
        self.curr_question_id = ""
        self.curr_startpoint_id = ""
        self.topActions = []
        self.curr_startpoint_id = ""

    def next_question_counter(self):
        self.question_counter += 1

    def set_current_rollout_step(self, value):
        self.current_rollout_step = value

    def _apply_action(self, action):
        """Appies ´action´ to the Environment
        and returns the corresponding reward

        Args:
            action: A value conforming action_spec that will be taken as action in the environment.

        Returns:
            a float value that is the reward received by the environment.
        """
        #possible_answer = self.paths[self.curr_startpoint_id][action[0].numpy()][2]
        possible_answer = self.paths[self.orig_startpoint_id][action[0].numpy()][2]

        if (self.curr_question_id, self.orig_startpoint_id) not in self.reasoning_paths:
            self.reasoning_paths[(self.curr_question_id, self.orig_startpoint_id)] = [(self.orig_startpoint_id, action[0].numpy(), possible_answer)]
        else:
            self.reasoning_paths[(self.curr_question_id, self.orig_startpoint_id)].append((self.curr_startpoint_id, action[0].numpy(), possible_answer))

        # check whether the predicted answer is in the gold answers
        gold_answers = self.all_answers[self.curr_question_id]
        if possible_answer in gold_answers:
            print(self.question_counter, self.curr_question_id, self.orig_startpoint_id, self.curr_startpoint_id, self.current_rollout_step, action[0].numpy(),
                  possible_answer, "Correct")
            self._is_right_answer_found = True
            return [1.0]
        else:
            # check if possible answer is entity or not, if yes, current episode ends.
            self.curr_startpoint_id = possible_answer
            print(self.question_counter, self.curr_question_id, self.orig_startpoint_id, self.curr_startpoint_id,
                  self.current_rollout_step, action[0].numpy(),
                  possible_answer, "Wrong")
            if self.alt_reward:
                return [-1.0]
            return [0.0]

    def _step(self, action):

        reward = self._apply_action(action)

        if self._is_right_answer_found:
            time_step = ts.termination(self._empty_observation(), reward)
            self._is_right_answer_found = False
        else:
            if self.current_rollout_step >= self.num_rollout_steps:
                time_step = ts.termination(self._empty_observation(), reward)
            else:
                time_step = ts.transition(self._get_observation(), reward, discount=self.discount)

        return time_step