from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts


"""KGQA Environment"""
class RLEnvironment(py_environment.PyEnvironment):

    def __init__(self,
                 observation_spec,
                 action_spec,
                 all_questions,
                 questionIds,
                 starts_per_question,
                 q_start_indices,
                 all_actions,
                 action_nbrs,
                 all_answers,
                 paths,
                 alt_reward):
        """
        :param observation_spec: observeration specification
        :param action_spec: action specification
        :param all_questions: question encodings
        :param questionIds: list with all question ids
        :param starts_per_question: context entities per startpoint
        :param q_start_indices: indices for qid and context entity number
        :param all_actions: action encodings
        :param action_nbrs: number of action (= number of paths per context entity)
        :param all_answers: gold label answer
        :param paths: KG paths from context entities
        """

        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self.questionIds = questionIds
        self.all_answers = all_answers
        self.q_start_indices = q_start_indices
        self.all_questions = all_questions
        self.all_actions = all_actions
        self.number_of_actions = action_nbrs

        self.starts_per_question = starts_per_question
        self.question_counter = 0

        self.paths = paths
        self.final_obs = False
        self._batch_size = 1

        self.alt_reward = alt_reward

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
            self.final_obs = True
            return self._empty_observation()

        # get next training ids for question and startpoints
        q_counter, start_counter = self.q_start_indices[self.question_counter]
        self.qId = self.questionIds[q_counter]

        self.start_id = self.starts_per_question[self.qId][start_counter]
        self.question_counter += 1

        # get pre-computed bert embeddings for the question
        encoded_question = self.all_questions[self.qId]

        # get action embeddings
        encoded_actions = self.all_actions[self.start_id]
        action_nbr = self.number_of_actions[self.start_id]

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
        self._done = False
        obs = self._get_observation()
        if self.final_obs:
            print("final obs inside reset")
            return ts.termination(self._empty_observation(), [0.0])
        return ts.restart(obs, batch_size=self._batch_size)

    def is_final_observation(self):
        return self.final_obs

    def set_is_rollout(self, rollout):
        self.rollout = rollout

    def reset_env(self):
        self.final_obs = False
        self.qId = ""
        self.start_id = ""
        self.topActions = []
        self.rollout = False
        self.question_counter = 0


    def _apply_action(self, action):
        """Appies ´action´ to the Environment
        and returns the corresponding reward

        Args:
            action: A value conforming action_spec that will be taken as action in the environment.

        Returns:
            a float value that is the reward received by the environment.
        """
        answer = self.paths[self.start_id][action[0].numpy()][2]

        # check whether the predicted answer is in the gold answers
        goldanswers = self.all_answers[self.qId]
        if answer in goldanswers:
            return [1.0]
        else:
            if self.alt_reward:
                return [-1.0]
            return [0.0]

    def _step(self, action):

        reward = self._apply_action(action)
        time_step = ts.termination(self._empty_observation(), reward)

        return time_step