from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

"""main evaluation file"""

with open(sys.argv[1], 'r') as config_file:
    config = json.load(config_file)

# set a seed value
seed_value = config["seed"]
# Set 'PYTHONHASHSEED' environment variable at a fixed value
import os
os.environ["PYTHONHASHSEED"] = str(seed_value)
# Set 'python' built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# Set 'numpy' pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# Set 'tensorflow' pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

from tf_agents.specs import array_spec, tensor_spec
from tf_agents.agents import ReinforceAgent
from tf_agents.trajectories import time_step as ts

import pickle
import operator

from policyNetwork import KGActionDistNet
sys.path.append("../utils")
import utils

tf.compat.v1.enable_v2_behavior()
train_step_counter = tf.compat.v2.Variable(0)
learning_rate = 1e-3

# load data from config file
filename = config["filename"]

# pre-computed context entities (=startpoints) for questions from testset
with open(config["startpoints"], "r") as start_file:
    startpoints = json.load(start_file)

# available paths for each starting point
with open(config["contextPaths"], "r") as path_file:
    contextPaths = json.load(path_file)

# KG node labels
with open(config["labels_dict"], "r") as labelFile:
    labels_dict = json.load(labelFile)

# BERT embedded questions
with open(config["bert_questions"], "rb") as q_file:
    bert_questions = pickle.load(q_file)

# BERT embedded actions
with open(config["bert_actions"], "rb") as a_file:
    bert_actions = pickle.load(a_file)

# number of actions available for a given startpoint
with open(config["action_nbrs"], "r") as nbr_file:
    action_nbrs = json.load(nbr_file)

# aggregation type for final ranking
if "agg_type" in config.keys():
    agg_type = config["agg_type"]
else:
    agg_type = "add"


# number of actions sampled per agent
nbr_sample_actions = config["nbr_sample_actions"]

action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=999, name='action')
observation_spec = array_spec.ArraySpec(shape=(config["observation_spec_shape_x"], config["observation_spec_shape_y"]), dtype=np.float32, name='observation')

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

# initialize policy network
actor_network = KGActionDistNet(
    seed_value,
    tensor_spec.from_spec(observation_spec),
    tensor_spec.from_spec(action_spec)
)

# restore trained policy network
checkpoint = tf.train.Checkpoint(actor_net=actor_network)
checkpoint.restore(config["checkpoint_path"] + "-seed-" + str(config["seed"]) + "/ckpt-" + str(config["checkpoint_nbr"]))

rfAgent = ReinforceAgent(
    tensor_spec.from_spec(ts.time_step_spec(observation_spec)),
    tensor_spec.from_spec(action_spec),
    actor_network, optimizer,
    train_step_counter=train_step_counter
)
rfAgent.initialize()

#get gready eval policy
eval_policy = rfAgent.policy

# check for existential question
def isExistential(question_start):
    existential_keywords = ['is', 'are', 'was', 'were', 'am', 'be', 'being', 'been', 'did', 'do', 'does', 'done', 'doing', 'has', 'had', 'having']
    if question_start in existential_keywords:
        return True
    return False

# calculate P@1
def getPrecisionAt1(answers, goldanswers):
    if "Yes" in goldanswers:
        print("Yes")
    if "No" in goldanswers:
        print("No")
    goldanswers_lower = [ga.lower() for ga in goldanswers]
    for answer in answers:
        if answer[-1] > 1:
            return 0.0
        if answer[0].lower() in goldanswers_lower:
            return 1.0
    return 0.0

# calculate Hit@5
def getHitsAt5(answers, goldanswers):
    if "Yes" in goldanswers:
        print("Yes")
    if "No" in goldanswers:
        print("No")
    goldanswers_lower = [ga.lower() for ga in goldanswers]
    for answer in answers:
        if answer[-1] > 5:
            return 0.0
        if answer[0].lower() in goldanswers_lower:
            return 1.0
    return 0.0

# calculate MRR
def getMRR(answers, goldanswers):
    if "Yes" in goldanswers:
        print("Yes")
    if "No" in goldanswers:
        print("No")
    goldanswers_lower = [ga.lower() for ga in goldanswers]
    i = 0
    for answer in answers:
        if answer[0].lower() in goldanswers_lower:
            return 1.0 / answer[-1]
        i += 1
    return 0.0

def call_rl(timesteps, start_ids):
    """apply trained policy network to get top answers (performed in parallel for all context entities/paths"""
    answers = dict()
    # perform one step with evaluation policy
    action_step = eval_policy.action(timesteps)
    all_action = np.arange(1000)
    all_actions = tf.expand_dims(all_action, axis=1)
    # get action distribution from policy network
    # we need entire distribution over all action since we want to get top k actions (not only top-1)
    distribution = actor_network.get_distribution()
    # calculate probability scores and sample the top-k actions
    log_probability_scores = distribution.log_prob(all_actions)
    log_probability_scores = tf.transpose(log_probability_scores)
    top_log_scores, topActions = tf.math.top_k(log_probability_scores, nbr_sample_actions)
    # get respective answers by following path described by the selected action
    for i in range(len(start_ids)):
        for j in range(len(topActions[i])):
            if j == 0:
                answers[start_ids[i]] = []
            if not start_ids[i] in contextPaths.keys():
                answers[start_ids[i]].append("")
                continue
            paths = contextPaths[start_ids[i]]
            if topActions[i][j].numpy() >= len(paths):
                answers[start_ids[i]].append("")
                continue
            answerpath = paths[topActions[i][j]]
            if len(answerpath) < 3:
                answers[start_ids[i]].append("")
                continue
            answer = answerpath[2]
            answers[start_ids[i]].append(answer)
    return (answers, top_log_scores.numpy())

def create_observation(startId, encoded_question):
    """create input to policy network"""

    if not startId in bert_actions.keys():
        print("Warning: No available actions for starting point with id: ", startId)
        return None

    encoded_actions = bert_actions[startId]
    action_nbr = action_nbrs[startId]

    mask = tf.ones(action_nbr)

    zeros = tf.zeros((1001-action_nbr))
    mask = tf.keras.layers.concatenate([mask, zeros], axis=0)
    mask = tf.expand_dims(mask, 0)
    mask = tf.expand_dims(mask, -1) # [1, 1001, 1]

    observation = tf.keras.layers.concatenate([encoded_question, encoded_actions], axis=0) # [1001, 768]

    observation = tf.expand_dims(observation, 0) # [1, 1001, 768]
    observation = tf.keras.layers.concatenate([observation, mask], axis=2) # [1, 1001, 769]
    tf.dtypes.cast(observation, tf.float32)

    return observation

def findAnswer(currentid):
    """
    main method for retrieving answers
    for all available context entities get predicitions from trained policy network (= parallel agents walk)
    and take the top-k answers per agent
    """

    if not currentid in startpoints.keys():
        return []
    currentStarts = startpoints[currentid]
    if len(currentStarts) == 0:
        return []

    timeSteps = None
    observations = None
    startIds = []

    #prepare input for policy network for each start point
    for startNode in currentStarts:
        observation = create_observation(startNode, bert_questions[currentid])

        if observation is None:
            continue
        if observations is None:
            observations = observation
        else:
            observations = np.concatenate((observations, observation))
        startIds.append(startNode)

    if not observations is None:
        timeSteps = ts.restart(observations, batch_size=observations.shape[0])

    if not timeSteps is None:
        # get predictions from policy network
        answers, log_probs = call_rl(timeSteps, startIds)
    else:
        return []

    i = 0
    answer_scores = dict()
    majority_answer_scores = dict()
    maxmajo_answer_scores = dict()
    additive_answer_scores = dict() # default

    for sId in startIds:
        for j in range(len(log_probs[i])):
            score = np.exp(log_probs[i][j])

            curr_answer = answers[sId][j]
            if curr_answer == "":
                print("empty answer for startid: ", sId, "with log prob: ", log_probs[i][j])
                continue
            # for additive aggregation type, take sum of scores when several agents have same answer entity
            if agg_type == "add":
                if curr_answer in additive_answer_scores.keys():
                    additive_answer_scores[curr_answer] += score
                else:
                    additive_answer_scores[curr_answer] = score

            # take maximal answer score
            elif agg_type == "max":
                if curr_answer in answer_scores.keys():
                    if score > answer_scores[curr_answer]:
                        answer_scores[curr_answer] = score
                else:
                    answer_scores[curr_answer] = score
            # store both: score and count of how many agent arrive at entity
            elif agg_type == "majo":
                if curr_answer in majority_answer_scores.keys():
                    if score > majority_answer_scores[curr_answer][0]:
                        majority_answer_scores[curr_answer][0] = score
                    majority_answer_scores[curr_answer][1] += 1
                else:
                    majority_answer_scores[curr_answer] = [score, 1]
            elif agg_type == "maxmajo":
                if curr_answer in maxmajo_answer_scores.keys():
                    if score > maxmajo_answer_scores[curr_answer][0]:
                        maxmajo_answer_scores[curr_answer][0] = score
                    maxmajo_answer_scores[curr_answer][1] += 1
                else:
                    maxmajo_answer_scores[curr_answer] = [score, 1]
        i += 1

    if agg_type == "add":
        return sorted(additive_answer_scores.items(), key=lambda item: item[1], reverse=True)
    elif agg_type == "max":
        return sorted(answer_scores.items(), key=lambda item: item[1], reverse=True)
    # first use count as sorting criterion, then score
    elif agg_type == "majo":
        return [(y, majority_answer_scores[y][1], majority_answer_scores[y][0]) for y in sorted(majority_answer_scores, key=lambda x: (majority_answer_scores[x][1], majority_answer_scores[x][1]), reverse=True)]
    # use score as first sorting criterion, count as second
    elif agg_type == "maxmajo":
        return [(y, maxmajo_answer_scores[y][0], maxmajo_answer_scores[y][1]) for y in sorted(maxmajo_answer_scores, key=lambda x: (maxmajo_answer_scores[x][0], maxmajo_answer_scores[x][1]), reverse=True)]

    additive_answer_scores = sorted(additive_answer_scores.items(), key=lambda item: item[1], reverse=True)
    return additive_answer_scores


def calculateAnswerRanks(answer_scores):
    """calculate ranks, several results can share the same rank if they have the same score"""
    if len(answer_scores) == 0:
        return []

    rank = 0
    same_ranked = 0
    prev_score = ["", ""]

    for score in answer_scores:
        if score[1] == prev_score[1]:
            # for these aggregation types we need to check two scores
            if agg_type == "majo" or agg_type == "maxmajo":
                if score[2] == prev_score[2]:
                    same_ranked += 1
                else:
                    rank += (1 + same_ranked)
                    same_ranked = 0
            else:
                same_ranked += 1
        else:
            rank += (1 + same_ranked)
            same_ranked = 0
        score.append(rank)
        prev_score = score
    return answer_scores


def formatAnswer(answer):
    if len(answer) == 0:
        return answer
    best_answer = answer
    if answer[0] == "Q" and "-" in answer:
        best_answer = answer.split("-")[0]
    elif utils.is_timestamp(answer):
        best_answer = utils.convertTimestamp(answer)
    return best_answer


if __name__ == '__main__':
    with open(config["filename"] + "-seed-" + str(config["seed"]) + ".csv", "w") as resFile:
        header = "QID,QUESTION,ANSWER,GOLD_ANSWER,PRECISION@1,HITS@5,MRR" + "\n"
        resFile.write(header)

        with open("../processed_data/LC-QuAD2/test_data/all_questions_testset.json", "r") as q_file:
            with open("../processed_data/LC-QuAD2/test_data/all_answers_testset.json", "r") as a_file:
                questionList = json.load(q_file)
                gold_answerList = json.load(a_file)

                avg_prec = 0.0
                avg_hits_5 = 0.0
                avg_mrr = 0.0

                question_count = 0
                # store how many questions are answered in total
                result_dict = dict()

                for question_id, question_text in questionList.items():
                    policy_state = eval_policy.get_initial_state(batch_size=1)

                    question_count += 1
                    question_start = question_text.strip().split(" ")[0].lower()
                    gold_answers = gold_answerList[question_id]
                    #find answer for current question
                    answer_scores = findAnswer(question_id)
                    answer_scores = [list(a) for a in answer_scores]

                    for answer in answer_scores:
                        if agg_type == "majo":
                            answer[2] = format(answer[2], '.3f')
                        else:
                            answer[1] = format(answer[1], '.3f')
                        answer[0] = formatAnswer(answer[0])
                    calculateAnswerRanks(answer_scores)

                    if isExistential(question_start):
                        answer_scores = [["Yes", 1.0, 1], ["No", 0.5, 2]]

                    result_dict[question_id] = answer_scores

                    # calculate metrics
                    prec = getPrecisionAt1(answer_scores, gold_answers)
                    hits_5 = getHitsAt5(answer_scores, gold_answers)
                    mrr = getMRR(answer_scores, gold_answers)

                    # write results to csv file
                    if answer_scores == []:
                        resFile.write(question_id+","+question_text+","+str([])+","+str(gold_answers)+","+str(prec)+","+str(hits_5)+","+str(mrr)+"\n")
                    else:
                        resFile.write(question_id+","+question_text+","+answer_scores[0][0]+","+str(gold_answers)+","+str(prec)+","+str(hits_5)+","+str(mrr)+"\n")
                    resFile.flush()


                    avg_prec += prec
                    avg_hits_5 += hits_5
                    avg_mrr += mrr

                # write results to file
                resFile.write("AVG results: , , , , "+str(format(avg_prec/question_count, '.3f'))+","+str(format(avg_hits_5/question_count, '.3f'))+","+str(format(avg_mrr/question_count, '.3f'))+"\n")
                resFile.write("total number of questions in the test: "+str(question_count)+"\n")

