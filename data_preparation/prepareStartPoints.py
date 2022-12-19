import json
import sys

sys.path.append("../utils")
import utils

"""Prepare startpoints for training"""
### only used for TRAINING data not for test data ####

# load retrieved startpoints
with open("../processed_data/LC-QuAD2/train_data/startPoints_trainset.json", "r") as s_file:
    startPoints = json.load(s_file)

# load corresponding paths (max 1000) for each start point
with open("../processed_data/LC-QuAD2/train_data/contextPaths_trainset.json", "r") as path_file:
    paths = json.load(path_file)

# load all gold answers
with open("../processed_data/LC-QuAD2/train_data/all_answers_trainset.json", "r") as answer_file:
    answers = json.load(answer_file)

# get a list of question ids
with open("../processed_data/LC-QuAD2/train_data/questionId_list_trainset.json", "r") as qid_file:
    qList = json.load(qid_file)

# limits the startpoints to those from which the answer is reachable in one hop to improve training (otherwise agent could only have bad actions to choose from)
def getReachableStartpoints():
    number_of_training_samples = 0
    number_of_samples_answer_reachable = 0
    reachable_start_points = dict()
    for qid in startPoints.keys():
        for stp in startPoints[qid]:
            number_of_training_samples += 1
            for path in paths[stp]:
                pathend = path[2]
                if utils.is_timestamp(pathend):
                    pathend = utils.convertTimestamp(pathend)
                # check if path lead to answer
                if qid in answers and pathend in answers[qid]:
                    number_of_samples_answer_reachable += 1
                    # add startpoint to reachable ones
                    if qid in reachable_start_points.keys():
                        reachable_start_points[qid].append(stp)
                    else:
                        reachable_start_points[qid] = [stp]
                    # sufficient if answer can be found for one path from particular startpoint
                    break
    # print stats
    print("number of training samples: ", number_of_training_samples)
    print("number of training samples where answer reachable from startpoint: ", number_of_samples_answer_reachable)
    return reachable_start_points

# store indices for question and the respective startpoints for easier processing later
def createQuestionStartpointList(starts):
    q_start_indices = []
    for i in range(len(qList)):
        if qList[i] in starts.keys():
            for j in range(len(starts[qList[i]])):
                q_start_indices.append([i, j])
    return q_start_indices

reachable_start_points = getReachableStartpoints()
with open("../processed_data/LC-QuAD2/train_data/reachable_startpoints_trainset.json", "w") as stp_file:
    json.dump(reachable_start_points, stp_file)

q_start_indices = createQuestionStartpointList(reachable_start_points)
with open("../processed_data/LC-QuAD2/train_data/question_start_indices_trainset.json", "w") as qfile:
    json.dump(q_start_indices, qfile)
