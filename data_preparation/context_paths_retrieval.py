
import json
import re
import os
from spacy.lang.en import English
from neo4j_connection import KGEnvironment
import numpy as np
import argparse
from time import sleep
from tqdm import tqdm

#BLINK_PATH = "/data2/jiang/BLINK/"
BLINK_PATH = "/export/home/jiang/workspace/BLINK/"

import sys
sys.path.append(BLINK_PATH)
import elq.main_dense as main_dense

"""
Get context entities per question along KG paths starting from there;
Candidate entities are scored by four different scores (lexical match, neighbor overlap, ned score, kg prior);
ELQ is used as NED tool
"""

ENTITY_PATTERN = re.compile('Q[0-9]+')
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

startPoints = dict()
globalSeenContextNodes = dict()

env = KGEnvironment()

# cut off for KG prior
MAX_PRIOR = 100

# config for ELQ NED
models_path = BLINK_PATH+"models/" # the path where you stored the ELQ models
config = {
    "interactive": False,
    "biencoder_model": models_path+"elq_wiki_large.bin",
    "biencoder_config": models_path+"elq_large_params.txt",
    "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "output_path": "logs/", # logging directory
    "faiss_index": "hnsw",
    "index_path": models_path+"faiss_hnsw_index.pkl",
    "num_cand_mentions": 10,
    "num_cand_entities": 10,
    "threshold_type": "joint",
    "threshold": -4.5,
}
elq_args = argparse.Namespace(**config)
models = main_dense.load_models(elq_args, logger=None)
id2wikidata = json.load(open(BLINK_PATH+"models/id2wikidata.json"))


class ContextNode(object):
    def __init__(self, id):
        self.id = id
    def setNeighbors(self, neighbors):
        self.neighbors = neighbors
    def getNeighbors(self):
        return self.neighbors
    def setOneHopPaths(self, paths):
        self.oneHopPaths = paths
    def getOneHopPaths(self):
        return self.oneHopPaths

    def __str__(self):
        return "id: " + str(self.id) + ", neighbors: " + str(self.neighbors) + ", oneHopPaths: " + str(self.oneHopPaths)


def getElqPredictions(question_id, question):
    data_to_link = [{"id": question_id, "text": question}]
    # run elq to get predictions for current question
    predictions = main_dense.run(elq_args, None, *models, test_data=data_to_link)
    elq_predictions = []
    for prediction in predictions:
        pred_scores = prediction["scores"]
        # get entity ids from wikidata
        pred_ids = [id2wikidata.get(wikipedia_id) for (wikipedia_id, a, b) in prediction['pred_triples']]
        p = 0
        for predId in pred_ids:
            if predId is None:
                continue
            # normalize the score
            score = np.exp(pred_scores[p])
            i = 0
            modified = False
            # potentially update score if same entity is matched multiple times.
            for tup in elq_predictions:
                if tup[0] == predId:
                    modified = True
                    if score > tup[1]:
                        elq_predictions[i] = (predId, score)
                i += 1
            # store entity id along its normalized score
            if not modified:
                elq_predictions.append((predId, score))
            p += 1
    return elq_predictions


def retrieveContextEntities(question, question_id, context_nodes, elq_predictions):
    startPoints[question_id] = list(context_nodes.keys())

    # check if we don't have any context entities yet, then use the ones predicted from the NED tool as initial entities
    if not context_nodes:
        for entry in elq_predictions:
            if not entry[0] in startPoints[question_id]:
                startPoints[question_id].append(entry[0])
            updateContext(context_nodes, entry[0])

    # score entities in one hop neighborhood and retrieve new context nodes
    #new_starts = expandStartingPoints(context_nodes, question_id, question, elq_predictions)
    #for newId in new_starts:
    #    updateContext(context_nodes, newId)
    #    if not newId in startPoints[question_id]:
    #        startPoints[question_id].append(newId)
    return


# find further context entities for a given question in one hop neighborhood
def expandStartingPoints(context_nodes, question_id, question, elq_predictions):
    candidates = dict()
    # go over existing context entity for this question so far:
    for node in context_nodes.keys():
        neighbors = context_nodes[node].getNeighbors()
        # go over 1 hop neighbors
        for neighbor in neighbors:
            if len(neighbor) < 2:
                continue
            if not re.match(ENTITY_PATTERN, neighbor):
                continue
            if neighbor in context_nodes.keys():
                continue
            # check if candidate is in neighborhood of several context entities
            if neighbor in candidates.keys():
                candidates[neighbor]["count"] += 1
                continue
            candidates[neighbor] = dict()
            candidates[neighbor]["count"] = 1.0

            # use number of triples where neighbor appears as subject (= number of outgoing paths) from KG (neo4j database) as KG prior
            neighbor_count = env.get_number_of_neighbors(neighbor)
            # this is cut off and normalized by MAX_PRIOR
            if neighbor_count > MAX_PRIOR:
                candidates[neighbor]["prior"] = 1.0
            else:
                candidates[neighbor]["prior"] = neighbor_count/MAX_PRIOR

            # calculate sim between candidate and question
    return []




def updateContext(context_nodes, newId):
    if len(newId) < 2:
        return
    # only add entities to context
    if not re.match(ENTITY_PATTERN, newId):
        return
    #check if we already have id in context
    if newId in context_nodes.keys():
        return

    newNode = ContextNode(newId)
    # get paths starting from entity
    if newId in globalSeenContextNodes.keys():
        paths = globalSeenContextNodes[newId]
    else:
        # for new ids: get paths from neo4j database
        paths = env.get_one_hop_nodes(newId)
        globalSeenContextNodes[newId] = paths
    neighbors = []
    for path in paths:
        neighbors.append(path[-1])

    newNode.setNeighbors(list(set(neighbors)))
    context_nodes[newId] = newNode
    return



def processData(data):
    # go over each question to retrieve the context entities
    idx = 0
    for question_info in data:

        question_id = str(question_info["uid"])
        question = question_info["question"]

        if not question or question.strip() == "":
            continue

        context_nodes = dict()
        elq_predictions = getElqPredictions(question_id, question)
        retrieveContextEntities(question, question_id, context_nodes, elq_predictions)
        idx += 1
        print("Processed: {0}, {1}/{2}: ".format(question_id, idx, len(data)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="the dataset name to be preprocessed.")
    parser.add_argument("--dest_dir", required=True, type=str, help="the destination directory for saving data.")

    args = parser.parse_args()

    if args.dataset == "lcq2":  # wikidata
        train_data = json.load(open("../data/LC-QuAD2/train.json"))
        processData(train_data)
        with open(os.path.join(args.dest_dir, "train_data/startPoints_trainset.json"), "w") as start_file:
            json.dump(startPoints, start_file)
        with open(os.path.join(args.dest_dir, "train_data/contextPaths_trainset.json"), "w") as path_file:
            json.dump(globalSeenContextNodes, path_file)

        # reset startpoints and contextpaths
        startPoints = dict()
        globalSeenContextNodes = dict()

        test_data = json.load(open("../data/LC-QuAD2/test.json"))
        processData(test_data)
        with open(os.path.join(args.dest_dir, "test_data/startPoints_testset.json"), "w") as start_file:
            json.dump(startPoints, start_file)
        with open(os.path.join(args.dest_dir, "test_data/contextPaths_testset.json"), "w") as path_file:
            json.dump(globalSeenContextNodes, path_file)