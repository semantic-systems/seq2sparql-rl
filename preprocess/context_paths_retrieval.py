import sys
sys.path.append("..")

import json
import re
import numpy as np
import argparse

#BLINK_PATH = "/data2/jiang/BLINK/"
BLINK_PATH = "/export/home/jiang/workspace/BLINK/"

sys.path.append(BLINK_PATH)
import elq.main_dense as main_dense

from utils.db_utils import Neo4JConnector
from utils.knowledge_graph import WikidataKG
from utils.utils import ENTITY_PATTERN

"""
Get context entities per question along KG paths starting from there;
Candidate entities are scored by four different scores (lexical match, neighbor overlap, ned score, kg prior);
ELQ is used as NED tool
"""

#env = Neo4JConnector()
wikidata = WikidataKG("https://skynet.coypu.org/wikidata/")

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


def get_ELQ_predictions(question_id, question_string):

    data_to_link = [{"id": question_id, "text": question_string}]
    # run elq to get predictions for current question
    predictions = main_dense.run(elq_args, None, *models, test_data=data_to_link)
    elq_predictions = []
    for prediction in predictions:
        pred_scores = prediction["scores"]
        # get entity ids from wikidata
        pred_ids = [id2wikidata.get(wikipedia_id) for (wikipedia_id, a, b) in prediction['pred_triples']]
        p = 0
        for pred_id in pred_ids:
            if pred_id is None:
                continue
            # normalize the score
            score = np.exp(pred_scores[p])
            i = 0
            modified = False
            # potentially update score if same entity is matched multiple times.
            for tup in elq_predictions:
                if tup[0] == pred_id:
                    modified = True
                    if score > tup[1]:
                        elq_predictions[i] = (pred_id, score)
                i += 1
            # store entity id along its normalized score
            if not modified:
                elq_predictions.append((pred_id, score))
            p += 1

    return elq_predictions


def retrieve_context_paths(data):

    start_points = dict()
    global_seen_context_nodes = dict()

    idx = 0
    for question_id, question_string in data.items():

        elq_predictions = get_ELQ_predictions(question_id, question_string)

        if question_id not in start_points:
            start_points[question_id] = []

        for entry in elq_predictions:

            if entry[0] not in start_points[question_id]:
                start_points[question_id].append(entry[0])

            # only add entities
            if len(entry[0]) < 2 or not re.match(ENTITY_PATTERN, entry[0]):
                continue

            if entry[0] not in global_seen_context_nodes.keys():
                # get paths from neo4j database
                #paths = env.get_one_hop_nodes(entry[0])
                tmp_paths = wikidata.get_one_hop_paths(entry[0])
                paths = []
                for path in tmp_paths:
                    paths.append([path[0], [path[1]], path[2]])
                global_seen_context_nodes[entry[0]] = paths

        idx += 1
        print("Processed: {0}, {1}/{2}: ".format(question_id, idx, len(data)))

    return start_points, global_seen_context_nodes


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="the dataset name to be processed.")

    args = parser.parse_args()

    if args.dataset == "lcq2":  # wikidata

        train_data = json.load(open("../data/LC-QuAD2/train_data/all_questions_trainset.json"))
        train_start_points, train_global_seen_context_nodes = retrieve_context_paths(train_data)

        # with open("../data/LC-QuAD2/train_data/startPoints_trainset.json", "w") as startPoints_file:
        #     json.dump(train_start_points, startPoints_file)
        # with open("../data/LC-QuAD2/train_data/contextPaths_trainset.json", "w") as contextPaths_file:
        #     json.dump(train_global_seen_context_nodes, contextPaths_file)

        test_data = json.load(open("../data/LC-QuAD2/test_data/all_questions_testset.json"))
        test_start_points, test_global_seen_context_nodes = retrieve_context_paths(test_data)
        # with open("../data/LC-QuAD2/test_data/startPoints_testset.json", "w") as startPoints_file:
        #     json.dump(test_start_points, startPoints_file)
        # with open("../data/LC-QuAD2/test_data/contextPaths_testset.json", "w") as contextPaths_file:
        #     json.dump(test_global_seen_context_nodes, contextPaths_file)