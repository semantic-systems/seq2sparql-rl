import re
import os
import sys
sys.path.append("../utils")
sys.path.append("../KGQA-datasets")
import json
import argparse
from tqdm import tqdm
from time import sleep
import KB_query
import lcq2_utils

"""Process and store data from the corresponding benchmark for later usage."""

def preprocess_lcq2(data, kb_endpoint):

    all_questions = dict()
    all_answers = dict()
    all_sparqls = dict()
    qList = []

    tmp_data = data
    error_question_dict = dict()

    i = 0

    while len(tmp_data) != 0:

        sleep(2.0)

        question_info = tmp_data.pop(0)

        question = question_info["question"]
        question_id = str(question_info["uid"])

        if question_id in all_questions:
            continue

        if not question or question.strip() == "":
            print("Null: ", question_id)
            continue

        try:
            sparql_query = question_info["sparql_wikidata"]
            for p, v in lcq2_utils.PREFIXES_WIKIDATA.items():
                idx = sparql_query.find(p)
                if sparql_query.find(p) != -1:
                    if sparql_query.find(v) == -1:
                        sparql_query = v + " " + sparql_query

            if question_id == "18046":
                sparql_query = sparql_query.replace("t1410874016", '"t1410874016"')
            if question_id == "18212":
                sparql_query = sparql_query.replace("t1270953452", '"t1270953452"')
            if question_id == "5223" or question_id == "14226":
                continue

            extracted_answers = KB_query.kb_query(sparql_query, kb_endpoint)
            all_answers[question_id] = lcq2_utils.get_gold_answers(extracted_answers)

            qList.append(question_id)
            all_questions[question_id] = question
            all_sparqls[question_id] = sparql_query

            i += 1
        except Exception as e:
            print(e)
            print("Error: ", question_id)
            if question_id not in error_question_dict:
                error_question_dict[question_id] = 1
            else:
                if error_question_dict[question_id] <= 5:
                    tmp_data.append(question_info)
                    error_question_dict[question_id] += 1
            i -= 1

        print("Processing: {0}, {1}/{2} Completed...".format(question_id, i, len(tmp_data)))

    return all_questions, all_answers, qList, all_sparqls


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="the dataset name to be preprocessed.")
    parser.add_argument("--train_data", required=True, type=str, help="the data for model training.")
    parser.add_argument("--valid_data", type=str, help="the data for model validation.")
    parser.add_argument("--test_data", type=str, help="the data for model testing.")
    parser.add_argument("--dest_dir", required=True, type=str, help="the destination directory for saving data.")
    parser.add_argument("--kb_endpoint", required=True, type=str, help="the url of the kb endpoint")

    args = parser.parse_args()

    if args.dataset == "lcq2": # wikidata

        # train_data = json.load(open(args.train_data))
        # train_questions, train_answers, train_qList, train_sparqls = preprocess_lcq2(train_data, args.kb_endpoint)
        #
        # with open(os.path.join(args.dest_dir, "train_data/all_questions_trainset.json"), "w") as questions_file:
        #     json.dump(train_questions, questions_file)
        # with open(os.path.join(args.dest_dir, "train_data/all_answers_trainset.json"), "w") as answers_file:
        #     json.dump(train_answers, answers_file)
        # with open(os.path.join(args.dest_dir, "train_data/questionId_list_trainset.json"), "w") as qIds_file:
        #     json.dump(train_qList, qIds_file)
        # with open(os.path.join(args.dest_dir, "train_data/all_sparqls_trainset.json"), "w") as sparqls_file:
        #     json.dump(train_sparqls, sparqls_file)

        test_data = json.load(open(args.test_data))
        test_questions, test_answers, test_qList, test_sparqls = preprocess_lcq2(test_data, args.kb_endpoint)

        with open(os.path.join(args.dest_dir, "test_data/all_questions_testset.json"), "w") as questions_file:
            json.dump(test_questions, questions_file)
        with open(os.path.join(args.dest_dir, "test_data/all_answers_testset.json"), "w") as answers_file:
            json.dump(test_answers, answers_file)
        with open(os.path.join(args.dest_dir, "test_data/questionId_list_testset.json"), "w") as qIds_file:
            json.dump(test_qList, qIds_file)
        with open(os.path.join(args.dest_dir, "test_data/all_sparqls_testset.json"), "w") as sparqls_file:
            json.dump(test_sparqls, sparqls_file)

