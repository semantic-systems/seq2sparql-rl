import re
import os
import sys
sys.path.append("../utils")
import json
import argparse
from time import sleep
import KB_query
import utils


PREFIXES_WIKIDATA = {
    " p:": "PREFIX p: <http://www.wikidata.org/prop/>",
    "wdt:": "PREFIX wdt: <http://www.wikidata.org/prop/direct/>",
    "wd:": "PREFIX wd: <http://www.wikidata.org/entity/>",
    "xsd:": "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>",
    "pq:": "PREFIX pq: <http://www.wikidata.org/prop/qualifier/>",
    "ps:": "PREFIX ps: <http://www.wikidata.org/prop/statement/>",
    "rdfs:": "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"
}


"""Process and store data from the corresponding benchmark for later usage."""
def preprocess_lcq2(data, kb_endpoint):

    all_questions = dict()
    all_answers = dict()
    all_sparqls = dict()
    all_question_ids = []
    error_question_dict = dict()
    num_questions = len(data)

    maximum_retries = 5
    idx = 0

    while len(data) != 0:

        sleep(1.0)

        question_info = data.pop(0)
        question_string = question_info["question"]
        question_id = str(question_info["uid"])

        idx += 1

        try:

            sparql_query = question_info["sparql_wikidata"]
            sparql_query = utils.fill_missing_prefixes(PREFIXES_WIKIDATA, sparql_query)

            gold_answers = KB_query.kb_query(sparql_query, kb_endpoint)
            #gold_answers = lcq2_utils.get_gold_answers(gold_answers)

            all_answers[question_id] = gold_answers
            all_question_ids.append(question_id)
            all_questions[question_id] = question_string
            all_sparqls[question_id] = question_info["sparql_wikidata"]

            print("Processed {0}, {1}/{2}/{3} (Completed/Total/Errors)...".format(question_id, idx, num_questions,
                                                                                  len(error_question_dict)))
        except Exception as e:

            if question_id not in error_question_dict:
                error_question_dict[question_id] = {"retries": 1, "error_msg": e}
            else:
                error_question_dict[question_id]["retries"] += 1

            if error_question_dict[question_id]["retries"] <= maximum_retries:
                data = [question_info] + data
                idx -= 1
                print("Error: {0}, retry {1} times(s)...".format(question_id, error_question_dict[question_id]["retries"]))
            else:
                print("Processed {0}, {1}/{2}/{3} (Completed/Total/Errors)...".format(question_id, idx ,num_questions, len(error_question_dict)))

    return all_questions, all_answers, all_question_ids, all_sparqls, error_question_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="the dataset name to be processed.")
    parser.add_argument("--kb_endpoint", required=True, type=str, help="the url of the kb endpoint")

    args = parser.parse_args()

    if args.dataset == "lcq2": # wikidata

        train_data_dir = "../data/LC-QuAD2/train_data"
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)

        test_data_dir = "../data/LC-QuAD2/test_data"
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)

        train_data = json.load(open("../data/LC-QuAD2/train.json"))
        train_questions, train_answers, train_qList, train_sparqls, train_questions_error = preprocess_lcq2(train_data, args.kb_endpoint)

        with open(os.path.join(train_data_dir, "all_questions_trainset.json"), "w") as questions_file:
            json.dump(train_questions, questions_file)
        with open(os.path.join(train_data_dir, "all_answers_trainset.json"), "w") as answers_file:
            json.dump(train_answers, answers_file)
        with open(os.path.join(train_data_dir, "questionId_list_trainset.json"), "w") as qIds_file:
            json.dump(train_qList, qIds_file)
        with open(os.path.join(train_data_dir, "all_sparqls_trainset.json"), "w") as sparqls_file:
            json.dump(train_sparqls, sparqls_file)
        with open(os.path.join(train_data_dir, "all_error_questions_trainset.json"), "w") as error_file:
            json.dump(train_questions_error, error_file)

        test_data = json.load(open("../data/LC-QuAD2/test.json"))
        test_questions, test_answers, test_qList, test_sparqls, test_questions_error = preprocess_lcq2(test_data, args.kb_endpoint)

        with open(os.path.join(test_data_dir, "all_questions_testset.json"), "w") as questions_file:
            json.dump(test_questions, questions_file)
        with open(os.path.join(test_data_dir, "all_answers_testset.json"), "w") as answers_file:
            json.dump(test_answers, answers_file)
        with open(os.path.join(test_data_dir, "questionId_list_testset.json"), "w") as qIds_file:
            json.dump(test_qList, qIds_file)
        with open(os.path.join(test_data_dir, "all_sparqls_testset.json"), "w") as sparqls_file:
            json.dump(test_sparqls, sparqls_file)
        with open(os.path.join(test_data_dir, "all_error_questions_testset.json"), "w") as error_file:
            json.dump(train_questions_error, error_file)

