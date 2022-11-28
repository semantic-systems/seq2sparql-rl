import os
import sys
sys.path.append("../utils")
import json
import argparse
import KB_query
import lcq2_utils
import lcq_utils

"""Process and store data from the corresponding benchmark for later usage."""


def preprocess_lcq2(data, kb_endpoint):

    all_questions = dict()
    all_answers = dict()
    all_answer_texts = dict()
    all_sparqls = dict()
    qList = []

    for question in data:
        question_id = str(question["uid"])
        qList.append(question_id)
        all_questions[question_id] = question["question"]
        q_query = question["sparql_wikidata"]
        extracted_answers = KB_query.kb_query(q_query, kb_endpoint)
        all_answers[question_id] = lcq2_utils.get_gold_answers(extracted_answers)
        all_answer_texts[question_id] = lcq2_utils.get_ent_label(all_answers[question_id], kb_endpoint)
        all_sparqls[question_id] = q_query
    return all_questions, all_answers, all_answer_texts, qList, all_sparqls



def preprocess_lcq(data, kb_endpoint):

    all_questions = dict()
    all_answers = dict()
    all_sparqls = dict()
    qList = []

    for question in data:
        question_id = question["_id"]
        qList.append(question_id)
        all_questions[question_id] = question["corrected_question"]
        q_query = question["sparql_query"]
        all_sparqls[question_id] = q_query
        extracted_answers = KB_query.kb_query(q_query, kb_endpoint)
        all_answers[question_id] = lcq_utils.get_gold_answers(extracted_answers)
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

    if args.dataset == "webqsp": # freebase
        pass


    elif args.dataset == "lcq": # dbpedia

        train_data = json.load(open(args.train_data))
        train_questions, train_answers, train_qList, train_sparqls = preprocess_lcq(train_data, args.kb_endpoint)

        with open(os.path.join(args.dest_dir, "train_data/all_questions_trainset.json"), "w") as questions_file:
            json.dump(train_questions, questions_file)
        with open(os.path.join(args.dest_dir, "train_data/all_answers_trainset.json"), "w") as answers_file:
            json.dump(train_answers, answers_file)
        with open(os.path.join(args.dest_dir, "train_data/questionId_list_trainset.json"), "w") as qIds_file:
            json.dump(train_qList, qIds_file)
        with open(os.path.join(args.dest_dir, "train_data/all_sparqls_trainset.json"), "w") as sparqls_file:
            json.dump(train_sparqls, sparqls_file)

        test_data = json.load(open(args.test_data))
        test_questions, test_answers, test_qList, test_sparqls = preprocess_lcq(test_data, args.kb_endpoint)

        with open(os.path.join(args.dest_dir, "test_data/all_questions_testset.json"), "w") as questions_file:
            json.dump(test_questions, questions_file)
        with open(os.path.join(args.dest_dir, "test_data/all_answers_testset.json"), "w") as answers_file:
            json.dump(test_answers, answers_file)
        with open(os.path.join(args.dest_dir, "test_data/questionId_list_testset.json"), "w") as qIds_file:
            json.dump(test_qList, qIds_file)
        with open(os.path.join(args.dest_dir, "test_data/all_sparqls_testset.json"), "w") as sparqls_file:
            json.dump(test_sparqls, sparqls_file)


    elif args.dataset == "lcq2": # wikidata

        train_data = json.load(open(args.train_data))
        train_questions, train_answers, train_answer_texts, train_qList, train_sparqls = preprocess_lcq2(train_data, args.kb_endpoint)

        with open(os.path.join(args.dest_dir, "train_data/all_questions_trainset.json"), "w") as questions_file:
            json.dump(train_questions, questions_file)
        with open(os.path.join(args.dest_dir, "train_data/all_answers_trainset.json"), "w") as answers_file:
            json.dump(train_answers, answers_file)
        with open(os.path.join(args.dest_dir, "train_data/all_answer_texts_trainset.json"), "w") as answer_texts_file:
            json.dump(train_answer_texts, answer_texts_file)
        with open(os.path.join(args.dest_dir, "train_data/questionId_list_trainset.json"), "w") as qIds_file:
            json.dump(train_qList, qIds_file)
        with open(os.path.join(args.dest_dir, "train_data/all_sparqls_trainset.json"), "w") as sparqls_file:
            json.dump(train_sparqls, sparqls_file)

        test_data = json.load(open(args.test_data))
        test_questions, test_answers, test_answer_texts, test_qList, test_sparqls = preprocess_lcq2(test_data, args.kb_endpoint)

        with open(os.path.join(args.dest_dir, "test_data/all_questions_testset.json"), "w") as questions_file:
            json.dump(test_questions, questions_file)
        with open(os.path.join(args.dest_dir, "test_data/all_answers_testset.json"), "w") as answers_file:
            json.dump(test_answers, answers_file)
        with open(os.path.join(args.dest_dir, "test_data/all_answer_texts_testset.json"), "w") as answer_texts_file:
            json.dump(test_answer_texts, answer_texts_file)
        with open(os.path.join(args.dest_dir, "test_data/questionId_list_testset.json"), "w") as qIds_file:
            json.dump(test_qList, qIds_file)
        with open(os.path.join(args.dest_dir, "test_data/all_sparqls_testset.json"), "w") as sparqls_file:
            json.dump(test_sparqls, sparqls_file)

