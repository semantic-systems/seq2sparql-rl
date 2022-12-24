import sys
sys.path.append("..")

import json
import pickle
import argparse
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from utils import utils

"""Encode questions and actions with pre-trained BERT model"""

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_encoding = TFBertModel.from_pretrained('bert-base-uncased', return_dict=True)


# get pre-trained BERT embeddings for question
def encode_question(question):

    tokenized_input = bert_tokenizer(question, return_tensors="tf", padding=True)
    encoded_input = bert_encoding(tokenized_input, output_hidden_states=True)
    # take average over all hidden layers
    all_layers = [encoded_input.hidden_states[l] for l in range(1, 13)]
    encoder_layer = tf.concat(all_layers, 1)
    pooled_output = tf.reduce_mean(encoder_layer, axis=1)

    return pooled_output


# get pre-trained BERT embeddings for actions
def encode_actions(actions):

    try:
        tokenized_actions = bert_tokenizer(actions, return_tensors='tf', padding=True, truncation=True, max_length=50)
    except Exception as e:
        print("error: ", e)
        print("actions not working: ", actions)
        return None

    encoded_actions = bert_encoding(tokenized_actions, output_hidden_states=True)
    # take average over all hidden layers
    all_layers = [encoded_actions.hidden_states[l] for l in range(1, 13)]
    encoder_layer = tf.concat(all_layers, 1)
    pooled_output = tf.reduce_mean(encoder_layer, axis=1)

    return pooled_output


# get node labels for each path (this can be adapted to also include start (paths) and endpoint as action)
def get_action_labels(paths):

    action_labels = dict()
    for idx, key in enumerate(paths.keys()):
        action_labels[key] = []
        actions = paths[key]
        for a in actions:
            p_labels = ""
            for aId in a[1]:
                p_labels += utils.get_label(aId) + " "
            action_labels[key].append(p_labels)
        print("Getting action labels: {0}, {1}/{2}".format(key, idx, len(paths)))

    return action_labels


# get all action embeddings for paths in the dataset
def get_action_encodings(action_labels):

    all_encoded_paths = dict()
    action_nbrs = dict()
    for idx, start in enumerate(action_labels.keys()):
        # store how many paths are available per startpoint
        action_nbrs[start] = len(action_labels[start])
        if action_nbrs[start] == 0:
            continue
        encoded_paths = None
        first = True
        j = -1

        # encode paths batchwise
        for i in range(action_nbrs[start]):
            j += 1
            if j == 64:
                if first:
                    encoded_paths = encode_actions(action_labels[start][i-j:i+1])
                    if encoded_paths is None:
                        j = -1
                        continue
                    first = False
                else:
                    encoded_actions = encode_actions(action_labels[start][i-j:i+1])
                    if encoded_actions is None:
                        j = -1
                        continue
                    encoded_paths = tf.keras.layers.concatenate([encoded_paths, encoded_actions], axis=0)
                j = -1
        encoded_actions = encode_actions(action_labels[start][i-j:i+1])
        if encoded_actions is None and encoded_paths is None:
            continue
        if not encoded_actions is None:
            if first:
                encoded_paths = encoded_actions
            else:
                encoded_paths = tf.keras.layers.concatenate([encoded_paths, encoded_actions], axis=0)
        # pad all paths to length of 1000
        if len(encoded_paths) < 1000:
            zeros = tf.zeros((1000-action_nbrs[start], 768))
            encoded_paths = tf.keras.layers.concatenate([encoded_paths, zeros], axis=0)
        all_encoded_paths[start] = encoded_paths
        print("Encoding action: {0}, {1}/{2}".format(start, idx, len(action_labels)))

    return all_encoded_paths, action_nbrs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str, help="the dataset name to be processed.")

    args = parser.parse_args()

    if args.dataset == "lcq2": # wikidata

        # load the train set of questions
        with open("../data/LC-QuAD2/train_data/all_questions_trainset.json", "r") as questions_file:
            train_questions = json.load(questions_file)
        # load all paths for each startpoints per question in the train set
        with open("../data/LC-QuAD2/train_data/contextPaths_trainset.json", "r") as contextPaths_file:
            train_paths = json.load(contextPaths_file)

        # load the test set of questions
        with open("../data/LC-QuAD2/test_data/all_questions_testset.json", "r") as questions_file:
            test_questions = json.load(questions_file)
        # load all paths for each startpoint per question in the test set
        with open("../data/LC-QuAD2/test_data/contextPaths_testset.json", "r") as contextPaths_file:
            test_paths = json.load(contextPaths_file)

        # encode the train set of questions
        encoded_train_questions = dict()

        for idx, qId in enumerate(train_questions.keys()):
            encoded_train_questions[qId] = encode_question(train_questions[qId])
            print("{0}, Completed: {1}/{2}".format(qId, idx, len(train_questions.keys())))

        with open("../data/LC-QuAD2/train_data/encoded_questions_trainset.pickle", "wb") as encoded_questions_file:
            pickle.dump(encoded_train_questions, encoded_questions_file)

        # encode the test set of questions
        encoded_test_questions = dict()
        for idx, qId in enumerate(test_questions.keys()):
            encoded_test_questions[qId] = encode_question(test_questions[qId])
            print("{0}, Completed: {1}/{2}".format(qId, idx, len(test_questions.keys())))

        with open("../data/LC-QuAD2/test_data/encoded_questions_testset.pickle", "wb") as encoded_questions_file:
            pickle.dump(encoded_test_questions, encoded_questions_file)

        print("Question Encoding Done!")

        # get action labels (needed for action encoding)
        train_action_labels = get_action_labels(train_paths)
        with open("../data/LC-QuAD2/train_data/action_labels_trainset.json", "w") as qfile:
            json.dump(train_action_labels, qfile)

        test_action_labels = get_action_labels(test_paths)
        with open("../data/LC-QuAD2/test_data/action_labels_testset.json", "w") as qfile:
            json.dump(test_action_labels, qfile)

        # encode actions
        encoded_train_paths, train_action_nbrs = get_action_encodings(train_action_labels)
        with open("../data/LC-QuAD2/train_data/encoded_paths_trainset.pickle", "wb") as encoded_contextPaths_file:
            pickle.dump(encoded_train_paths, encoded_contextPaths_file)
        with open("../data/LC-QuAD2/train_data/action_numbers_trainset.json", "w") as acntion_nbr_file:
            json.dump(train_action_nbrs, acntion_nbr_file)

        encoded_test_paths, test_action_nbrs = get_action_encodings(test_action_labels)
        with open("../data/LC-QuAD2/test_data/encoded_paths_testset.pickle", "wb") as encoded_contextPaths_file:
            pickle.dump(encoded_test_paths, encoded_contextPaths_file)
        with open("../data/LC-QuAD2/test_data/action_numbers_testset.json", "w") as acntion_nbr_file:
            json.dump(test_action_nbrs, acntion_nbr_file)

        print("Action Encoding Done!")