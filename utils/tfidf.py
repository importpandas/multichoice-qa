"""
tfidf.py

TF-IDF Baseline (running as judge - takes debate logs as input, returns persuasiveness accuracy)
"""
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

import argparse
import json
import os
import numpy as np
import tqdm
import transformers
import torch
from transformers import AutoTokenizer
import random


ANS2IDX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
DEBATE2IDX = {'Ⅰ': 0, 'Ⅱ': 1, 'Ⅲ': 2, 'Ⅳ': 3}


def parse_args():
    p = argparse.ArgumentParser(description='TF-IDF Judge')
    p.add_argument("-m", "--mode", default='cross-model', help='Mode to run in < judge | cross-model >')
    p.add_argument("-d", "--dataset", default='dream', help='Dataset to run on < race | dream >')

    p.add_argument("-t", "--train", required=True, help='Path to raw train data to compute TF-IDF')
    p.add_argument("-v", "--val", nargs='+', help='Paths to debate logs for each agent.')
    p.add_argument("-q", "--with_question", default=False, action='store_true', help='TF-IDF with Question + Answer')

    p.add_argument("-p", "--pretrained", default='datasets/bert/uncased_L-12_H-768_A-12/vocab.txt')

    return p.parse_args()


def compute_tf(p_a):
    """Given tensor of [ndoc, words], compute Term Frequence (BoW) Representation [ndoc, voc_sz]"""

    # Compute Vocabulary
    vocab = set()
    print('\nCreating Vocabulary...')
    for doc in tqdm.tqdm(p_a):
        vocab |= set(doc)
    vocab = {w: i for i, w in enumerate(list(vocab))}

    # Compute Count TF Matrix
    tf = np.zeros((len(p_a), len(vocab)), dtype=int)
    print('\nComputing TF Matrix...')
    for i, doc in tqdm.tqdm(enumerate(p_a)):
        for w in doc:
            tf[i][vocab[w]] += 1

    return tf


def race_judge(args, idf, keys):
    """Run and Compute Accuracy on Baseline QA Model"""
    if args.mode == 'judge':
        levels = [os.path.join(args.val[0], x) for x in os.listdir(args.val[0])]
        correct, total = 0, 0
        for level in levels:
            passages = [os.path.join(level, x) for x in os.listdir(level)]
            print('\nRunning Debates for %s...' % level)
            for p in tqdm.tqdm(passages):
                # Get Key Stub
                k, cur_question = os.path.relpath(p, args.val[0]), 0
                while os.path.join(k, str(cur_question)) in keys:
                    key = os.path.join(k, str(cur_question))
                    d = keys[key]

                    # Compute Scores
                    passage_idx = d['passage_idx'][0]
                    opt_idxs = d['option_idx']

                    opt_scores = cosine_similarity(idf[opt_idxs], idf[passage_idx]).flatten()
                    best_opt = np.argmax(opt_scores)

                    # Score
                    if best_opt == d['answer']:
                        correct += 1

                    total += 1
                    cur_question += 1
        print("\nJudge Accuracy: %.5f out of %d Total Examples" % (correct / total, total))

    else:
        correct, total = 0, 0
        for key in keys:
            d = keys[key]

            # Compute Scores
            passage_idx = d['passage_idx'][0]
            opt_idxs = d['option_idx']

            opt_scores = cosine_similarity(idf[opt_idxs], idf[passage_idx]).flatten()
            best_opt = np.argmax(opt_scores)

            # Score
            if best_opt == d['answer']:
                correct += 1

            total += 1

        print("\nPersuasion Accuracy: %.5f out of %d Total Examples" % (correct / total, total))


def dream_judge(args, idf, keys):
    """Run and Compute Accuracy on Baseline QA Model"""
    if args.mode == 'judge':
        # Load and Iterate through Data
        with open(args.val[0], 'rb') as f:
            data = json.load(f)

        correct, total = 0, 0
        for i, article in enumerate(data):
            for idx in range(len(article[1])):
                # Get Key
                key = os.path.join(article[2], str(idx))

                d = keys[key]

                # Compute Scores
                passage_idx = d['passage_idx'][0]
                opt_idxs = d['option_idx']

                opt_scores = cosine_similarity(idf[opt_idxs], idf[passage_idx]).flatten()
                best_opt = np.argmax(opt_scores)

                # Score
                if best_opt == d['answer']:
                    correct += 1

                total += 1

        print("\nJudge Accuracy: %.5f out of %d Total Examples" % (correct / total, total))
    else:
        correct, total = 0, 0
        for key in keys:
            d = keys[key]

            # Compute Scores
            passage_idx = d['passage_idx'][0]
            opt_idxs = d['option_idx']

            opt_scores = cosine_similarity(idf[opt_idxs], idf[passage_idx]).flatten()
            best_opt = np.argmax(opt_scores)

            # Score
            if best_opt == d['answer']:
                correct += 1
            total += 1

        print("\nPersuasion Accuracy: %.5f out of %d Total Examples" % (correct / total, total))


def parse_race_data(data_path, tokenizer):
    # Create Tracking Variables
    pid, p_a, contexts = [], [], []

    # Iterate through Train Data First!
    levels = [os.path.join(data_path, x) for x in os.listdir(data_path)]
    for level in levels:
        passages = [os.path.join(level, x) for x in os.listdir(level)]

        print('\nProcessing %s...' % level)
        for p in tqdm.tqdm(passages):
            # Get Key Stub

            # Read File
            with open(p, 'rb') as f:
                data = json.load(f)

            # Create State Variables
            passage_idx = []

            # Tokenize Passage
            context = data['article']

            # Tokenize and Add to P_A
            tokens = tokenizer.tokenize(context)
            pid.append(data["id"].replace(".txt", ""))
            p_a.append(tokens)
            contexts.append(context)

    return pid, p_a, contexts


def parse_dream_data(data_path, tokenizer):
    # Create Tracking Variables
    pid, p_a, passages = [], [], []

    # Iterate through Train Data First!
    with open(data_path, 'rb') as f:
        data = json.load(f)

    for i, article in enumerate(data):

        # Tokenize Passage
        context = " ".join(article[0])

        # Tokenize and Add to P_A
        tokens = tokenizer.tokenize(context)
        pid.append(article[2])
        p_a.append(tokens)
        passages.append(context)

    return pid, p_a, passages

def compute_similarity_with_tfidf(dataset, data_path, tokenizer, top_n=10):
    if dataset == 'race':
        pid_list, PA, passages = parse_race_data(data_path, tokenizer)

    elif dataset == 'dream':
        pid_list, PA, passages = parse_dream_data(data_path, tokenizer)

    # Compute TF Matrix
    TF = compute_tf(PA)

    # Compute TF-IDF Matrix
    print('\nComputing TF-IDF Matrix...')
    transformer = TfidfTransformer()
    TF_IDF = transformer.fit_transform(TF)
    assert (TF_IDF.shape[0] == len(PA) == len(TF))
    scores = cosine_similarity(TF_IDF, TF_IDF)
    np.fill_diagonal(scores, 0)
    max_scores = torch.sort(torch.tensor(scores), descending=True)[0][:, :top_n]
    max_score_idxs = torch.sort(torch.tensor(scores), descending=True)[1][:, :top_n]
    similarity_dict = {}
    score_level = np.arange(0, 1.1, 0.1)
    for pid, max_score_list, max_idx_list in zip(pid_list, max_scores, max_score_idxs):
        similarity_dict[pid] = [(max_score.item(), max_idx.item()) for max_score, max_idx in zip(max_score_list, max_idx_list)]
    for low_score, high_score in zip(score_level[:-1], score_level[1:]):
        idx_list = torch.where((max_scores[:, 0] >= low_score) & (max_scores[:, 0] < high_score))[0].tolist()
        print(f"***** Examples between {low_score:.2f} and {high_score:.2f} total {len(idx_list)} of {len(pid_list)}")
        sample_num = min(len(idx_list), 20)
        for idx in random.sample(idx_list, sample_num):
            print(passages[idx], '\n',passages[max_score_idxs[:, 0][idx]])
            print("\n")
        print("\n\n")
    return similarity_dict


if __name__ == "__main__":
    # Parse Args
    arguments = parse_args()

    # Load BERT Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(arguments.pretrained, do_lower_case=True)

    print(len(compute_similarity_with_tfidf(arguments.dataset, arguments.train, tokenizer)))


        # Run Appropriate Accuracy Scorer
        #dream_judge(arguments, TF_IDF, D)
