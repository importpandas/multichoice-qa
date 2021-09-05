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
import spacy
nlp = spacy.load("en_core_web_sm")
from mcmrc.data_utils.processors import process_text


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


def parse_race_data(data_path, tokenizer):
    # Create Tracking Variables
    pid, p_a, examples = [], [], []

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

            # Tokenize Passage
            article = process_text(data["article"])
            doc = nlp(article)
            article_sent_start = [sent.start_char for sent in doc.sents]

            # Tokenize and Add to P_A
            tokens = tokenizer.tokenize(article)
            pid.append(data["id"].replace(".txt", ""))
            p_a.append(tokens)
            examples.append([article, article_sent_start])

    return pid, p_a, examples


def parse_dream_data(data_path, tokenizer):
    # Create Tracking Variables
    pid, p_a, examples = [], [], []

    # Iterate through Train Data First!
    with open(data_path, 'rb') as f:
        data = json.load(f)

    for i, dialogue in enumerate(data):

        # Tokenize Passage
        dialogue_text = '\n'.join(dialogue[0])
        article = process_text(dialogue_text)
        doc = nlp(article)
        article_sent_start = [sent.start_char for sent in doc.sents]
        for i, sent_start in enumerate(article_sent_start):
            if i > 0 and sent_start - article_sent_start[i - 1] <= 3:
                article_sent_start.pop(i)

        # Tokenize and Add to P_A
        tokens = tokenizer.tokenize(article)
        pid.append(dialogue[2])
        p_a.append(tokens)
        examples.append([article, article_sent_start])

    return pid, p_a, examples

def compute_similarity_with_tfidf(dataset, data_path, tokenizer, top_n=10, lower_bound=0.3, upper_bound=0.8):
    if dataset == 'race':
        pid_list, PA, examples = parse_race_data(data_path, tokenizer)

    elif dataset == 'dream':
        pid_list, PA, examples = parse_dream_data(data_path, tokenizer)

    # Compute TF Matrix
    TF = compute_tf(PA)

    # Compute TF-IDF Matrix
    print('\nComputing TF-IDF Matrix...')
    transformer = TfidfTransformer()
    TF_IDF = transformer.fit_transform(TF)
    assert (TF_IDF.shape[0] == len(PA) == len(TF))
    scores = cosine_similarity(TF_IDF, TF_IDF)
    np.fill_diagonal(scores, 0)
    scores = np.where((scores > lower_bound) & (scores < upper_bound), scores, 0)
    max_scores = torch.sort(torch.tensor(scores), descending=True)[0][:, :top_n]
    max_score_idxs = torch.sort(torch.tensor(scores), descending=True)[1][:, :top_n]
    similarity_dict = {}
    score_level = np.arange(0, 1.1, 0.1)
    qualified_num = 0
    for pid, max_score_list, max_idx_list in zip(pid_list, max_scores, max_score_idxs):
        if max_score_list[0] > 0:
            similarity_dict[pid] = [(max_score.item(), pid_list[max_idx.item()]) for max_score, max_idx in zip(max_score_list, max_idx_list) if max_score > 0]
            qualified_num += 1
        else:
            similarity_dict[pid] = []
    for low_score, high_score in zip(score_level[:-1], score_level[1:]):
        idx_list = torch.where((max_scores[:, 0] >= low_score) & (max_scores[:, 0] < high_score))[0].tolist()
        print(f"***** Examples between {low_score:.2f} and {high_score:.2f} total {len(idx_list)} of {len(pid_list)}")
        sample_num = min(len(idx_list), 5)
        for idx in random.sample(idx_list, sample_num):
            print(examples[idx][0], '\n\n', examples[max_score_idxs[:, 0][idx]][0])
            print("\n\n")
        print("\n\n\n")
    examples_dict = {pid: passage for pid, passage in zip(pid_list, examples)}
    return similarity_dict, examples_dict, qualified_num / len(pid_list)


if __name__ == "__main__":
    # Parse Args
    arguments = parse_args()

    # Load BERT Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(arguments.pretrained, do_lower_case=True)

    print(len(compute_similarity_with_tfidf(arguments.dataset, arguments.train, tokenizer)))


        # Run Appropriate Accuracy Scorer
        #dream_judge(arguments, TF_IDF, D)
