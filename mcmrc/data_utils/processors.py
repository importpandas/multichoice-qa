import six
import unicodedata
import torch
import random
import numpy as np
import json
import spacy
import re
nlp = spacy.load("en_core_web_sm")
chinese_nlp = spacy.load("zh_core_web_sm")


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def load_pseudo_label(pseudo_label_path):
    pseudo_label = torch.load(pseudo_label_path)
    pseudo_label_merged = {}
    for item in pseudo_label.keys():
        pseudo_label_merged[item] = {}
        for split in pseudo_label[item].keys():
            pseudo_label_merged[item].update(pseudo_label[item][split])
    return pseudo_label_merged


def load_evidence_logits(evidence_logits_path):
    with open(evidence_logits_path, "r") as f:
        evidence_logits = json.load(f)
    return evidence_logits


def load_exp_race_data(exp_race_file, load_evidence=False, use_chinese_nlp=False):
    print(exp_race_file)
    all_examples = dict.fromkeys(["example_id", "article", 'article_sent_start', "question", "answer", "options"], None)
    for k in all_examples.keys():
        all_examples[k] = []
    if load_evidence:
        all_examples['evidence'] = []
    less_option_num = 0
    with open(exp_race_file) as f:
        race_data = json.load(f)['data']
        for data in race_data:
            questions = data["questions"]
            answers = data["answers"]
            options = data["options"]
            article = process_text(data["article"])
            if 'article_sent_start' in data.keys():
                article_sent_start = data['article_sent_start']
            else:
                if use_chinese_nlp:
                    doc = chinese_nlp(article)
                else:
                    doc = nlp(article)
                article_sent_start = [sent.start_char for sent in doc.sents]

            if load_evidence:
                if 'evidences' in data:
                    evidences = data['evidences']
                else:
                    evidences = [''] * len(answers)

            for i in range(len(questions)):
                question = questions[i]
                question = re.sub("(_+ _+)", "_", question)
                answer = answers[i]
                option = options[i]
                if len(option) < 4:
                    option = option + [""] * (4 - len(option))
                    less_option_num += 1
                all_examples["example_id"].append(data["id"] + '-' + str(i))
                all_examples["article"].append(article)
                all_examples["article_sent_start"].append(article_sent_start)
                all_examples["question"].append(question)
                all_examples["answer"].append(answer)
                all_examples["options"].append(option)
                if load_evidence:
                    all_examples["evidence"].append(evidences[i])
                # if len(all_examples["example_id"]) > 4:
                #     return all_examples
    print(f"total {len(all_examples['example_id'])} less {less_option_num}")
    return all_examples


def load_adv_race_data(adv_race_file):
    print(adv_race_file)
    all_examples = dict.fromkeys(["example_id", "article", 'article_sent_start', "question", "answer", "options"], None)
    for k in all_examples.keys():
        all_examples[k] = []
    less_option_num = 0
    with open(adv_race_file) as f:
        race_data = json.load(f)
        for example_id, data in race_data.items():
            question = data["question"]
            answer = chr(data["label"] + ord("A"))
            option = data["options"]
            article = process_text(data["context"])
            doc = nlp(article)
            article_sent_start = [sent.start_char for sent in doc.sents]
            all_examples["example_id"].append(example_id)
            all_examples["article"].append(article)
            all_examples["article_sent_start"].append(article_sent_start)
            all_examples["question"].append(question)
            all_examples["answer"].append(answer)
            all_examples["options"].append(option)
            # if len(all_examples["example_id"]) > 2:
            #     return all_examples
    print(f"total {len(all_examples['example_id'])} less {less_option_num}")
    return all_examples


def process_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    if six.PY2 and isinstance(outputs, str):
        try:
            outputs = six.ensure_text(outputs, "utf-8")
        except UnicodeDecodeError:
            outputs = six.ensure_text(outputs, "latin-1")

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    outputs = outputs.replace("`", "'")
    outputs = outputs.replace("''", '"')

    if lower:
        outputs = outputs.lower()

    return outputs


def get_orig_chars_to_bounded_chars_mapping(tokens_char_span, total_len):
    all_chars_to_start_chars = []
    all_chars_to_end_chars = []

    prev_token_end_char = 0

    for token_start_char, token_end_char in tokens_char_span:
        if prev_token_end_char == token_end_char:
            continue
        token_start_char = prev_token_end_char if token_start_char < prev_token_end_char else token_start_char
        all_chars_to_start_chars += [token_start_char] * (token_end_char - prev_token_end_char)
        all_chars_to_end_chars += [prev_token_end_char - 1] * (token_start_char - prev_token_end_char)
        all_chars_to_end_chars += [token_start_char] * (token_end_char - token_start_char)
        prev_token_end_char = token_end_char

    if prev_token_end_char != total_len:
        all_chars_to_start_chars += [prev_token_end_char - 1] * (total_len - prev_token_end_char)
        all_chars_to_end_chars += [prev_token_end_char - 1] * (total_len - prev_token_end_char)

    assert len(all_chars_to_start_chars) == len(all_chars_to_end_chars) == total_len
    return all_chars_to_start_chars, all_chars_to_end_chars


def concat_question_option(question, option, dataset='dream'):
    orig_question = question
    question = re.sub("(_+)", "_", question)
    underline_count = len(re.findall("_", question))
    if dataset == 'dream' or underline_count == 0:
        qa_cat = " ".join([orig_question, option])
    elif underline_count == 1:
        qa_cat = question.replace("_", " " + option + " ", 1)
    else:
        qa_cat = ""
        option_split = list(filter(lambda x: x.strip(), option.split(";")))
        question_split = question.split("_")
        if len(question_split) != len(option_split) + 1:
            qa_cat = " ".join([question, option])
        else:
            for i in range(len(question_split) - 1):
                qa_cat += question_split[i] + option_split[i]
            qa_cat += question_split[-1]
    return qa_cat


# Preprocessing the datasets.
def prepare_features(examples, tokenizer=None, data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']

    labels = []
    qa_list = []
    processed_contexts = []

    num_choices = len(options[0])

    for i in range(len(answers)):
        label = ord(answers[i]) - ord("A")
        labels.append(label)
        processed_contexts.append([process_text(contexts[i])] * num_choices)

        question = process_text(questions[i])
        qa_pairs = []
        for j in range(num_choices):
            option = process_text(options[i][j])

            qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
            qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
            qa_pairs.append(qa_cat)
        qa_list.append(qa_pairs)

    first_sentences = sum(processed_contexts, [])
    second_sentences = sum(qa_list, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=512,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples = {k: [v[i: i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}

    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = example_ids

    # Un-flatten
    return tokenized_examples


# Preprocessing the datasets.
def prepare_features_for_evaluating_evidence(examples, evidence_sentences=None, tokenizer=None, data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']

    labels = []
    qa_list = []
    processed_contexts = []
    all_example_ids = []

    num_choices = len(options[0])

    for i in range(len(answers)):
        full_context = contexts[i]

        # label = ord(answers[i]) - ord("A")

        question = process_text(questions[i])
        for j in range(num_choices):
            labels.append(j)
            example_id = example_ids[i] + '_' + str(j)
            all_example_ids.append(example_id)

            evidence = evidence_sentences[example_id]

            processed_contexts.append([evidence] * num_choices)

            qa_pairs = []
            for k in range(num_choices):
                option = process_text(options[i][k])

                if "_" in question:
                    question = question.replace("_", "")
                    qa_cat = " ".join([question, option])
                else:
                    qa_cat = " ".join([question, option])
                qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
                qa_pairs.append(qa_cat)
            qa_list.append(qa_pairs)

    first_sentences = sum(processed_contexts, [])
    second_sentences = sum(qa_list, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples = {k: [v[i: i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}

    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = all_example_ids

    # Un-flatten
    return tokenized_examples


# Preprocessing the datasets.
def prepare_features_for_generate_pickout_pseudo_label(examples, tokenizer=None, data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    qa_list = []
    labels = []
    all_example_ids = []
    all_sent_idxs = []
    processed_contexts = []
    num_choices = len(options[0])

    for i in range(len(answers)):
        full_context = contexts[i]
        example_id = example_ids[i]
        label = ord(answers[i]) - ord("A")

        question = process_text(questions[i])
        qa_pairs = []
        for j in range(num_choices):
            option = process_text(options[i][j])

            qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
            qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
            qa_pairs.append(qa_cat)

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        for sent_idx in range(len(per_example_sent_starts) - 1):
            sent_start = per_example_sent_starts[sent_idx]
            sent_end = per_example_sent_starts[sent_idx + 1]
            pickout_sent = full_context[sent_start: sent_end]

            processed_contexts.append([process_text(pickout_sent)] * num_choices)
            qa_list.append(qa_pairs)
            all_sent_idxs.append(sent_idx)
            all_example_ids.append(example_id)
            labels.append(label)

    first_sentences = sum(processed_contexts, [])
    second_sentences = sum(qa_list, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=128,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples = {k: [v[i: i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}

    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = all_example_ids
    tokenized_examples['sent_idx'] = all_sent_idxs

    # Un-flatten
    return tokenized_examples



# Preprocessing the datasets.
def prepare_features_for_generate_pseudo_label(examples, tokenizer=None, data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    features = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'sent_bound_token': [], 'example_ids': [],
                'label': []}

    num_choices = len(options[0])

    for i in range(len(answers)):

        # processed_contexts.append([process_text(contexts[i])] * 4)
        processed_context = contexts[i]
        example_id = example_ids[i]
        label = ord(answers[i]) - ord("A")
        question = process_text(questions[i])
        per_example_sent_starts = sent_starts[i]
        per_example_sent_ends = [char_idx - 1 for char_idx in per_example_sent_starts[1:]] + [len(processed_context) - 1]

        choices_inputs = []
        all_text_b = []
        all_text_b_len = []
        for j in range(num_choices):
            option = process_text(options[i][j])

            qa_cat = concat_question_option(question, option, dataset=data_args.dataset)

            # truncated_qa_cat = tokenizer.tokenize(qa_cat, add_special_tokens=False, max_length=data_args.max_qa_length)
            truncated_text_b_id = tokenizer.encode(qa_cat, truncation=True, max_length=data_args.max_qa_length,
                                                   add_special_tokens=False)
            truncated_text_b = tokenizer.decode(truncated_text_b_id, clean_up_tokenization_spaces=False)
            all_text_b.append(truncated_text_b)
            all_text_b_len.append(len(tokenizer.encode(truncated_text_b, add_special_tokens=False)))

        for j in range(num_choices):
            text_b = all_text_b[j] + tokenizer.pad_token * (max(all_text_b_len) - all_text_b_len[j])

            inputs = tokenizer(
                processed_context,
                text_b,
                add_special_tokens=True,
                max_length=data_args.max_seq_length,
                padding="max_length" if data_args.pad_to_max_length else False,
                truncation='only_first',
                stride=data_args.max_seq_length - 3 - 128 - max(all_text_b_len),
                return_overflowing_tokens=True,
            )
            choices_inputs.append(inputs)
        assert len(set([len(x["input_ids"]) for x in choices_inputs])) == 1

        per_example_feature_num = len(choices_inputs[0]["input_ids"])

        tokens_char_span = tokenizer(processed_context, return_offsets_mapping=True, add_special_tokens=False)[
            'offset_mapping']
        all_chars_to_start_chars, all_chars_to_end_chars = get_orig_chars_to_bounded_chars_mapping(tokens_char_span,
                                                                                                   len(processed_context))

        for j in range(per_example_feature_num):
            input_ids = [x["input_ids"][j] for x in choices_inputs]
            attention_mask = [x["attention_mask"][j] for x in choices_inputs]
            token_type_ids = [x["token_type_ids"][j] for x in choices_inputs]
            sent_bound_token = []
            for sent_idx, (sent_start, sent_end) in enumerate(zip(per_example_sent_starts, per_example_sent_ends)):
                new_sent_start, new_sent_end = all_chars_to_start_chars[sent_start], all_chars_to_end_chars[sent_end]
                if not (choices_inputs[0].char_to_token(j, new_sent_start) and choices_inputs[0].char_to_token(j,
                                                                                                               new_sent_end)):
                    continue
                sent_bound_token.append((sent_idx, choices_inputs[0].char_to_token(j, new_sent_start),
                                         choices_inputs[0].char_to_token(j, new_sent_end)))
            features['input_ids'].append(input_ids)
            features['attention_mask'].append(attention_mask)
            features['token_type_ids'].append(token_type_ids)
            features['sent_bound_token'].append(sent_bound_token)
            features['example_ids'].append(example_id)
            features['label'].append(label)

    # Un-flatten
    return features


def prepare_features_for_generating_multi_turn_pseudo_label(examples, tokenizer=None, data_args=None,
                                                            pseudo_label_path="", max_sent_num=5):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    if pseudo_label_path:
        all_pseudo_label = load_pseudo_label(pseudo_label_path)
        pseudo_logit = all_pseudo_label['logit']
        acc = all_pseudo_label['acc']

    all_feature_example_ids = []
    sent_sequence = []

    processed_contexts = []
    qa_list = []
    labels = []
    for i in range(len(answers)):

        # processed_contexts.append([process_text(contexts[i])] * 4)
        full_context = contexts[i]
        example_id = example_ids[i]
        label = ord(answers[i]) - ord("A")
        per_example_sent_starts = sent_starts[i]
        per_example_sent_ends = [char_idx - 1 for char_idx in per_example_sent_starts[1:]] + [len(full_context) - 1]

        qa_concat = process_text(questions[i])
        for j in range(4):
            option = process_text(options[i][j])
            qa_concat += " [SEP]"
            qa_concat += option
        qa_concat = " ".join(whitespace_tokenize(qa_concat)[- data_args.max_qa_length:])

        inputs = tokenizer(
            full_context,
            qa_concat,
            add_special_tokens=True,
            max_length=data_args.max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation='only_first',
            return_overflowing_tokens=False,
        )

        tokens_char_span = tokenizer(full_context, return_offsets_mapping=True, add_special_tokens=False)[
            'offset_mapping']
        all_chars_to_start_chars, all_chars_to_end_chars = get_orig_chars_to_bounded_chars_mapping(tokens_char_span,
                                                                                                   len(full_context))

        sent_logits = {}
        for sent_idx, (sent_start, sent_end) in enumerate(zip(per_example_sent_starts, per_example_sent_ends)):
            new_sent_start, new_sent_end = all_chars_to_start_chars[sent_start], all_chars_to_end_chars[sent_end]
            if not (inputs.char_to_token(new_sent_start) and inputs.char_to_token(new_sent_end)):
                continue
            sent_logits[sent_idx] = pseudo_logit[example_id][sent_idx]

        for sent_num in range(1, max_sent_num + 1):
            per_example_evidence_sent_idxs = sorted(sent_logits.keys(),
                                                    key=lambda x: abs(sent_logits[x]), reverse=True)[: sent_num]

            evidence_concat = ""
            for evidence_sent_idx in sorted(per_example_evidence_sent_idxs):
                sent_start = per_example_sent_starts[evidence_sent_idx]
                sent_end = per_example_sent_ends[evidence_sent_idx]
                evidence_concat += full_context[sent_start: sent_end + 1]

            processed_contexts.append([evidence_concat] * 4)

            question = process_text(questions[i])
            qa_pairs = []
            for j in range(4):
                option = process_text(options[i][j])

                qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
                qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
                qa_pairs.append(qa_cat)
            qa_list.append(qa_pairs)
            all_feature_example_ids.append(example_id)
            sent_sequence.append(sorted(per_example_evidence_sent_idxs))
            labels.append(label)

    first_sentences = sum(processed_contexts, [])
    second_sentences = sum(qa_list, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=data_args.max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    tokenized_examples['example_ids'] = all_feature_example_ids
    tokenized_examples['sent_sequence'] = sent_sequence
    tokenized_examples['labels'] = labels

    # Un-flatten
    return tokenized_examples


def prepare_features_for_initializing_complex_evidence_selector(examples, tokenizer=None, data_args=None,
                                                                pseudo_label_path=""):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    if pseudo_label_path:
        all_pseudo_label = load_pseudo_label(pseudo_label_path)
        pseudo_logit = all_pseudo_label['logit']
        acc = all_pseudo_label['acc']

    features = {}
    features['input_ids'] = []
    features['attention_mask'] = []
    features['token_type_ids'] = []
    features['sent_bound'] = []
    if pseudo_label_path:
        features['label'] = []
    features['example_ids'] = []

    for i in range(len(answers)):

        # processed_contexts.append([process_text(contexts[i])] * 4)
        processed_context = contexts[i]
        example_id = example_ids[i]
        question = process_text(questions[i])
        per_example_sent_starts = sent_starts[i]
        per_example_sent_ends = [char_idx - 1 for char_idx in per_example_sent_starts[1:]] + [
            len(processed_context) - 1]

        qa_concat = process_text(questions[i])
        for j in range(4):
            option = process_text(options[i][j])
            qa_concat += " [SEP]"
            qa_concat += option
        qa_concat = " ".join(whitespace_tokenize(qa_concat)[- data_args.max_qa_length:])

        inputs = tokenizer(
            processed_context,
            qa_concat,
            add_special_tokens=True,
            max_length=data_args.max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
            truncation='only_first',
            return_overflowing_tokens=False,
        )

        tokens_char_span = tokenizer(processed_context, return_offsets_mapping=True, add_special_tokens=False)[
            'offset_mapping']
        all_chars_to_start_chars, all_chars_to_end_chars = get_orig_chars_to_bounded_chars_mapping(tokens_char_span,
                                                                                                   len(processed_context))

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        sent_bound_token = []
        if pseudo_label_path:
            sent_label = []
        for sent_idx, (sent_start, sent_end) in enumerate(zip(per_example_sent_starts, per_example_sent_ends)):
            new_sent_start, new_sent_end = all_chars_to_start_chars[sent_start], all_chars_to_end_chars[sent_end]
            if not (inputs.char_to_token(new_sent_start) and inputs.char_to_token(new_sent_end)):
                continue
            sent_bound_token.append((sent_idx,
                                     inputs.char_to_token(new_sent_start),
                                     inputs.char_to_token(new_sent_end)))
            if pseudo_label_path:
                if data_args.filter_label_with_ground_truth:
                    sent_label.append(pseudo_logit[example_id][sent_idx])
                else:
                    sent_label.append(abs(pseudo_logit[example_id][sent_idx]))
        features['input_ids'].append(input_ids)
        features['attention_mask'].append(attention_mask)
        features['token_type_ids'].append(token_type_ids)
        features['sent_bound'].append(sent_bound_token)
        features['example_ids'].append(example_id)
        if pseudo_label_path:
            features['label'].append(sent_label)

    # Un-flatten
    return features


def prepare_features_for_initializing_simple_evidence_selector(examples, evidence_len=2, tokenizer=None, data_args=None,
                                                               pseudo_label_path=""):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    all_pseudo_label = load_pseudo_label(pseudo_label_path)

    pseudo_logit = all_pseudo_label['logit']
    acc = all_pseudo_label['acc']

    qa_list = []
    labels = []
    processed_contexts = []

    for i in range(len(answers)):
        full_context = contexts[i]
        example_id = example_ids[i]

        qa_concat = process_text(questions[i])
        for j in range(4):
            option = process_text(options[i][j])
            qa_concat += "[SEP]"
            qa_concat += option

        evidence_len = evidence_len if evidence_len <= len(pseudo_logit[example_id]) else len(pseudo_logit[example_id])
        if data_args.filter_label_with_ground_truth:
            per_example_evidence_sent_idxs = sorted(pseudo_logit[example_id].keys(),
                                                    key=lambda x: pseudo_logit[example_id][x], reverse=True)[: evidence_len]
        else:
            per_example_evidence_sent_idxs = sorted(pseudo_logit[example_id].keys(),
                                                    key=lambda x: abs(pseudo_logit[example_id][x]), reverse=True)[: evidence_len]

        if data_args.train_with_adversarial_examples:
            per_example_noise_sent_idxs = sorted(pseudo_logit[example_id].keys(),
                                                 key=lambda x: pseudo_logit[example_id][x])[: evidence_len]
        else:
            per_example_noise_sent_idxs = []

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        for evidence_sent_idx in per_example_evidence_sent_idxs:
            sent_start = per_example_sent_starts[evidence_sent_idx]
            sent_end = per_example_sent_starts[evidence_sent_idx + 1]
            evidence_sent = full_context[sent_start: sent_end]
            processed_contexts.append(evidence_sent)
            qa_list.append(qa_concat)
            labels.append(1)

        for evidence_sent_idx in per_example_noise_sent_idxs:
            sent_start = per_example_sent_starts[evidence_sent_idx]
            sent_end = per_example_sent_starts[evidence_sent_idx + 1]
            evidence_sent = full_context[sent_start: sent_end]
            processed_contexts.append(evidence_sent)
            qa_list.append(qa_concat)
            labels.append(2)

        all_irre_sent_idxs = list(
            filter(lambda x: x not in per_example_evidence_sent_idxs and x not in per_example_noise_sent_idxs,
                   list(range(len(per_example_sent_starts) - 1))))
        negative_sent_num = evidence_len if evidence_len <= len(all_irre_sent_idxs) else len(all_irre_sent_idxs)
        for irre_sent_idx in random.sample(all_irre_sent_idxs, negative_sent_num):
            sent_start = per_example_sent_starts[irre_sent_idx]
            sent_end = per_example_sent_starts[irre_sent_idx + 1]
            irre_sent = full_context[sent_start: sent_end]
            processed_contexts.append(irre_sent)
            qa_list.append(qa_concat)
            labels.append(0)

    tokenized_examples = tokenizer(
        processed_contexts,
        qa_list,
        truncation=True,
        max_length=data_args.max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    tokenized_examples['label'] = labels

    # Un-flatten
    return tokenized_examples


def prepare_features_for_initializing_bidirectional_evidence_selector(examples,
                                                        evidence_sampling_num=1,
                                                        negative_sampling_ratio=1.0,
                                                        tokenizer=None,
                                                        data_args=None,
                                                        pseudo_label_path="",
                                                        jump_wrong_examples=False,
                                                        polarity_by_answer="none"):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    all_pseudo_label = load_pseudo_label(pseudo_label_path)

    options_logit_diff = all_pseudo_label['options_logit_diff']
    orig_logit = all_pseudo_label['orig_logit']

    qa_list = []
    labels = []
    all_example_ids = []
    processed_contexts = []

    num_choices = len(options[0])

    for i in range(len(answers)):
        full_context = contexts[i]
        example_id = example_ids[i]
        qa_label = ord(answers[i]) - ord('A')
        qa_pred = int(np.argmax(orig_logit[example_id]))

        if jump_wrong_examples and qa_pred != qa_label:
            continue

        per_example_options_prob_diff = options_logit_diff[example_id]

        processed_question = process_text(questions[i])

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        evidence_sampling_num = evidence_sampling_num if evidence_sampling_num <= len(
            per_example_options_prob_diff) else len(per_example_options_prob_diff)

        if polarity_by_answer == "none" or polarity_by_answer == "remove":
            sorted_sents_by_positive_score = sorted(per_example_options_prob_diff.items(), key=lambda x: abs(min(x[1])),
                                                    reverse=True)
            sorted_sents_by_negative_score = sorted(per_example_options_prob_diff.items(), key=lambda x: max(x[1]),
                                                    reverse=True)
        elif polarity_by_answer == "force":
            sorted_sents_by_positive_score = sorted(per_example_options_prob_diff.items(), key=lambda x: -x[1][qa_label],
                                                    reverse=True)
            sorted_sents_by_negative_score = sorted(per_example_options_prob_diff.items(),
                                                    key=lambda x: max([logit for j, logit in enumerate(x[1]) if j != qa_label]),
                                                    reverse=True)

        positive_evidence_set = []
        for positive_evidence, evidence_score in sorted_sents_by_positive_score[: evidence_sampling_num]:
            sent_start = per_example_sent_starts[positive_evidence]
            sent_end = per_example_sent_starts[positive_evidence + 1]
            evidence_sent = full_context[sent_start: sent_end]

            if np.min(evidence_score) >= 0:
                continue
            if polarity_by_answer == "force":
                option_for_evidence = qa_label
                if evidence_score[qa_label] >= 0:
                    continue
            else:
                option_for_evidence = np.argmin(evidence_score)
                if polarity_by_answer == "remove" and option_for_evidence != qa_label:
                    continue

            option = process_text(options[i][option_for_evidence])
            qa_concat = processed_question + "[SEP]"
            qa_concat += option

            processed_contexts.append(evidence_sent)
            qa_list.append(qa_concat)
            labels.append(1)
            positive_evidence_set.append(positive_evidence)
            all_example_ids.append(example_id+f'_{positive_evidence}_{option_for_evidence}')

        negative_evidence_set = []
        for negative_evidence, evidence_score in sorted_sents_by_negative_score[: evidence_sampling_num]:
            sent_start = per_example_sent_starts[negative_evidence]
            sent_end = per_example_sent_starts[negative_evidence + 1]
            evidence_sent = full_context[sent_start: sent_end]

            if np.max(evidence_score) <= 0:
                continue
            if polarity_by_answer == "force":
                option_for_evidence = np.argmax([logit if j != qa_label else -100 for j, logit in enumerate(evidence_score)])
                if evidence_score[option_for_evidence] <= 0:
                    continue
            else:
                option_for_evidence = np.argmax(evidence_score)
                if polarity_by_answer == "remove" and option_for_evidence == qa_label:
                    continue

            option = process_text(options[i][option_for_evidence])
            qa_concat = processed_question + "[SEP]"
            qa_concat += option

            processed_contexts.append(evidence_sent)
            qa_list.append(qa_concat)
            labels.append(2)
            negative_evidence_set.append(negative_evidence)
            all_example_ids.append(example_id + f'_{negative_evidence}_{option_for_evidence}')

        all_irre_sent_idxs = list(filter(lambda x: x not in positive_evidence_set + negative_evidence_set,
                                         list(range(len(per_example_sent_starts) - 1))))
        if evidence_sampling_num * negative_sampling_ratio <= len(all_irre_sent_idxs):
            negative_sent_num = evidence_sampling_num * negative_sampling_ratio
            if 0 < negative_sent_num < 1 and random.random() > negative_sent_num:
                negative_sent_num = 1
            else:
                negative_sent_num = int(negative_sent_num)
        else:
            negative_sent_num = len(all_irre_sent_idxs)

        if negative_sent_num >= 1:
            irre_options = random.sample(list(range(num_choices)), negative_sent_num)
            for idx, irre_sent_idx in enumerate(random.sample(all_irre_sent_idxs, negative_sent_num)):
                sent_start = per_example_sent_starts[irre_sent_idx]
                sent_end = per_example_sent_starts[irre_sent_idx + 1]
                irre_sent = full_context[sent_start: sent_end]
                option = process_text(options[i][irre_options[idx]])
                qa_concat = processed_question + "[SEP]"
                qa_concat += option

                processed_contexts.append(irre_sent)
                qa_list.append(qa_concat)
                labels.append(0)
                all_example_ids.append(example_id + f'_{irre_sent_idx}_{irre_options[idx]}')

    tokenized_examples = tokenizer(
        processed_contexts,
        qa_list,
        truncation=True,
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = all_example_ids

    # Un-flatten
    return tokenized_examples


def prepare_features_for_initializing_evidence_selector(examples,
                                                        evidence_sampling_num=2,
                                                        negative_sampling_ratio=1.0,
                                                        hard_negative_sampling=False,
                                                        tokenizer=None,
                                                        data_args=None,
                                                        pseudo_label_path=""):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    all_pseudo_label = load_pseudo_label(pseudo_label_path)

    pseudo_logit = all_pseudo_label['kv_div']
    options_prob_diff = all_pseudo_label['options_logit_diff']

    qa_list = []
    labels = []
    processed_contexts = []

    num_choices = len(options[0])

    for i in range(len(answers)):
        full_context = contexts[i]
        example_id = example_ids[i]

        per_example_options_prob_diff = options_prob_diff[example_id]

        processed_question = process_text(questions[i])

        evidence_sampling_num = evidence_sampling_num if evidence_sampling_num <= len(
            pseudo_logit[example_id]) else len(pseudo_logit[example_id])
        send_idxs_sorted_by_importance = sorted(pseudo_logit[example_id].keys(),
                                                key=lambda x: abs(pseudo_logit[example_id][x]), reverse=True)

        if len(sent_starts[i]) != len(send_idxs_sorted_by_importance):
            continue
        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        hard_sampling_num = evidence_sampling_num * negative_sampling_ratio if hard_negative_sampling else 0
        if 0 < hard_sampling_num < 1 and random.random() > hard_sampling_num:
            hard_sampling_num = 1
        else:
            hard_sampling_num = int(hard_sampling_num)

        for evidence_sent_idx in send_idxs_sorted_by_importance[: evidence_sampling_num]:
            sent_start = per_example_sent_starts[evidence_sent_idx]
            sent_end = per_example_sent_starts[evidence_sent_idx + 1]
            evidence_sent = full_context[sent_start: sent_end]
            option_for_evidence = np.argmin(per_example_options_prob_diff[evidence_sent_idx])
            option = process_text(options[i][option_for_evidence])
            qa_concat = processed_question + "[SEP]"
            qa_concat += option

            processed_contexts.append(evidence_sent)
            qa_list.append(qa_concat)
            labels.append(1)

            if hard_sampling_num >= 1:
                disturb_option_for_evidence = np.argmax(per_example_options_prob_diff[evidence_sent_idx])
                option = process_text(options[i][disturb_option_for_evidence])
                qa_concat = processed_question + "[SEP]"
                qa_concat += option

                processed_contexts.append(evidence_sent)
                qa_list.append(qa_concat)
                labels.append(0)
                hard_sampling_num -= 1

        all_irre_sent_idxs = list(filter(lambda x: x not in send_idxs_sorted_by_importance[:2],
                                         list(range(len(per_example_sent_starts) - 1))))
        if evidence_sampling_num * negative_sampling_ratio <= len(all_irre_sent_idxs):
            negative_sent_num = evidence_sampling_num * negative_sampling_ratio
            if 0 < negative_sent_num < 1 and random.random() > negative_sent_num:
                negative_sent_num = 1
            else:
                negative_sent_num = int(negative_sent_num)
        else:
            negative_sent_num = len(all_irre_sent_idxs)

        if negative_sent_num >= 1:
            option_num = len(list(filter(lambda x:x, options[i])))
            irre_options = random.sample(list(range(option_num)), negative_sent_num)
            for idx, irre_sent_idx in enumerate(random.sample(all_irre_sent_idxs, negative_sent_num)):
                sent_start = per_example_sent_starts[irre_sent_idx]
                sent_end = per_example_sent_starts[irre_sent_idx + 1]
                irre_sent = full_context[sent_start: sent_end]
                option = process_text(options[i][irre_options[idx]])
                qa_concat = processed_question + "[SEP]"
                qa_concat += option

                processed_contexts.append(irre_sent)
                qa_list.append(qa_concat)
                labels.append(0)

    if len(processed_contexts) == 0:
        return None

    tokenized_examples = tokenizer(
        processed_contexts,
        qa_list,
        truncation=True,
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    tokenized_examples['label'] = labels

    # Un-flatten
    return tokenized_examples


def prepare_features_for_generating_evidence_using_selector(examples, tokenizer=None, data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    qa_list = []
    processed_contexts = []
    all_example_ids = []
    all_sent_idx = []

    for i in range(len(answers)):
        full_context = contexts[i]

        qa_concat = process_text(questions[i])
        for j in range(4):
            option = process_text(options[i][j])
            qa_concat += "[SEP]"
            qa_concat += option

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        for j in range(len(per_example_sent_starts) - 1):
            sent_start = per_example_sent_starts[j]
            sent_end = per_example_sent_starts[j + 1]
            sent = full_context[sent_start: sent_end]
            processed_contexts.append(sent)
            qa_list.append(qa_concat)
            all_example_ids.append(example_ids[i])
            all_sent_idx.append(j)

    tokenized_examples = tokenizer(
        processed_contexts,
        qa_list,
        truncation=True,
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    tokenized_examples['example_ids'] = all_example_ids
    tokenized_examples['sent_idx'] = all_sent_idx

    # Un-flatten
    return tokenized_examples


def prepare_features_for_generating_optionwise_evidence(examples, tokenizer=None, data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    qa_list = []
    processed_contexts = []
    all_example_ids = []
    all_sent_idx = []

    num_choices = len(options[0])

    for i in range(len(answers)):
        full_context = contexts[i]

        processed_question = process_text(questions[i])

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        for j in range(len(per_example_sent_starts) - 1):
            sent_start = per_example_sent_starts[j]
            sent_end = per_example_sent_starts[j + 1]
            sent = full_context[sent_start: sent_end]
            for k in range(num_choices):
                option = process_text(options[i][k])
                qa_concat = processed_question + "[SEP]"
                qa_concat += option
                processed_contexts.append(sent)
                qa_list.append(qa_concat)
                all_example_ids.append(example_ids[i] + '_' + str(k))
                all_sent_idx.append(j)

    tokenized_examples = tokenizer(
        processed_contexts,
        qa_list,
        truncation=True,
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    tokenized_examples['example_ids'] = all_example_ids
    tokenized_examples['sent_idx'] = all_sent_idx

    # Un-flatten
    return tokenized_examples


def prepare_features_for_reading_evidence(examples, evidence_logits=None, pseudo_label_or_not=True,
                                          run_pseudo_label_with_options=False,
                                          evidence_len=2, tokenizer=None, data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    labels = []
    qa_list = []
    processed_contexts = []

    for i in range(len(answers)):
        full_context = contexts[i]
        example_id = example_ids[i]
        label = ord(answers[i]) - ord("A")
        labels.append(label)

        per_example_evidence_logits = evidence_logits[example_id]
        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        evidence_len = evidence_len if evidence_len <= len(evidence_logits[example_id]) else len(
            evidence_logits[example_id])
        if run_pseudo_label_with_options:
            if data_args.filter_label_with_ground_truth or not pseudo_label_or_not:
                per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                        key=lambda x: per_example_evidence_logits[x][label])[
                                                 : evidence_len]
            else:
                per_example_evidence_sent_idxs = list(set(sum([sorted(per_example_evidence_logits.keys(),
                                                                      key=lambda x: per_example_evidence_logits[x][
                                                                          option_idx])[
                                                               : evidence_len] for option_idx in range(4)], [])))

        else:
            if data_args.filter_label_with_ground_truth or not pseudo_label_or_not:
                per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                        key=lambda x: per_example_evidence_logits[x], reverse=True)[
                                                 : evidence_len]
            else:
                per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                        key=lambda x: abs(per_example_evidence_logits[x]),
                                                        reverse=True)[
                                                 : evidence_len]

        evidence_concat = ""
        for evidence_sent_idx in sorted(per_example_evidence_sent_idxs):
            sent_start = per_example_sent_starts[evidence_sent_idx]
            sent_end = per_example_sent_starts[evidence_sent_idx + 1]
            evidence_concat += full_context[sent_start: sent_end]

        processed_contexts.append([evidence_concat] * 4)

        question = process_text(questions[i])
        qa_pairs = []
        for j in range(4):
            option = process_text(options[i][j])

            qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
            qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
            qa_pairs.append(qa_cat)
        qa_list.append(qa_pairs)

    first_sentences = sum(processed_contexts, [])
    second_sentences = sum(qa_list, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    tokenized_examples['label'] = labels

    # Un-flatten
    return tokenized_examples


def prepare_features_for_reading_optionwise_evidence(examples, evidence_logits=None, evidence_len=2, tokenizer=None,
                                                     data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    labels = []
    qa_list = []
    processed_contexts = []
    all_example_ids = []
    evidence_sentences = []

    num_choices = len(options[0])

    for i in range(len(answers)):
        full_context = contexts[i]

        # label = ord(answers[i]) - ord("A")

        question = process_text(questions[i])
        for j in range(num_choices):
            if options[i][j] == "":
                continue
            labels.append(j)
            example_id = example_ids[i] + '_' + str(j)
            all_example_ids.append(example_id)

            per_example_evidence_logits = evidence_logits[example_id]
            per_example_sent_starts = sent_starts[i]
            per_example_sent_starts.append(len(full_context))

            evidence_len = evidence_len if evidence_len <= len(evidence_logits[example_id]) else len(
                evidence_logits[example_id])
            per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                    key=lambda x: per_example_evidence_logits[x], reverse=True)[
                                             : evidence_len]
            evidence_concat = ""
            for evidence_sent_idx in sorted(per_example_evidence_sent_idxs):
                sent_start = per_example_sent_starts[evidence_sent_idx]
                sent_end = per_example_sent_starts[evidence_sent_idx + 1]
                evidence_concat += full_context[sent_start: sent_end]
            evidence_sentences.append(evidence_concat)

            processed_contexts.append([evidence_concat] * num_choices)

            qa_pairs = []
            for k in range(num_choices):
                option = process_text(options[i][k])

                qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
                qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
                qa_pairs.append(qa_cat)
            qa_list.append(qa_pairs)

    first_sentences = sum(processed_contexts, [])
    second_sentences = sum(qa_list, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples = {k: [v[i: i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}

    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = all_example_ids
    tokenized_examples['evidence_sentence'] = evidence_sentences

    # Un-flatten
    return tokenized_examples


def get_competitive_score_for_example(evidence_logits, example_id, label, prediction=None, score_method='max_wrong_sub_right',
                                 topn_scoring_scope=1):

    sent_num = len(evidence_logits[example_id + '_0'])
    topn_scoring_scope = topn_scoring_scope if sent_num > topn_scoring_scope > 0 else sent_num

    per_example_noisy_scores = [
        sum(sorted(evidence_logits[example_id + '_' + str(choice)].values(), reverse=True)[: topn_scoring_scope])
        for choice in range(4) if example_id + '_' + str(choice) in evidence_logits.keys()]

    if score_method == 'max_wrong':
        per_example_noisy_scores[label] = -1000
        per_example_noisy_score = max(per_example_noisy_scores)
    elif score_method == 'max_wrong_sub_right':
        right_score = per_example_noisy_scores[label]
        per_example_noisy_scores[label] = -1000
        per_example_noisy_score = max(per_example_noisy_scores) - right_score
    elif score_method == 'max_wrong_sub_predict':
        pred_choice = np.argmax(prediction)
        pred_score = per_example_noisy_scores[pred_choice]
        per_example_noisy_scores[pred_choice] = -1000
        per_example_noisy_score = max(per_example_noisy_scores) - pred_score
    else:
        raise ValueError()
    # noisy_score[eid] = per_example_noisy_score - max(evidence_logits[eid + '_' + str(ord(label) - ord('A'))].values())
    return per_example_noisy_score


def get_evidence_sent_idx(sent_starts, sent_ends, article, evidence):
    processed_evidence = process_text(evidence)
    span_start = article.find(processed_evidence)
    span_end = span_start + len(processed_evidence) - 1

    sent_start_idx = 0
    sent_end_idx = 0
    found_start = False
    found_end = False
    for i, (sent_start, sent_end) in enumerate(zip(sent_starts, sent_ends)):
        # print(sent_start, sent_end)
        if span_start >= sent_start and span_start <= sent_end:
            sent_start_idx = i
            found_start = True
        if span_end >= sent_start and span_end <= sent_end:
            sent_end_idx = i
            found_end = True
    if not (found_start and found_end):
        # print(evidence)
        return -1, -1
    return sent_start_idx, sent_end_idx


def prepare_features_for_bidirectional_answer_verifier(
        examples,
        train_verifier_with_option=True,
        evidence_logits=None,
        positive_evidence_len=1,
        add_polarity_hint=False,
        negative_evidence_len=1,
        evidence_label=None,
        tokenizer=None,
        data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    labels = []
    qa_list = []
    processed_contexts = []
    evidence = []

    num_choices = len(options[0])

    for i in range(len(answers)):
        processed_context = contexts[i]

        label = ord(answers[i]) - ord("A")
        labels.append(label)

        question = process_text(questions[i])
        context_list = []
        qa_pairs = []

        per_example_sent_starts = sent_starts[i]
        per_example_sent_ends = [char_idx - 1 for char_idx in per_example_sent_starts[1:]] + [
            len(processed_context) - 1]

        per_example_evidence = []
        sent_num = len(evidence_logits[example_ids[i] + '_' + str(0)])
        for j in range(num_choices):
            option = process_text(options[i][j])
            qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
            qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
            qa_pairs.append(qa_cat)

            optionwise_example_id = example_ids[i] + '_' + str(j)

            per_example_evidence_logits = evidence_logits[optionwise_example_id]

            sents_with_evidential_score = {f'{sent_idx}_{polarity}': score
                                           for sent_idx, sent_logit in per_example_evidence_logits.items()
                                           for polarity, score in enumerate(sent_logit)}
            sorted_sents_by_evidential_score = sorted(sents_with_evidential_score.items(), key=lambda x: x[1],
                                                      reverse=True)

            positive_evidence, negative_evidence = '', ''
            positive_evidence_len = positive_evidence_len if positive_evidence_len <= sent_num else sent_num
            negative_evidence_len = negative_evidence_len if negative_evidence_len <= sent_num else sent_num

            # 1 means positive evidence, 2 means negative evidence
            evidence_len = {1: positive_evidence_len, 2: negative_evidence_len}
            evidence_idxs = {1: [], 2: []}

            if evidence_label is not None:
                per_example_evidence_label = evidence_label[example_ids[i]]
                if j == label and per_example_evidence_label[0] is not None:
                    positive_evidence_label = per_example_evidence_label[0]
                    sent_start_idx, sent_end_idx = get_evidence_sent_idx(per_example_sent_starts, per_example_sent_ends,
                                                                         processed_context, positive_evidence_label[0])
                    for evi_sent_idx in range(sent_start_idx, sent_end_idx + 1):
                        evidence_idxs[1].append(evi_sent_idx)
                        evidence_len[1] -= 1

                if j != label and per_example_evidence_label[1] is not None and j in per_example_evidence_label[1]:
                    negative_evidence_label = per_example_evidence_label[1][j]
                    sent_start_idx, sent_end_idx = get_evidence_sent_idx(per_example_sent_starts, per_example_sent_ends,
                                                                         processed_context, negative_evidence_label[0])
                    for evi_sent_idx in range(sent_start_idx, sent_end_idx + 1):
                        evidence_idxs[2].append(evi_sent_idx)
                        evidence_len[2] -= 1

            already_evidence_set = []

            for evidence_idx_with_polarity, evidence_score in sorted_sents_by_evidential_score:
                evidence_idx, polarity = evidence_idx_with_polarity.split("_")
                evidence_idx = int(evidence_idx)
                polarity = int(polarity)
                if evidence_idx in already_evidence_set:
                    continue
                if polarity == 0:
                    if data_args.dynamic_evidence_len:
                        already_evidence_set.append(evidence_idx)
                    continue
                if not data_args.dynamic_evidence_len and len(evidence_idxs[polarity]) >= evidence_len[polarity]:
                    continue
                if evidence_idx in evidence_idxs[polarity]:
                    if evidence_idx not in already_evidence_set:
                        already_evidence_set.append(evidence_idx)
                    continue
                if not data_args.dynamic_evidence_len and  \
                        len(evidence_idxs[1]) >= evidence_len[1] and len(evidence_idxs[2]) >= evidence_len[2]:
                    break
                already_evidence_set.append(evidence_idx)
                sent_start = per_example_sent_starts[evidence_idx]
                sent_end = per_example_sent_ends[evidence_idx]
                if polarity == 1 and positive_evidence == '':
                    positive_evidence = processed_context[sent_start: sent_end + 1]
                elif polarity == 2 and negative_evidence == '':
                    negative_evidence = processed_context[sent_start: sent_end + 1]

                evidence_idxs[polarity].append(evidence_idx)

            evidence_concat = ""
            if add_polarity_hint and len(evidence_idxs[1]) > 0:
                evidence_concat += "Positive Evidence: "
            for positive_evidence_idx in sorted(evidence_idxs[1]):
                sent_start = per_example_sent_starts[positive_evidence_idx]
                sent_end = per_example_sent_ends[positive_evidence_idx]
                evidence_concat += processed_context[sent_start: sent_end + 1]

            if add_polarity_hint and len(evidence_idxs[2]) > 0 and len(evidence_idxs[1]) > 0:
                evidence_concat += " [SEP] Negative Evidence: "
            elif len(evidence_idxs[2]) > 0 and len(evidence_idxs[1]) > 0:
                evidence_concat += " [SEP] "
            for negative_evidence_idx in sorted(evidence_idxs[2]):
                sent_start = per_example_sent_starts[negative_evidence_idx]
                sent_end = per_example_sent_ends[negative_evidence_idx]
                evidence_concat += processed_context[sent_start: sent_end + 1]

            per_example_evidence.append([positive_evidence, negative_evidence])
            context_list.append(evidence_concat)

        evidence.append(per_example_evidence)
        processed_contexts.append(context_list)
        qa_list.append(qa_pairs)

    first_sentences = sum(processed_contexts, [])
    second_sentences = sum(qa_list, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples = {k: [v[i: i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}
    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = example_ids
    tokenized_examples['evidence'] = evidence

    # Un-flatten
    return tokenized_examples


def prepare_features_for_eve_mrc(
        examples,
        evidence_logits=None,
        evidence_len=3,
        tokenizer=None,
        data_args=None):

    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    num_choices = len(options[0])

    features = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'example_ids': [],
                'label': [], 'positive_mask': [], 'negative_mask': [], 'evidence': []}

    for i in range(len(answers)):
        processed_context = contexts[i]
        example_id = example_ids[i]
        label = ord(answers[i]) - ord("A")
        question = process_text(questions[i])
        per_example_sent_starts = sent_starts[i]
        per_example_sent_ends = [char_idx - 1 for char_idx in per_example_sent_starts[1:]] + [
            len(processed_context) - 1]

        tokens_char_span = tokenizer(processed_context, return_offsets_mapping=True, add_special_tokens=False)[
            'offset_mapping']
        all_chars_to_start_chars, all_chars_to_end_chars = get_orig_chars_to_bounded_chars_mapping(tokens_char_span,
                                                                                                   len(processed_context))

        sent_num = len(evidence_logits[example_ids[i] + '_' + str(0)])
        evidence_len = evidence_len if evidence_len <= sent_num else sent_num

        choices_inputs = []
        for j in range(num_choices):
            option = process_text(options[i][j])
            optionwise_example_id = example_ids[i] + '_' + str(j)

            per_example_evidence_logits = evidence_logits[optionwise_example_id]

            sorted_sents_by_positive_score = sorted(per_example_evidence_logits.items(), key=lambda x: x[1][1],
                                                    reverse=True)
            sorted_sents_by_negative_score = sorted(per_example_evidence_logits.items(), key=lambda x: x[1][2],
                                                    reverse=True)

            # positive_mask =

            qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
            qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])

            inputs = tokenizer(
                processed_context,
                qa_cat,
                truncation="only_first",
                max_length=512,
                padding="max_length" if data_args.pad_to_max_length else False,
            )

            # positive_mask = inputs['token_type_ids']

            evidence_len = evidence_len if evidence_len <= sent_num and not data_args.dynamic_evidence_len else sent_num

            def _get_evidence_mask(sorted_sents_by_evi_score, evidence_len, polarity="positive"):
                evidence_mask = np.array(inputs['token_type_ids'])
                evidence_mask[0] = 1
                evidence_sent = ""
                polarity_to_idx = {'positive': 1, 'negative': 2, 'non_evi': 0}
                for evidence_idx, evidence_score in sorted_sents_by_evi_score[: evidence_len]:
                    if data_args.dynamic_evidence_len and np.argmax(evidence_score) != polarity_to_idx[polarity]:
                        continue
                    sent_start = per_example_sent_starts[evidence_idx]
                    sent_end = per_example_sent_ends[evidence_idx]
                    evidence_sent = processed_context[sent_start: sent_end + 1] if evidence_sent == "" else evidence_sent
                    new_sent_start, new_sent_end = all_chars_to_start_chars[sent_start], all_chars_to_end_chars[
                        sent_end]
                    # print(sent_start, sent_end, new_sent_start, new_sent_end)
                    if not (inputs.char_to_token(0, new_sent_start) and inputs.char_to_token(0, new_sent_end)):
                        continue
                    token_start = inputs.char_to_token(0, new_sent_start)
                    token_end = inputs.char_to_token(0, new_sent_end)
                    evidence_mask[token_start: token_end] = 1
                    #print(processed_context[sent_start: sent_end], tokenizer.convert_ids_to_tokens
                    #        (input_ids[token_start: token_end]))
                return evidence_mask.tolist(), evidence_sent

            positive_mask, positive_evidence = _get_evidence_mask(sorted_sents_by_positive_score, evidence_len, polarity="positive")
            negative_mask, negative_evidence = _get_evidence_mask(sorted_sents_by_negative_score, evidence_len, polarity="negative")
            assert len(positive_mask) == len(negative_mask) == len(inputs['attention_mask'])
            # print(positive_mask, negative_mask)
            inputs['positive_mask'] = positive_mask
            inputs['negative_mask'] = negative_mask
            inputs['evidence'] = [positive_evidence, negative_evidence]

            # question_ids = filter(lambda x: token_type_ids[x[0]], enumerate(input_ids))
            choices_inputs.append(inputs)

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = [x["attention_mask"] for x in choices_inputs]
        token_type_ids = [x["token_type_ids"] for x in choices_inputs]
        positive_mask = [x["positive_mask"] for x in choices_inputs]
        negative_mask = [x["negative_mask"] for x in choices_inputs]
        evidence = [x["evidence"] for x in choices_inputs]

        features['input_ids'].append(input_ids)
        features['attention_mask'].append(attention_mask)
        features['token_type_ids'].append(token_type_ids)
        features['example_ids'].append(example_id)
        features['label'].append(label)
        features['positive_mask'].append(positive_mask)
        features['negative_mask'].append(negative_mask)
        features['evidence'].append(evidence)

    # Un-flatten
    return features


def prepare_features_for_eve_mrc_with_relation_embedding(
        examples,
        evidence_logits=None,
        evidence_len=3,
        tokenizer=None,
        data_args=None):

    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    num_choices = len(options[0])

    features = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'example_ids': [],
                'label': [], 'evidence_type': [], 'evidence': []}

    for i in range(len(answers)):
        processed_context = contexts[i]
        example_id = example_ids[i]
        label = ord(answers[i]) - ord("A")
        question = process_text(questions[i])
        per_example_sent_starts = sent_starts[i]
        per_example_sent_ends = [char_idx - 1 for char_idx in per_example_sent_starts[1:]] + [
            len(processed_context) - 1]

        tokens_char_span = tokenizer(processed_context, return_offsets_mapping=True, add_special_tokens=False)[
            'offset_mapping']
        all_chars_to_start_chars, all_chars_to_end_chars = get_orig_chars_to_bounded_chars_mapping(tokens_char_span,
                                                                                                   len(processed_context))

        sent_num = len(evidence_logits[example_ids[i] + '_' + str(0)])
        evidence_len = evidence_len if evidence_len <= sent_num else sent_num

        choices_inputs = []
        for j in range(num_choices):
            option = process_text(options[i][j])
            optionwise_example_id = example_ids[i] + '_' + str(j)

            per_example_evidence_logits = evidence_logits[optionwise_example_id]


            sents_with_evidential_score = {f'{sent_idx}_{polarity}': score
                                           for sent_idx, sent_logit in per_example_evidence_logits.items()
                                           for polarity, score in enumerate(sent_logit)}
            sorted_sents_by_evidential_score = sorted(sents_with_evidential_score.items(), key=lambda x: x[1],
                                                      reverse=True)

            # positive_mask =

            qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
            qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])

            inputs = tokenizer(
                processed_context,
                qa_cat,
                truncation="only_first",
                max_length=512,
                padding="max_length" if data_args.pad_to_max_length else False,
            )

            positive_evidence, negative_evidence = None, None
            evidence_len = evidence_len if evidence_len <= sent_num else sent_num

            # 1 means positive evidence, 2 means negative evidence
            evidence_num = {1: 0, 2: 0}
            evidence_type = np.array(inputs['token_type_ids'])
            already_evidence_set = []
            for evidence_idx_with_polarity, evidence_score in sorted_sents_by_evidential_score:
                evidence_idx, polarity = evidence_idx_with_polarity.split("_")
                evidence_idx = int(evidence_idx)
                polarity = int(polarity)
                if evidence_idx in already_evidence_set:
                    continue
                if polarity == 0:
                    if data_args.dynamic_evidence_len:
                        already_evidence_set.append(evidence_idx)
                    continue
                if not data_args.dynamic_evidence_len and evidence_num[polarity] >= evidence_len:
                    continue
                already_evidence_set.append(evidence_idx)
                sent_start = per_example_sent_starts[evidence_idx]
                sent_end = per_example_sent_ends[evidence_idx]
                if polarity == 1 and positive_evidence is None:
                    positive_evidence = processed_context[sent_start: sent_end + 1]
                elif polarity == 2 and negative_evidence is None:
                    negative_evidence = processed_context[sent_start: sent_end + 1]

                new_sent_start, new_sent_end = all_chars_to_start_chars[sent_start], all_chars_to_end_chars[
                    sent_end]
                if not (inputs.char_to_token(0, new_sent_start) and inputs.char_to_token(0, new_sent_end)):
                    continue
                token_start = inputs.char_to_token(0, new_sent_start)
                token_end = inputs.char_to_token(0, new_sent_end)
                evidence_type[token_start: token_end] = polarity + 1
                evidence_num[polarity] += 1
                #print(processed_context[sent_start: sent_end], tokenizer.convert_ids_to_tokens
                #        (input_ids[token_start: token_end]))

            inputs['evidence_type'] = evidence_type
            inputs['evidence'] = [positive_evidence, negative_evidence]

            # question_ids = filter(lambda x: token_type_ids[x[0]], enumerate(input_ids))
            choices_inputs.append(inputs)

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = [x["attention_mask"] for x in choices_inputs]
        token_type_ids = [x["token_type_ids"] for x in choices_inputs]
        evidence_type = [x["evidence_type"] for x in choices_inputs]
        evidence = [x["evidence"] for x in choices_inputs]

        features['input_ids'].append(input_ids)
        features['attention_mask'].append(attention_mask)
        features['token_type_ids'].append(token_type_ids)
        features['example_ids'].append(example_id)
        features['label'].append(label)
        features['evidence_type'].append(evidence_type)
        features['evidence'].append(evidence)

    # Un-flatten
    return features


def prepare_features_for_answer_verifier(
        examples,
        train_verifier_with_option=False,
        train_verifier_with_non_overlapping_evidence=False,
        train_verifier_with_sample_weighting=False,
        score_method="max_wrong_sub_right",
        evidence_logits=None,
        evidence_len=2,
        tokenizer=None,
        data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    labels = []
    example_id_list = []
    qa_list = []
    processed_contexts = []
    competitive_scores = []
    evidence_list = []


    num_choices = len(options[0])
    evidence_with_logits = []

    for i in range(len(answers)):
        full_context = contexts[i]
        eid = example_ids[i]

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        if len(per_example_sent_starts) != len(evidence_logits[eid + '_0']) + 1:
            continue

        label = ord(answers[i]) - ord("A")
        labels.append(label)
        example_id_list.append(eid)

        if train_verifier_with_sample_weighting:
            competitive_score = get_competitive_score_for_example(evidence_logits, eid, label, score_method=score_method)
            competitive_scores.append(competitive_score)

        question = process_text(questions[i])
        context_list = []

        if train_verifier_with_option:
            qa_pairs = []
            for j in range(num_choices):
                option = process_text(options[i][j])

                qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
                qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
                qa_pairs.append(qa_cat)
        else:
            qa_pairs = [question] * num_choices

        per_example_evidence_with_logits = [[] for _ in range(num_choices)]
        per_example_evidence_sent_idxs = [[] for _ in range(num_choices)]
        sent_num = len(evidence_logits[example_ids[i] + '_' + str(0)])
        if train_verifier_with_non_overlapping_evidence:
            all_sorted_evidence_idxes = []
            max_evidence_num = num_choices * evidence_len if num_choices * evidence_len <= sent_num else sent_num

            for j in range(num_choices):
                optionwise_example_id = example_ids[i] + '_' + str(j)

                per_option_evidence_logits = evidence_logits[optionwise_example_id]

                per_option_sorted_evidence_idxes = sorted(per_option_evidence_logits.items(),
                                                          key=lambda x: x[1], reverse=True)[: max_evidence_num]
                all_sorted_evidence_idxes.append(per_option_sorted_evidence_idxes)
            all_sorted_evidence_idxes = torch.tensor(all_sorted_evidence_idxes)

            while sum([len(l) for l in per_example_evidence_sent_idxs]) < max_evidence_num:
                max_score_idx = torch.argmax(all_sorted_evidence_idxes[:, :, 1])
                max_score_idx = (max_score_idx // all_sorted_evidence_idxes.size(1),
                                 max_score_idx % all_sorted_evidence_idxes.size(1))
                max_score_option_idx = max_score_idx[0].item()
                max_score_evidence = all_sorted_evidence_idxes[max_score_idx][0].item()
                if len(per_example_evidence_sent_idxs[max_score_option_idx]) < evidence_len:
                    per_example_evidence_sent_idxs[max_score_option_idx].append(int(max_score_evidence))
                    all_sorted_evidence_idxes[torch.where(all_sorted_evidence_idxes == max_score_evidence)[:-1] + (1, )] = -1
                else:
                    all_sorted_evidence_idxes[max_score_idx][1] = -1
            for j in range(num_choices):
                evidence_concat = ""
                optionwise_example_id = example_ids[i] + '_' + str(j)

                for evidence_sent_idx in sorted(per_example_evidence_sent_idxs[j]):
                    sent_start = per_example_sent_starts[evidence_sent_idx]
                    sent_end = per_example_sent_starts[evidence_sent_idx + 1]
                    evidence_concat += full_context[sent_start: sent_end]
                    per_example_evidence_with_logits[j].append([evidence_logits[optionwise_example_id][evidence_sent_idx],
                                                                full_context[sent_start: sent_end]])
                context_list.append(evidence_concat)
        else:
            per_example_evidence = []
            for j in range(num_choices):
                optionwise_example_id = example_ids[i] + '_' + str(j)

                per_example_evidence_logits = evidence_logits[optionwise_example_id]

                evidence_len = evidence_len if evidence_len <= len(evidence_logits[optionwise_example_id]) else len(
                    evidence_logits[optionwise_example_id])
                per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                        key=lambda x: per_example_evidence_logits[x], reverse=True)[
                                                 : evidence_len]

                evidence_start = per_example_sent_starts[per_example_evidence_sent_idxs[0]]
                evidence_end = per_example_sent_starts[per_example_evidence_sent_idxs[0] + 1]
                per_example_evidence.append([full_context[evidence_start: evidence_end]])
                
                evidence_concat = ""
                for evidence_sent_idx in sorted(per_example_evidence_sent_idxs):
                    sent_start = per_example_sent_starts[evidence_sent_idx]
                    sent_end = per_example_sent_starts[evidence_sent_idx + 1]
                    evidence_concat += full_context[sent_start: sent_end]
                    per_example_evidence_with_logits[j].append([evidence_logits[optionwise_example_id][evidence_sent_idx],
                                                                full_context[sent_start: sent_end]])

                context_list.append(evidence_concat)

        evidence_with_logits.append(per_example_evidence_with_logits)
        processed_contexts.append(context_list)
        qa_list.append(qa_pairs)
        evidence_list.append(per_example_evidence)

    first_sentences = sum(processed_contexts, [])
    second_sentences = sum(qa_list, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples = {k: [v[i: i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}
    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = example_id_list
    tokenized_examples['evidence'] = evidence_list
    if train_verifier_with_sample_weighting:
        tokenized_examples['competitive_scores'] = competitive_scores

    # Un-flatten
    return tokenized_examples


def prepare_features_for_training_answer_verifier(
        examples,
        answer_logits=None,
        evidence_logits=None,
        train_answer_verifier_with_option=False,
        downsampling=False,
        is_training=False,
        evidence_len=2,
        tokenizer=None,
        data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    labels = []
    qa_list = []
    processed_contexts = []
    filtered_example_ids = []

    for i in range(len(answers)):
        full_context = contexts[i]
        example_id = example_ids[i]

        if example_id not in answer_logits.keys():
            continue

        prediction = np.argmax(answer_logits[example_id])
        ground_truth = ord(answers[i]) - ord("A")
        if downsampling and not int(prediction != ground_truth) and random.random() > 0.5 and is_training:
            continue
        labels.append(int(prediction != ground_truth))

        question = process_text(questions[i])

        if train_answer_verifier_with_option:
            option = process_text(options[i][prediction])

            qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
            qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
        else:
            qa_cat = question
        qa_list.append(qa_cat)

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        optionwise_example_id = example_ids[i] + '_' + str(prediction)

        per_example_evidence_logits = evidence_logits[optionwise_example_id]

        evidence_len = evidence_len if evidence_len <= len(evidence_logits[optionwise_example_id]) else len(evidence_logits[optionwise_example_id])
        per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                key=lambda x: per_example_evidence_logits[x], reverse=True)[
                                         : evidence_len]
        evidence_concat = ""
        for evidence_sent_idx in sorted(per_example_evidence_sent_idxs):
            sent_start = per_example_sent_starts[evidence_sent_idx]
            sent_end = per_example_sent_starts[evidence_sent_idx + 1]
            evidence_concat += full_context[sent_start: sent_end]

        processed_contexts.append(evidence_concat)
        filtered_example_ids.append(example_id)

    tokenized_examples = tokenizer(
        processed_contexts,
        qa_list,
        truncation="only_first",
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = filtered_example_ids

    # Un-flatten
    return tokenized_examples


def prepare_features_for_training_mc_style_answer_verifier(
        examples,
        answer_logits=None,
        evidence_logits=None,
        evidence_len=2,
        is_training=False,
        tokenizer=None,
        data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    labels = []
    qa_list = []
    processed_contexts = []
    filtered_example_ids = []

    for i in range(len(answers)):
        full_context = contexts[i]
        example_id = example_ids[i]

        if example_id not in answer_logits.keys():
            continue

        label = ord(answers[i]) - ord("A")
        labels.append(label)

        question = process_text(questions[i])

        qa_pairs = []
        for j in range(4):
            option = process_text(options[i][j])

            qa_cat = concat_question_option(question, option, dataset=data_args.dataset)
            qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
            qa_pairs.append(qa_cat)
        qa_list.append(qa_pairs)

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        evidence_len = evidence_len if evidence_len <= len(evidence_logits[example_ids[i] + '_' + str(0)]) else len(
            evidence_logits[example_ids[i] + '_' + str(0)])
        if data_args.verifier_evidence_type == "prediction":
            prediction = np.argmax(answer_logits[example_id])
            optionwise_example_id = example_ids[i] + '_' + str(prediction)

            per_example_evidence_logits = evidence_logits[optionwise_example_id]

            per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                    key=lambda x: per_example_evidence_logits[x], reverse=True)[
                                             : evidence_len]
        elif data_args.verifier_evidence_type == "strongest":
            per_example_evidence_logits = {}
            for option in range(4):
                optionwise_example_id = example_ids[i] + '_' + str(option)
                per_example_optionwise_evidence_logits = evidence_logits[optionwise_example_id]
                for sent_idx, sent_logit in per_example_optionwise_evidence_logits.items():
                    if sent_idx not in per_example_evidence_logits.keys():
                        per_example_evidence_logits[sent_idx] = sent_logit
                    else:
                        per_example_evidence_logits[sent_idx] = sent_logit if sent_logit > per_example_evidence_logits[sent_idx] \
                            else per_example_evidence_logits[sent_idx]
            per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                    key=lambda x: per_example_evidence_logits[x], reverse=True)[
                                             : evidence_len]

        elif data_args.verifier_evidence_type == "optionwise":
            per_example_evidence_sent_idxs = []
            for option in range(4):
                optionwise_example_id = example_ids[i] + '_' + str(option)
                per_example_optionwise_evidence_logits = evidence_logits[optionwise_example_id]

                per_example_optionwise_evidence_sent_idxs = sorted(per_example_optionwise_evidence_logits.keys(),
                                                        key=lambda x: per_example_optionwise_evidence_logits[x], reverse=True)[
                                                 : evidence_len]
                per_example_evidence_sent_idxs += per_example_optionwise_evidence_sent_idxs
            per_example_evidence_sent_idxs = list(set(per_example_evidence_sent_idxs))
        else:
            raise ValueError("verifier evidence type error!")

        evidence_concat = ""
        for evidence_sent_idx in sorted(per_example_evidence_sent_idxs):
            sent_start = per_example_sent_starts[evidence_sent_idx]
            sent_end = per_example_sent_starts[evidence_sent_idx + 1]
            evidence_concat += full_context[sent_start: sent_end]

        processed_contexts.append([evidence_concat] * 4)
        filtered_example_ids.append(example_id)

    first_sentences = sum(processed_contexts, [])
    second_sentences = sum(qa_list, [])

    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation="only_first",
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = filtered_example_ids

    # Un-flatten
    return tokenized_examples
