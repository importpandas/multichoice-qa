import six
import unicodedata
import torch
import random
import spacy


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
    pseudo_label_merged['acc'] = pseudo_label['acc']
    # pseudo_label_merged['acc'] = dict(**pseudo_label['acc']['train'],
    #                                   **pseudo_label['acc']['validation'], **pseudo_label['acc']['test'])
    pseudo_label_merged['logit'] = dict(**pseudo_label['pseudo_label']['train'],
                                      **pseudo_label['pseudo_label']['validation'], **pseudo_label['pseudo_label']['test'])
    return pseudo_label_merged


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
    if lower:
        outputs = outputs.lower()

    return outputs


# Preprocessing the datasets.
def prepare_features(examples, tokenizer=None, data_args=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']

    labels = []
    qa_list = []
    processed_contexts = []

    for i in range(len(answers)):
        label = ord(answers[i]) - ord("A")
        labels.append(label)
        processed_contexts.append([process_text(contexts[i])] * 4)

        question = process_text(questions[i])
        qa_pairs = []
        for j in range(4):
            option = process_text(options[i][j])

            if "_" in question:
                qa_cat = question.replace("_", option)
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
        max_length=data_args.max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    tokenized_examples['label'] = labels

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


    features = {}
    features['input_ids'] = []
    features['attention_mask'] = []
    features['token_type_ids'] = []
    features['sent_bound_token'] = []
    features['example_ids'] = []
    features['label'] = []

    for i in range(len(answers)):


        #processed_contexts.append([process_text(contexts[i])] * 4)
        processed_context = contexts[i]
        example_id = example_ids[i]
        label = ord(answers[i]) - ord("A")
        question = process_text(questions[i])
        per_example_sent_starts = sent_starts[i]
        per_example_sent_ends = [char_idx for char_idx in per_example_sent_starts[1:]] + [len(processed_context) - 1]

        choices_inputs = []
        all_text_b = []
        all_text_b_len = []
        for j in range(4):
            option = process_text(options[i][j])

            if "_" in question:
                qa_cat = question.replace("_", option)
            else:
                qa_cat = " ".join([question, option])
            #truncated_qa_cat = tokenizer.tokenize(qa_cat, add_special_tokens=False, max_length=data_args.max_qa_length)
            truncated_text_b_id = tokenizer.encode(qa_cat, truncation=True, max_length=data_args.max_qa_length, add_special_tokens=False)
            truncated_text_b = tokenizer.decode(truncated_text_b_id, clean_up_tokenization_spaces=False)
            all_text_b.append(truncated_text_b)
            all_text_b_len.append(len(tokenizer.encode(truncated_text_b, add_special_tokens=False)))

        for j in range(4):
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
        for j in range(per_example_feature_num):
            input_ids = [x["input_ids"][j] for x in choices_inputs]
            attention_mask = [x["attention_mask"][j] for x in choices_inputs]
            token_type_ids = [x["token_type_ids"][j] for x in choices_inputs]
            sent_bound_token = []
            for sent_idx, (sent_start, sent_end) in enumerate(zip(per_example_sent_starts, per_example_sent_ends)):
                if not (choices_inputs[0].char_to_token(j, sent_start) and choices_inputs[0].char_to_token(j, sent_end)):
                    continue
                if sent_idx != len(per_example_sent_starts) - 1:
                    sent_bound_token.append((sent_idx, choices_inputs[0].char_to_token(j, sent_start), choices_inputs[0].char_to_token(j, sent_end) - 1))
                else:
                    sent_bound_token.append((sent_idx, choices_inputs[0].char_to_token(j, sent_start), choices_inputs[0].char_to_token(j, sent_end)))
            features['input_ids'].append(input_ids)
            features['attention_mask'].append(attention_mask)
            features['token_type_ids'].append(token_type_ids)
            features['sent_bound_token'].append(sent_bound_token)
            features['example_ids'].append(example_id)
            features['label'].append(label)

    # Un-flatten
    return features


def prepare_features_for_initializing_complex_evidence_selector(examples, tokenizer=None, data_args=None, pseudo_label_path=""):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    all_pseudo_label = load_pseudo_label(pseudo_label_path)

    pseudo_logit = all_pseudo_label['logit']
    acc = all_pseudo_label['acc']


    features = {}
    features['input_ids'] = []
    features['attention_mask'] = []
    features['token_type_ids'] = []
    features['sent_label'] = []

    for i in range(len(answers)):


        #processed_contexts.append([process_text(contexts[i])] * 4)
        processed_context = contexts[i]
        example_id = example_ids[i]
        question = process_text(questions[i])
        per_example_sent_starts = sent_starts[i]
        per_example_sent_ends = [char_idx for char_idx in per_example_sent_starts[1:]] + [len(processed_context) - 1]

        choices_inputs = []
        all_text_b = []
        all_text_b_len = []
        for j in range(4):
            option = process_text(options[i][j])

            if "_" in question:
                qa_cat = question.replace("_", option)
            else:
                qa_cat = " ".join([question, option])
            #truncated_qa_cat = tokenizer.tokenize(qa_cat, add_special_tokens=False, max_length=data_args.max_qa_length)
            truncated_text_b_id = tokenizer.encode(qa_cat, truncation=True, max_length=data_args.max_qa_length, add_special_tokens=False)
            truncated_text_b = tokenizer.decode(truncated_text_b_id, clean_up_tokenization_spaces=False)
            all_text_b.append(truncated_text_b)
            all_text_b_len.append(len(tokenizer.encode(truncated_text_b, add_special_tokens=False)))

        for j in range(4):
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

        #per_example_feature_num = len(choices_inputs[0]["input_ids"])
        for j in range(1):
            input_ids = [x["input_ids"][j] for x in choices_inputs]
            attention_mask = [x["attention_mask"][j] for x in choices_inputs]
            token_type_ids = [x["token_type_ids"][j] for x in choices_inputs]
            sent_bound_token = []
            for sent_idx, (sent_start, sent_end) in enumerate(zip(per_example_sent_starts, per_example_sent_ends)):
                if not (choices_inputs[0].char_to_token(j, sent_start) and choices_inputs[0].char_to_token(j, sent_end)):
                    continue
                if sent_idx != len(per_example_sent_starts) - 1:
                    sent_bound_token.append((sent_idx, pseudo_logit[example_id][sent_idx],
                        choices_inputs[0].char_to_token(j, sent_start), choices_inputs[0].char_to_token(j, sent_end) - 1))
                else:
                    sent_bound_token.append((sent_idx, pseudo_logit[example_id][sent_idx],
                        choices_inputs[0].char_to_token(j, sent_start), choices_inputs[0].char_to_token(j, sent_end)))
            features['input_ids'].append(input_ids)
            features['attention_mask'].append(attention_mask)
            features['token_type_ids'].append(token_type_ids)
            features['sent_label'].append(sent_bound_token)

    # Un-flatten
    return features


def prepare_features_for_initializing_simple_evidence_selector(examples, evidence_len=2, tokenizer=None, data_args=None, pseudo_label_path=""):
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
            if data_args.train_with_adversarial_examples:
                per_example_noise_sent_idxs = sorted(pseudo_logit[example_id].keys(),
                                                        key=lambda x: pseudo_logit[example_id][x])[
                                                 : evidence_len]
            else:
                per_example_noise_sent_idxs = []
        else:
            per_example_evidence_sent_idxs = sorted(pseudo_logit[example_id].keys(),
                                                    key=lambda x: abs(pseudo_logit[example_id][x]), reverse=True)[: evidence_len]
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

        all_irre_sent_idxs = list(filter(lambda x: x not in per_example_evidence_sent_idxs and x not in per_example_noise_sent_idxs,
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
        max_length=data_args.max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    tokenized_examples['example_ids'] = all_example_ids
    tokenized_examples['sent_idx'] = all_sent_idx

    # Un-flatten
    return tokenized_examples

def prepare_features_for_reading_evidence(examples, evidence_logits=None, pseudo_label_or_not=True, evidence_len=2, tokenizer=None, data_args=None):
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


        evidence_len = evidence_len if evidence_len <= len(evidence_logits[example_id]) else len(evidence_logits[example_id])
        if data_args.filter_label_with_ground_truth or not pseudo_label_or_not:
            per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                    key=lambda x: per_example_evidence_logits[x], reverse=True)[
                                             : evidence_len]
        else:
            per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                    key=lambda x: abs(per_example_evidence_logits[x]), reverse=True)[
                                             : evidence_len]

        evidence_concat = ""
        for evidence_sent_idx in per_example_evidence_sent_idxs:
            sent_start = per_example_sent_starts[evidence_sent_idx]
            sent_end = per_example_sent_starts[evidence_sent_idx + 1]
            evidence_concat += full_context[sent_start: sent_end]

        processed_contexts.append([evidence_concat] * 4)

        question = process_text(questions[i])
        qa_pairs = []
        for j in range(4):
            option = process_text(options[i][j])

            if "_" in question:
                qa_cat = question.replace("_", option)
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
        max_length=data_args.max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    tokenized_examples['label'] = labels

    # Un-flatten
    return tokenized_examples

