import six
import unicodedata
import torch
import random
import numpy as np
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
                                        **pseudo_label['pseudo_label']['validation'],
                                        **pseudo_label['pseudo_label']['test'])
    if 'options_prob_diff' in pseudo_label.keys():
        pseudo_label_merged['options_prob_diff'] = dict(**pseudo_label['options_prob_diff']['train'],
                                                        **pseudo_label['options_prob_diff']['validation'],
                                                        **pseudo_label['options_prob_diff']['test'])

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
    tokenized_examples['example_ids'] = example_ids

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
        for j in range(4):
            option = process_text(options[i][j])

            if "_" in question:
                qa_cat = question.replace("_", option)
            else:
                qa_cat = " ".join([question, option])
            # truncated_qa_cat = tokenizer.tokenize(qa_cat, add_special_tokens=False, max_length=data_args.max_qa_length)
            truncated_text_b_id = tokenizer.encode(qa_cat, truncation=True, max_length=data_args.max_qa_length,
                                                   add_special_tokens=False)
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

                if "_" in question:
                    qa_cat = question.replace("_", option)
                else:
                    qa_cat = " ".join([question, option])
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


def prepare_features_for_initializing_extensive_evidence_selector(examples, evidence_sampling_num=2, tokenizer=None,
                                                                  data_args=None, pseudo_label_path=""):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']

    all_pseudo_label = load_pseudo_label(pseudo_label_path)

    pseudo_logit = all_pseudo_label['logit']
    acc = all_pseudo_label['acc']
    options_prob_diff = all_pseudo_label['options_prob_diff']

    qa_list = []
    labels = []
    processed_contexts = []

    for i in range(len(answers)):
        full_context = contexts[i]
        example_id = example_ids[i]

        per_example_options_prob_diff = options_prob_diff[example_id]

        processed_question = process_text(questions[i])

        evidence_sampling_num = evidence_sampling_num if evidence_sampling_num <= len(
            pseudo_logit[example_id]) else len(pseudo_logit[example_id])
        per_example_evidence_sent_idxs = sorted(pseudo_logit[example_id].keys(),
                                                key=lambda x: abs(pseudo_logit[example_id][x]), reverse=True)[
                                         : evidence_sampling_num]

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        for evidence_sent_idx in per_example_evidence_sent_idxs:
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

        all_irre_sent_idxs = list(filter(lambda x: x not in per_example_evidence_sent_idxs,
                                         list(range(len(per_example_sent_starts) - 1))))
        negative_sent_num = evidence_sampling_num if evidence_sampling_num <= len(all_irre_sent_idxs) else len(
            all_irre_sent_idxs)
        irre_options = random.sample(list(range(4)), negative_sent_num)
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
        max_length=data_args.max_seq_length,
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

    for i in range(len(answers)):
        full_context = contexts[i]

        processed_question = process_text(questions[i])

        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        for j in range(len(per_example_sent_starts) - 1):
            sent_start = per_example_sent_starts[j]
            sent_end = per_example_sent_starts[j + 1]
            sent = full_context[sent_start: sent_end]
            for k in range(4):
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
                                                                          option_num])[
                                                               : evidence_len] for option_num in range(4)], [])))

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

    for i in range(len(answers)):
        full_context = contexts[i]

        # label = ord(answers[i]) - ord("A")

        question = process_text(questions[i])
        for j in range(4):
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

            processed_contexts.append([evidence_concat] * 4)

            qa_pairs = []
            for k in range(4):
                option = process_text(options[i][k])

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
        max_length=data_args.max_evidence_seq_length,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = all_example_ids

    # Un-flatten
    return tokenized_examples


def prepare_features_for_intensive_evidence_selector(
        examples,
        train_intensive_selector_with_option=False,
        train_intensive_selector_with_non_overlapping_evidence=False,
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
    qa_list = []
    processed_contexts = []

    for i in range(len(answers)):
        full_context = contexts[i]

        label = ord(answers[i]) - ord("A")
        labels.append(label)

        question = process_text(questions[i])
        context_list = []

        if train_intensive_selector_with_option:
            qa_pairs = []
            for j in range(4):
                option = process_text(options[i][j])

                if "_" in question:
                    qa_cat = question.replace("_", option)
                else:
                    qa_cat = " ".join([question, option])
                qa_cat = " ".join(whitespace_tokenize(qa_cat)[- data_args.max_qa_length:])
                qa_pairs.append(qa_cat)
        else:
            qa_pairs = [question] * 4


        per_example_sent_starts = sent_starts[i]
        per_example_sent_starts.append(len(full_context))

        per_example_evidence_sent_idxs = [[] for _ in range(4)]
        sent_num = len(evidence_logits[example_ids[i] + '_' + str(0)])
        if train_intensive_selector_with_non_overlapping_evidence:
            all_sorted_evidence_idxes = []
            max_evidence_num = 4 * evidence_len if 4 * evidence_len <= sent_num else sent_num

            for j in range(4):
                optionwise_example_id = example_ids[i] + '_' + str(j)

                per_option_evidence_logits = evidence_logits[optionwise_example_id]

                per_option_sorted_evidence_idxes = sorted(per_option_evidence_logits.items(),
                                                          key=lambda x: x[1], reverse=True)[: max_evidence_num]
                all_sorted_evidence_idxes.append(per_option_sorted_evidence_idxes)
            all_sorted_evidence_idxes = torch.tensor(all_sorted_evidence_idxes)

            while sum([len(l) for l in per_example_evidence_sent_idxs]) < max_evidence_num:
                max_score_idx = torch.argmax(all_sorted_evidence_idxes[:, :, 1])
                max_score_idx = (max_score_idx // max_evidence_num, max_score_idx % max_evidence_num)
                max_score_option_idx = max_score_idx[0]
                max_score_evidence = all_sorted_evidence_idxes[max_score_idx][0]
                if len(per_example_evidence_sent_idxs[max_score_option_idx]) < 2:
                    per_example_evidence_sent_idxs[max_score_option_idx].append(int(max_score_evidence.item()))
                    all_sorted_evidence_idxes[torch.where(all_sorted_evidence_idxes == max_score_evidence)[:-1] + (1, )] = -1
                else:
                    all_sorted_evidence_idxes[max_score_idx][1] = -1
            for j in range(4):
                evidence_concat = ""
                for evidence_sent_idx in sorted(per_example_evidence_sent_idxs[j]):
                    sent_start = per_example_sent_starts[evidence_sent_idx]
                    sent_end = per_example_sent_starts[evidence_sent_idx + 1]
                    evidence_concat += full_context[sent_start: sent_end]
                context_list.append(evidence_concat)
        else:
            for j in range(4):
                optionwise_example_id = example_ids[i] + '_' + str(j)

                per_example_evidence_logits = evidence_logits[optionwise_example_id]

                evidence_len = evidence_len if evidence_len <= len(evidence_logits[optionwise_example_id]) else len(
                    evidence_logits[optionwise_example_id])
                per_example_evidence_sent_idxs = sorted(per_example_evidence_logits.keys(),
                                                        key=lambda x: per_example_evidence_logits[x], reverse=True)[
                                                 : evidence_len]
                evidence_concat = ""
                for evidence_sent_idx in sorted(per_example_evidence_sent_idxs):
                    sent_start = per_example_sent_starts[evidence_sent_idx]
                    sent_end = per_example_sent_starts[evidence_sent_idx + 1]
                    evidence_concat += full_context[sent_start: sent_end]

                context_list.append(evidence_concat)

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
    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    tokenized_examples['label'] = labels
    tokenized_examples['example_ids'] = example_ids

    # Un-flatten
    return tokenized_examples
