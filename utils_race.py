import six
import unicodedata
import torch
import spacy


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


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

    qa_list = []
    processed_contexts = []

    for i in range(len(answers)):

        processed_contexts.append([process_text(contexts[i])] * 4)

        question = process_text(questions[i])
        qa_pairs = []
        for j in range(4):
            option = process_text(options[i][j])

            if "_" in question:
                qa_cat = question.replace("_", option)
            else:
                qa_cat = " ".join([question, option])
            #truncated_qa_cat = tokenizer.tokenize(qa_cat, add_special_tokens=False, max_length=data_args.max_qa_length)
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
    sent_start_token = [[v for v in [tokenized_examples.char_to_token(i*4 + j, sent_start) for sent_start in one_sent_starts] if v]
                        for i, one_sent_starts in enumerate(sent_starts) for j in range(4)]


    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}

    tokenized_examples['sent_start_token'] = []
    for i in range(0, len(sent_start_token), 4):
        min_sent_num = min([len(l) for l in sent_start_token[i: i + 4]])
        tokenized_examples['sent_start_token'].append([l[:min_sent_num] for l in sent_start_token[i: i + 4]])

    tokenized_examples['example_ids'] = example_ids

    # Un-flatten
    return tokenized_examples

# Preprocessing the datasets.
def prepare_features_with_pseudo_label(examples, evidence_len=2, tokenizer=None, data_args=None, all_pseudo_label: dict=None):
    contexts = examples['article']
    answers = examples['answer']
    options = examples['options']
    questions = examples['question']
    example_ids = examples['example_id']
    sent_starts = examples['article_sent_start']
    evidence_sents = [sorted(torch.argsort(torch.tensor(all_pseudo_label[example_id]))[-1 * evidence_len:].tolist()) for example_id in example_ids]

    qa_list = []
    labels = []
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
            #truncated_qa_cat = tokenizer.tokenize(qa_cat, add_special_tokens=False, max_length=data_args.max_qa_length)
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
    sent_start_tokens = [[v for v in [tokenized_examples.char_to_token(i*4 + j, sent_start) for sent_start in one_sent_starts] if v]
                        for i, one_sent_starts in enumerate(sent_starts) for j in range(4)]

    for i, (input_ids, token_type_ids, attention_mask, sent_start_token) in enumerate(zip(tokenized_examples['input_ids'], tokenized_examples['token_type_ids'],
                                                                            tokenized_examples['attention_mask'], sent_start_tokens)):
        evidence_sent = evidence_sents[i // 4]
        sep_pos = torch.where(torch.tensor(input_ids) == tokenizer.sep_token_id)[0][0].item()
        seq_len = len(input_ids)
        sent_start_token.append(sep_pos)
        sent_start_token.append(seq_len)
        new_input_ids = []
        new_token_type_ids = []
        new_attention_mask = []
        new_input_ids.append(input_ids[0])
        new_token_type_ids.append(token_type_ids[0])
        new_attention_mask.append(attention_mask[0])
        for evidence_start in evidence_sent + [len(sent_start_token) - 2]:
            token_start = sent_start_token[evidence_start]
            token_end = sent_start_token[evidence_start + 1]
            new_input_ids += input_ids[token_start: token_end]
            new_token_type_ids += token_type_ids[token_start: token_end]
            new_attention_mask += [1] * (token_end - token_start)
        tokenized_examples['input_ids'][i] = new_input_ids
        tokenized_examples['token_type_ids'][i] = new_token_type_ids
        tokenized_examples['attention_mask'][i] = new_attention_mask




    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    tokenized_examples['label'] = labels

    # Un-flatten
    return tokenized_examples
