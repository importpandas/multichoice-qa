# coding=utf8
import os
import re
import sys
import time
import logging

logger = logging.getLogger(__name__)

EXP = 'exp'


def hyperparam_path_for_initializing_evidence_selector(model_args, data_args, training_args):
    dataset_name = f'dataset_{data_args.dataset}'
    model_type = f'model_{model_args.model_name_or_path.split("/")[-1]}'
    now_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
    exp_name = hyperparam_base(model_args, data_args, training_args)
    if hasattr(data_args, 'evidence_len'):
        exp_name += f'__evidence_len_{data_args.evidence_len}'
    exp_name += f'__pseudo_path_{data_args.pseudo_label_path.split("/")[-1].replace(".pt", "")}'
    if hasattr(data_args, 'filter_label_with_ground_truth'):
        exp_name += f'__filtered_label_{data_args.filter_label_with_ground_truth}'
    if hasattr(data_args, 'train_with_adversarial_examples'):
        exp_name += f'__train_with_adver_examples_{data_args.train_with_adversarial_examples}'
    if hasattr(model_args, 'sentence_pooling_type'):
        exp_name += f'__pooling_type_{model_args.sentence_pooling_type}'
    exp_path = os.path.join(training_args.output_dir, dataset_name, model_type, exp_name, now_time)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path_for_two_stage_evidence_selector(model_args, data_args, training_args):
    dataset_name = f'dataset_{data_args.dataset}'
    model_type = f'model_{model_args.model_name_or_path.split("/")[-1]}'
    now_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
    exp_name = hyperparam_base(model_args, data_args, training_args)
    # exp_name += f'__max_evi_seq_len_{data_args.max_evidence_seq_length}'
    # exp_name += f'__max_seq_len_{data_args.max_seq_length}'
    if training_args.train_evidence_selector:
        exp_name = re.sub(r"__epoch_[\d.]+", "", exp_name)
        exp_name += f'__sel_epochs_{training_args.num_train_selector_epochs}'
        exp_name += f'__evi_sam_num_{data_args.evidence_sampling_num}'
        exp_name += f'__neg_sam_ratio_{data_args.negative_sampling_ratio}'
        exp_name += f'__hard_sam_{data_args.hard_negative_sampling}'
        if data_args.selector_jump_wrong_examples:
            exp_name += f'__jump_wrong_{data_args.selector_jump_wrong_examples}'
        if data_args.evidence_polarity_by_answer_correctness:
            exp_name += f'__evidence_polarity_{data_args.evidence_polarity_by_answer_correctness}'
    if training_args.train_answer_verifier:
        exp_name = re.sub(r"__epoch_[\d.]+", "", exp_name)
        # if not training_args.train_evidence_selector:
        #     exp_name += f'__evi_sam_num_{data_args.evidence_sampling_num}'
        exp_name += f'__veri_epochs_{training_args.num_train_verifier_epochs}'
        if data_args.dynamic_evidence_len:
            exp_name += f'__dynamic_evi_len_{data_args.dynamic_evidence_len}'
        else:
            if data_args.positive_evidence_len >= 0 or data_args.negative_evidence_len >= 0:
                exp_name += f'__pos_evi_len_{data_args.positive_evidence_len}'
                exp_name += f'__neg_evi_len_{data_args.negative_evidence_len}'
            else:
                exp_name += f'__veri_evi_len_{data_args.verifier_evidence_len}'
        if data_args.polarity_hint:
            exp_name += f'__polarity_hint_{data_args.polarity_hint}'
        if not data_args.train_verifier_with_option:
            exp_name += f'__train_ise_with_opt_{data_args.train_verifier_with_option}'
        if data_args.train_verifier_with_non_overlapping_evidence:
            exp_name += f'__veri_with_no_overlap_evi_{data_args.train_verifier_with_non_overlapping_evidence}'
        if model_args.initialize_verifier_from_reader:
            exp_name += f'__init_veri_from_reader_{model_args.initialize_verifier_from_reader}'
        if data_args.train_verifier_with_sample_weighting:
            exp_name += f'__weighting_temperature_{data_args.weighting_temperature}'
            exp_name += f'__weighting_method_{data_args.weighting_method}'
    # try:
    #     if training_args.train_answer_verifier:
    #         exp_name += f'__veri_type_{model_args.verifier_type}'
    #         exp_name += f'__veri_evi_len_{data_args.verifier_evidence_len}'
    #         if model_args.verifier_type == "classification":
    #             exp_name += f'__veri_with_opt_{data_args.train_answer_verifier_with_option}'
    #             exp_name += f'__downsampling_{data_args.train_verifier_with_downsampling}'
    #             exp_name += f'__logits_path_{data_args.answer_logits_path.split("/")[-1].replace(".json", "")}'
    #         elif model_args.verifier_type == "multi_choice":
    #             exp_name += f'__veri_evi_type_{data_args.verifier_evidence_type}'
    # except:
    #     pass

    exp_path = os.path.join(training_args.output_dir, dataset_name, model_type, exp_name, now_time)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path_for_eve_mrc(model_args, data_args, training_args):
    dataset_name = f'dataset_{data_args.dataset}'
    model_type = f'model_{model_args.model_name_or_path.split("/")[-1]}'
    now_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
    exp_name = hyperparam_base(model_args, data_args, training_args)
    if training_args.train_answer_verifier:
        exp_name = re.sub(r"__epoch_[\d.]+", "", exp_name)
        # if not training_args.train_evidence_selector:
        #     exp_name += f'__evi_sam_num_{data_args.evidence_sampling_num}'
        if data_args.dynamic_evidence_len:
            exp_name += f'__dynamic_evi_len_{data_args.dynamic_evidence_len}'
        else:
            exp_name += f'__veri_evi_len_{data_args.verifier_evidence_len}'
        if training_args.large_learning_rate > 0:
            exp_name += f'__large_lr_{training_args.large_learning_rate}'
        exp_name += f'__eve_layer_{model_args.num_eve_layers}'
        exp_name += f'__pool_type_{model_args.output_pooling_type}'
        if not model_args.attention_score_scaling:
            exp_name += f'__attention_scaling_{model_args.attention_score_scaling}'
        if model_args.eve_with_relation_embedding:
            exp_name += f'__relation_type_num_{model_args.relation_type_num}'
            exp_name += f'__relation_method_{model_args.relation_encoding_method}'
            exp_name += f'__relation_head_share_{model_args.share_relation_across_head}'
            exp_name += f'__relation_layer_share_{model_args.share_relation_across_layer}'
        else:
            exp_name += f'__eve_head_num_{model_args.eve_head_num}'
            exp_name += f'__eve_att_head_{model_args.eve_att_head}'

    exp_path = os.path.join(training_args.output_dir, dataset_name, model_type, exp_name, now_time)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path_for_baseline(model_args, data_args, training_args):
    dataset_name = f'dataset_{data_args.dataset}'
    model_type = f'model_{model_args.model_name_or_path.split("/")[-1]}'
    now_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
    exp_name = hyperparam_base(model_args, data_args, training_args)
    try:
        if data_args.split_train_dataset:
            exp_name += f'__n_fold_{data_args.n_fold}'
            exp_name += f'__holdout_set_{data_args.holdout_set}'
    except AttributeError:
        logger.error("'data_args' has no attribute 'split_train_dataset'")
    try:
        if data_args.shuffle_train_dataset:
            exp_name += f'__shuffled_data_{data_args.shuffle_train_dataset}'
    except AttributeError:
        logger.error("'data_args' has no attribute 'split_train_dataset'")
    try:
        if model_args.loss_function:
            exp_name += f'__loss_function_{model_args.loss_function}'
    except AttributeError:
        logger.error("'model_args' has no attribute 'loss_function'")
    if data_args.pad_to_max_length:
        exp_name += f'__pad_maxlen_{data_args.pad_to_max_length}'

    exp_path = os.path.join(training_args.output_dir, dataset_name, model_type, exp_name, now_time)

    if training_args.do_train and not os.path.exists(exp_path):
        os.makedirs(exp_path)
        return exp_path
    else:
        return model_args.model_name_or_path


def hyperparam_base(model_args, data_args, training_args):
    exp_name = ''
    exp_name += f'lr_{training_args.learning_rate}__'
    if training_args.weight_decay > 0:
        exp_name += f'wd_{training_args.weight_decay}__'
    exp_name += f'bs_{training_args.train_batch_size * training_args.gradient_accumulation_steps}__'
    exp_name += f'wr_{training_args.warmup_ratio}__'
    exp_name += f'epoch_{training_args.num_train_epochs}__'
    exp_name += f'seed_{training_args.seed}'
    return exp_name
