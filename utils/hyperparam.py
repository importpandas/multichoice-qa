# coding=utf8
import os, sys
import time

EXP = 'exp'


def hyperparam_path_for_initializing_evidence_selector(model_args, data_args, training_args):
    dataset_name = f'dataset_{data_args.dataset}'
    model_type = f'model_{model_args.model_name_or_path.split("/")[-1]}'
    now_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
    exp_name = hyperparam_base(model_args, data_args, training_args)
    exp_name += f'__evidence_len_{data_args.evidence_len}'
    exp_name += f'__pseudo_path_{data_args.pseudo_label_path.split("/")[-1].replace(".pt", "")}'
    exp_name += f'__filtered_label_{data_args.filter_label_with_ground_truth}'
    try:
        if data_args.train_with_adversarial_examples:
            exp_name += f'__train_with_adver_examples_{data_args.train_with_adversarial_examples}'
    except AttributeError:
        pass
    exp_path = os.path.join(training_args.output_dir, dataset_name, model_type, exp_name, now_time)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path


def hyperparam_path_for_baseline(model_args, data_args, training_args):
    dataset_name = f'dataset_{data_args.dataset}'
    model_type = f'model_{model_args.model_name_or_path.split("/")[-1]}'
    now_time = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime())
    exp_name = hyperparam_base(model_args, data_args, training_args)
    exp_path = os.path.join(training_args.output_dir, dataset_name, model_type, exp_name, now_time)

    if training_args.do_train and not os.path.exists(exp_path):
        os.makedirs(exp_path)
        return exp_path
    else:
        return model_args.model_name_or_path


def hyperparam_base(model_args, data_args, training_args):
    exp_name = ''
    exp_name += f'lr_{training_args.learning_rate}__'
    exp_name += f'per_device_bs_{training_args.per_device_train_batch_size}__'
    exp_name += f'gradacc_{training_args.gradient_accumulation_steps}__'
    exp_name += f'wr_{training_args.warmup_ratio}__'
    exp_name += f'epoch_{training_args.num_train_epochs}'
    return exp_name
