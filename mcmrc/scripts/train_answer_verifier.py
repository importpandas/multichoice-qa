# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.
import json
import os
import sys
import logging
from pathlib import Path
import datasets
from datasets import load_dataset, ReadInstruction, Dataset
from functools import partial
from objprint import add_objprint

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
import numpy as np

from utils.hyperparam import hyperparam_path_for_two_stage_evidence_selector
from utils.initialization import setup_root_logger
from utils.utils_distributed_training import is_main_process

from ..trainer.trainer import Trainer
from ..data_utils.collator import *
from ..data_utils.eval_expmrc import evaluate_multi_choice
from ..cli.argument import BasicModelArguments, BasicDataTrainingArguments
from ..trainer.trainer_utils import compute_mc_metrics, compute_classification_metrics, compute_verifier_metrics

from mcmrc.data_utils.processors import (
    prepare_features_for_initializing_extensive_evidence_selector,
    prepare_features_for_generating_optionwise_evidence, prepare_features_for_reading_optionwise_evidence,
    prepare_features_for_intensive_evidence_selector,
    prepare_features,
    load_exp_race_data
)

logger = logging.getLogger(__name__)


@add_objprint(color=False)
@dataclass
class ModelArguments(BasicModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    extensive_evidence_selector_path: str = field(
        default="",
        metadata={"help": "Path to extensive evidence selector"}
    )
    intensive_evidence_selector_path: str = field(
        default="",
        metadata={"help": "Path to intensive evidence selector"}
    )
    evidence_reader_path: str = field(
        default="",
        metadata={"help": "Path to pretrained MRC system 2 for answering questions using evidence"}
    )


@add_objprint(color=False)
@dataclass
class DataTrainingArguments(BasicDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    pseudo_label_path: str = field(
        default="",
        metadata={"help": "Path to pseudo evidence label"}
    )
    answer_logits_path: str = field(
        default="",
        metadata={"help": "Path to answer prediction for MRC model trained with cross validation"}
    )
    max_evidence_seq_length: int = field(
        default=200,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    evidence_sampling_num: int = field(
        default=2,
        metadata={
            "help": "number of sentences of each evidence"
        },
    )
    intensive_evidence_len: int = field(
        default=2,
        metadata={
            "help": "number of sentences of each evidence"
        },
    )
    train_intensive_selector_with_option: bool = field(
        default=False,
        metadata={
            "help": "Whether to train intensive selector with question-option pair"
        },
    )
    train_intensive_selector_with_non_overlapping_evidence: bool = field(
        default=False,
        metadata={
            "help": "Whether to train intensive selector with non overlapping evidence"
        },
    )
    exp_race_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate model on exp_race_file"},
    )


@add_objprint(color=False)
@dataclass
class AllTrainingArguments(TrainingArguments):
    train_extensive_evidence_selector: bool = field(
        default=False,
        metadata={"help": "Whether to train extensive evidence reader."})
    train_intensive_evidence_selector: bool = field(
        default=False,
        metadata={"help": "Whether to train intensive evidence reader."})
    eval_extensive_evidence_selector: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate extensive evidence reader."})
    eval_intensive_evidence_selector: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate intensive evidence reader."})
    eval_on_exp_race: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate intensive evidence reader on Exp RACE dev set."})
    num_train_selector_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs of evidence selector to perform."})
    num_train_verifier_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs to perform."})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AllTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    checkpoint_dir = hyperparam_path_for_two_stage_evidence_selector(model_args, data_args, training_args)
    ckpt_dir = Path(checkpoint_dir)
    postfix = ""
    if training_args.train_extensive_evidence_selector or training_args.train_intensive_evidence_selector:
        postfix += "_train"
    else:
        postfix += "_eval"
    setup_root_logger(ckpt_dir, training_args.local_rank, debug=False, postfix=postfix)

    training_args.output_dir = checkpoint_dir

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Model parameters %s", model_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the [datasets]: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    if data_args.dataset not in ['race', 'dream']:
        raise ValueError("Dataset should be race or dream.")

    if training_args.eval_on_exp_race and data_args.exp_race_file is None and data_args.dataset == 'race':
        raise ValueError("exp_race_file must be specified")

    if data_args.dataset == 'dream':
        training_args.eval_on_exp_race = False

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    data_files = {'train': data_args.train_file if data_args.train_file is not None else None,
                  'validation': data_args.validation_file if data_args.validation_file is not None else None,
                  'test': data_args.test_file if data_args.test_file is not None else None}

    # datasets = load_dataset(data_args.dataload_script, data_args.dataload_split,
    #                         data_files=data_files if data_files['train'] is not None else None,
    #                         data_dir=data_args.data_dir,
    #                         split={'train': ReadInstruction('train', from_=0, to=5, unit='abs'),
    #                                'validation': ReadInstruction('validation', from_=0, to=5, unit='abs'),
    #                                'test': ReadInstruction('test', from_=0, to=5, unit='abs')})
    datasets = load_dataset(data_args.dataload_script, data_args.dataload_split,
                            data_files=data_files if data_files['train'] is not None else None,
                            data_dir=data_args.data_dir)
    if training_args.eval_on_exp_race:
        exp_dataset = Dataset.from_dict(load_exp_race_data(data_args.exp_race_file))


    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    extensive_evidence_selector_path = model_args.extensive_evidence_selector_path \
        if model_args.extensive_evidence_selector_path else model_args.model_name_or_path
    intensive_evidence_selector_path = model_args.intensive_evidence_selector_path \
        if model_args.intensive_evidence_selector_path else model_args.model_name_or_path
    evidence_reader_path = model_args.evidence_reader_path \
        if model_args.evidence_reader_path else model_args.model_name_or_path

    extensive_selector_config = AutoConfig.from_pretrained(
        extensive_evidence_selector_path,
        cache_dir=model_args.cache_dir,
    )
    intensive_selector_config = AutoConfig.from_pretrained(
        intensive_evidence_selector_path,
        cache_dir=model_args.cache_dir,
    )
    evidence_reader_config = AutoConfig.from_pretrained(
        evidence_reader_path,
        cache_dir=model_args.cache_dir,
    )

    extensive_evidence_selector = AutoModelForSequenceClassification.from_pretrained(
        extensive_evidence_selector_path,
        config=extensive_selector_config,
        cache_dir=model_args.cache_dir,
    )
    intensive_evidence_selector = AutoModelForMultipleChoice.from_pretrained(
        intensive_evidence_selector_path,
        config=intensive_selector_config,
        cache_dir=model_args.cache_dir,
    )
    evidence_reader = AutoModelForMultipleChoice.from_pretrained(
        evidence_reader_path,
        config=evidence_reader_config,
        cache_dir=model_args.cache_dir,
    )

    if training_args.train_extensive_evidence_selector:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    pprepare_features_for_initializing_evidence_selector = partial(
        prepare_features_for_initializing_extensive_evidence_selector,
        evidence_sampling_num=data_args.evidence_sampling_num,
        tokenizer=tokenizer,
        data_args=data_args,
        pseudo_label_path=data_args.pseudo_label_path)

    pprepare_features_for_generating_optionwise_evidence = partial(
        prepare_features_for_generating_optionwise_evidence,
        tokenizer=tokenizer,
        data_args=data_args)

    pprepare_features_for_reading_optionwise_evidence = partial(
        prepare_features_for_reading_optionwise_evidence,
        tokenizer=tokenizer,
        data_args=data_args)

    pprepare_features_for_intensive_evidence_selector = partial(
        prepare_features_for_intensive_evidence_selector,
        evidence_len=data_args.intensive_evidence_len,
        train_intensive_selector_with_option=data_args.train_intensive_selector_with_option,
        train_intensive_selector_with_non_overlapping_evidence=data_args.train_intensive_selector_with_non_overlapping_evidence,
        tokenizer=tokenizer,
        data_args=data_args)

    pprepare_features_for_multiple_choice = partial(
        prepare_features,
        tokenizer=tokenizer,
        data_args=data_args)

    training_args.num_train_epochs = training_args.num_train_selector_epochs
    extensive_trainer = Trainer(
        model=extensive_evidence_selector,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSequenceClassification(tokenizer=tokenizer),
        compute_metrics=compute_mc_metrics,
    )

    training_args.num_train_epochs = training_args.num_train_verifier_epochs
    intensive_trainer = Trainer(
        model=intensive_evidence_selector,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_mc_metrics,
    )

    mc_trainer = Trainer(
        model=evidence_reader,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_mc_metrics,
    )

    if training_args.eval_intensive_evidence_selector:
        multiple_choice_datasets = {k: datasets[k].map(
            pprepare_features_for_multiple_choice,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        ) for k in datasets.keys()}

    if training_args.train_extensive_evidence_selector or training_args.eval_extensive_evidence_selector:
        train_extensive_evidence_selector_datasets = {k: datasets[k].map(
            pprepare_features_for_initializing_evidence_selector,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        ) for k in datasets.keys() if k != "train" or training_args.train_extensive_evidence_selector}

    if training_args.train_extensive_evidence_selector:
        extensive_trainer.train_dataset = train_extensive_evidence_selector_datasets["train"]
        extensive_trainer.eval_dataset = train_extensive_evidence_selector_datasets["validation"]
        train_result = extensive_trainer.train()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Extensive Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    # generate extensive evidence logits
    if training_args.train_intensive_evidence_selector:
        extensive_evidence_logits = {
            k: extensive_trainer.evidence_generating(v, pprepare_features_for_generating_optionwise_evidence) for k, v
            in datasets.items()}
    elif training_args.eval_intensive_evidence_selector:
        extensive_evidence_logits = {
            k: extensive_trainer.evidence_generating(v, pprepare_features_for_generating_optionwise_evidence)
            for k, v in datasets.items() if k != "train"}

    # prepare features for intensive evidence selector
    if training_args.train_intensive_evidence_selector or training_args.eval_intensive_evidence_selector:
        train_intensive_evidence_selector_datasets = {}
        extensive_evidence_sentences = {}
        for split in datasets.keys():
            if not training_args.train_intensive_evidence_selector and split == 'train':
                continue

            intensive_dataset = datasets[split].map(
                partial(pprepare_features_for_intensive_evidence_selector,
                        evidence_logits=extensive_evidence_logits[split]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            evidence_sentences = {eid: [[(logit, sent) for sent, logit in zip(option_sents, option_logits)]
                                        for option_sents, option_logits in zip(evidence_sent, evidence_logit)]
                                        for eid, evidence_sent, evidence_logit in
                                  zip(intensive_dataset['example_ids'],
                                      intensive_dataset['evidence_sentence'],
                                      intensive_dataset['evidence_logit'])}
            train_intensive_evidence_selector_datasets[split] = intensive_dataset.remove_columns(["evidence_sentence", "evidence_logit"])
            if split != "train":
                extensive_evidence_sentences[split] = evidence_sentences

        if training_args.eval_on_exp_race:
            multiple_choice_datasets['exp'] = exp_dataset.map(
                pprepare_features_for_multiple_choice,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            exp_extensive_evidence_logits = extensive_trainer.evidence_generating(
                exp_dataset, pprepare_features_for_generating_optionwise_evidence)

            processed_exp_dataset = exp_dataset.map(
                partial(pprepare_features_for_intensive_evidence_selector,
                        evidence_logits=exp_extensive_evidence_logits),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            evidence_sentences = {eid: [[(logit, sent) for sent, logit in zip(option_sents, option_logits)]
                                        for option_sents, option_logits in zip(evidence_sent, evidence_logit)]
                                        for eid, evidence_sent, evidence_logit in
                                  zip(processed_exp_dataset['example_ids'],
                                      processed_exp_dataset['evidence_sentence'],
                                      processed_exp_dataset['evidence_logit'])}
            processed_exp_dataset = processed_exp_dataset.remove_columns(["evidence_sentence", "evidence_logit"])
            train_intensive_evidence_selector_datasets['exp'] = processed_exp_dataset
            extensive_evidence_sentences['exp'] = evidence_sentences

    if training_args.train_intensive_evidence_selector:
        intensive_trainer.train_dataset = train_intensive_evidence_selector_datasets["train"]
        intensive_trainer.eval_dataset = train_intensive_evidence_selector_datasets["validation"]

        train_result = intensive_trainer.train()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "a+") as writer:
            logger.info("***** Intensive Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    # Evaluation
    # To use the best checkpoint model at end, use the aruguments
    # load_best_model_at_end, metric_for_best_model, evaluation_strategy steps
    # --load_best_model_at_end \
    # --metric_for_best_model accuracy \
    # --evaluation_strategy steps \

    if training_args.eval_extensive_evidence_selector:

        for split in ["validation", "test"]:
            logger.info(f"*** Evaluate {split} set ***")
            results = extensive_trainer.evaluate(train_extensive_evidence_selector_datasets[split]).metrics
            fulleval_results, all_evidence_sentences = extensive_trainer.evaluate_extensive_selector_with_explicit_reader(
                evidence_reader=evidence_reader,
                eval_dataset=datasets[split],
                feature_func_for_evidence_reading=pprepare_features_for_reading_optionwise_evidence,
                feature_func_for_evidence_generating=pprepare_features_for_generating_optionwise_evidence)

            metrics = {**results, **fulleval_results}
            output_eval_file = os.path.join(training_args.output_dir, f"{split}_extensive_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Extensive Eval results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            output_evidence_file = os.path.join(training_args.output_dir, f"{split}_evidence.json")
            with open(output_evidence_file, "w") as f:
                json.dump(all_evidence_sentences, f)

    if training_args.eval_intensive_evidence_selector:

        sets = ["validation", "test"]
        if training_args.eval_on_exp_race and data_args.dataset == "race":
            sets.append("exp")

        for split in sets:
            logger.info(f"*** Evaluate {split} set ***")
            metrics, predictions = intensive_trainer.evaluate_intensive_selector_with_explicit_reader(
                evidence_reader=evidence_reader,
                multiple_choice_dataset=multiple_choice_datasets[split],
                intensive_selector_dataset=train_intensive_evidence_selector_datasets[split])

            output_prediction_file = os.path.join(training_args.output_dir, f"{split}_intensive_predictions.json")
            with open(output_prediction_file, "w") as f:
                json.dump(predictions, f)
            if training_args.eval_on_exp_race and split == "exp":
                ground_truth_file = json.load(open(data_args.exp_race_file, 'rb'))
                for ratio, merge_prediction in predictions.items():
                    prediction_file = {}
                    for eid, probs in merge_prediction.items():
                        pred_option = np.argmax(probs)
                        pred_evidence = sorted(extensive_evidence_sentences['exp'][eid][pred_option], key=lambda x: x[0], reverse=True)[0][1]
                        prediction_file[eid] = {"answer": chr(pred_option + ord("A")), "evidence": pred_evidence}
                    all_f1, ans_f1, evi_f1, total_count, skip_count = evaluate_multi_choice(ground_truth_file,
                                                                                            prediction_file)
                    metrics[f"merge_{ratio}_all_f1"] = all_f1
                    metrics[f"merge_{ratio}_ans_f1"] = ans_f1
                    metrics[f"merge_{ratio}_evi_f1"] = evi_f1
                    metrics[f"merge_{ratio}_total_count"] = total_count
                    metrics[f"merge_{ratio}_skip_count"] = skip_count

            output_eval_file = os.path.join(training_args.output_dir, f"{split}_intensive_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info(f"***** Eval {split} results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        output_evidence_file = os.path.join(training_args.output_dir, f"all_extensive_evidence.json")
        with open(output_evidence_file, "w") as f:
            json.dump(extensive_evidence_sentences, f)





if __name__ == "__main__":
    main()
