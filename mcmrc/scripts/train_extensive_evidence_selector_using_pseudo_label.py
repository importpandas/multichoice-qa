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
from datasets import load_dataset, ReadInstruction
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

from utils.hyperparam import hyperparam_path_for_two_stage_evidence_selector
from utils.initialization import setup_root_logger
from utils.utils_distributed_training import is_main_process

from ..trainer.trainer import Trainer, evaluate_verifier_with_reader_and_iselector, evaluate_mc_style_verifier_with_reader_and_iselector
from ..data_utils.collator import *
from ..cli.argument import BasicModelArguments, BasicDataTrainingArguments
from ..trainer.trainer_utils import compute_mc_metrics, compute_classification_metrics, compute_verifier_metrics

from mcmrc.data_utils.processors import (
    prepare_features_for_initializing_extensive_evidence_selector,
    prepare_features_for_generating_optionwise_evidence, prepare_features_for_reading_optionwise_evidence,
    prepare_features_for_intensive_evidence_selector,
    prepare_features_for_training_answer_verifier,
    prepare_features_for_training_mc_style_answer_verifier,
    prepare_features
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
    answer_verifier_path: str = field(
        default="",
        metadata={"help": "Path to pretrained MRC system 2 for answering questions using evidence"}
    )
    verifier_type: str = field(
        default="classification",
        metadata={"help": "'classification' model or 'multi_choice' model"}
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
    verifier_evidence_len: int = field(
        default=2,
        metadata={
            "help": "number of sentences of each evidence"
        },
    )
    train_verifier_with_downsampling: bool = field(
        default=False,
        metadata={
            "help": "Whether to train answer verifier with downsampling"
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
    train_answer_verifier_with_option: bool = field(
        default=False,
        metadata={
            "help": "Whether to train answer verifier with question-option pair"
        },
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
    train_answer_verifier: bool = field(
        default=False,
        metadata={"help": "Whether to train answer verifier."})
    eval_extensive_evidence_selector: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate extensive evidence reader."})
    eval_intensive_evidence_selector: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate intensive evidence reader."})
    eval_answer_verifier: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate answer verifier."})


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
    answer_verifier_path = model_args.answer_verifier_path \
        if model_args.answer_verifier_path else model_args.model_name_or_path

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
    answer_verifier_config = AutoConfig.from_pretrained(
        answer_verifier_path,
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
    if model_args.verifier_type == "classification":
        answer_verifier = AutoModelForSequenceClassification.from_pretrained(
            answer_verifier_path,
            config=answer_verifier_config,
            cache_dir=model_args.cache_dir,
        )
    elif model_args.verifier_type == "multi_choice":
        answer_verifier = AutoModelForMultipleChoice.from_pretrained(
            answer_verifier_path,
            config=answer_verifier_config,
            cache_dir=model_args.cache_dir,
        )

    if training_args.train_extensive_evidence_selector:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    pprepare_features_for_initializing_evidence_selector = partial(
        prepare_features_for_initializing_extensive_evidence_selector,
        evidence_len=data_args.evidence_sampling_num,
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

    if model_args.verifier_type == "classification":
        pprepare_features_for_training_answer_verifier = partial(
            prepare_features_for_training_answer_verifier,
            evidence_len=data_args.verifier_evidence_len,
            train_answer_verifier_with_option=data_args.train_answer_verifier_with_option,
            downsampling=data_args.train_verifier_with_downsampling,
            tokenizer=tokenizer,
            data_args=data_args)
    elif model_args.verifier_type == "multi_choice":
        pprepare_features_for_training_answer_verifier = partial(
            prepare_features_for_training_mc_style_answer_verifier,
            evidence_len=data_args.verifier_evidence_len,
            tokenizer=tokenizer,
            data_args=data_args)

    extensive_trainer = Trainer(
        model=extensive_evidence_selector,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSequenceClassification(tokenizer=tokenizer),
        compute_metrics=compute_mc_metrics,
    )

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

    verifier_trainer = Trainer(
        model=answer_verifier,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSequenceClassification(tokenizer=tokenizer)
        if model_args.verifier_type == "classification" else DataCollatorForMultipleChoice(tokenizer=tokenizer),
        compute_metrics=compute_classification_metrics if model_args.verifier_type == "classification" else compute_mc_metrics,
    )

    if training_args.train_answer_verifier or training_args.eval_intensive_evidence_selector or training_args.eval_answer_verifier:
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
    if training_args.train_intensive_evidence_selector or training_args.train_answer_verifier:
        extensive_evidence_logits = {
            k: extensive_trainer.evidence_generating(v, pprepare_features_for_generating_optionwise_evidence) for k, v
            in datasets.items()}
    elif training_args.eval_intensive_evidence_selector or training_args.eval_answer_verifier:
        extensive_evidence_logits = {
            k: extensive_trainer.evidence_generating(v, pprepare_features_for_generating_optionwise_evidence)
            for k, v in datasets.items() if k != "train"}

    # prepare features for intensive evidence selector
    if training_args.train_intensive_evidence_selector or training_args.eval_intensive_evidence_selector \
            or training_args.eval_answer_verifier:
        train_intensive_evidence_selector_datasets = {k: datasets[k].map(
            partial(pprepare_features_for_intensive_evidence_selector,
                    evidence_logits=extensive_evidence_logits[k]),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        ) for k in datasets.keys() if k != "train" or training_args.train_intensive_evidence_selector}

    # prepare features for answer verifier
    if training_args.train_answer_verifier or training_args.eval_answer_verifier:
        mc_label_dict = {split: {example['example_ids']: example['label'] for example in multiple_choice_datasets[split]}
                         for split in datasets.keys() if split != "train" or training_args.train_answer_verifier}
        reader_output = {split: mc_trainer.evaluate(multiple_choice_datasets[split]) for split in datasets.keys()
                         if split != "train" or training_args.train_answer_verifier}
        answer_logits = {split: {example_id: prediction.tolist() for prediction, label_id, example_id in zip(*reader_output[split][: -1])}
                         for split in datasets.keys() if split != "train"}
        if data_args.answer_logits_path:
            logger.info(f"loading answer logits from {data_args.answer_logits_path}")
            with open(data_args.answer_logits_path) as f:
                trainset_answer_logits = json.load(f)
            answer_logits['train'] = trainset_answer_logits

        train_answer_verifier_datasets = {k: datasets[k].map(
            partial(pprepare_features_for_training_answer_verifier, answer_logits=answer_logits[k],
                    evidence_logits=extensive_evidence_logits[k]),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        ) for k in datasets.keys() if k != "train" or training_args.train_answer_verifier}
        if training_args.train_answer_verifier:
            logger.info(f"total {sum(train_answer_verifier_datasets['train']['label'])} positive example for training verifier")

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

    if training_args.train_answer_verifier:
        verifier_trainer.train_dataset = train_answer_verifier_datasets["train"]
        verifier_trainer.eval_dataset = train_answer_verifier_datasets["validation"]

        train_result = verifier_trainer.train()

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
            fulleval_results = extensive_trainer.evaluate_extensive_selector_with_explicit_reader(
                evidence_reader=evidence_reader,
                eval_dataset=datasets[split],
                feature_func_for_evidence_reading=pprepare_features_for_reading_optionwise_evidence,
                feature_func_for_evidence_generating=pprepare_features_for_generating_optionwise_evidence)

            metrics = {**results, **fulleval_results}
            output_eval_file = os.path.join(training_args.output_dir, f"{split}_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Extensive Eval results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    if training_args.eval_intensive_evidence_selector:

        for split in ["validation", "test"]:
            logger.info(f"*** Evaluate {split} set ***")
            metrics = intensive_trainer.evaluate_intensive_selector_with_explicit_reader(
                evidence_reader=evidence_reader,
                multiple_choice_dataset=multiple_choice_datasets[split],
                intensive_selector_dataset=train_intensive_evidence_selector_datasets[split])
            output_eval_file = os.path.join(training_args.output_dir, f"{split}_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Extensive Eval results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    if training_args.eval_answer_verifier:

        selector_output = {k: intensive_trainer.evaluate(train_intensive_evidence_selector_datasets[k])
                           for k in datasets.keys() if k != "train"}
        selector_logits = {k: {example_id: prediction.tolist() for prediction, label_id, example_id
                             in zip(*selector_output[k][: -1])} for k in datasets.keys() if k != "train"}

        for split in ["validation", "test"]:
            logger.info(f"*** Evaluate {split} set ***")
            results = verifier_trainer.evaluate(train_answer_verifier_datasets[split])
            verifier_logits = {example_id: prediction.tolist()
                               for prediction, label_id, example_id in zip(*results[: -1])}
            metrics = results.metrics

            if model_args.verifier_type == "classification":
                if split == 'validation':
                    fulleval_metrics = evaluate_verifier_with_reader_and_iselector(
                        reader_logits=answer_logits[split],
                        selector_logits=selector_logits[split],
                        verifier_logits=verifier_logits,
                        label_dict=mc_label_dict[split])
                    val_verify_thresholds = {k: v for k, v in fulleval_metrics.items() if "thresh" in k}
                else:
                    fulleval_metrics = evaluate_verifier_with_reader_and_iselector(
                        reader_logits=answer_logits[split],
                        selector_logits=selector_logits[split],
                        verifier_logits=verifier_logits,
                        label_dict=mc_label_dict[split],
                        threshold=val_verify_thresholds)
            else:
                fulleval_metrics = evaluate_mc_style_verifier_with_reader_and_iselector(
                    reader_logits=answer_logits[split],
                    selector_logits=selector_logits[split],
                    verifier_logits=verifier_logits,
                    label_dict=mc_label_dict[split])

            metrics = {**metrics, **fulleval_metrics}
            output_eval_file = os.path.join(training_args.output_dir, f"{split}_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Verifier Eval results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            verifier_prediction = {'verifier_logits': verifier_logits, 'reader_logits': answer_logits[split],
                                   'selector_logits': selector_logits[split]}
            output_prediction_file = os.path.join(training_args.output_dir, f"{split}_verifier_prediction.json")
            with open(output_prediction_file, "w") as f:
                json.dump(verifier_prediction, f)


if __name__ == "__main__":
    main()
