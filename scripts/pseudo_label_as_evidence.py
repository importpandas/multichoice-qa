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

import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from functools import partial

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from utils.utils_distributed_training import is_main_process

from utils.hyperparam import hyperparam_path_for_initializing_evidence_selector
from utils.initialization import setup_root_logger
from utils.utils_race import load_pseudo_label
from data_utils.collator import DataCollatorForMultipleChoice

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    pseudo_label_path: str = field(
        metadata={"help": "Path to pseudo evidence label"}
    )
    filter_label_with_ground_truth: bool = field(
        default=True,
        metadata={"help": "Whether to use pseudo label filtered by ground truth"},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file"}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "the local path of input data"})
    dataset: str = field(
        default='race',
        metadata={"help": "name of the used dataset, race or dream. Default: race."}
    )
    dataload_script: Optional[str] = field(
        default=None,
        metadata={"help": "path to the dataset processing script with the dataset builder. Can be either:a local path "
                          "to processing script or the directory containing the script (if the script has the same "
                          "name as the directory),e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'"}
    )
    dataload_split: Optional[str] = field(
        default=None,
        metadata={"help": "the type (or say 'category') needs to be loaded. For 'race' dataset, it can be chosen from "
                          "'middle', 'high' or 'all'. For 'dream' dataset, it should be 'plain_text'. May be more "
                          "dataset will be included."}
    )
    eval_dataset: Optional[str] = field(
        default="all",
        metadata={"help": "the eval dataset,'dev', 'test' or 'all' (means both 'dev' and 'test'). default: all"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    run_pseudo_label_with_options: bool = field(
        default=False, metadata={"help": "Whether run the pseudo label with options tag"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_evidence_len: int = field(
        default=6,
        metadata={"help": "The maximum length of input evidence sentences"},
    )
    max_qa_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after WordPiece tokenization. "
                    "Sequences longer than this will be truncated, and sequences shorter "
                    "than this will be padded."
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    checkpoint_dir = hyperparam_path_for_initializing_evidence_selector(model_args, data_args, training_args)
    ckpt_dir = Path(checkpoint_dir)
    postfix = ""
    if training_args.do_train:
        postfix += "_train"
    elif training_args.do_eval:
        postfix += "_eval"
    setup_root_logger(ckpt_dir, training_args.local_rank, debug=False, postfix=postfix)

    training_args.output_dir = checkpoint_dir

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the [datasets]: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    if data_args.dataset not in ['race', 'dream']:
        raise ValueError("Dataset should be race or dream.")
    else:
        if data_args.dataset == 'race':
            from utils.utils_race import prepare_features_for_reading_evidence
        if data_args.dataset == 'dream':
            pass

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    data_files = {}
    data_files['train'] = data_args.train_file if data_args.train_file is not None else None
    data_files['validation'] = data_args.validation_file if data_args.validation_file is not None else None
    data_files['test'] = data_args.test_file if data_args.test_file is not None else None

    datasets = load_dataset(data_args.dataload_script, data_args.dataload_split,
                            data_files=data_files if data_files['train'] is not None else None,
                            data_dir=data_args.data_dir)

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    all_pseudo_label = load_pseudo_label(data_args.pseudo_label_path)

    if data_args.run_pseudo_label_with_options:
        pseudo_logit = all_pseudo_label['options_prob_diff']
    else:
        pseudo_logit = all_pseudo_label['logit']
    acc = all_pseudo_label['acc']

    # Data collator
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    eval_on_dev = (data_args.eval_dataset == "all" or data_args.eval_dataset == "dev") and training_args.do_eval
    eval_on_test = (data_args.eval_dataset == "all" or data_args.eval_dataset == "test") and training_args.do_eval

    train_results = {}
    eval_results = {}
    test_results = {}
    for evidence_num in range(1, data_args.max_evidence_len):

        pprepare_features_for_using_pseudo_label_as_evidence = partial(prepare_features_for_reading_evidence,
                                                                       run_pseudo_label_with_options=data_args.run_pseudo_label_with_options,
                                                                       evidence_logits=pseudo_logit,
                                                                       evidence_len=evidence_num,
                                                                       tokenizer=tokenizer, data_args=data_args)
        tokenized_datasets = datasets.map(
            pprepare_features_for_using_pseudo_label_as_evidence,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
            eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        if training_args.do_train:
            train_result = trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
            trainer.save_model()  # Saves the tokenizer too for easy upload

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, f"evidence_{evidence_num}_trainer_state.json"))
            for key in list(train_result.metric.keys()):
                train_results[f'evidence{evidence_num}_{key}'] = train_result.metric[key]

        if eval_on_dev:
            logger.info("*** Evaluate ***")
            results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
            for key in list(results.keys()):
                eval_results[f'evidence{evidence_num}_{key}'] = results[key]

        if eval_on_test:
            logger.info("*** Test ***")
            results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
            for key in list(results.keys()):
                test_results[f'evidence{evidence_num}_{key}'] = results[key]

    if eval_on_dev:
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(eval_results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    if eval_on_test:
        output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results *****")
            for key, value in sorted(test_results.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    if training_args.do_train:
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_results.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
