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
import json

import numpy as np
from datasets import load_dataset, ReadInstruction, Dataset, concatenate_datasets
from functools import partial
from pathlib import Path
from objprint import add_objprint
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from ..model.auto_model import AutoModelForMultipleChoice

from utils.utils_distributed_training import is_main_process
from utils.hyperparam import hyperparam_path_for_baseline
from utils.initialization import setup_root_logger

from ..data_utils.collator import DataCollatorForMultipleChoice
from ..cli.argument import BasicModelArguments, BasicDataTrainingArguments
from ..trainer.trainer import Trainer

logger = logging.getLogger(__name__)


@add_objprint(color=False)
@dataclass
class DataTrainingArguments(BasicDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    output_prediction_file: bool = field(
        default=True, metadata={"help": "whether to output model prediction"}
    )
    evidence_file_path: str = field(
        default="",
        metadata={"help": "Path to pseudo evidence label"}
    )
    max_evidence_seq_length: int = field(
        default=200,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )


def evidence_reading(evidence_file_path):
    evidence_sentences = {}
    with open(evidence_file_path, "r") as f:
        evidence_file = json.load(f)
        for k, data in evidence_file.items():
            example_id = k.split("/")[-3] + k.split("/")[-2].replace(".txt", "_") + k[-1]
            for i, sentence in enumerate(data['sentences_chosen']):
                evidence_sentences[example_id + '_' + str(i)] = sentence
    return evidence_sentences


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((BasicModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    checkpoint_dir = hyperparam_path_for_baseline(model_args, data_args, training_args)
    ckpt_dir = Path(checkpoint_dir)
    postfix = ""
    if training_args.do_train:
        postfix += "_train"
    if training_args.do_eval:
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    if data_args.dataset not in ['race', 'dream']:
        raise ValueError("Dataset should be race or dream.")
    else:
        from mcmrc.data_utils.processors import prepare_features_for_evaluating_evidence

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    data_files = {'train': data_args.train_file if data_args.train_file is not None else None,
                  'validation': data_args.validation_file if data_args.validation_file is not None else None,
                  'test': data_args.test_file if data_args.test_file is not None else None}

    # datasets = load_dataset(data_args.dataload_script, data_args.dataload_split,
    #                         data_files=data_files if data_files['train'] is not None else None,
    #                         data_dir=data_args.data_dir)
    datasets = load_dataset(data_args.dataload_script, data_args.dataload_split,
                            data_files=data_files if data_files['train'] is not None else None,
                            data_dir=data_args.data_dir,
                            split={'train': ReadInstruction('train', from_=0, to=5, unit='abs'),
                                   'validation': ReadInstruction('validation', from_=0, to=100, unit='abs'),
                                   'test': ReadInstruction('test', from_=0, to=5, unit='abs')})

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
    evidence_reader = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    evidence_sentences = evidence_reading(data_args.evidence_file_path)

    pprepare_features = partial(prepare_features_for_evaluating_evidence, evidence_sentences=evidence_sentences,
                                tokenizer=tokenizer, data_args=data_args)

    # Data collator
    data_collator = (
        default_data_collator if data_args.pad_to_max_length else DataCollatorForMultipleChoice(tokenizer=tokenizer)
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    trainer = Trainer(
        model=evidence_reader,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    for split in [k for k in datasets.keys() if k != "train"]:
        if (split == "validation" and not training_args.do_eval) or (split == "test" and not training_args.do_predict):
            continue
        logger.info(f"*** Evaluate {split} set ***")
        output, _ = trainer.evidence_reading(evidence_reader, datasets[split], pprepare_features,
                                             metric_key_prefix=f'fulleval1')

        output_eval_file = os.path.join(training_args.output_dir, f"evidence_{split}_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Evidence Eval results *****")
            for key, value in sorted(output.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")


if __name__ == "__main__":
    main()
