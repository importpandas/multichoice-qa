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

import os
import sys
import logging
from pathlib import Path
import numpy as np
from datasets import load_dataset
from functools import partial

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from ..model.evidence_selector import AlbertForEvidenceSelection

from utils.hyperparam import hyperparam_path_for_initializing_evidence_selector
from utils.initialization import setup_root_logger
from utils.utils_distributed_training import is_main_process

from ..trainer.trainer import Trainer
from ..data_utils.collator import *
from ..cli.argument import BasicModelArguments, BasicDataTrainingArguments

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments(BasicModelArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    evidence_reader_path: str = field(
        metadata={"help": "Path to pretrained MRC system 2 for answering questions using evidence"}
    )
    sentence_pooling_type: Optional[str] = field(
        default="average",
        metadata={"help": "Pooling type to generate sentence vectors, either 'average' or 'max'"}
    )
    evidence_selector_type: Optional[str] = field(
        default="simple",
        metadata={"help": "Model type of evidence selector, 'simple' or 'complex'"}
    )


@dataclass
class DataTrainingArguments(BasicDataTrainingArguments):
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
    evidence_len: int = field(
        default=2,
        metadata={
            "help":     "number of sentences of each evidence"
        },
    )


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
            from mcmrc.data_utils.processors import prepare_features_for_initializing_complex_evidence_selector, \
                prepare_features_for_generating_evidence_using_selector, prepare_features_for_reading_evidence
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

    datasets = load_dataset(data_args.dataload_script, data_args.dataload_split, data_files=data_files if data_files['train'] is not None else None,
                            data_dir=data_args.data_dir)

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    config.pooling_type = model_args.sentence_pooling_type
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    evidence_selector = AlbertForEvidenceSelection.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    evidence_reader = AutoModelForMultipleChoice.from_pretrained(
        model_args.evidence_reader_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )


    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    pprepare_features_for_initializing_evidence_selector = partial(prepare_features_for_initializing_complex_evidence_selector,
                                    tokenizer=tokenizer, data_args=data_args, pseudo_label_path=data_args.pseudo_label_path)


    initializing_evidence_selector_datasets = datasets.map(
        pprepare_features_for_initializing_evidence_selector,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    pprepare_features_for_generating_evidence_using_selector = partial(prepare_features_for_generating_evidence_using_selector,
                                tokenizer=tokenizer, data_args=data_args)

    pprepare_features_for_reading_evidence = partial(prepare_features_for_reading_evidence, pseudo_label_or_not=False, tokenizer=tokenizer, data_args=data_args)




    # Data collator
    data_collator = DataCollatorForInitializingComplexEvidenceSelector(tokenizer=tokenizer)


    # Initialize our Trainer
    trainer = Trainer(
        model=evidence_selector,
        args=training_args,
        train_dataset=initializing_evidence_selector_datasets["train"] if training_args.do_train else None,
        eval_dataset=initializing_evidence_selector_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    if training_args.do_train:
        train_result = trainer.train()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

    # Evaluation
    # To use the best checkpoint model at end, use the aruguments
    # load_best_model_at_end, metric_for_best_model, evaluation_strategy steps
    # --load_best_model_at_end \
    # --metric_for_best_model accuracy \
    # --evaluation_strategy steps \
    eval_on_dev = (data_args.eval_dataset == "all" or data_args.eval_dataset == "dev") and training_args.do_eval
    eval_on_test = (data_args.eval_dataset == "all" or data_args.eval_dataset == "test") and training_args.do_eval

    if eval_on_dev:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate(initializing_evidence_selector_datasets["validation"]).metrics
        fulleval_results = trainer.evaluate_with_explicit_reader(evidence_reader, datasets["validation"], pprepare_features_for_reading_evidence,
                                                                 initializing_evidence_selector_datasets["validation"],
                                                                 evidence_generating_data_collator=data_collator)

        metrics = {**results, **fulleval_results}
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in sorted(metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
    if eval_on_test:
        logger.info("*** Test ***")

        results = trainer.evaluate(initializing_evidence_selector_datasets["test"]).metrics
        fulleval_results = trainer.evaluate_with_explicit_reader(evidence_reader, datasets["test"], pprepare_features_for_reading_evidence,
                                                                 initializing_evidence_selector_datasets["test"],
                                                                 evidence_generating_data_collator=data_collator)

        metrics = {**results, **fulleval_results}
        output_test_file = os.path.join(training_args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Test results *****")
            for key, value in sorted(metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
