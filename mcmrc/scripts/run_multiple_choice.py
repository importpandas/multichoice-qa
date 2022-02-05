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
    AutoModelForMultipleChoice,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

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
    debug_mode:  bool = field(
        default=False, metadata={"help": "whether to load a subset of data for debug"}
    )
    split_train_dataset:  bool = field(
        default=False, metadata={"help": "whether to split training dataset "
                                         "like cross validation for evidence generating"}
    )
    n_fold: int = field(
        default=5,
        metadata={"help": "split fold num of training dataset"},
    )
    holdout_set: int = field(
        default=0,
        metadata={"help": "split fold num of training dataset"},
    )
    output_prediction_file:  bool = field(
        default=True, metadata={"help": "whether to output model prediction"}
    )
    evidence_logits_path: str = field(
        default="",
        metadata={"help": "Path to evidence label"}
    )


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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
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
    if not 0 <= data_args.holdout_set < data_args.n_fold:
        raise ValueError("Test fold must be in [0, n_fold)")

    if data_args.dataset not in ['race', 'dream']:
        raise ValueError("Dataset should be race or dream.")
    else:
        from mcmrc.data_utils.processors import prepare_features

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if data_args.debug_mode:
        datasets = load_dataset(data_args.dataload_script, data_args.dataload_split,
                                data_dir=data_args.data_dir,
                                split={'train': ReadInstruction('train', from_=0, to=5, unit='abs'),
                                       'validation': ReadInstruction('validation', from_=0, to=5, unit='abs'),
                                       'test': ReadInstruction('test', from_=0, to=5, unit='abs')})
    else:
        datasets = load_dataset(data_args.dataload_script, data_args.dataload_split,
                                data_dir=data_args.data_dir)

    if data_args.split_train_dataset:
        holdout_set_start = int(len(datasets['train']) / data_args.n_fold * data_args.holdout_set)
        holdout_set_end = int(len(datasets['train']) / data_args.n_fold * (data_args.holdout_set + 1))
        shuffled_train_set = datasets['train'].shuffle(seed=training_args.seed)
        if holdout_set_start == 0:
            new_train_set = Dataset.from_dict(shuffled_train_set[holdout_set_end:])
        elif holdout_set_end == len(datasets['train']):
            new_train_set = Dataset.from_dict(shuffled_train_set[:holdout_set_start])
        else:
            new_train_set = concatenate_datasets([Dataset.from_dict(shuffled_train_set[:holdout_set_start]),
                                                Dataset.from_dict(shuffled_train_set[holdout_set_end:])])

        new_holdout_set = Dataset.from_dict(shuffled_train_set[holdout_set_start: holdout_set_end])
        assert new_train_set.num_rows + new_holdout_set.num_rows == shuffled_train_set.num_rows
        datasets['train'] = new_train_set
        datasets['holdout_set'] = new_holdout_set


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

    pprepare_features = partial(prepare_features, tokenizer=tokenizer, data_args=data_args)
    tokenized_datasets = datasets.map(
        pprepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

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
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
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
    if training_args.do_eval:

        if training_args.load_best_model_at_end:
            best_model = AutoModelForMultipleChoice.from_pretrained(
                            training_args.output_dir,
                            from_tf=bool(".ckpt" in model_args.model_name_or_path),
                            config=config,
                            cache_dir=model_args.cache_dir,
                        )

        for split in [k for k in datasets.keys() if k != "train"]:
            logger.info(f"*** Evaluate {split} set ***")
            results = trainer.evaluate(tokenized_datasets[split])
            if training_args.load_best_model_at_end:
                final_model = trainer.model
                trainer.model = best_model
                best_model_results = trainer.evaluate(tokenized_datasets[split])
                trainer.model = final_model

            output_eval_file = os.path.join(training_args.output_dir, f"{split}_results.txt")
            with open(output_eval_file, "a+") as writer:
                logger.info("***** Extensive Eval results *****")
                if not training_args.do_train:
                    writer.write(f"eval checkpoint {model_args.model_name_or_path}\n")
                for key, value in sorted(results.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")
                if training_args.load_best_model_at_end:
                    writer.write(f"best model on dev set\n")
                    for key, value in sorted(best_model_results.metrics.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
            if data_args.output_prediction_file or data_args.split_train_dataset:
                prediction = {example_id: prediction.tolist() for prediction, label_id, example_id in zip(*results[: -1])}
                if split == "holdout_set":
                    output_prediction_file = os.path.join(training_args.output_dir, f"holdout_{data_args.n_fold}_{data_args.holdout_set}_prediction.json")
                else:
                    output_prediction_file = os.path.join(training_args.output_dir, f"{split}_prediction.json")
                with open(output_prediction_file, "w") as f:
                    json.dump(prediction, f)


if __name__ == "__main__":
    main()
