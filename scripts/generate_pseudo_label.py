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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import tqdm

from datasets import load_dataset
from functools import partial

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from utils.utils_distributed_training import is_main_process
from data_utils.collator import DataCollatorForGeneratingEvidenceLabel



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
    dataload_script: str = field(
        metadata={"help": "path to the dataset processing script with the dataset builder. Can be either:a local path "
                          "to processing script or the directory containing the script (if the script has the same "
                          "name as the directory),e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'"}
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
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_qa_length: int = field(
        default=64,
        metadata={
            "help":     "The maximum total input sequence length after WordPiece tokenization. "
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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    if data_args.dataset not in ['race', 'dream']:
        raise ValueError("Dataset should be race or dream.")
    else:
        if data_args.dataset == 'race':
            from utils.utils_race import prepare_features_for_generate_pseudo_label
        if data_args.dataset == 'dream':
            pass

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    data_files['train'] = data_args.train_file if data_args.train_file is not None else None
    data_files['validation'] = data_args.validation_file if data_args.validation_file is not None else None
    data_files['test'] = data_args.test_file if data_args.test_file is not None else None

    datasets = load_dataset(data_args.dataload_script, data_args.dataload_split, data_files=data_files if data_files['train'] is not None else None,
                            data_dir=data_args.data_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

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

    column_names = datasets["train"].column_names


    pprepare_features_for_generate_pseudo_label = partial(prepare_features_for_generate_pseudo_label, tokenizer=tokenizer, data_args=data_args)
    tokenized_datasets = datasets.map(
        pprepare_features_for_generate_pseudo_label,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    data_collator = (
        default_data_collator if data_args.pad_to_max_length else DataCollatorForGeneratingEvidenceLabel(tokenizer=tokenizer)
    )

    device = training_args.device
    model.to(device)
    model.eval()
    if training_args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    pseudo_label = {}
    acc = {}
    for train_test_or_eval, dataset in tokenized_datasets.items():
        dataloader = DataLoader(
            dataset,
            batch_size=training_args.eval_batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=data_collator,
            num_workers=0
        )

        pseudo_label_split = {}
        acc_split = {}
        print(f'{train_test_or_eval}', len(dataloader))
        for step, batch in tqdm.tqdm(enumerate(dataloader)):
            with torch.no_grad():
                origin_inputs = {
                    "input_ids": batch['input_ids'].to(device),
                    "attention_mask": batch['attention_mask'].to(device),
                    "token_type_ids": batch['token_type_ids'].to(device),
                }
                origin_logits = model(**origin_inputs).logits.detach().cpu()

            example_ids = batch['example_ids']
            sent_bounds = batch['sent_bound_token']


            for i, one_example_sent_bounds in enumerate(sent_bounds):

                if example_ids[i] not in pseudo_label_split.keys():
                    kl_div_per_example = {}
                    pseudo_label_split[example_ids[i]] = kl_div_per_example
                else:
                    kl_div_per_example = pseudo_label_split[example_ids[i]]

                one_example_logit = origin_logits[i]
                one_example_sent_bounds = torch.tensor(one_example_sent_bounds, device=device)
                one_example_attention_mask = batch['attention_mask'][i]
                one_example_input_ids = batch['input_ids'][i]
                one_example_token_type_ids = batch['token_type_ids'][i]
                one_example_label = batch['labels'][i]
                sent_num = one_example_sent_bounds.size()[0]


                for j in range(0, sent_num, training_args.eval_batch_size):
                    batch_start = j
                    batch_end = j + training_args.eval_batch_size if j < sent_num - training_args.eval_batch_size else sent_num
                    batched_sent_bound = torch.stack((one_example_sent_bounds[batch_start: batch_end, 1],
                                                      one_example_sent_bounds[batch_start: batch_end, 2])).unsqueeze(1).permute(2, 1, 0)

                    batched_attention_mask = one_example_attention_mask.unsqueeze(0).expand(batch_end - batch_start, -1,
                                                                                            -1).clone().to(device)

                    pos_matrix = torch.arange(batched_attention_mask.size()[-1], device=device).view(1, 1, -1)
                    if_in_sent = torch.logical_and(batched_sent_bound[:,:,0].unsqueeze(-1) <= pos_matrix,
                                                   pos_matrix <= batched_sent_bound[:,:,1].unsqueeze(-1))

                    batched_attention_mask = torch.where(if_in_sent, torch.tensor((0), device=device), batched_attention_mask)
                    batched_input_ids = one_example_input_ids.expand(batch_end - batch_start, -1, -1).contiguous()
                    batched_token_type_ids = one_example_token_type_ids.expand(batch_end - batch_start, -1, -1).contiguous()

                    with torch.no_grad():
                        masked_inputs = {
                            "input_ids": batched_input_ids.to(device),
                            "attention_mask": batched_attention_mask.to(device),
                            "token_type_ids": batched_token_type_ids.to(device),
                        }
                        masked_logits = model(**masked_inputs).logits.detach().cpu()
                        kl_divs = torch.sum(F.kl_div(F.log_softmax(masked_logits, dim=-1), F.softmax(one_example_logit, dim=-1), reduction='none'), dim=-1)

                    for k, kl_div in enumerate(kl_divs.detach().cpu().tolist()):
                        sent_idx = one_example_sent_bounds[batch_start + k, 0].item()
                        evidence_or_noise = 1 if F.softmax(masked_logits[k])[one_example_label].item() < F.softmax(one_example_logit)[one_example_label].item() else -1
                        if sent_idx in kl_div_per_example.keys():
                            kl_div_per_example[sent_idx] = (evidence_or_noise * kl_div) if kl_div > abs(kl_div_per_example[sent_idx])\
                                                                                    else kl_div_per_example[sent_idx]
                        else:
                            kl_div_per_example[sent_idx] = evidence_or_noise * kl_div

                acc_split[example_ids[i]] = 1 if torch.argmax(one_example_logit).item() == one_example_label.item() else 0

        pseudo_label[train_test_or_eval] = pseudo_label_split
        acc[train_test_or_eval] = acc_split

    label = {
        'pseudo_label': pseudo_label,
        'acc': acc
    }
    torch.save(label, data_args.dataset + f"_pseudo_label_{config.model_type}_{config.hidden_size}.pt")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
