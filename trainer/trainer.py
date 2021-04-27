import collections
import gc
import inspect
import math
import os
import re
import shutil
import sys
import time
import warnings
import logging
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from datasets import load_dataset
from functools import partial

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.trainer import Trainer
from transformers.trainer_utils import speed_metrics

from data_utils.collator import *

logger = logging.getLogger(__name__)


class EvidenceSelectorTrainer(Trainer):

    def evidence_reading(
            self,
            evidence_reader,
            eval_dataset,
            prepare_feature_func,
            evidence_logits,
    ):
        evidence_reader = evidence_reader.to(self.args.device)
        evidence_reader = self._wrap_model(evidence_reader, training=False)

        evidence_reading_data_collator = DataCollatorForMultipleChoice(tokenizer=self.tokenizer)

        column_names = eval_dataset.column_names

        pprepare_feature_func = partial(prepare_feature_func, evidence_logits=evidence_logits)
        processed_datasets = eval_dataset.map(
            pprepare_feature_func,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

        eval_sampler = SequentialSampler(processed_datasets)
        evidence_reading_dataloader = DataLoader(
            processed_datasets,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=evidence_reading_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        start_time = time.time()
        evidence_generator = self.model
        self.model = evidence_reader
        output = self.prediction_loop(
            evidence_reading_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=None,
            metric_key_prefix="fulleval",
        )
        self.model = evidence_generator

        n_samples = len(processed_datasets)
        output.metrics.update(speed_metrics("fulleval", start_time, n_samples))
        self.log(output.metrics)
        return output
        #
        # batch_size = evidence_reading_dataloader.batch_size
        # num_examples = self.num_examples(evidence_reading_dataloader)
        # logger.info(f"***** Running evidence evaluating *****")
        # logger.info(f"  Num examples = {num_examples}")
        # logger.info(f"  Batch size = {batch_size}")
        #
        # for step, batch in enumerate(evidence_reading_dataloader):
        #     with torch.no_grad():
        #         inputs = {
        #             "input_ids": batch['input_ids'].to(evidence_evaluator.device),
        #             "attention_mask": batch['attention_mask'].to(evidence_evaluator.device),
        #             "token_type_ids": batch['token_type_ids'].to(evidence_evaluator.device),
        #         }
        #         logits = evidence_evaluator(**inputs).logits.detach().cpu()

    def evaluate_with_explicit_reader(
            self,
            evidence_reader,
            eval_dataset,
            prepare_feature_func,
            evidence_generating_dataset,
            ignore_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :param ignore_keys:
                :param evidence_generating_dataset:
                :param prepare_feature_func:
                :param eval_dataset:
                :param evidence_reader:
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        evidence_generating_data_collator = DataCollatorForGeneratingEvidenceUsingSelector(tokenizer=self.tokenizer)

        if evidence_generating_dataset is not None and not isinstance(evidence_generating_dataset,
                                                                      collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        evidence_selector = self._wrap_model(self.model, training=False)

        eval_sampler = SequentialSampler(evidence_generating_dataset)

        evidence_generating_dataloader = DataLoader(
            evidence_generating_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=evidence_generating_data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        # start_time = time.time()

        evidence_logits = {}
        for step, batch in enumerate(evidence_generating_dataloader):
            with torch.no_grad():
                inputs = {
                    "input_ids": batch['input_ids'].to(self.args.device),
                    "attention_mask": batch['attention_mask'].to(self.args.device),
                    "token_type_ids": batch['token_type_ids'].to(self.args.device),
                }
                logits = evidence_selector(**inputs).logits.detach().cpu()
            example_ids = batch['example_ids']
            sent_idxs = batch['sent_idx']

            for i, (example_id, sent_idx) in enumerate(zip(example_ids, sent_idxs)):
                if example_id not in evidence_logits.keys():
                    evidence_logits[example_id] = {}
                evidence_logits[example_id][sent_idx] = logits[i][1].item()

        output = self.evidence_reading(evidence_reader, eval_dataset,  prepare_feature_func, evidence_logits)

        return output.metrics
        # return output.metrics
