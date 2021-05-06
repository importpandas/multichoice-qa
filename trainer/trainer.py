import os, re
import logging
import math
from functools import partial
import timeit
import collections
import time

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, \
    get_polynomial_decay_schedule_with_warmup

from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from trainer.common import Timer
from data_utils.collator import DataCollatorForMultipleChoice, DataCollatorForGeneratingEvidenceUsingSelector

from trainer.checkpoint import save_checkpoint

from utils.param import iter_parameters_of_optimizer

from packaging import version
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify
)
from transformers.trainer_utils import EvalPrediction, denumpify_detensorize, PredictionOutput, speed_metrics, TrainOutput

_is_native_amp_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, args, model, tokenizer, train_dataset, eval_dataset,
                 data_collator=None, compute_metrics=None, rank=0, world_size=1):

        self.args = args
        self.model_name_or_path = ""
        self.world_size = world_size  # torch.distributed3.get_world_size()
        self.checkpoint_dir = args.output_dir

        self.label_names = ["labels"]

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)

        self.data_collator = data_collator if data_collator is not None else default_collator

        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None

        self.compute_metrics = compute_metrics

        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.num_train_epochs = int(args.num_train_epochs)

        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps
        self.eval_steps = args.eval_steps

        self.weight_decay = args.weight_decay
        self.learning_rate = args.learning_rate
        self.layerwise_lr_decay = 1
        self.adam_epsilon = args.adam_epsilon
        self.max_grad_norm = args.max_grad_norm

        self.device = args.device
        model = model.to(args.device)

        self.model = model

        self.tokenizer = tokenizer
        self.optimizer = self._build_optimizer()
        self.rank = rank
        self.timer = Timer(builtin_keys=('wall', 'io', 'gpu', 'merge', 'cv'))

        self.evaluate_during_training = True

        self.output_hidden_states = None
        self.output_attentions = None

    def _build_optimizer(self):
        # Local optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        large_lr = ['classifier']
        optimizer_grouped_parameters = []
        for n, p in self.model.named_parameters():
            params_group = {}
            params_group['params'] = p
            params_group['weight_decay'] = 0.0 if any(nd in n for nd in no_decay) or any(
                ll in n for ll in large_lr) else self.weight_decay
            if any(ll in n for ll in large_lr):
                params_group['lr'] = 1e-3
            else:
                if 'electra.embedding' in n or 'bert.embedding' in n:
                    depth = 0
                elif 'electra.encoder.layer' in n:  # electra
                    depth = int(re.search(r"electra.encoder.layer.(\d+)", n).group(1)) + 1
                elif 'bert.encoder.layer' in n:  # bert, roberta-wwm-ext
                    depth = int(re.search(r"bert.encoder.layer.(\d+)", n).group(1)) + 1
                else:
                    depth = self.model.config.num_hidden_layers
                params_group['lr'] = self.learning_rate * \
                                     (self.layerwise_lr_decay ** (self.model.config.num_hidden_layers - depth))
            optimizer_grouped_parameters.append(params_group)

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)

        return optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _prepare_inputs(self, inputs):
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        return inputs

    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys=None,
    ):
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss.detach()

    def _wrap_model(self, model):
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        return model

    def train(self):
        train_sampler = RandomSampler(self.train_dataset) if self.world_size == 1 else DistributedSampler(
            self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset,
                                      sampler=train_sampler,
                                      batch_size=self.args.train_batch_size,
                                      collate_fn=self.data_collator,
                                      num_workers=self.args.dataloader_num_workers)

        # self.t_total = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps) * self.num_train_epochs # unexpected
        self.t_total = (
                                   len(train_dataloader) // self.gradient_accumulation_steps) * self.num_train_epochs  # last of each epoch will be dropped
        self.warmup_steps = int(self.t_total * 0.1)

        # Prepare optimizer and schedule (linear warmup and decay)

        # params = list(self.model.named_parameters())
        # all_trainable_params = divide_parameters(params, lr=self.learning_rate)
        # optimizer = BERTAdam(all_trainable_params, lr=self.learning_rate, warmup=0.1, t_total=self.t_total, schedule='slanted_triangular', s_opt1=1.0, s_opt2=0.0, s_opt3=1.0)

        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=self.warmup_steps, num_training_steps=self.t_total)

        optimizer_and_scheduler_states_exist = False
        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(self.model_name_or_path, "scheduler.pt")):
            optimizer_and_scheduler_states_exist = True
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(torch.load(os.path.join(self.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.model_name_or_path, "scheduler.pt")))

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.gradient_accumulation_steps
            * self.world_size,
        )
        logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.t_total)

        global_step = 1
        epochs_trained = 0
        # Check if continuing training from a checkpoint

        tr_loss, logging_loss = 0.0, 0.0
        best_eval_acc = 0.0

        model = self._wrap_model(self.model)
        metrics = {}

        self.optimizer.zero_grad()

        # Profiling timers
        self.timer.clear()
        self.timer['wall'].start()
        for epoch_idx in range(epochs_trained, self.num_train_epochs):
            self.optimizer.zero_grad()  # drop last
            for step, batch in enumerate(train_dataloader):
                # Skip past any already trained steps if resuming training

                with self.timer['gpu']:
                    loss = self.training_step(model, batch)  # backward in local machine
                tr_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Unscales the gradients of optimizer's assigned params in-place
                    # unscale_ should only be called once per optimizer per step call,
                    # Since the gradients of optimizer's assigned params are now unscaled, clips as usual.
                    # You may use the same value for max_norm here as you would without gradient scaling.
                    torch.nn.utils.clip_grad_norm_(iter_parameters_of_optimizer(self.optimizer), self.max_grad_norm)

                    with self.timer['merge']:
                        # grad sync
                        self.optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Log metrics of evalution
                    if self.eval_steps > 0 and global_step % self.eval_steps == 0:
                        # Do not waste multiple GPUs when self.distributed_eval_data_split is False
                        if self.evaluate_during_training and self.world_size == 1:
                            with self.timer['cv']:
                                results = self.evaluate(self.eval_dataset).metrics
                            for key, value in results.items():
                                logger.info(f'step{global_step} eval_{key}: {value}')
                                metrics[f'step{global_step} eval_{key}'] = value

                            _current_eval_acc = max([value for key, value in results.items() if 'acc' in key])
                            if best_eval_acc < _current_eval_acc:
                                best_eval_acc = _current_eval_acc
                                if self.rank == 0:
                                    # Take care of distributed/parallel training
                                    save_checkpoint(self.checkpoint_dir, model, self.tokenizer)
                                    # torch.save(args, os.path.join(self.checkpoint_dir, "training_args.bin"))
                                    logger.info(
                                        f"Saving model checkpoint with currently best validation acc {best_eval_acc:.5f}"
                                        f" to {self.checkpoint_dir}")

                    # Save model checkpoint
                    if self.rank == 0 and self.save_steps > 0 and global_step % self.save_steps == 0:
                        output_dir = os.path.join(self.checkpoint_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        save_checkpoint(output_dir, model, self.tokenizer)
                        # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        # remove previous optimizer.pt to save disk
                        previous_saved_optimizer_path = os.path.join(self.checkpoint_dir, "checkpoint-{}".format(
                            global_step - self.save_steps), "optimizer.pt")
                        if os.path.exists(previous_saved_optimizer_path):
                            os.remove(previous_saved_optimizer_path)
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)
                # the last interval will be retained
                self.timer['wall'].checkpoint()
                self.timer['wall'].start()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Log metrics
                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        logger.info(f'step{global_step} lr: {scheduler.get_lr()[0]}')
                        logger.info(f'step{global_step} loss: {(tr_loss - logging_loss) / self.logging_steps}\n\n')
                        logging_loss = tr_loss
                        logger.info(self.timer.last_result)
            logger.info(self.timer.total_result)

        if self.rank == 0:
            # Take care of distributed/parallel training
            output_dir = os.path.join(self.checkpoint_dir, "checkpoint-end")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_checkpoint(output_dir, model, self.tokenizer)
            logger.info("Saving model checkpoint at the end to %s", output_dir)

        return TrainOutput(global_step, tr_loss / global_step, metrics)



    def evaluate(self, dataset, data_collator=None, description="", metric_key_prefix="eval"):
        # predicition with single device

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset,
                                     sampler=eval_sampler,
                                     batch_size=self.args.eval_batch_size,
                                     collate_fn=self.data_collator if data_collator is None else data_collator,
                                     num_workers=self.args.dataloader_num_workers)

        batch_size = eval_dataloader.batch_size
        num_examples = len(eval_dataloader.dataset)
        logger.info("***** Running evaluation {} *****".format(description))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, self.args.world_size)
        prediction_loss_only = True if self.compute_metrics is None else None

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(eval_dataloader, "sampler") and isinstance(eval_dataloader.sampler,
                                                                  SequentialDistributedSampler):
                make_multiple_of = eval_dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model = self._wrap_model(self.model)
        model.eval()

        start_time = timeit.default_timer()
        for step, inputs in enumerate(eval_dataloader):
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only)

            if loss is not None:
                losses = loss.repeat(eval_dataloader.batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(nested_numpify(losses_host))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(nested_numpify(preds_host))
            labels_gatherer.add_arrays(nested_numpify(labels_host))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))


        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics).metrics

    def evidence_reading(
            self,
            evidence_reader,
            eval_dataset,
            prepare_feature_func,
            metric_key_prefix="fulleval"
    ):
        evidence_reader = evidence_reader.to(self.args.device)
        evidence_reader = self._wrap_model(evidence_reader, training=False)

        evidence_reading_data_collator = DataCollatorForMultipleChoice(tokenizer=self.tokenizer)

        column_names = eval_dataset.column_names

        processed_datasets = eval_dataset.map(
            prepare_feature_func,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=False,
        )

        start_time = time.time()
        evidence_generator = self.model
        self.model = evidence_reader
        output = self.evaluate(
            processed_datasets,
            data_collator=evidence_reading_data_collator,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=None,
            metric_key_prefix=metric_key_prefix,
        )
        self.model = evidence_generator

        n_samples = len(processed_datasets)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        return output

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

        metrics = {}
        for evidence_len in [2, 3, 4]:
            pprepare_feature_func = partial(prepare_feature_func, evidence_len=evidence_len, evidence_logits=evidence_logits)
            output = self.evidence_reading(evidence_reader, eval_dataset,  pprepare_feature_func, metric_key_prefix=f'fulleval{evidence_len}')
            metrics = {**metrics, **output.metrics}

        return metrics
        # return output.metrics

