import os
import math
import shutil
import random
import logging
import warnings
from typing import Literal, Any, cast
from collections.abc import Mapping

import torch
import datasets
import transformers
import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as metrics

from tap import Tap
from scipy.optimize import linear_sum_assignment
from transformers import (
    MODEL_MAPPING,  # type: ignore
    DataCollatorForLanguageModeling,  # type: ignore
    DataCollatorForWholeWordMask,
    SchedulerType,  # type: ignore
    get_scheduler,  # type: ignore
)
from transformers.models.distilbert import (
    DistilBertConfig,
    DistilBertTokenizerFast,
    DistilBertForMaskedLM
)
from transformers.utils import send_example_telemetry  # type: ignore
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, DatasetDict, Features, Value
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = tuple(MODEL_MAPPING)
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class Args(Tap):
    """Finetune a transformers model on a Masked Language Modeling task."""
    dataset_name: str | None = None
    """The name of the dataset to use (via the datasets library)."""
    n_classes: int = 4
    """Number of classes/clusters."""
    dataset_config_name: str | None = None
    """The configuration name of the dataset to use (via the datasets library)."""
    train_file: str
    """A csv or a json file containing the training data."""
    val_file: str
    """A csv or a json file containing the validation data."""
    validation_split_percentage: float | None = None
    """The percentage of the train set used as validation set in case there's no validation split."""
    pad_to_maxlen: bool = False
    """If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used."""
    model_name_or_path: str
    """Path to pretrained model or model identifier from huggingface.co/models."""
    config_name: str | None = None
    """Pretrained config name or path if not the same as model name."""
    tokenizer_name: str | None = None
    """Pretrained tokenizer name or path if not the same as model name."""
    per_device_train_batch_size: int = 8
    """Batch size (per device) for the training dataloader."""
    per_device_eval_batch_size: int = 8
    """Batch size (per device) for the evaluation dataloader."""
    lr: float = 5e-5
    """Initial learning rate (after the potential warmup period) to use."""
    weight_decay: float = 0.0
    """Weight decay to use."""
    num_train_epochs: int = 3
    """Total number of training epochs to perform."""
    max_train_steps: int | None = None
    """Total number of training steps to perform. If provided, overrides num_train_epochs."""
    gradient_accumulation_steps: int = 1
    """Number of updates steps to accumulate before performing a backward/update pass."""
    lr_scheduler_type: SchedulerType = SchedulerType.LINEAR
    """The scheduler type to use."""
    num_warmup_steps: int = 0
    """Number of steps for the warmup in the lr scheduler."""
    output_dir: str
    """Where to store the final model."""
    seed: int | None = 2025
    """Random seed for reproducible training."""
    model_type: Literal[MODEL_TYPES] | None = None  # type: ignore
    """Model type to use if training from scratch."""
    max_seq_length: int = 32
    """The maximum total input sequence length after tokenization. 
    Sequences longer than this will be truncated."""
    num_workers: int | None = None
    """The number of processes to use for the preprocessing."""
    overwrite_cache: bool = False
    """Overwrite the cached training and evaluation sets."""
    mlm_probability: float = 0.15
    """Ratio of tokens to mask for masked language modeling loss."""
    wwm: bool = False
    trust_remote_code: bool = False
    """Whether or not to allow for custom models defined on the Hub in their 
    own modeling files. This option should only be set to `True` 
    for repositories you trust and in which you have read the code, 
    as it will execute code present on the Hub on your local machine."""
    checkpointing_epochs: int = 200
    """Whether the various states should be saved at the end of every n steps, 
    or non-positive number for each epoch."""
    resume_from_checkpoint: str | None = None
    """If the training should continue from a checkpoint folder."""
    cluster_evaluating_epochs: int = 50
    with_tracking: bool = False
    """Whether to enable experiment trackers for logging."""
    report_to: Literal['all', 'tensorboard', 'wandb', 'comet_ml', 'clearml'] = 'all'
    """The integration to report the results and logs to. Supported platforms 
    are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. 
    Use `"all"` (default) to report to all integrations. 
    Only applicable when `--with_tracking` is passed."""
    low_cpu_mem_usage: bool = False
    """It is an option to create the model as an empty shell, then only 
    materialize its parameters when the pretrained weights are loaded. 
    If passed, LLM loading time and RAM consumption will be benefited."""
    early_stopping_loss_thresh: float = 0.0
    save_limit: int | None = None

    def configure(self) -> None:
        self.add_argument(
            '--lr_scheduler_type',
            type=SchedulerType,
            default='linear',
            choices=[
                'linear', 'cosine', 'cosine_with_restarts',
                'polynomial', 'constant', 'constant_with_warmup'
            ]
        )
    
    def process_args(self) -> None:
        if self.dataset_name is None and self.train_file is None and self.val_file is None:
            raise ValueError('Need either a dataset name or a training/validation file.')
        
        valid_extensions = ('.csv', '.json', '.txt')

        if self.train_file is not None and not self.train_file.endswith(valid_extensions):
            raise ValueError('training file should be a csv, json or txt file.')
        
        if self.val_file is not None and not self.val_file.endswith(valid_extensions):
            raise ValueError('validation file should be a csv, json or txt file.')

        if self.wwm:
            self.step_prefix = 'wwm_step_'
            self.epoch_prefix = 'wwm_epoch_'
        else:
            self.step_prefix = 'step_'
            self.epoch_prefix = 'epoch_'


def get_raw_datasets(args: Args) -> DatasetDict:
    """
    Get the datasets: you can either provide your own CSV/JSON/TXT
    training and evaluation files (see below) or just provide the name of
    one of the public datasets available on the hub at
    https://huggingface.co/datasets/
    (the dataset will be downloaded automatically from the datasets Hub).

    For CSV/JSON files, this script will use the column called 'text' or
    the first column if no column called 'text' is found.
    You can easily tweak this behavior (see below).
    
    In distributed training, the load_dataset function guarantee that
    only one local process can concurrently download the dataset.
    """
    features = dict(label='int64', text='string', text1='string', text2='string')
    features = Features({k: Value(v) for k, v in features.items()})
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        assert isinstance(raw_datasets, DatasetDict)
        if 'validation' not in raw_datasets:
            raw_datasets['validation'] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f'train[:{args.validation_split_percentage}%]',
                features=features
            )
            raw_datasets['train'] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f'train[{args.validation_split_percentage}%:]',
                features=features
            )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files['train'] = args.train_file
        if args.val_file is not None:
            data_files['validation'] = args.val_file
        extension = args.train_file.rsplit('.', maxsplit=1)[-1]
        if extension == 'txt':
            extension = 'text'
        raw_datasets = load_dataset(extension, data_files=data_files, features=features)
        assert isinstance(raw_datasets, DatasetDict)
        # If no validation data is there, validation_split_percentage 
        # will be used to divide the dataset.
        if 'validation' not in raw_datasets:
            raw_datasets['validation'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[:{args.validation_split_percentage}%]',
                features=features
            )
            raw_datasets['train'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[{args.validation_split_percentage}%:]',
                features=features
            )
    
    return raw_datasets


def load_tokenizer_and_model(args: Args) -> tuple[DistilBertTokenizerFast, DistilBertForMaskedLM]:
    """
    See more about loading any type of standard or custom dataset
    (from files, python dict, pandas DataFrame, etc) at
    https://huggingface.co/docs/datasets/loading_datasets.

    Load pretrained model and tokenizer.

    In distributed training, the .from_pretrained methods guarantee that
    only one local process can concurrently download model & vocab.
    """
    config_name = (
        args.model_name_or_path
        if args.config_name is None
        else args.config_name
    )
    config = DistilBertConfig.from_pretrained(
        config_name,
        trust_remote_code=args.trust_remote_code
    )

    tokenizer_name = (
        args.model_name_or_path
        if args.tokenizer_name is None
        else args.tokenizer_name
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code
    )

    model = DistilBertForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf='.ckpt' in args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )

    return tokenizer, model


def tokenize_datasets(
    raw_datasets, tokenizer, accelerator,
    column_names, text_column_name,
    max_seq_length, args: Args
):
    def tokenize(examples: Mapping[str, str]):
        # Remove empty lines
        examples[text_column_name] = [
            line
            for line in examples[text_column_name]
            if line.strip()
        ]
        return tokenizer(
            examples[text_column_name],
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling 
            # (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    with accelerator.main_process_first():
        return raw_datasets.map(
            tokenize,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc='Running tokenizer on dataset line_by_line',
        )


def resume_from_checkpoint(accelerator, train_loader, num_update_steps_per_epoch, args: Args):
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
    else:
        # Get the most recent checkpoint
        dirs = (entry for entry in os.scandir() if entry.is_dir())
        checkpoint_path = max(dirs, key=lambda d: d.stat().st_ctime).path
    
    path = os.path.basename(checkpoint_path)

    accelerator.print(f'Resumed from checkpoint: {checkpoint_path}')
    accelerator.load_state(checkpoint_path)
    # Extract `epoch_{i}` or `step_{i}`
    training_difference = os.path.splitext(path)[0]

    if training_difference.startswith(args.epoch_prefix):
        starting_epoch = int(training_difference.removeprefix(args.epoch_prefix)) + 1
        resume_step = None
        completed_steps = starting_epoch * num_update_steps_per_epoch
    else:
        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = int(training_difference.removeprefix(args.step_prefix)) * args.gradient_accumulation_steps
        starting_epoch = resume_step // len(train_loader)
        completed_steps = resume_step // args.gradient_accumulation_steps
        resume_step -= starting_epoch * len(train_loader)
    
    return completed_steps, starting_epoch, resume_step


def evaluate(
    model: DistilBertForMaskedLM,
    loader: DataLoader,
    accelerator: Accelerator,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    model.eval()
    losses = []
    embeddings = []
    for batch in loader:
        with torch.no_grad():
            outputs = model(**batch)
            batch.pop('labels')
            bert_output = model.distilbert(**batch).last_hidden_state
            attention_mask = batch['attention_mask'].unsqueeze(-1)
            embedding = (
                torch.sum(bert_output * attention_mask, dim=1)
                / torch.sum(attention_mask, dim=1)
            )
        
        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(len(batch['input_ids']))))
        embeddings.append(embedding)
    
    return losses, embeddings


def reassign_predicted_labels(y_pred, y_true):
    cm = metrics.confusion_matrix(y_true, y_pred)
    cost_matrix = cm.sum(axis=0, keepdims=True).T - cm.T
    _, assignments = linear_sum_assignment(cost_matrix)

    return assignments[y_pred]


def cluster_evaluate(embeddings, n_classes: int, random_state: int | None):
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.cat(embeddings, dim=0)

    if n_classes <= 20:
        clusterer = cluster.KMeans(
            n_clusters=n_classes,
            init='k-means++',
            n_init=10,
            random_state=random_state,
            max_iter=3000,
            tol=1e-2
        )
    else:
        clusterer = cluster.AgglomerativeClustering(
            n_clusters=n_classes,
            metric='cosine',
            linkage='average'
        )
    
    y_pred = clusterer.fit_predict(embeddings.numpy(force=True))

    y_true = pd.read_csv(args.val_file)['label'].to_numpy()
    if min_val := y_true.min():
        y_true -= min_val

    y_pred = utils.reassign_labels(y_pred, y_true, n_classes)

    acc = metrics.accuracy_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)

    return acc, nmi


class MyDataCollatorForWholeWordMask(DataCollatorForWholeWordMask):

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        batch['attention_mask'] = torch.stack([torch.as_tensor(example['attention_mask']) for example in examples])
        return batch


def main(args: Args) -> None:
    send_example_telemetry('run_mlm_no_trainer', args)

    # Initialize the accelerator. We will let the accelerator handle device
    # placement for us in this example. If we're using tracking, we also
    # need to initialize it here and it will by default pick up all
    # supported trackers in the environment

    if args.with_tracking:
        log_with = args.report_to
        project_dir = args.output_dir
    else:
        log_with = None
        project_dir = None
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=log_with,
        project_dir=project_dir
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(args.output_dir, 'log.log')
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    raw_datasets = get_raw_datasets(args)
    
    tokenizer, model = load_tokenizer_and_model(args)

    # We resize the embeddings only when necessary to avoid index errors.
    # If you are creating a model from scratch on a small vocab and
    # want a smaller embedding size, remove this test.
    embedding_size = model.config.hidden_size
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First, we tokenize all the texts.
    column_names = raw_datasets['train'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                'The chosen tokenizer supports a `model_max_length` that is '
                'longer than the default `block_size` value of 1024. '
                'If you would like to use a longer `block_size` up to '
                '`tokenizer.model_max_length` you can override this '
                'default with `--block_size xxx`.'
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f'The max_seq_length passed ({args.max_seq_length}) is larger than '
                f'the maximum length for the model ({tokenizer.model_max_length}). '
                f'Using max_seq_length={tokenizer.model_max_length}.'
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    raw_train_dataset = raw_datasets['train']
    n_samples = len(raw_train_dataset)

    # Log a few random samples from the training set:
    draw_indices = random.sample(range(n_samples), min(3, n_samples))

    for i in draw_indices:
        logger.info(f'Sample {i} of the raw training set: {raw_train_dataset[i]}.')

    tokenized_datasets = tokenize_datasets(
        raw_datasets, tokenizer, accelerator,
        column_names, text_column_name, max_seq_length, args
    )

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']

    for i in draw_indices:
        logger.info(f'Sample {i} of the tokenized training set: {train_dataset[i]}.')

    # Data collator
    # This one will take care of randomly masking the tokens.
    if args.wwm:
        data_collator_class = MyDataCollatorForWholeWordMask
    else:
        data_collator_class = DataCollatorForLanguageModeling
    
    data_collator = data_collator_class(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability
    )

    # DataLoaders creation:
    train_loader = DataLoader(
        train_dataset,  # type: ignore
        args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_loader = DataLoader(
        eval_dataset,  # type: ignore
        args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                param
                for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [
                param
                for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Note -> the training dataloader needs to be prepared before we grab 
    # his length below (cause its length will be shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.XLA:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the
    # training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = args.as_dict()
        # TensorBoard cannot log Enums, need the raw value
        experiment_config['lr_scheduler_type'] = experiment_config['lr_scheduler_type'].value
        accelerator.init_trackers('mlm_no_trainer', experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info('***** Running training *****')
    logger.info(f'  Num samples = {len(train_dataset)}')
    logger.info(f'  Num batches per epoch = {len(train_loader)}')
    logger.info(f'  Num epochs = {args.num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {args.per_device_train_batch_size}')
    logger.info(f'  Total train batch size '
                f'(w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), dynamic_ncols=True, disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None:
        completed_steps, starting_epoch, resume_step = resume_from_checkpoint(
            accelerator, train_loader, num_update_steps_per_epoch, args
        )

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    total_loss = 0
    checkpoints_tracker = []
    # losses = []
    # acc_scores = []
    # nmi_scores = []
    model.train()
    for epoch in range(starting_epoch + 1, args.num_train_epochs + 1):
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch + 1 and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_loader = accelerator.skip_first_batches(train_loader, resume_step)
        else:
            active_loader = train_loader
        
        for batch in active_loader:
            assert isinstance(batch, Mapping)
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # loss = loss.mean()
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.item() * len(batch['input_ids'])
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if not accelerator.sync_gradients:
                continue

            progress_bar.update()
            completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        eval_losses, embeddings = evaluate(model, eval_loader, accelerator)

        eval_loss = torch.cat(eval_losses).mean().item()
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = math.inf
        
        train_loss = total_loss / len(train_loader.dataset)

        msg = [
            f'epoch: {epoch:3}',
            f'step: {completed_steps:5}',
            f'train_loss: {train_loss:.8f}',
            f'perplexity: {perplexity:12.8f}',
            f'eval_loss: {eval_loss:.8f}'
        ]
        
        if args.checkpointing_epochs > 0 and epoch % args.checkpointing_epochs == 0:
            acc, nmi = cluster_evaluate(embeddings, args.n_classes, args.seed)
            msg += [
                f'acc: {acc:.2%}',
                f'nmi: {nmi:.2%}'
            ]

            output_dir = f'{args.epoch_prefix}{epoch}'
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            
            should_save = True
            if args.save_limit is not None and len(checkpoints_tracker) >= args.save_limit:
                worst_dir, worst_acc = checkpoints_tracker[-1]
                if acc > worst_acc:
                    checkpoints_tracker.pop()
                    shutil.rmtree(worst_dir)
                else:
                    should_save = False

            if should_save:
                accelerator.save_state(output_dir)
                checkpoints_tracker.append((output_dir, acc))
                checkpoints_tracker.sort(key=lambda item: item[1], reverse=True)

        logger.info(' '.join(msg))
        
        if args.with_tracking:
            accelerator.log(
                {
                    'perplexity': perplexity,
                    'eval_loss': eval_loss,
                    'train_loss': train_loss,
                    'epoch': epoch
                },
                step=completed_steps,
            )
        
        if eval_loss < args.early_stopping_loss_thresh:
            output_dir = 'early_stop'
            if args.wwm:
                output_dir = f'wwm_{output_dir}'
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            break

        # losses.append(eval_loss)
        # acc, nmi = cluster_eval(embeddings, args.n_classes, args)
        # acc_scores.append(acc)
        # nmi_scores.append(nmi)
        model.train()

    if args.with_tracking:
        accelerator.end_training()


if __name__ == '__main__':
    warnings.filterwarnings('ignore', 'DataCollatorForWholeWordMask*', UserWarning)
    args = Args().parse_args()
    main(args)
