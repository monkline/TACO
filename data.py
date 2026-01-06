from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset, Features, Value
from nlpaug import Augmenter
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def load_raw_dataset(filename: str) -> Dataset:
    extension = filename.rsplit('.', maxsplit=1)[-1]
    if extension == 'txt':
        extension = 'text'

    features = dict(
        label='int64', text='string', text1='string', text2='string'
    )

    raw_dataset = load_dataset(
        extension,
        data_files=filename, split='all',
        features=Features({k: Value(v) for k, v in features.items()})
    )
    assert isinstance(raw_dataset, Dataset)
    
    return raw_dataset


def default_fmt_key(name: str, key: str) -> str:
    return f'{name}.{key}'


def get_tokenize(
    column_names: str | list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
    fmt_key: Callable[[str, str], str]
) -> Callable[[Mapping[str, Any]], MutableMapping[str, Any]]:
    def tokenize_column(examples: Mapping[str, Any], column_name: str) -> MutableMapping[str, Any]:
        return tokenizer(
            examples[column_name],
            padding='max_length',
            truncation=True,
            max_length=max_seq_length
        )

    def tokenize(examples: Mapping[str, Any]) -> MutableMapping[str, Any]:
        if isinstance(column_names, str):
            return tokenize_column(examples, column_names)
        
        return {
            fmt_key(column_name, k): v
            for column_name in column_names
            for k, v in tokenize_column(examples, column_name).items()
        }

    return tokenize


def tokenize_dataset(
    raw_dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    accelerator: Accelerator,
    column_names: str | list[str],
    max_seq_length: int,
    remove_columns: bool | str | list[str] = True,
    fmt_key: Callable[[str, str], str] = default_fmt_key
):
    tokenize = get_tokenize(column_names, tokenizer, max_seq_length, fmt_key)
    
    if isinstance(remove_columns, bool):
        remove_columns = column_names if remove_columns else None
    
    with accelerator.main_process_first():
        return raw_dataset.map(
            tokenize,
            batched=True, remove_columns=remove_columns
        )


def make_train_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    accelerator: Accelerator,
    max_seq_length: int,
    fmt_key: Callable[[str, str], str] = default_fmt_key
):
    column_names = ['text', 'text1', 'text2']

    dataset = tokenize_dataset(
        dataset,
        tokenizer, accelerator,
        column_names, max_seq_length,
        fmt_key=fmt_key
    )

    # TODO: consider writing a class to implement the indices return
    dataset = dataset.add_column('index', range(len(dataset)))  # type: ignore

    return dataset


def load_train_dataset(
    filename: str,
    tokenizer: PreTrainedTokenizerFast,
    accelerator: Accelerator,
    max_seq_length: int
) -> Dataset:
    dataset = load_raw_dataset(filename)
    return make_train_dataset(dataset, tokenizer, accelerator, max_seq_length)
