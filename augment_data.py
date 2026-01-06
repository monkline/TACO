from pathlib import Path
from typing import Literal

import torch
import pandas as pd
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from nlpaug import Augmenter
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.distilbert import DistilBertTokenizerFast

from tap import Tap
from tqdm import tqdm

from common import Task, INITIAL_DISTILBERT_MODEL


def augment_one(
    aug: Augmenter,
    text: str,
    fallback: str,
    tokenizer: PreTrainedTokenizerFast,
    max_retries: int
) -> tuple[str, bool]:
    fallback_input = tokenizer(fallback, return_tensors='pt')['input_ids']

    for _ in range(max_retries):
        result: list[str] = aug.augment(text)
        if not result:
            continue

        augmented = result[0].strip().lower()
        if not augmented:
            continue

        augmented_input = tokenizer(augmented, return_tensors='pt')['input_ids']

        if not torch.equal(fallback_input, augmented_input):
            return augmented, False
    
    return fallback, True


class Args(Tap):
    tasks: list[Task]
    root: Path = Path('augmented-datasets')

    filename: str = 'original.csv'
    fallback_filename: str = 'trans_subst_10_trans_subst_10.csv'
    fallback_column: str = 'text'

    max_retires: int = 10

    model_root: Path = Path('pretrained-models')
    model_name: str

    method: Literal['contextual', 'summarize'] = 'contextual'
    aug_args: list[float]

    batch_numbers: tuple[int, ...] = (0,)

    overwrite: bool = False
    
    @property
    def model_path(self) -> Path:
        return self.model_root / self.model_name

    def filepath(self, task: Task) -> Path:
        return self.single_columns_dir(task) / self.filename
    
    def fallback_filepath(self, task: Task) -> Path:
        return self.root / task / self.fallback_filename
    
    def single_columns_dir(self, task: Task) -> Path:
        return self.root / task / 'single_columns'


def augment_on_task(args: Args, task: Task) -> None:
    dataset = pd.read_csv(args.filepath(task))
    texts = dataset['text'].str.lower().tolist()
    labels = dataset['label'].tolist()

    fallback_texts = pd.read_csv(args.fallback_filepath(task))[args.fallback_column].tolist()

    tokenizer = DistilBertTokenizerFast.from_pretrained(INITIAL_DISTILBERT_MODEL)

    for arg in args.aug_args:
        for batch_number in args.batch_numbers:
            name = args.model_name.partition('-')[0]
            if batch_number != 0:
                name = f'{name}{batch_number}'
            
            if args.method == 'contextual':
                method_name = f'{name}_subst_{int(arg * 100)}'
                aug = naw.ContextualWordEmbsAug(str(args.model_path), aug_p=arg, device='cuda')
            elif args.method == 'summarize':
                method_name = f'{name}_abst_{int(arg * 10)}'
                aug = nas.AbstSummAug(str(args.model_path), temperature=arg, max_length=64, device='cuda')
            else:
                raise ValueError
            
            output_path = args.single_columns_dir(task) / f'{method_name}.csv'
            if not args.overwrite and output_path.exists():
                continue

            augmented_texts: list[str] = []
            fallback_count = 0

            with tqdm(total=len(texts), desc=f'Augmenting with {method_name}', dynamic_ncols=True) as tbar:
                for text, fallback in zip(texts, fallback_texts):
                    augmented, use_fallback = augment_one(aug, text, fallback, tokenizer, args.max_retires)
                    
                    augmented_texts.append(augmented)
                    fallback_count += use_fallback

                    tbar.set_postfix(dict(fallback_count=fallback_count))
                    tbar.update()
            
            output = pd.DataFrame(dict(label=labels, text=augmented_texts))
            output.to_csv(output_path, index=False)
            print(f'Save augmented dataset to {output_path}')


def main(args: Args) -> None:
    for task in args.tasks:
        augment_on_task(args, task)


if __name__ == '__main__':
    main(Args().parse_args())
