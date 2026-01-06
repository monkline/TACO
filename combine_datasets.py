from pathlib import Path
from itertools import product

import pandas as pd
from tap import Tap

from common import Task


class Args(Tap):
    tasks: tuple[Task, ...]
    prefixes1: tuple[str, ...] = ('roberta_subst_10',)
    prefixes2: tuple[str, ...] = ('roberta1_subst_10',)
    overwrite: bool = False


def combine_on_task(args: Args, task: Task):
    root = Path('./augmented-datasets') / task
    input_dir = root / 'single_columns'
    output_dir = root

    aug1_paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        if path.name.startswith(args.prefixes1)
    ]

    aug2_paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        if path.name.startswith(args.prefixes2)
    ]

    orig_df = pd.read_csv(input_dir / 'original.csv')

    for aug1_path, aug2_path in product(aug1_paths, aug2_paths):
        if aug1_path == aug2_path:
            continue

        aug1_df = pd.read_csv(aug1_path)
        aug2_df = pd.read_csv(aug2_path)

        output_df = orig_df.copy()
        output_df['text1'] = aug1_df['text']
        output_df['text2'] = aug2_df['text']

        output_name = f'{aug1_path.stem}_{aug2_path.stem}.csv'
        output_path = output_dir / output_name

        if not args.overwrite and output_path.exists():
            continue

        output_df.to_csv(output_path, index=False)
        print(f'File save to {output_path}')


def main(args: Args) -> None:
    for task in args.tasks:
        combine_on_task(args, task)


if __name__ == '__main__':
    main(Args().parse_args())
