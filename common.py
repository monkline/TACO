from typing import Literal, TypeAlias, TypeGuard

Task: TypeAlias = Literal['AG', 'SO', 'SS', 'Bio', 'G-TS', 'G-T', 'G-S', 'Tweet']
TASKS: frozenset[Task] = frozenset({'AG', 'SO', 'SS', 'Bio', 'G-TS', 'G-T', 'G-S', 'Tweet'})

N_CLASSES: dict[Task, int] = {
    'AG': 4,
    'SO': 20,
    'SS': 8,
    'Bio': 20,
    'G-TS': 152,
    'G-T': 152,
    'G-S': 152,
    'Tweet': 89
}

SOURCE_DATASET_FILES: dict[Task, str] = {
    'AG': './augmented-datasets/AG/single_columns/original.csv',
    'SO': './augmented-datasets/SO/single_columns/original.csv',
    'SS': './augmented-datasets/SS/single_columns/original.csv',
    'Bio': './augmented-datasets/Bio/single_columns/original.csv',
    'G-TS': './augmented-datasets/G-TS/single_columns/original.csv',
    'G-T': './augmented-datasets/G-T/single_columns/original.csv',
    'G-S': './augmented-datasets/G-S/single_columns/original.csv',
    'Tweet': './augmented-datasets/Tweet/single_columns/original.csv',
}

DATASET_FILES: dict[Task, str] = {
    'AG': './augmented-datasets/AG/roberta_subst_10_roberta1_subst_10.csv',
    'SO': './augmented-datasets/SO/roberta_subst_10_roberta1_subst_10.csv',
    'SS': './augmented-datasets/SS/roberta_subst_10_roberta1_subst_10.csv',
    'Bio': './augmented-datasets/Bio/roberta_subst_10_roberta1_subst_10.csv',
    'G-TS': './augmented-datasets/G-TS/roberta_subst_10_roberta1_subst_10.csv',
    'G-T': './augmented-datasets/G-T/roberta_subst_10_roberta1_subst_10.csv',
    'G-S': './augmented-datasets/G-S/roberta_subst_10_roberta1_subst_10.csv',
    'Tweet': './augmented-datasets/Tweet/roberta_subst_10_roberta1_subst_10.csv'
}

PRETRAINED_MODELS: dict[Task, str] = {
    'AG': './pretrained-models/AG/wwm_early_stop',
    'SO': './pretrained-models/SO/wwm_early_stop',
    'SS': './pretrained-models/SS/wwm_early_stop',
    'Bio': './pretrained-models/Bio/wwm_early_stop',
    'G-TS': './pretrained-models/G-TS/wwm_early_stop',
    'G-T': './pretrained-models/G-T/wwm_early_stop',
    'G-S': './pretrained-models/G-S/wwm_early_stop',
    'Tweet': './pretrained-models/Tweet/wwm_early_stop'
}

INITIAL_DISTILBERT_MODEL = './pretrained-models/distilbert-base-nli-stsb-mean-tokens'

MAX_SEQ_LENGTHS: dict[Task, int] = {
    'AG': 42,
    'SO': 25,
    'SS': 32,
    'Bio': 45,
    'G-TS': 40,
    'G-T': 16,
    'G-S': 32,
    'Tweet': 20
}


def is_task(s: str) -> TypeGuard[Task]:
    return s in TASKS
