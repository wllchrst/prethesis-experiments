from dataclasses import dataclass

@dataclass
class DatasetInformation:
    link: str
    hf_dataset: bool = True
    augment: bool = True
    with_balancing: bool = True

MODELS = [
    # "distilbert-base-uncased",
    # "roberta-base",
    # "bert-base-uncased",
    # "xlnet-base-cased",
    "indobenchmark/indobert-base-p1",
    "indobenchmark/indobert-large-p1"
]

DATASETS: list[DatasetInformation] = [
    # DatasetInformation('dair-ai/emotion', True, True),
    DatasetInformation('../Experiments/Datasets/indo-data-review.csv', False, False, False),
    # DatasetInformation('../Experiments/Datasets/process_data.csv', False, True),
]