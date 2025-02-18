from dataclasses import dataclass

@dataclass
class DatasetInformation:
    link: str
    hf_dataset: True
    augment: True

DISTIL_BERT = "distilbert-base-uncased"

MODELS = [
    "distilbert-base-uncased",
    "roberta-base",
    "bert-base-uncased",
    "xlnet-base-cased"
]

DATASETS: list[DatasetInformation] = [
    DatasetInformation('dair-ai/emotion', True, True),
    DatasetInformation('../Experiments/Datasets/indo-data-review.csv', False, False),
    DatasetInformation('../Experiments/Datasets/process_data.csv', False, True),
]