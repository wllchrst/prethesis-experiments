from dataclasses import dataclass

@dataclass
class DatasetInformation:
    link: str
    labels_dropped: list[str]
    hf_dataset: bool = True
    augment: bool = True
    with_balancing: bool = True
    is_indonesian: bool = False
    
MODELS = [
    "distilbert-base-uncased",
    "roberta-base",
    "bert-base-uncased",
    # "xlnet-base-cased",
    # "indobenchmark/indobert-base-p1",
]

DATASETS: list[DatasetInformation] = [
    DatasetInformation(link='dair-ai/emotion' , labels_dropped=['love', 'surprise'],hf_dataset=True, augment=True, with_balancing=True, is_indonesian=False),
    # DatasetInformation('../Experiments/Datasets/indo-data-review.csv', False, False, False),
    # DatasetInformation('../Experiments/Datasets/process_data.csv', False, True),
]