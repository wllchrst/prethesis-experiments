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
    # "distilbert-base-uncased",
    "roberta-base",
    "bert-base-uncased",
    # "xlnet-base-cased",
    # "indobenchmark/indobert-base-p1",
]

DATASETS: list[DatasetInformation] = [
    # DatasetInformation(link='dair-ai/emotion' , labels_dropped=[],hf_dataset=True, augment=True, with_balancing=True, is_indonesian=False),
    DatasetInformation(link='../DataScript/indonesian_dataset_augmented_mapped.csv', labels_dropped=[], augment=False, hf_dataset=False, with_balancing=False, is_indonesian=True),
    # DatasetInformation(link='../Experiments/Datasets/indo-data-review.csv', labels_dropped=[], augment=False, hf_dataset=False, with_balancing=False, is_indonesian=True),
    # DatasetInformation(link='../Experiments/Datasets/process_data.csv', labels_dropped=[], hf_dataset=False, augment=True, with_balancing=True, is_indonesian=False),
]