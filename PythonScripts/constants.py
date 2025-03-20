from dataclasses import dataclass
from PythonScripts.training.training import TrainingInformation

@dataclass
class DatasetInformation:
    link: str
    labels_dropped: list[str]
    hf_dataset: bool = True
    augment: bool = True
    with_balancing: bool = True
    is_indonesian: bool = False

@dataclass
class BaggingConfig:
    model_paths: list[str]
    num_model: int
    training_information: TrainingInformation
    tokenizer: str
    
MODELS = [
    # "distilbert-base-uncased",
    # "roberta-base",
    # "bert-base-uncased",
    # "xlnet-base-cased",
    "indobenchmark/indobert-base-p1",
    "AptaArkana/indonesian-distilbert-base-cased-finetuned-indonlu"
]

DATASETS: list[DatasetInformation] = [
    # DatasetInformation(link='dair-ai/emotion' , labels_dropped=[],hf_dataset=True, augment=True, with_balancing=True, is_indonesian=False),
    DatasetInformation(link='DataScript/indonesian_dataset_augmented_mapped.csv', labels_dropped=[], augment=False, hf_dataset=False, with_balancing=False, is_indonesian=True),
    # DatasetInformation(link='../Experiments/Datasets/indo-data-review.csv', labels_dropped=[], augment=False, hf_dataset=False, with_balancing=False, is_indonesian=True),
    # DatasetInformation(link='../Experiments/Datasets/process_data.csv', labels_dropped=[], hf_dataset=False, augment=True, with_balancing=True, is_indonesian=False),
]

TEST_CONFIG = TrainingInformation(
    pretrained_model='',
    dataset_name='IndonesiaDataset',
    epoch=1,
    dropout_probability=0.1,
    batch_size=8,
    folder_name_from_info=True    
)

TRAIN_CONFIGS = [
    # ! BASE
    TrainingInformation(
       pretrained_model='',
       dataset_name='IndonesiaDataset',
       epoch=6,
       dropout_probability=0.4,
       batch_size=16,
       folder_name_from_info=True,
       weight_decay=0.3
    ),
    # ! WITH DROP OUT 0.3
    TrainingInformation(
        pretrained_model='',
        dataset_name='IndonesiaDataset',
        epoch=3,
        dropout_probability=0.3,
        batch_size=8,
        folder_name_from_info=True
    ),
    # ! WITH DROP OUT 0.5
    TrainingInformation(
        pretrained_model='',
        dataset_name='IndonesiaDataset',
        epoch=3,
        dropout_probability=0.5,
        batch_size=8,
        folder_name_from_info=True
    ),
    # ! BATCH SIZE CHANGE
    TrainingInformation(
        pretrained_model='',
        dataset_name='IndonesiaDataset',
        epoch=3,
        dropout_probability=0.1,
        batch_size=16,
        folder_name_from_info=True
    ),
    # ! CHANGE WEIGHT DECAY
    TrainingInformation(
        pretrained_model='',
        dataset_name='IndonesiaDataset',
        epoch=3,
        dropout_probability=0.1,
        batch_size=8,
        folder_name_from_info=True,
        weight_decay=0.3
    ),
    TrainingInformation(
        pretrained_model='',
        dataset_name='IndonesiaDataset',
        epoch=3,
        dropout_probability=0.3,
        batch_size=32,
        folder_name_from_info=True,
    ),
]

ENSEMBLE_CONFIG = BaggingConfig(
    model_paths=['indobenchmark/indobert-base-p1', 'indobenchmark/indobert-base-p1'],
    num_model=2,
    training_information=TRAIN_CONFIGS[0],
    tokenizer='indobenchmark/indobert-base-p1'
)
