from constants import DISTIL_BERT, DATASETS
from data_loader import DataLoaderSettings, DataLoader
from custom_dataset import CustomDataset, DatasetSettings
from training import train_model, TrainingInformation

dataset_info = DATASETS[1]

loader_settings = DataLoaderSettings(
    dataset_link=dataset_info.link,
    keys=['train', 'test', 'validation'],
    label_col='label',
    text_col='text',
    with_augmentation=dataset_info.augment,
    hf_dataset=dataset_info.hf_dataset
)

data_loader = DataLoader(loader_settings)

dataset_settings = DatasetSettings(
    label_col='label',
    text_col='text',
    tokenizer_link=DISTIL_BERT
)

dataset = CustomDataset(dataset_settings, data_loader)

training_information = TrainingInformation(
    dataset_name=dataset_info.link,
    epoch=1,
    pretrained_model=DISTIL_BERT
)

train_model(dataset, training_information)