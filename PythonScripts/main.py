from constants import DISTIL_BERT, DATASETS_LINKS
from data_loader import DataLoaderSettings, DataLoader
from custom_dataset import CustomDataset, DatasetSettings
from training import train_model, TrainingInformation

loader_settings = DataLoaderSettings(
    dataset_link=DATASETS_LINKS[1],
    keys=['train', 'test', 'validation'],
    label_col='label',
    text_col='text',
    with_augmentation=True
)

data_loader = DataLoader(loader_settings)

dataset_settings = DatasetSettings(
    label_col='label',
    text_col='text',
    tokenizer_link=DISTIL_BERT
)

dataset = CustomDataset(dataset_settings, data_loader)

training_information = TrainingInformation(
    dataset_name=DATASETS_LINKS[1],
    epoch=1,
    pretrained_model=DISTIL_BERT
)

train_model(dataset, training_information)