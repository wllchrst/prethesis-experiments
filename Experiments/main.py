from data_loader import DataLoader, DataLoaderSettings
from custom_dataset import CustomDataset, DatasetSettings
from training import train_model, TrainingInformation

loader_settings = DataLoaderSettings(
    dataset_link='dair-ai/emotion',
    keys=['train', 'validation', 'test'],
    label_col='label',
    text_col='text'
)

loader = DataLoader(loader_settings=loader_settings)
dataset_settings = DatasetSettings(
    tokenizer_link='xlm-roberta-base',
    label_col='label',
    text_col='text'
)

dataset = CustomDataset(
    data_loader=loader,
    dataset_settings=dataset_settings
)

training_information = TrainingInformation(
    pretrained_model='xlm-roberta-base'
)

# print(dataset.train[0])
# print(dataset.count_unique_labels())

train_model(dataset, training_information=training_information)