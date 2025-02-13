from data_loader import DataLoader, DataLoaderSettings
from custom_dataset import CustomDataset, DatasetSettings
from training import train_model, TrainingInformation
from constants import DATASETS_LINKS, DISTIL_BERT

datasets: list[CustomDataset] = []

for dataset_link in DATASETS_LINKS:
    print(dataset_link)
    print('')

    loader_settings = DataLoaderSettings(
        dataset_link=dataset_link,
        keys=['train', 'validation', 'test'],
        label_col='label',
        text_col='text'
    )

    loader = DataLoader(loader_settings=loader_settings)
    print(loader.dataset[0])

    dataset_settings = DatasetSettings(
        tokenizer_link=DISTIL_BERT,
        label_col='label',
        text_col='text'
    )

    dataset = CustomDataset(
        data_loader=loader,
        dataset_settings=dataset_settings
    )

    datasets.append(dataset)

for idx in range(len(datasets)):
    print(DATASETS_LINKS[idx])
    
    training_information = TrainingInformation(
        pretrained_model=DISTIL_BERT
    )

    train_model(datasets[idx], training_information=training_information)