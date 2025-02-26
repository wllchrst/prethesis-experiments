from constants import MODELS, DATASETS
from data_loader import DataLoaderSettings, DataLoader
from custom_dataset import CustomDataset, DatasetSettings
from training import train_model, TrainingInformation

if __name__ == "__main__":

    for dataset_info in DATASETS:

        loader_settings = DataLoaderSettings(
                dataset_link=dataset_info.link,
                keys=['train', 'test', 'validation'],
                label_col='label',
                text_col='text',
                columns_to_drop=[],
                from_cache=False,
                with_augmentation=dataset_info.augment,
                hf_dataset=dataset_info.hf_dataset,
                with_balancing=dataset_info.with_balancing,
                is_indonesian=dataset_info.is_indonesian,
                with_details=True,
            )

        data_loader = DataLoader(loader_settings)
        for model in MODELS:
            dataset_settings = DatasetSettings(
                label_col='label',
                text_col='text',
                tokenizer_link=model
            )

            dataset = CustomDataset(dataset_settings, data_loader)

            training_information = TrainingInformation(
                dataset_name=dataset_info.link,
                epoch=4,
                pretrained_model=model
            )

            train_model(dataset, training_information)