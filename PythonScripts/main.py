import nltk
from PythonScripts.constants import MODELS, DATASETS, TRAIN_CONFIGS, TEST_CONFIG
from PythonScripts.dataset.data_loader import DataLoaderSettings, DataLoader
from PythonScripts.dataset.custom_dataset import CustomDataset, DatasetSettings
from PythonScripts.training.training import train_model
from PythonScripts.ensemble.bagging import Bagging

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt_tab")

if __name__ == "__main__":
    testing = False
    
    for dataset_info in DATASETS:
        loader_settings = DataLoaderSettings(
                dataset_link=dataset_info.link,
                keys=['train', 'test', 'validation'],
                label_col='label',
                text_col='text',
                labels_to_drop=[],
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
            
            bagging = Bagging("indobenchmark/indobert-base-p1", 3, dataset, TRAIN_CONFIGS[0])
            bagging.train()
            break
            
            if testing:
                TEST_CONFIG.pretrained_model = model
                train_model(dataset, TEST_CONFIG, labels_dropped=dataset_info.labels_dropped)
                break
            else:
                for config in TRAIN_CONFIGS:
                    config.pretrained_model = model

                    train_model(dataset, config, labels_dropped=dataset_info.labels_dropped)
