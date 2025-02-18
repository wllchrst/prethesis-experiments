'''
Python Script for loading data from pandas(Need to be installed first) or even kaggle.
'''
import pandas as pd
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets
from data_processor import DataProcessor
from datasets import Dataset, ClassLabel
from collections import Counter

@dataclass
class DataLoaderSettings:
    '''
    Settings or attributes that is going to be past in the data loader class contains information for dataset link, keys from the dataset (train, test, validation), text_col and label_col of the dataset
    '''
    dataset_link: str
    keys: list[str]
    text_col: str
    label_col: str
    with_augmentation: bool
    hf_dataset: bool = True

class DataLoader:
    '''
    Class for load data from hugging or csv file
    '''
    def __init__(self, loader_settings: DataLoaderSettings, save_path='../Experiments/Datasets/'):
        self.save_path = save_path
        self.settings = loader_settings
        self.file_name = self.settings.dataset_link.replace("/", " ")
        self.data_processor = DataProcessor()
        self.loaded = self.load_dataset()
        self.processed = self.process_dataset(self.settings.with_augmentation)
        
        print(f'Loaded: {self.loaded}, Processed: {self.processed}')
        
    def process_dataset(self, with_augmentation=True) -> bool:
        '''
        Process dataset using the data_processor class
        '''
        if not hasattr(self, 'dataset'):
            raise AttributeError("Dataset has not been loaded")
        print(self.dataset.features[self.settings.label_col])
        self.class_names = self.dataset.features[self.settings.label_col].names
        
        self.dataset = self.data_processor.process_text\
            (dataset=self.dataset, text_col=self.settings.text_col)
            
        self.dataset = self.data_processor.balance_dataset\
            (dataset=self.dataset, dataset_name=self.settings.dataset_link.replace('/', ' '), with_augmentation=with_augmentation, text_col='text')
        
        self.dataset = self.data_processor.convert_labels_to_classlabel\
            (dataset=self.dataset, label_col=self.settings.label_col)
        return True

    def load_dataset(self) -> bool:
        '''
        Load Dataset from a csv file or hugging face dataset
        '''
        if not self.settings.hf_dataset:
            try:
                df = pd.read_csv(self.settings.dataset_link)
                
                labels = list(df['label'].unique())
                class_label = ClassLabel(names=labels)
                df['label'] = df['label'].apply(lambda x: labels.index(x))
                
                self.dataset = Dataset.from_pandas(df)
                self.dataset = self.dataset.cast_column("label", class_label)
                return True
            except Exception as e:
                print(f"Loading dataset went wrong: {e}")
                return False
        try:
            dataset = load_dataset(self.settings.dataset_link, trust_remote_code=True)
            merged_dataset = None

            if not isinstance(dataset, dict):
                self.dataset = dataset
                return True

            for key in self.settings.keys:
                if key not in dataset:
                    continue

                dataset_partition = dataset[key]
                
                if merged_dataset == None:
                    merged_dataset = dataset_partition
                else:
                    merged_dataset = concatenate_datasets([merged_dataset, dataset_partition])
            
            self.dataset = merged_dataset
            return True
        except Exception as e:
            print(f'Loading {self.settings.dataset_link} Error: {e}')
            return False