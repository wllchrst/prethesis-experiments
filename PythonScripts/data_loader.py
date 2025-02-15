'''
Python Script for loading data from pandas(Need to be installed first) or even kaggle.
'''
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets

@dataclass
class DataLoaderSettings:
    '''
    Settings or attributes that is going to be past in the data loader class contains information for dataset link, keys from the dataset (train, test, validation), text_col and label_col of the dataset
    '''
    dataset_link: str
    keys: list[str]
    text_col: str
    label_col: str

class DataLoader:
    '''
    Class for load data from hugging or csv file
    '''
    def __init__(self, loader_settings: DataLoaderSettings):
        self.settings = loader_settings
        self.loaded = self.load_dataset()

        print(f'Loaded: {self.loaded}')

    def load_dataset(self) -> bool:
        '''
        Load Dataset from a csv file or hugging face dataset
        '''

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
            print(f'Something went wrong when trying to load dataset from link {self.settings.dataset_link}')
            print(f'Got error {e}')
            return False
