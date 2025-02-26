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
    columns_to_drop: list[str]
    hf_dataset: bool = True
    with_balancing: bool = True
    with_details: bool = True
    from_cache: bool = False
    is_indonesian: bool = False

class DataLoader:
    '''
    Class for load data from hugging or csv file

    Steps of the data loader.
    - Load dataset
    - Process dataset
        - Drop columns
        - Process text from dataset
        - Balance dataset either with augmentation or dropping data
        - convert the label to ClassLabel type
        - Print Dataset Detail
    '''
    def __init__(self, loader_settings: DataLoaderSettings, save_path='../Experiments/Datasets/'):
        self.save_path = save_path
        self.settings = loader_settings
        self.file_name = self.settings.dataset_link.replace("/", " ")
        self.data_processor = DataProcessor()
        self.loaded = self.load_dataset()
        self.processed = self.process_dataset(self.settings.with_augmentation)
        self.dataset_detail()
        
    def process_dataset(self, with_augmentation=True) -> bool:
        '''
        Process dataset using the data_processor class
        '''
        print("Before Processing")
        self.dataset_detail()
        if not hasattr(self, 'dataset'):
            raise AttributeError("Dataset has not been loaded")

        self.dataset = self.data_processor.drop_column_from_dataset(self.settings.columns_to_drop, self.dataset)

        self.dataset = self.data_processor.process_text\
            (dataset=self.dataset, text_col=self.settings.text_col)
        
        if self.settings.with_balancing:
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
                self.class_names = self.dataset.features[self.settings.label_col].names
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
            self.class_names = self.dataset.features[self.settings.label_col].names
            return True
        except Exception as e:
            print(f'Loading {self.settings.dataset_link} Error: {e}')
            return False

    def dataset_detail(self) -> None:
        """
        Print Out Details of the dataset
        """
        if not self.settings.with_details:
            return None

        print("="*80)
        print("Dataset Detail\n")

        label_counts = Counter(self.dataset[self.settings.label_col])
        
        print("Label counts:")
        for label, count in label_counts.items():
            print(f"{label}: {count}")

        print("\nExamples of Text for Each Label:")
        
        for label_key in label_counts.keys():
            # Get indices where label matches
            indices = [i for i, lbl in enumerate(self.dataset[self.settings.label_col]) if lbl == label_key]
            
            # Ensure there are enough examples to display
            selected_examples = [self.dataset[self.settings.text_col][i] for i in indices[:3]]
            
            print(f"\nLabel: {self.class_names[label_key]}")
            for i, example in enumerate(selected_examples, 1):
                print(f"  {i}. {example}")

        print("="*80)

