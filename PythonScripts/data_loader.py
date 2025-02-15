'''
Python Script for loading data from pandas(Need to be installed first) or even kaggle.
'''
from dataclasses import dataclass
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset, concatenate_datasets, ClassLabel

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
    def __init__(self, loader_settings: DataLoaderSettings):
        self.stop_words = stopwords.words('english')
        self.settings = loader_settings
        self.loaded = self.load_dataset()
        self.processed = self.process()
        self.convert_labels_to_classlabel()

        print(f'Loaded: {self.loaded}')
        print(f'Processed: {self.processed}')

    def load_dataset(self) -> bool:
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
    
    def convert_labels_to_classlabel(self):
        unique_labels = list(set(self.dataset[self.settings.label_col]))
        class_label_feature = ClassLabel(num_classes=len(unique_labels), names=[str(label) for label in unique_labels])

        self.dataset = self.dataset.map(lambda example: {self.settings.label_col: class_label_feature.str2int(str(example[self.settings.label_col]))})
        self.dataset = self.dataset.cast_column(self.settings.label_col, class_label_feature)
        print(type(self.dataset[0]['label']))
        
    def process(self) -> bool:
        if not self.loaded:
            raise ValueError("Dataset has not been loaded")
        
        def process_text(sample) -> str:
            text = sample[self.settings.text_col]
            words = self.tokenize(text)
            words = self.remove_stopwords(words)

            sample[self.settings.text_col] = ' '.join(words)
            return sample

        try:
            self.dataset = self.dataset.map(process_text)
            return True
        except Exception as e:
            print(e)        
            return False

    def tokenize(self, text) -> list[str]:
        words = word_tokenize(text)
        return words

    def remove_stopwords(self, words) -> list[str]:
        words = [word for word in words if word not in self.stop_words and word.isalpha()]
        return words
