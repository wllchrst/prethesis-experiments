from interfaces import DataLoaderSettings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset, concatenate_datasets

class DataLoader:
    def __init__(self, loader_settings: DataLoaderSettings):
        self.stop_words = stopwords("english")
        self.settings = loader_settings
        pass

    def load_dataset(self):
        try:
            dataset = load_dataset(self.settings.dataset_link)
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
        except FileNotFoundError:
            print(f'Something went wrong when trying to load dataset from link: {self.settings.dataset_link}')
            return False
        

    def tokenize(self, text) -> list[str]:
        words = word_tokenize(text)
        return words

    def remove_stopwords(self, words) -> list[str]:
        words = [word for word in words if word not in self.stop_words and word.isalpha()]
        return words

    def process_text(self, text) -> str:
        words = self.tokenize(text)
        words = self.remove_stopwords(words)

        return ' '.join(words)