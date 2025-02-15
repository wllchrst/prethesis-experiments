from datasets import ClassLabel, Dataset
from nltk import word_tokenize 
from nltk.corpus import stopwords

class DataProcessor:
    '''
    Separate Class that works with data loader for processing the data (text processing, text augmentation and other important part of the data loading process)
    '''

    def __init__(self):
        self.stop_words = stopwords.words("english")

    def convert_labels_to_classlabel(self, dataset: Dataset, label_col: str) -> Dataset:
        '''
        Convert labels from a dataset into Class Label
        
        Args:
        - dataset: Dataset, dataset that is going to be converted into class label
        - label_col: str, name of the column that is going to be changed

        Returns:
        - dataset: Dataset, dataset that have been processed
        '''
        unique_labels = list(set(dataset[label_col]))
        class_label_feature = ClassLabel(num_classes=len(unique_labels), names=[str(label) for label in unique_labels])

        dataset = dataset.map(lambda example: {label_col: class_label_feature.str2int(str(example[label_col]))})
        dataset = dataset.cast_column(label_col, class_label_feature)

        return dataset

    def process_text(self, dataset: Dataset, text_col: str) -> bool:
        '''
        Process text column from the dataset, actions: Remove Stop Words, and non alphabetic words
        
        Args:
        - dataset: Dataset, dataset that is going to be processed
        - text_col: str, name of the text column
        '''
        def process_text(sample) -> str:
            text = sample[text_col]
            words = self.tokenize(text)
            words = self.remove_stopwords(words)

            sample[text_col] = ' '.join(words)
            return sample

        dataset = dataset.map(process_text)
        return True

    def tokenize(self, text: str) -> list[str]:
        '''
        Tokenize text into words
        
        Args:
        - text: str, text that is going to be tokenize
        
        Returns:
        - list[str], result of the tokenize
        '''

        words = word_tokenize(text)
        return words

    def remove_stopwords(self, words: list[str]) -> list[str]:
        '''
        Remove words that is in the list of stop words and not alphabetic

        Args:
        - words: list[str]
        
        Returns:
        - list[str]
        '''

        words = [word for word in words if word not in self.stop_words and word.isalpha()]
        return words
