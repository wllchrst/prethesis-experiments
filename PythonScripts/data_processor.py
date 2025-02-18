import nltk
import random
import os
import pandas as pd
from datasets import ClassLabel, Dataset
from nltk import word_tokenize 
from nltk.corpus import stopwords
from eda import eda
from collections import Counter

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt_tab")

class DataProcessor:
    '''
    Separate Class that works with data loader for processing the data (text processing, text augmentation and other important part of the data loading process)
    '''

    def __init__(self, save_path='../Experiments/Datasets/'):
        self.stop_words = stopwords.words("english")
        self.save_path = save_path
    
    def balance_dataset(self, dataset: Dataset, dataset_name: str, with_augmentation: bool, text_col='text', label_col='label'):
        '''
        Balance dataset by randomly discarding data or augmenting data.

        Args:
        - dataset: Dataset, the dataset to balance.
        - dataset_name: str, name of the dataset file.
        - with_augmentation: bool, whether to use augmentation for balancing.
        - text_col: str, column name containing text data.
        - label_col: str, column name containing labels.

        Returns:
        - Dataset, balanced dataset.
        '''
        augmented_desc = "_augmented" if with_augmentation else ""
        dataset_name += '.csv'
        file_name = self.save_path + augmented_desc + dataset_name
        if os.path.exists(file_name) and with_augmentation:
            df = pd.read_csv(file_name)
            return Dataset.from_pandas(df)

        # Count occurrences of each label
        label_counts = Counter(dataset[label_col])
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())  # Get the smallest label count

        balanced_data = []
        for label, count in label_counts.items():
            # Extract rows with the current label
            subset = [row for row in dataset if row[label_col] == label]

            if with_augmentation:
                # Augment the data to match the max count
                while len(subset) < max_count:
                    sample = random.choice(subset)
                    augmented_texts = eda(sample[text_col])  # Apply augmentation
                    for aug_text in augmented_texts:
                        if len(subset) < max_count:
                            augmented_sample = sample.copy()
                            augmented_sample[text_col] = aug_text
                            subset.append(augmented_sample)
                        else:
                            break
            else:
                # Randomly downsample if necessary
                if count > min_count:  
                    subset = random.sample(subset, min_count)

            balanced_data.extend(subset)

        df_balanced = pd.DataFrame(balanced_data)
        df_balanced.to_csv(file_name, index=False)

        return Dataset.from_list(balanced_data)


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
        class_label_feature = ClassLabel\
            (num_classes=len(unique_labels), names=[str(label) for label in unique_labels])

        dataset = dataset\
            .map(lambda example: {label_col: class_label_feature.str2int(str(example[label_col]))})
        dataset = dataset.cast_column(label_col, class_label_feature)

        return dataset

    def process_text(self, dataset: Dataset, text_col: str) -> Dataset:
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
        return dataset

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
