import nltk
import random
import os
import asyncio
import pandas as pd
from datasets import ClassLabel, Dataset
from nltk import word_tokenize 
from nltk.corpus import stopwords
from eda import eda
from collections import Counter
from indonesia_eda import indonesia_eda

class DataProcessor:
    '''
    Separate Class that works with data loader for processing the data (text processing, text augmentation and other important part of the data loading process)
    '''

    def __init__(self, save_path='../Experiments/Datasets/'):
        self.stop_words = stopwords.words("english")
        self.save_path = save_path
    
    async def balance_dataset(self, dataset: Dataset, dataset_name: str, with_augmentation: bool, text_col='text', label_col='label', from_cache=False, is_indonesian=False):
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
        if os.path.exists(file_name) and with_augmentation and from_cache:
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
                    if sample[text_col] == '':
                        continue

                    augmented_texts = []
                    
                    if is_indonesian:
                        augmented_texts = await indonesia_eda(sample[text_col])
                    else:
                        augmented_texts = eda(sample[text_col], num_aug=3)

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
    
    def convert_labels_to_classlabel(self, dataset: Dataset, label_col: str, class_names: list[str]) -> Dataset:
        '''
        Convert labels from a dataset into Class Label with correct feature mapping.

        Args:
        - dataset: Dataset, dataset that is going to be converted into class label
        - label_col: str, name of the column that is going to be changed

        Returns:
        - dataset: Dataset, dataset that has been processed
        '''
        unique_labels = sorted(list(set(dataset[label_col])))
        new_class_names = []
        for label in unique_labels:
            new_class_names.append(class_names[label])

        label_mapping = {old_label: new_index for new_index, old_label in enumerate(unique_labels)}

        dataset = dataset.map(lambda example: {label_col: label_mapping[example[label_col]]})

        class_label_feature = ClassLabel(num_classes=len(new_class_names), names=new_class_names)

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
            try:
                text = sample[text_col]
                words = self.tokenize(text)
                words = self.remove_stopwords(words)

                sample[text_col] = ' '.join(words)
                return sample
            except Exception as e:
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

    def drop_label_from_dataset(self, labels: list[int], label_col: str, dataset: Dataset) -> Dataset:
        """
        Drop all samples from the dataset that match any label in the provided list.

        Args:
            labels (list[str]): List of labels to be removed.
            dataset (Dataset): The dataset to filter.

        Returns:
            Dataset: A new dataset with the specified labels removed.
        """
        # Convert labels to a set for faster lookup
        labels_to_remove = set(labels)

        # Debug: Check unique labels in the dataset
        unique_labels = set(dataset[label_col])
        print(f"Unique labels before filtering: {unique_labels}")
        print(f"Labels to remove: {labels_to_remove}")

        # Check for potential type mismatch issues
        sample_label = dataset[0][label_col]
        print(f"Sample label type: {type(sample_label)}, Label values: {sample_label}")

        # Ensure the labels are of the same type as dataset labels
        if isinstance(sample_label, int):
            labels_to_remove = {int(label) for label in labels_to_remove}
        elif isinstance(sample_label, str):
            labels_to_remove = {str(label) for label in labels_to_remove}

        # Perform filtering
        filtered_dataset = dataset.filter(lambda example: example[label_col] not in labels_to_remove)

        # Debug: Check unique labels after filtering
        unique_labels_after = set(filtered_dataset[label_col])
        print(f"Unique labels after filtering: {unique_labels_after}")

        return filtered_dataset