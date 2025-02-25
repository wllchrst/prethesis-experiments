from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from dataclasses import dataclass
from data_loader import DataLoader

@dataclass
class DatasetSettings:
    label_col: str
    text_col: str
    tokenizer_link: str

class CustomDataset:
    """
    Custom dataset class for tokenizing text data.

    Attributes:
    - dataset: DataFrame loaded from CSV
    - tokenizer: Tokenizer for text processing
    - max_length: Maximum token length
    """
    
    def __init__(self, dataset_settings: DatasetSettings, data_loader: DataLoader, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(dataset_settings.tokenizer_link)
        self.settings = dataset_settings
        self.data_loader = data_loader
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.splitted = self.split_dataset()
        self.tokenized = self.tokenize_datasets()
        print(f'Splitted: {self.splitted}')
        print(f'Tokenized: {self.tokenized}')
    
    def count_unique_labels(self) -> int:
        """
        Counts the number of unique labels in the dataset.

        Returns:
        - int: The number of unique labels.
        """
        try:
            label_column = self.settings.label_col

            # Get unique labels from dataset
            unique_labels = set(self.train[label_column])

            print(f"Number of unique labels: {len(unique_labels)}")
            return len(unique_labels)

        except Exception as e:
            print(f"Error counting unique labels: {e}")
            return 0

    def __len__(self):
        return len(self.data)
    
    def get_class_names(self):
        return self.data_loader.class_names

    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42) -> bool:
        """
        Split the dataset into training, validation, and testing sets.

        Args:
        - train_ratio (float): Proportion of the dataset to use for training.
        - val_ratio (float): Proportion of the dataset to use for validation.
        - test_ratio (float): Proportion of the dataset to use for testing.
        - random_state (int): Seed for reproducibility.

        Returns:
        - bool: True if the split was successful, False otherwise.
        """
        if not hasattr(self, "data_loader"):
            print("Dataset is not loaded.")
            return False

        if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1 and train_ratio + val_ratio + test_ratio == 1):
            print("Invalid split ratios. Ensure they sum to 1.")
            return False

        dataset = self.data_loader.dataset
        try:
            # First, split into train and temp (val + test)
            train_test_split = dataset.train_test_split(test_size=(1 - train_ratio), seed=seed, stratify_by_column=self.settings.label_col)
            train_data = train_test_split["train"]
            temp_data = train_test_split["test"]

            # Compute relative validation split
            val_size = val_ratio / (val_ratio + test_ratio)  # Normalize val/test split
            val_test_split = temp_data.train_test_split(test_size=(1 - val_size), seed=seed, stratify_by_column=self.settings.label_col)

            self.train = train_data
            self.val = val_test_split["train"]
            self.test = val_test_split["test"]

            print(f"Dataset split complete: Train({len(self.train)}), Val({len(self.val)}), Test({len(self.test)})")
            return True

        except Exception as e:
            print(f"Error splitting dataset: {e}")
            return False

    def tokenize_datasets(self):
        """
        Tokenizes the train, validation, and test datasets using the tokenizer.

        This function modifies self.train, self.val, and self.test in-place.
        """
        if not hasattr(self, "train") or self.train is None:
            print("Training dataset is not loaded.")
            return False
        if not hasattr(self, "val") or self.val is None:
            print("Validation dataset is not loaded.")
            return False
        if not hasattr(self, "test") or self.test is None:
            print("Testing dataset is not loaded.")
            return False

        try:
            text_column = self.settings.text_col
            label_column = self.settings.label_col
            
            # Tokenization function
            def tokenize_function(example):
                encoding = self.tokenizer(
                    example[text_column],
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                encoding["labels"] = [torch.tensor(label, dtype=torch.long) for label in example[label_column]]
                return encoding

            self.train = self.train.map(tokenize_function, batched=True)
            self.val = self.val.map(tokenize_function, batched=True)
            self.test = self.test.map(tokenize_function, batched=True)

            print("Tokenization complete for train, val, and test datasets.")
            return True

        except Exception as e:
            print(f"Error tokenizing datasets: {e}")
            return False