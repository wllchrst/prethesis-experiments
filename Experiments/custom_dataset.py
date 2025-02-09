from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from interfaces import DatasetInformation


class CustomDataset(Dataset):
    """
    Custom dataset class for tokenizing text data.

    Attributes:
    - dataset: DataFrame loaded from CSV
    - tokenizer: Tokenizer for text processing
    - max_length: Maximum token length
    """
    
    def __init__(self, dataset_information: DatasetInformation, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(dataset_information.tokenizer_link)
        self.dataset_information = dataset_information
        self.max_length = max_length
        self.label_encoder = LabelEncoder()
        self.loaded = self.load_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.loaded == False:
            raise ValueError("The dataset has not been loaded")
            
        sample = self.train[idx]
        text = sample[self.dataset_information.text_col]
        label = sample[self.dataset_information.label_col]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        print(f'Tokenizing Result: {encoding}')

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(label), dtype=torch.long)  # Convert label to tensor
        }

    def load_dataset(self) -> bool:
        """
        Load dataset from a link can be from kaggle or from hugging face apis

        Args: 
        - 

        Returns:
        - boolean
        """
        try:
            self.data = load_dataset(self.dataset_information.dataset_link)

            self.train = self.data['train']
            self.validation = self.data['validation']
            self.test = self.data['test']

            return True
        except FileNotFoundError:
            print(f"Error when trying to get dataset from this link: {self.dataset_information.dataset_link}")
            return False