from dataclasses import dataclass

@dataclass
class DatasetInformation:
    tokenizer_link: str

@dataclass
class DataLoaderSettings:
    dataset_link: str
    keys: list[str]
    text_col: str
    label_col: str