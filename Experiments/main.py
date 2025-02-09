from custom_dataset import CustomDataset
from interfaces import DatasetInformation

dataset_information = DatasetInformation(
  dataset_link='dair-ai/emotion',
  dataset_type='HF', # from hugging face.
  label_col='label',
  text_col='text',
  tokenizer_link='xlm-roberta-base'
)

dataset = CustomDataset(dataset_information=dataset_information)

sample = dataset[0]

print(sample)