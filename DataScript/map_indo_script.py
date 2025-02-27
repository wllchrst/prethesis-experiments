import pandas as pd

df = pd.read_csv("./indonesian_dataset_augmented.csv")

df.drop(columns=['Unnamed: 0'], inplace=True)


# Map Categories
category_map = {0: 'Sadness', 1: 'Anger', 2: 'Love', 3: 'Happy', 4: 'Fear'}
df['label'] = df['label'].astype(int)
df['label'] = df['label'].map(category_map)

print(df['label'].unique())

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('./indonesian_dataset_augmented_mapped.csv')