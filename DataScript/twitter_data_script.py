"""
Dataset LINK: https://www.kaggle.com/code/shainy/twitter-emotion-analysis/input
"""
import pandas as pd

df = pd.read_csv("../Experiments/Datasets/data.csv")

df['label'] = df['Feeling']
df['text'] = df['Tweets']

df.drop(columns=['Tweets', 'Sl no', 'Search key', 'Feeling'], inplace=True)

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("../Experiments/Datasets/process_data.csv")