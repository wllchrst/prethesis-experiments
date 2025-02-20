import pandas as pd
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df_indo = pd.read_csv('../Experiments/Datasets/PRDECT-ID Dataset.csv')

drop_columns = ['Category', 'Product Name', 'Location', 'Price', 'Overall Rating', 'Number Sold', 'Total Review', 'Customer Rating', 'Sentiment', 'Customer Review', 'Emotion']

df_indo['text'] = df_indo['Customer Review']
df_indo['label'] = df_indo['Emotion']
df_indo.drop(columns=drop_columns, inplace=True)
# df_indo['label'] = label_encoder.fit_transform(df_indo['label'])

df_indo = df_indo.sample(frac=1).reset_index(drop=True)
df_indo.to_csv("../Experiments/Datasets/indo-data-review.csv")