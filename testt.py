import pandas as pd
import utils
from sklearn.utils import shuffle
import random

#lst = ['a','b','c']
#dct = {k:v for k,v in enumerate(lst)}
#print(dct)

#df = pd.read_csv('data/test_data.csv')
#sample_idx = random.sample(range(df.shape[0]), 2000)

#print(df.iloc[sample_idx])
#print(sample_idx)
#print(data.shape)
#X_test = data.drop(utils.GENRE_TOP_NAME, axis=1)
#y_test = data[utils.GENRE_TOP_NAME]
#print(y_test)

#['hiphop','rock','alternative']

data = pd.read_csv('data/test_data.csv', index_col=0)
print(data)
print(data.iloc[0])
print(data.iloc[0].to_numpy())
#X_test = data.drop(utils.GENRE_TOP_NAME, axis=1)
#y_test = data[utils.EXTERNAL_VARIABLES_NAMES]
#true_labels = y_test[utils.GENRE_TOP_NAME]
#print(true_labels)
