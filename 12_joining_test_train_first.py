import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk.corpus
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model
import scipy.sparse as sps

# Read and join train and test data.

df_train = pd.read_csv('../input/train_sample_10000.csv')
df_test = pd.read_csv('../input/test_sample_100.csv')

df_test_train = df_train.append(df_test)

print('train and test read and joined')


# Process product data.

products = pd.read_csv("../input/producto_tabla.csv")

print('product csv read complete')

products['short_name'] = products.NombreProducto.str.extract('^(\D*)', expand=False)
products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', expand=False)
w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
products['weight'] = w[0].astype('float') * w[1].map({'Kg': 1000, 'g': 1})
products['pieces'] = products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')

products['short_name_processed'] = (products['short_name'].map(
    lambda x: " ".join([i for i in x.lower().split() if i not in nltk.corpus.stopwords.words("spanish")])))
stemmer = SnowballStemmer("spanish")
products['short_name_processed'] = (
products['short_name_processed'].map(lambda x: " ".join([stemmer.stem(i) for i in x.lower().split()])))

short_name_processed_list = products['short_name_processed'].unique()


products = pd.concat([products.drop('short_name', axis=1), pd.get_dummies(short_name_processed_list)], axis=1)

print('product features generated')

# Join data and products
df_test_train = df_test_train.join(products, on='Producto_ID', lsuffix='_t')
df_test_train.fillna(value=0, inplace=True)

print('product features added to table')

cliente_id_list = df_test_train['Cliente_ID']
df_test_train = pd.concat([df_test_train.drop('Cliente_ID', axis=1), pd.get_dummies(cliente_id_list)], axis=1)

print('client ids added to table')

# agencia_id_list = df_test_train['Agencia_ID']
# df_test_train = pd.concat([df_test_train.drop('Agencia_ID', axis=1), pd.get_dummies(agencia_id_list)], axis = 1)
#
# print('agencia ids added to table')

columns = list(short_name_processed_list) + list(cliente_id_list)

# extract train and test back from test_train df
df_train = (df_test_train.loc[df_test_train['id'] <= 0])
df_test = (df_test_train.loc[df_test_train['id'] > 0])

print('test and train split back')

labels = df_train['Demanda_uni_equil'].values
features = df_train[columns].values

regr = linear_model.LinearRegression()
lin_model = regr.fit(features, labels)

print('model built')
test_features = df_test[columns].values
prediction = lin_model.predict(test_features)

print('model applied')

pred_int = prediction.astype(int)
pred_df = pd.DataFrame({'Demanda_uni_equil': pred_int})

pred_df = pred_df.fillna(value=2)
pred_df = pred_df.replace(to_replace=0, value=2)
pred_df = pred_df.replace(to_replace=-1, value=2)

pred_df.to_csv("../output/100sample_test.csv", index=False)

print('written to file')



print('done!')
