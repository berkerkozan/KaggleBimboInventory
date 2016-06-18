import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk.corpus
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model

# Process product data.

products = pd.read_csv("../input/producto_tabla.csv")
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

vectorizer = CountVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None, \
                             max_features=1000)

products = pd.concat([products.drop('short_name', axis=1), pd.get_dummies(short_name_processed_list)], axis=1)

# Process training data.
df_train = pd.read_csv('../input/train_sample_100.csv')

print("train data read")
# Make client ID's feature
cliente_id_list = df_train['Cliente_ID']  ## todo client id should be obtained from client_tabla instead.
# agencia_id_list = df_train['Agencia_ID']  ## todo this can be obtained from the town_state if needed.

# df_train = pd.concat([df_train.drop('Agencia_ID', axis=1), pd.get_dummies(agencia_id_list)],axis=1)
df_train = pd.concat([df_train.drop('Cliente_ID', axis=1), pd.get_dummies(cliente_id_list)], axis=1)

# Join train and products
df_train = df_train.join(products, on='Producto_ID', lsuffix='_t')
df_train.fillna(value=0, inplace=True)

print("train data joined")

# Get test data.
# df_test = pd.read_csv('../input/test_sample_100.csv')
# df_test = pd.concat([df_test.drop('Agencia_ID', axis=1), pd.get_dummies(agencia_id_list)],axis=1)
# df_test = pd.concat([df_test.drop('Cliente_ID', axis=1), pd.get_dummies(cliente_id_list)],axis=1)
# df_test = df_test.join(products, on='Producto_ID', lsuffix='_t')
# df_test.fillna(value=0, inplace=True)


columns = list(short_name_processed_list) + list(cliente_id_list)

labels = df_train['Demanda_uni_equil'].values
features = df_train[columns].values

# test_features = df_test[list(columns)].values

# Fit models
regr = linear_model.LinearRegression()

# Train the model using the training sets
our_wonderful_model = regr.fit(features, labels)

print("model built")

pred_df = pd.DataFrame(columns=['id', 'Demanda_uni_equil'])
print(pred_df.columns)

# Load, join & predict test.csv line by line. todo something is wrong. can't write the predictions to teh pred_df.
for df_test in pd.read_csv('../input/test_sample_1000.csv', chunksize=100):
    df_test = pd.concat([df_test.drop('Cliente_ID', axis=1), pd.get_dummies(cliente_id_list)], axis=1)
    df_test = df_test.join(products, on='Producto_ID', lsuffix='_t')
    df_test.fillna(value=0, inplace=True)
    test_features = df_test[columns].values
    prediction = our_wonderful_model.predict(test_features)
    pred_int = prediction.astype(int)
    print("id",df_test['id'])
    pred_df_tmp = pd.DataFrame({'id': df_test['id'], 'Demanda_uni_equil': pred_int})
    print('pred_df_tmp', pred_df_tmp)
    pred_df.append(pred_df_tmp)
    print(pred_df)

# prediction = our_wonderful_model.predict(test_features)
# pred_int = prediction.astype(int)
# pred_df = pd.DataFrame({'Demanda_uni_equil': pred_int})

pred_df = pred_df.fillna(value=2)
pred_df = pred_df.replace(to_replace=0, value=2)
pred_df = pred_df.replace(to_replace=-1, value=2)

#result = pd.concat([df_test['id'], pred_df], axis=1)
pred_df.to_csv("../output/cliente_featurized_test_1000.csv", index=False)
