import numpy as np
from numpy import *
import xgboost as xgb
import pandas as pd
import math

import nltk.corpus
from nltk.stem.snowball import SnowballStemmer

from sklearn.cross_validation import train_test_split

from ml_metrics import rmsle

import scipy.sparse as sp



print ('')
print ('Loading Data...')

def evalerror(preds, dtrain):

    labels = dtrain.get_label()
    assert len(preds) == len(labels)
    labels = labels.tolist()
    preds = preds.tolist()
    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0,preds[i]) + 1)) ** 2.0 for i,pred in enumerate(labels)]
    return 'error', (sum(terms_to_sum) * (1.0/len(preds))) ** 0.5

train = pd.read_csv('../input/train_sample_100.csv')
small_temp_test = pd.read_csv('../input/test_sample_100.csv')
temp_test = pd.read_csv('../input/test.csv', chunksize=100000)


# Process product data.

products = pd.read_csv("../input/producto_tabla.csv")

print('product csv read complete')

products['short_name'] = products.NombreProducto.str.extract('^(\D*)', expand=False)
# products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', expand=False)
# w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
# products['weight'] = w[0].astype('float') * w[1].map({'Kg': 1000, 'g': 1})
products['pieces'] = products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')

products['short_name_processed'] = (products['short_name'].map(
    lambda x: " ".join([i for i in x.lower().split() if i not in nltk.corpus.stopwords.words("spanish")])))
stemmer = SnowballStemmer("spanish")
products['short_name_processed'] = (
products['short_name_processed'].map(lambda x: " ".join([stemmer.stem(i) for i in x.lower().split()])))

short_name_processed_list = products['short_name_processed'].unique()


products = pd.concat([products.drop(['short_name', 'short_name_processed', 'NombreProducto'], axis=1), pd.get_dummies(short_name_processed_list)], axis=1)

print('product features generated')

# Join data and products
train = train.join(products, on='Producto_ID', lsuffix='_t')
train.fillna(value=0, inplace=True)

print('train data joined')



# test = pd.DataFrame()



print('real test created')

count = 0
for chunk in temp_test:


    df_chunk = pd.DataFrame(chunk)
    df_chunk = df_chunk.join(products, on='Producto_ID', lsuffix='_t')
    # array_chunk = array(df_chunk)
    csr_chunk = sp.csr_matrix(df_chunk)
    # csr_chunk = sps.csr_matrix(array_chunk)
    # csr_chunk = sps.csr_matrix((df_chunk, indices, indptr), [shape = (M, N)])
    if count == 0:
        test = csr_chunk
    else:
        test = sp.vstack((test, csr_chunk), format='csr')

    # test = pd.concat([test, df_chunk], axis=1)
    #test = test.append(df_chunk, ignore_index=True)
    count += 1
    print('test shape', test.shape)
    print('chunk shape', df_chunk.shape)

print('test data joined')
print('test shape', test.shape)
print('chunk shape', df_chunk.shape)


print('product features added to table')


print ('')
print ('Training_Shape:', train.shape)

ids = test['id']
test = test.drop(['id'],axis = 1)

y = train['Demanda_uni_equil']
X = train[test.columns.values]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1729)

print ('Division_Set_Shapes:', X.shape, y.shape)
print ('Validation_Set_Shapes:', X_train.shape, X_test.shape)

params = {}
params['objective'] = "reg:linear"
params['eta'] = 0.025
params['max_depth'] = 5
params['subsample'] = 0.8
params['colsample_bytree'] = 0.6
params['silent'] = True

print ('')

test_preds = np.zeros(test.shape[0])
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test)

watchlist = [(xg_train, 'train')]
num_rounds = 100

xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, feval = evalerror, early_stopping_rounds= 20, verbose_eval = 10)
preds = xgclassifier.predict(xg_test, ntree_limit=xgclassifier.best_iteration)

print ('RMSLE Score:', rmsle(y_test, preds))

fxg_test = xgb.DMatrix(test)
fold_preds = np.around(xgclassifier.predict(fxg_test, ntree_limit=xgclassifier.best_iteration), decimals = 1)
test_preds += fold_preds

submission = pd.DataFrame({'id':ids, 'Demanda_uni_equil': test_preds})
submission.to_csv('submission.csv', index=False)

print ('done')