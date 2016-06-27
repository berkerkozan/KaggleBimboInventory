import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk.corpus
from nltk.stem.snowball import SnowballStemmer
import scipy.sparse as sps
from sklearn  import preprocessing


df_train = pd.read_csv('../input/train_sample_100.csv')

prod_client_totals = df_train.groupby(['Cliente_ID', 'Producto_ID']).agg({'Demanda_uni_equil': np.sum})

prod_client_totals = pd.DataFrame(prod_client_totals.reset_index())

cliente_list = list(set(df_train['Cliente_ID']))
producto_list = list(set(df_train['Producto_ID']))

prod_client_totals = pd.concat([prod_client_totals.get_dummies(producto_list)], axis=1)


sparse_client_totals = sps.csr_matrix(prod_client_totals)  # todo onehot encoder conversion to create product id columns

enc = preprocessing.OneHotEncoder()

sparse_client_totals.getcol(0)

# zip(X.row, X.col, X.data)[:10]

cliente_encoded = enc.fit(sparse_client_totals.getcol(0).toarray())



# client_matrix = pd.concat([prod_client_totals.drop('Producto_ID', axis = 1), pd.get_dummies(prod_client_totals['Producto_ID'])], axis=1)




# client_matrix.iloc[:,2:] = (client_matrix['Demanda_uni_equil']*client_matrix.iloc[:,2:].T).T
#
# client_matrix.drop('Demanda_uni_equil', axis=1, inplace=True)
#
# print(client_matrix.head())


