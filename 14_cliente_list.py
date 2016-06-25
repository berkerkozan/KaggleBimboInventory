import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk.corpus
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model

df_cliente = pd.read_csv('../input/cliente_tabla.csv')

cliente_id_list = df_cliente['Cliente_ID']

df_cliente.describe()

cliente_id_list = list(set(cliente_id_list))

len(cliente_id_list)