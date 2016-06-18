# coding: utf-8

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#output_notebook()

def get_product_agg(cols):
    df_train = pd.read_csv('input/train_sample_10000.csv', usecols = ['Cliente_ID','Producto_ID'] + cols,
                           dtype  = {'Cliente_ID': 'int32',
                                     'Producto_ID':'int32',
                                     'Demanda_uni_equil':'float32'})
    agg  = df_train.groupby(['Cliente_ID','Producto_ID'], as_index=False).agg(['mean'])
    agg.columns  =  ['_'.join(col).strip() for col in agg.columns.values]
    del(df_train)
    return agg

def get_product_agg_test(cols):
    df_train = pd.read_csv('input/test.csv', usecols = ['Cliente_ID','Producto_ID'] + cols,
                           dtype  = {'Cliente_ID': 'int32',
                                     'Producto_ID':'int32',
                                     'Demanda_uni_equil':'float32'})
    agg  = df_train.groupby(['Cliente_ID','Producto_ID'], as_index=False).agg(['mean'])
    agg.columns  =  ['_'.join(col).strip() for col in agg.columns.values]
    del(df_train)
    return agg

agg1 = get_product_agg(['Demanda_uni_equil'])
agg2 = get_product_agg_test(['Demanda_uni_equil'])




products  =  pd.read_csv("input/producto_tabla.csv")
products['short_name'] = products.NombreProducto.str.extract('^(\D*)', expand=False)
products['brand'] = products.NombreProducto.str.extract('^.+\s(\D+) \d+$', expand=False)
w = products.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
products['weight'] = w[0].astype('float')*w[1].map({'Kg':1000, 'g':1})
products['pieces'] =  products.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')

products.short_name.value_counts(dropna=False)

from nltk.corpus import stopwords
print(stopwords.words("spanish"))

products['short_name_processed'] = (products['short_name']
                                        .map(lambda x: " ".join([i for i in x.lower()
                                                                 .split() if i not in stopwords.words("spanish")])))

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("spanish")

print(stemmer.stem("Tortillas"))

products['short_name_processed'] = (products['short_name_processed']
                                        .map(lambda x: " ".join([stemmer.stem(i) for i in x.lower().split()])))

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 1000) 

product_bag_words = vectorizer.fit_transform(products.short_name_processed).toarray()


product_bag_words = pd.concat([products.Producto_ID, 
                               pd.DataFrame(product_bag_words, 
                                            columns= vectorizer.get_feature_names(), index = products.index)], axis=1)

product_bag_words.drop('Producto_ID', axis=1).sum().sort_values(ascending=False).head(100)

df = (pd.merge(agg1.reset_index(), products, on='Producto_ID', how='left').
      groupby('short_name')['Demanda_uni_equil_sum'].sum().sort_values(ascending=False))
