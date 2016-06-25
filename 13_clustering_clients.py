import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk.corpus
from nltk.stem.snowball import SnowballStemmer



df_train = pd.read_csv('../input/train.csv')
df_train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId', \
                    'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek', \
                    'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']
prod_client_median = df_train.groupby(['ProductId', 'ClientId']).agg({'AdjDemand': np.median})

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

products = pd.concat([products.drop('short_name', axis=1), pd.get_dummies(short_name_processed_list)], axis=1)

prod_client_median = prod_client_median.join(products, on='Producto_ID', lsuffix='_t')

prod_client_median.to_csv('../input/client_prod_median.csv')
