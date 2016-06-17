# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 19:50:23 2016

@author: can
"""

import pandas as pd


products = pd.read_csv('input/producto_tabla.csv')
products.info()
products['grams'] = products.NombreProducto.str.extract('.* (\d+)g.*', expand=False)
products['ml'] = products.NombreProducto.str.extract('.* (\d+)ml.*', expand=False)
products['inches'] = products.NombreProducto.str.extract('.* (\d+)in.*', expand=False)
products['pct'] = products.NombreProducto.str.extract('.* (\d+)pct.*', expand=False)
products['pieces'] = products.NombreProducto.str.extract('.* (\d+)p.*', expand=False)
products['bim'] = products.NombreProducto.str.contains('bim', case=False)
products['mla'] = products.NombreProducto.str.contains('mla', case=False)
products['mta'] = products.NombreProducto.str.contains('mta', case=False)
products['tab'] = products.NombreProducto.str.contains('tab', case=False)
products['pan'] = products.NombreProducto.str.contains('pan', case=False)
products['mtb'] = products.NombreProducto.str.contains('mtb', case=False)
products['lar'] = products.NombreProducto.str.contains('lar', case=False)
products['gbi'] = products.NombreProducto.str.contains('gbi', case=False)
products['won'] = products.NombreProducto.str.contains('won', case=False)
products['duo'] = products.NombreProducto.str.contains('duo', case=False)
products['tubo'] = products.NombreProducto.str.contains('tubo', case=False)
products['tr'] = products.NombreProducto.str.contains('tr', case=False)
products['cu'] = products.NombreProducto.str.contains('cu', case=False)
products['sp'] = products.NombreProducto.str.contains('sp', case=False)
products['prom'] = products.NombreProducto.str.contains('prom', case=False)
products['fresa'] = products.NombreProducto.str.contains('fresa', case=False)
products['dh'] = products.NombreProducto.str.contains('dh', case=False)
products['vainilla'] = products.NombreProducto.str.contains('vainilla', case=False)
products['deliciosas'] = products.NombreProducto.str.contains('deliciosas', case=False)
products['blanco'] = products.NombreProducto.str.contains('blanco', case=False)
products['chocolate'] = products.NombreProducto.str.contains('chocolate', case=False)


#labels = products.NombreProducto.str.extract('([^\d]+) \d+.*', expand=False)
pr = pd.concat([products.drop('NombreProducto', axis=1)], axis=1)
pr.info()
pr.to_csv('input/products_features.csv', index=False)


#[('g', 2458), ('p', 1128), ('bim', 679), ('mla', 657), ('mta', 545), ('tab', 269),
# ('tr', 258), ('cu', 246), ('pan', 219), ('mtb', 208), ('sp', 205), ('prom', 197),
# ('lar', 182), ('fresa', 152), ('gbi', 130), ('won', 117), ('duo', 101),
# ('tubo', 100), ('dh', 95), ('vainilla', 94), ('deliciosas', 92), ('blanco', 91),
# ('chocolate', 90), ('tnb', 85), ('tira', 84), ('lon', 83), ('cj', 81), 
#('gansito', 80), ('suavicremas', 69), ('san', 67), ('galleta', 66), ('nuez', 65),
# ('mr', 64), ('multigrano', 63), ('pina', 61), ('tortilla', 60), ('tostada', 59),
# ('barritas', 59), ('frut', 58), ('bran', 57), ('cr', 57), ('lata', 56), 
#('me', 55), ('principe', 55), ('mantecadas', 53), ('pct', 52), ('mini', 52), 
#('triki', 51), ('kc', 51), ('tostado', 50), ('mg', 49), ('ml', 49), ('roles', 48),
# ('integral', 48), ('bollos', 46), ('tortillinas', 45), ('bimbollos', 45), 
#('oro', 44), ('medias', 44), ('canelitas', 44), ('noches', 43), ('fibra', 43),
# ('cc', 42), ('con', 41), ('barra', 41), ('cjm', 40), ('de', 40), ('hna', 39),
# ('mi', 37), ('super', 36), ('tartinas', 35), ('trakes', 34), ('mas', 33), 
#('kg', 33), ('surtido', 33), ('sl', 32), ('bar', 32), ('canela', 31), ('avena', 31)
#, ('nito', 30), ('in', 29), ('chocochispas', 28), ('sandwich', 28), 
#('plativolos', 28), ('choco', 28), ('doble', 28), ('pack', 27), ('submarinos', 26),
# ('salmas', 25), ('silueta', 24), ('hot', 24), ('maiz', 24), ('clasico', 24),
# ('donas', 24), ('linaza', 24), ('lors', 23), ('harina', 23), ('bco', 23),
# ('wonder', 22), ('panera', 22)]