# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 11:33:10 2016

@author: can
"""

import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.preprocessing import OneHotEncoder

from subprocess import check_output
import nltk
from nltk.book import *
from nltk.collocations import *


import re
file = open('input/producto_tabla.csv', 'r')
# .lower() returns a version with all upper case characters replaced with lower case characters.
text = file.read().lower()
file.close()
# replaces anything that is not a lowercase letter, a space, or an apostrophe with a space:
text = re.sub('[^a-z\ \']+', " ", text)

words = nltk.tokenize.word_tokenize(text)
fdist = FreqDist(words)

print(fdist.most_common(100))

#bigram_measures = nltk.collocations.BigramAssocMeasures()
#
#finder = BigramCollocationFinder.from_words(words)
#finder.apply_freq_filter(3)
#print finder.nbest(bigram_measures.pmi, 100)