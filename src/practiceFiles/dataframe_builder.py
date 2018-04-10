import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
word_list = brown.words()
## List of all words in English. Doesn't contain all words though!
word_set = set(word_list)
filereader = pd.read_csv('./training_set_rel3.csv', encoding="ISO-8859-1")
example_essay = filereader.loc[:,["essay","domain1_score"]]

        

