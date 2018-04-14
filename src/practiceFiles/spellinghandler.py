import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import brown
from collections import Counter
import matplotlib.pyplot as plt
import re

word_list = []
with open('./words_alpha.txt') as f:
    for line in f:
        word = line.split(None, 1)[0]
        word_list.append(word.lower())
        
word_set = set(word_list)
stop_words = set(stopwords.words('english'))
file_reader = pd.read_csv('./training_set_rel3.csv', encoding="ISO-8859-1")
##  Set-1 Essays loaded onto essayreader along with their scores.(Scale 2-12)
#essay_reader = file_reader.loc[:1782,["essay"]]
#training_df = file_reader.loc[:1782,["domain1_score"]]
essay_reader = file_reader['essay'][12297]


##  Write regular expressions to match Named Entity Recognition
##  The entitities identified by NER are: 
##  "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT"
##  Other replacements made: 
##  "MONTH" (any month name not tagged as a date by the NER), 
##  "EMAIL" (anything that looks like an e-mail address), 
##  "NUM" (word containing digits or non-alphanumeric symbols), 
##   and "CAPS" (any capitalized word that doesn't begin a sentence, except in 
##  essays where more than 20% of the characters are capitalized letters), 
##  "DR" (any word following "Dr." with or without the period, with any capitalization, 
##  that doesn't fall into any of the above), 
##  "CITY" and "STATE" (various cities and states).

## For each essay this array has to be different. Thsese two arrays needs to be reset
##  after each iteration.  
filtered_essay = []
wrong_spellings = []
 
count_wrong_spellings = []
count_words = []  
#word_tokens = word_tokenize(row)
tokenizer = RegexpTokenizer(r'\w+')
ner = re.findall("[@]\w+", essay_reader)
temp = []
for item in ner:
    temp.append(item[1:])
ner_set = set(temp)
word_tokens = tokenizer.tokenize(essay_reader)
word_tokens = [e for e in word_tokens if e not in ner_set]
for w in word_tokens:
    if w not in stop_words:
        if w.lower() not in word_set:
            wrong_spellings.append(w)                
        else:
            filtered_essay.append(w)
   

