import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
word_list = brown.words()
##  List of all words in English. Doesn't contain all words though!
word_set = set(word_list)
##  Set of all stopwords in English
stop_words = set(stopwords.words('english'))
##  Complete dataset loaded in filereader
filereader = pd.read_csv('./training_set_rel3.csv', encoding="ISO-8859-1")
##  Set-1 Essays loaded onto essayreader along with their scores.(Scale 2-12)
essayreader = filereader.loc[:1782,["essay","domain1_score"]]

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
for row in essayreader['essay']:    
    word_tokens = word_tokenize(row)
    #for w in word_tokens:
     #   if w not in word_set:
      #      wrong_spellings.append(w)       
        
    for w in word_tokens:
        if w not in stop_words:
            if w not in word_set:
                wrong_spellings.append(w)                
            else:
                filtered_essay.append(w)
    
    print(len(wrong_spellings))
    ## reset array
    wrong_spellings = []
    print("\n\n")
    #print(filtered_essay)
    #print("\n\n")
    #print(row)
    #print("\n\n\n")
        

