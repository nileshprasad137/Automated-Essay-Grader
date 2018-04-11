import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from collections import Counter
word_list = brown.words()
##  List of all words in English. Doesn't contain all words though!
word_set = set(word_list)
##  Set of all stopwords in English
stop_words = set(stopwords.words('english'))
##  Complete dataset loaded in filereader
file_reader = pd.read_csv('./training_set_rel3.csv', encoding="ISO-8859-1")
##  Set-1 Essays loaded onto essayreader along with their scores.(Scale 2-12)
essay_reader = file_reader.loc[:1782,["essay","domain1_score"]]

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
normalized_noun_count = []
normalized_adverb_count = []
normalized_verb_count = []
normalized_adjective_count = []
 
count_wrong_spellings = []
count_words = []

for row in essay_reader['essay']:    
    word_tokens = word_tokenize(row)
    
    for w in word_tokens:
        if w not in stop_words:
            if w not in word_set:
                wrong_spellings.append(w)                
            else:
                filtered_essay.append(w)
    
    count_wrong_spellings.append(len(wrong_spellings))
    count_words.append(len(filtered_essay))
    ## reset arrays
    filtered_essay = []
    wrong_spellings = []       

## These two add two new columns in essayreader dataframe
essay_reader['wrong_spellings'] = count_wrong_spellings
essay_reader['word_count'] = count_words

for row in essay_reader['essay']:
    word_tokens = word_tokenize(row)
    tags = nltk.pos_tag(word_tokens)
    counts = Counter(tag for word,tag in tags)
    total_pos = sum(counts.values()) 
    normalized_noun_count = round(((counts['NN']+counts['NNS']+counts['NNP']+counts['NNPS'])/total_pos)*100)    
    normalized_adjective_count = round(((counts['JJ']+counts['JJR']+counts['JJS'])/total_pos)*100)
    normalized_verb_count = round(((counts['VB']+counts['VBD']+counts['VBG']+counts['VBN']+counts['VBP']+counts['VBZ'])/total_pos)*100)
    normalized_adverb_count = round(((counts['RB']+counts['RBR']+counts['RBS'])/total_pos)*100)
    print(str(normalized_noun_count) + " " + str(normalized_adjective_count) + " " + str(normalized_verb_count) + " " + str(normalized_adverb_count))
    print("\n\n")
    