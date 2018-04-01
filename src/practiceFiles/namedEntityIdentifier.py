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
example_essay = filereader['essay'][1]
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example_essay)
filtered_essay = [w for w in word_tokens if not w in stop_words]
filtered_essay = []
wrong_spellings = []
for w in word_tokens:
    if w not in stop_words:
        filtered_essay.append(w)
for w in word_tokens:
    if w not in word_set:
        wrong_spellings.append(w)
#print(word_tokens)
#print(set(filtered_essay))
#print(set(wrong_spellings))

## Now, find most frequent words.

##Build a freq distribution of words
all_words = nltk.FreqDist(filtered_essay)
print(all_words.most_common(15))
        

