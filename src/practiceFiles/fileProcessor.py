# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:00:24 2018

@author: Nilesh Prasad
"""

import sys
import pandas as pd
import numpy as np
#reload(sys)
#sys.setdefaultencoding("ISO-8859-1")
filereader = pd.read_csv('./training_set_rel3.csv', encoding="ISO-8859-1")
#a.encode('utf-8').strip()
#print(sys.getdefaultencoding())
print(filereader['essay'][1])