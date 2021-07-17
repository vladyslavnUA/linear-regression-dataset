import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 
from sklearn.datasets import load_boston
boston_dataset = load_boston()

dict_keys(['data', 'target', 'feature_names', 'DESCR'])

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()