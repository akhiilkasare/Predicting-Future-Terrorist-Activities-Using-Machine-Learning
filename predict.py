import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cleaned_data.csv', index_col=False)
df = df.drop('Unnamed: 0', axis=1)

x = df.drop(['attacktype1'], axis=1)
y = df['attacktype1']


Pkl_Filename = "/home/akhil/Downloads/machine_learning/gtd/Pickle_RL_Model.pkl"  

# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(rf, file)

with open(Pkl_Filename, 'rb') as file:  
    Pickled_RF_Model = pickle.load(file)

y_pred = Pickled_RF_Model.predict([[6,1,2021,1,13,1,15,1565,32086,12,179,1944,
                                   58,0,58,0,1,1]])

def predict(text):
    if text == 1:
        return 'Assassination'
    elif text == 2:
        return 'Armed assault'
    elif text == 3:
        return 'Bombing/explosion'
    elif text == 4:
        return 'Hijacking'
    elif text == 5:
        return 'Hostage taking (barricade incident)'
    elif text == 6:
        return 'Hostage taking (kidnapping)'
    elif text == 7:
        return 'Facility/infrastructure attack'
    elif text == 8:
        return 'Unarmed assaults'
    else:
        return 'Unknown'
    
print("The Future Terrorist Activity will be of Type  : ", predict(y_pred))

