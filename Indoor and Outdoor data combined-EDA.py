#!/usr/bin/env python
# coding: utf-8

# In[1698]:


import pandas as pd
from pandas import Series
import numpy as np
from numpy import mean
import scipy
import math
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from imblearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, LeavePOut, ShuffleSplit, LeaveOneOut,RandomizedSearchCV, KFold,StratifiedKFold


# In[1699]:


# Load dataset

Outdoor = pd.read_csv('C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\outdoor_activity.csv')
del Outdoor['Unnamed: 0']
print(len(Outdoor))
Indoor= pd.read_csv('C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\indoor_activity.csv')
del Indoor['Unnamed: 0']
print(len(Indoor))


df=pd.DataFrame()
for i in (1,2,3,4,6,7):
    User1 = Outdoor.loc[Outdoor["tagId"]==i]
    User1['Day'] = pd.DatetimeIndex(User1['datetime']).day
    a = (User1['cluster_name'].value_counts()/np.float(len(User1)))*100
    a=a.to_frame()
    a=a.reset_index()
    a = a.assign(User=i)
    if i==1:
        a = a.assign(UCLA=3)
        a = a.assign(Lubben=12)
    elif i==2:
        a = a.assign(UCLA=3)
        a = a.assign(Lubben=14)
    elif i==3:
        a = a.assign(UCLA=4)
        a = a.assign(Lubben=28)
    elif i==4:
        a = a.assign(UCLA=6)
        a = a.assign(Lubben=18)
    elif i==6:
        a = a.assign(UCLA=7)
        a = a.assign(Lubben=8)
    elif i==7:
        a = a.assign(UCLA=7)
        a = a.assign(Lubben=9)
    df = df.append(a, ignore_index = True)
df.rename(columns={"index":"cluster","cluster_name":"Average_percentage_of_timespent"},inplace=True)
print(df)



import seaborn as sns
sns.set(rc={'figure.figsize':(8,6)})
df['cluster'] = df.cluster.map( {'Keinupuitso_nearbyarea':'Place1' , 'Hervantakeskus':'Place10' ,'Other_area':'Place11','PRISMA':'Place2','Lidl_lakalaiva':'Place3','Peltolammi':'Place6','Lielahti':'Place4','Kaukajärvi':'Place5','Tesoma':'Place7','Hervanta':'Place8','Kauppi':'Place9','Kaleva':'Place12','Ratina':'Place14','Keskustori':'Place13'})
df['User'] = pd.Series(df['User'], dtype="string")
print(df)

sns.set(rc={'figure.figsize':(8,6)})
sns.set(font_scale=2)
sns.factorplot(x='User', y='Average_percentage_of_timespent',hue='cluster',
                        size=8,  aspect=1.5,
                        kind='bar', 
                        data=df, palette='muted')
plt.title("Time spent in Outdoors by Users")
plt.show()


df=pd.DataFrame()
for i in (1,2,3,4,7):
    User1 = Indoor.loc[Indoor["tagId"]==i]
    User1['Date'] = pd.DatetimeIndex(User1['time']).day
    a = (User1['zone_name'].value_counts()/np.float(len(User1)))*100
    a=a.to_frame()
    a=a.reset_index()
    a = a.assign(User=i)
    if i==1:
        a = a.assign(UCLA=3)
        a = a.assign(Lubben=12)
    elif i==2:
        a = a.assign(UCLA=3)
        a = a.assign(Lubben=14)
    elif i==3:
        a = a.assign(UCLA=4)
        a = a.assign(Lubben=28)
    elif i==4:
        a = a.assign(UCLA=6)
        a = a.assign(Lubben=18)
    elif i==6:
        a = a.assign(UCLA=7)
        a = a.assign(Lubben=8)
    elif i==7:
        a = a.assign(UCLA=7)
        a = a.assign(Lubben=9)
    df = df.append(a, ignore_index = True)
df.rename(columns={"index":"cluster","zone_name":"Average_percentage_of_timespent"},inplace=True)
print(df)



import seaborn as sns
df['cluster'] = df.cluster.map( {'cafeteria':'Place1' , 'paivatoiminta':'Place3' ,'gymnasium':'Place4','no activity':'Place2'})
df['User'] = pd.Series(df['User'], dtype="string")
print(df)


sns.set(rc={'figure.figsize':(8,6)})
sns.set(font_scale=2)
sns.factorplot(x='User', y='Average_percentage_of_timespent',hue='cluster',
                        size=8,  aspect=1.5,
                        kind='bar', 
                        data=df, palette='muted')
#plt.xlabel('User', fontsize=25)
#plt.ylabel('Average_percentage_of_timespent', fontsize=25)
plt.title("Time spent in Indoors by Users")
#plt.legend(loc='upper left',bbox_to_anchor =(1.00, 1.00),fontsize='x-large')
plt.show()

