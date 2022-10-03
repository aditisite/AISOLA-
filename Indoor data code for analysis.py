#!/usr/bin/env python
# coding: utf-8

# Import basic libraries
import re
import pandas as pd
import json
import datetime
import glob
import os
import numpy as np

# Create Dataframe
pos_data = pd.DataFrame()


#Loading all the files

dir = "C:\\Users\\chadsi\\Downloads\\OneDrive_1_3-21-2022\\"
all_files = glob.glob(dir + "/*.txt")
for filename in all_files:
    print(filename)
    a_file = open(filename)
    file = os.path.basename(filename)
    file = file.rsplit( ".", 1 )[0]
    file = file.split('_')[1]
    replacement = ''
    date = file.replace(file[0:4], replacement)
    data = []
    lines = a_file.readlines()
    a =0
    for line in lines:
        line = line.replace("\n", "")
        l = re.sub(r'^.*?{', '{', line)
        l = l[:-1]
        st = []
        for match in re.finditer('version', l):
            st.append(match.start())
        if len(st)>1:
            for i in range(len(st)):
                if i==len(st)-1:
                    b = l[st[i]-2:]
                else:
                    b = l[st[i]-2:st[i+1]-3]
                val = json.loads(b)
                data.append(val)
        else:
            val = json.loads(l)
            data.append(val)
        a = a+1
    df = pd.DataFrame(data)
    df = df.assign(Date=date)
    pos_data = pos_data.append(df, ignore_index = True)
pos_data


# Removing all entries where success is not true
pos_data.drop(pos_data.loc[pos_data['success']==False].index, inplace=True)
pos_data = pos_data.reset_index()
del pos_data['index']
pos_data


# Transforming json entries into dataframe

pos_data["data"].apply(pd.Series)
pos_data = pd.concat([pos_data, pos_data["data"].apply(pd.Series)], axis=1)
pos_data.rename(columns={"timestamp":"time"},inplace=True)

##############################################################################

del pos_data['data']
pos_data["coordinates"].apply(pd.Series)
pos_data = pd.concat([pos_data, pos_data["coordinates"].apply(pd.Series)], axis=1)

###############################################################################

del pos_data['coordinates']
pos_data["tagData"].apply(pd.Series)
pos_data = pd.concat([pos_data, pos_data["tagData"].apply(pd.Series)], axis=1)

##############################################################################

del pos_data["tagData"]
pos_data['val'] = pos_data["zones"].apply(pd.Series)
pos_data["val"].apply(pd.Series)
pos_data = pd.concat([pos_data, pos_data["val"].apply(pd.Series)], axis=1)
del pos_data["zones"]

#############################################################################

del pos_data['val']
pos_data["metrics"].apply(pd.Series)
pos_data = pd.concat([pos_data, pos_data["metrics"].apply(pd.Series)], axis=1)

#############################################################################
pos_data


#Selecting columns which are necessary

selected_columns = pos_data[["tagId","time","Date","x","y","z","accelerometer","status","id","name"]]
pos_data_ = selected_columns.copy()
pos_data_.status = pos_data_.status.astype(int)
pos_data_.rename(columns={"x":"tag_x_coordinate","y":"tag_y_coordinate","z":"tag_z_coordinate","id":"zone_id","name":"zone_name"},inplace=True)
pos_data_['zone_name'].isnull().sum()
pos_data_.zone_name = pos_data_.zone_name.fillna('no activity')
pos_data_['zone_tag'] = pos_data_['zone_name'] 
pos_data_.zone_tag = pos_data_.zone_tag.map( {'no activity':0 , 'gymnasium':1, 'cafeteria':2, 'paivatoiminta':3} )
pos_data_['status_id'] = pos_data_['status']
pos_data_.status = pos_data_.status.map( {0: 'Neutral',1:'Happy', 2:'Sad'} )
pos_data_


# Changing certain datatypes

pos_data_['tagId'] = pd.Series(pos_data_['tagId'], dtype="string")
pos_data_['zone_name'] = pd.Series(pos_data_['zone_name'], dtype="string")
pos_data_['status_id'] = pd.Series(pos_data_['status_id'], dtype="category")
pos_data_['zone_tag'] = pd.Series(pos_data_['zone_tag'], dtype="category")
pos_data_


# Checking if there is any null value

del pos_data_['zone_id']
null_values = pos_data_.isnull().sum()
print(null_values)
#pos_data_ = pos_data_.dropna()


# Preprocessing accelerometer data

pos_data_[['val1','val2','val3','val4','val5','val6','val7']]= pos_data_["accelerometer"].apply(pd.Series)
pos_data_[['acc_x','acc_y','acc_z']]= pos_data_["val1"].apply(pd.Series)
pos_data_

del pos_data_['accelerometer']
del pos_data_['val1']
del pos_data_['val2']
del pos_data_['val3']
del pos_data_['val4']
del pos_data_['val5']
del pos_data_['val6']
del pos_data_['val7']
pos_data_

pos_data_.sort_values(by=['tagId'])

pos_data_ = pos_data_.drop(pos_data_[pos_data_.tagId == '5'].index)
pos_data_

pos_data_ = pos_data_.reset_index()
del pos_data_['index']
pos_data_

pos_data_.time = pos_data_.time.apply(lambda d: datetime.datetime.fromtimestamp(int(d)).strftime('%Y-%m-%d %H:%M:%S'))
pos_data_
pos_data_ = pos_data_.sort_values("time")


#Calculating indoor speed and distance travelled by user

#User 1

User1 = pos_data_.loc[pos_data_["tagId"]=='1']
User1 = User1.reset_index()
print(User1)
del User1['index']

User1F = pd.DataFrame()
selected_columns = User1[["time","tag_x_coordinate","tag_y_coordinate","tag_z_coordinate"]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

User1S.tag_x_coordinate = User1S.tag_x_coordinate.apply(lambda d: d/1000)
User1S.tag_y_coordinate = User1S.tag_y_coordinate.apply(lambda d: d/1000)
User1S.tag_z_coordinate = User1S.tag_z_coordinate.apply(lambda d: d/1000)

User1S.time = pd.to_datetime(User1S.time, format='%Y-%m-%d %H:%M:%S')
diff = User1S.diff()

diff.time = diff.time.apply(lambda d: d.total_seconds())
print(diff.time)

coords = [c for c in User1S.columns if not 'time' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['time']
velocity1[velocity1==np.inf] = 0.0

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User1F = pd.concat([User1, df, df1], axis=1)
User1F.Estimated_Speed = User1F.Estimated_Speed.round(2)
#User1F.Estimated_Speed[User1F.Estimated_Speed > 1.5] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
User1F.time = pd.to_datetime(User1F.time, format='%Y-%m-%d %H:%M:%S')
#plt.scatter(User1F['time'],User1F['Estimated_Speed'])
plt.plot(User1F['Estimated_Speed'], 'o',label = 'Estimated_Speed')
#plt.plot(velocity1, 'x')
plt.title('Estimated speed for User1')
plt.xticks(rotation=45)
plt.legend()
plt.show()


#User 2

User2 = pos_data_.loc[pos_data_["tagId"]=='2']
User2 = User2.reset_index()
print(User2)
del User2['index']

User2F = pd.DataFrame()
selected_columns = User2[["time","tag_x_coordinate","tag_y_coordinate","tag_z_coordinate"]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

User1S.tag_x_coordinate = User1S.tag_x_coordinate.apply(lambda d: d/1000)
User1S.tag_y_coordinate = User1S.tag_y_coordinate.apply(lambda d: d/1000)
User1S.tag_z_coordinate = User1S.tag_z_coordinate.apply(lambda d: d/1000)

User1S.time = pd.to_datetime(User1S.time, format='%Y-%m-%d %H:%M:%S')
diff = User1S.diff()

diff.time = diff.time.apply(lambda d: d.total_seconds())
print(diff.time)

coords = [c for c in User1S.columns if not 'time' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['time']
velocity1[velocity1==np.inf] = 0.0

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User2F = pd.concat([User2, df, df1], axis=1)
User2F.Estimated_Speed = User2F.Estimated_Speed.round(2)
#User1F.Estimated_Speed[User1F.Estimated_Speed > 1.5] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
User2F.time = pd.to_datetime(User2F.time, format='%Y-%m-%d %H:%M:%S')
#plt.scatter(User1F['time'],User1F['Estimated_Speed'])
plt.plot(User2F['Estimated_Speed'], 'o',label = 'Estimated_Speed')
#plt.plot(velocity1, 'x')
plt.title('Estimated speed for User2')
plt.xticks(rotation=45)
plt.legend()
plt.show()


#User 3

User3 = pos_data_.loc[pos_data_["tagId"]=='3']
User3 = User3.reset_index()
print(User3)
del User3['index']

User3F = pd.DataFrame()
selected_columns = User3[["time","tag_x_coordinate","tag_y_coordinate","tag_z_coordinate"]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

User1S.tag_x_coordinate = User1S.tag_x_coordinate.apply(lambda d: d/1000)
User1S.tag_y_coordinate = User1S.tag_y_coordinate.apply(lambda d: d/1000)
User1S.tag_z_coordinate = User1S.tag_z_coordinate.apply(lambda d: d/1000)

User1S.time = pd.to_datetime(User1S.time, format='%Y-%m-%d %H:%M:%S')
diff = User1S.diff()

diff.time = diff.time.apply(lambda d: d.total_seconds())
print(diff.time)

coords = [c for c in User1S.columns if not 'time' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['time']
velocity1[velocity1==np.inf] = 0.0

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User3F = pd.concat([User3, df, df1], axis=1)
User3F.Estimated_Speed = User3F.Estimated_Speed.round(2)
#User1F.Estimated_Speed[User1F.Estimated_Speed > 1.5] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
User3F.time = pd.to_datetime(User3F.time, format='%Y-%m-%d %H:%M:%S')
#plt.scatter(User1F['time'],User1F['Estimated_Speed'])
plt.plot(User3F['Estimated_Speed'], 'o',label = 'Estimated_Speed')
#plt.plot(velocity1, 'x')
plt.title('Estimated speed for User3')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# User 4

User4 = pos_data_.loc[pos_data_["tagId"]=='4']
User4 = User4.reset_index()
print(User4)
del User4['index']

User4F = pd.DataFrame()
selected_columns = User4[["time","tag_x_coordinate","tag_y_coordinate","tag_z_coordinate"]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

User1S.tag_x_coordinate = User1S.tag_x_coordinate.apply(lambda d: d/1000)
User1S.tag_y_coordinate = User1S.tag_y_coordinate.apply(lambda d: d/1000)
User1S.tag_z_coordinate = User1S.tag_z_coordinate.apply(lambda d: d/1000)

User1S.time = pd.to_datetime(User1S.time, format='%Y-%m-%d %H:%M:%S')
diff = User1S.diff()

diff.time = diff.time.apply(lambda d: d.total_seconds())
print(diff.time)

coords = [c for c in User1S.columns if not 'time' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['time']
velocity1[velocity1==np.inf] = 0.0

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User4F = pd.concat([User4, df, df1], axis=1)
User4F.Estimated_Speed = User4F.Estimated_Speed.round(2)
#User1F.Estimated_Speed[User1F.Estimated_Speed > 1.5] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
User4F.time = pd.to_datetime(User4F.time, format='%Y-%m-%d %H:%M:%S')
#plt.scatter(User1F['time'],User1F['Estimated_Speed'])
plt.plot(User4F['Estimated_Speed'], 'o',label = 'Estimated_Speed')
#plt.plot(velocity1, 'x')
plt.title('Estimated speed Vs. for User4')
plt.xticks(rotation=45)
plt.legend()
plt.show()


#User 7

User7 = pos_data_.loc[pos_data_["tagId"]=='7']
User7 = User7.reset_index()
print(User7)
del User7['index']

User7F = pd.DataFrame()
selected_columns = User7[["time","tag_x_coordinate","tag_y_coordinate","tag_z_coordinate"]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

User1S.tag_x_coordinate = User1S.tag_x_coordinate.apply(lambda d: d/1000)
User1S.tag_y_coordinate = User1S.tag_y_coordinate.apply(lambda d: d/1000)
User1S.tag_z_coordinate = User1S.tag_z_coordinate.apply(lambda d: d/1000)

User1S.time = pd.to_datetime(User1S.time, format='%Y-%m-%d %H:%M:%S')
diff = User1S.diff()

diff.time = diff.time.apply(lambda d: d.total_seconds())
print(diff.time)

coords = [c for c in User1S.columns if not 'time' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['time']
velocity1[velocity1==np.inf] = 0.0

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User7F = pd.concat([User7, df, df1], axis=1)
User7F.Estimated_Speed = User7F.Estimated_Speed.round(2)
#User1F.Estimated_Speed[User1F.Estimated_Speed > 1.5] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
User7F.time = pd.to_datetime(User7F.time, format='%Y-%m-%d %H:%M:%S')
#plt.scatter(User1F['time'],User1F['Estimated_Speed'])
plt.plot(User7F['Estimated_Speed'], 'o',label = 'Estimated_Speed')
#plt.plot(velocity1, 'x')
plt.title('Estimated speed Vs.for User7')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# Merging the previous column values to the speed and distance estimated and creating final dataframe

print("**********User1************")
print(User1F)
print("**********User2************")
print(User2F)
print("**********User3************")
print(User3F)
print("**********User4************")
print(User4F)
print("**********User7************")
print(User7F)


#Combining all the dataframes

User1F['UCLA'] = '3'
User1F['Lubben'] = '12'
User2F['UCLA'] = '3'
User2F['Lubben'] = '14'
User3F['UCLA'] = '4'
User3F['Lubben'] = '28'
User4F['UCLA'] = '6'
User4F['Lubben'] = '18'
User7F['UCLA'] = '7'
User7F['Lubben'] = '9'
User_combined = pd.concat([User1F, User2F,User3F,User4F,User7F], axis=0)
User_combined


#Checking for null values
null_values = User_combined.isnull().sum()
print(null_values)


# Viusualizing per day speed of user indoors

import seaborn as sns
sns.set(rc={'figure.figsize':(10,6)})
sns.scatterplot(data=User_combined, y="Estimated_Speed", x="time", hue="tagId",palette="bright")



######################################################### DATA ANALYSIS########################################################


#Importing libraries for data analysis, can ignore libraries already imported

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
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, LeavePOut, ShuffleSplit, LeaveOneOut,RandomizedSearchCV, KFold,StratifiedKFold


selected_columns = User_combined[["tagId","tag_x_coordinate","tag_y_coordinate","tag_z_coordinate","acc_x","acc_y","acc_z",'zone_tag','status_id','Estimated_Speed','Distance','UCLA','Lubben']]
User_combined_analysis = selected_columns.copy()
User_combined_analysis

User_combined_analysis = User_combined_analysis.reset_index()
del User_combined_analysis['index']
User_combined_analysis

import matplotlib.pyplot as plt
import seaborn as sns
print(pos_data_['tagId'].value_counts())
print(pos_data_['tagId'].value_counts()/np.float(len(pos_data_)))
#figure(figsize=(8, 6), dpi=80)
sns.countplot(pos_data_['tagId'],palette="bright")
plt.show()

#checking for null values and filling null values 
null_values = User_combined_analysis.isnull().sum()
print(null_values)
User_combined_analysis = User_combined_analysis.fillna(method="bfill")
is_NaN = User_combined_analysis.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = User_combined_analysis[row_has_NaN]
print(rows_with_NaN)
print(User_combined_analysis)

# Userwise creating dataframe
X1 = User_combined_analysis[User_combined_analysis.tagId == '1']
X2 = User_combined_analysis[User_combined_analysis.tagId == '2']
X3 = User_combined_analysis[User_combined_analysis.tagId == '3']
X4 = User_combined_analysis[User_combined_analysis.tagId == '4']
X7 = User_combined_analysis[User_combined_analysis.tagId == '7']

# function for extracting time domain features from accelerometer data
def acc_data_feature_extraction(X_train, label):
    x_list = []
    y_list = []
    z_list = []
    avg_dis = []
    avg_speed = []
    avs = []
    state = []
    zone = []
    window_size = 10
    step_size = 10
    ucla = []
    lubben = []
    # creating overlaping windows of size window-size 100
    for i in range(0, X_train.shape[0] - window_size,step_size):
        xs = X_train['acc_x'].values[i: i + 10]
        ys = X_train['acc_y'].values[i: i + 10]
        zs = X_train['acc_z'].values[i: i + 10]
        xt = X_train['tag_x_coordinate'].values[i: i + 10]
        yt = X_train['tag_y_coordinate'].values[i: i + 10]
        vt = X_train['Estimated_Speed'].values[i: i + 10]
        status = stats.mode(X_train['status_id'][i: i + 10])[0][0]
        zones = stats.mode(X_train['zone_tag'][i: i + 10])[0][0]
        uc = stats.mode(X_train['UCLA'][i: i + 10])[0][0]
        lu = stats.mode(X_train['Lubben'][i: i + 10])[0][0]
        
        dis=[]
        m = [x - xt[i - 1] for i, x in enumerate(xt)][1:]
        n = [x - yt[i - 1] for i, x in enumerate(yt)][1:]
        for x, y in zip(m,n):
            distance = math.sqrt(x**2 + y**2)
            dis.append(distance)
        avg = sum(dis)/len(dis)
        
        x_list.append(xs)
        y_list.append(ys)
        z_list.append(zs)
        state.append(status)
        zone.append(zones)
        avg_dis.append(avg)
        avs.append(vt)
        ucla.append(uc)
        lubben.append(lu)
    
    # Statistical Features on raw x, y and z in time domain
    X_train = pd.DataFrame()
    
    for i in range(len(avs)):
        vp = sum(avs[i])/len(avs[i])
        avg_speed.append(vp)
    
    # mean
    X_train['x_mean'] = pd.Series(x_list).apply(lambda x: x.mean())
    X_train['y_mean'] = pd.Series(y_list).apply(lambda x: x.mean())
    X_train['z_mean'] = pd.Series(z_list).apply(lambda x: x.mean())

    # std dev
    X_train['x_std'] = pd.Series(x_list).apply(lambda x: x.std())
    X_train['y_std'] = pd.Series(y_list).apply(lambda x: x.std())
    X_train['z_std'] = pd.Series(z_list).apply(lambda x: x.std())

    # median
    X_train['x_median'] = pd.Series(x_list).apply(lambda x: np.median(x))
    X_train['y_median'] = pd.Series(y_list).apply(lambda x: np.median(x))
    X_train['z_median'] = pd.Series(z_list).apply(lambda x: np.median(x))

    # median abs dev 
    X_train['x_mad'] = pd.Series(x_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['y_mad'] = pd.Series(y_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))
    X_train['z_mad'] = pd.Series(z_list).apply(lambda x: np.median(np.absolute(x - np.median(x))))

    # interquartile range
    X_train['x_IQR'] = pd.Series(x_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['y_IQR'] = pd.Series(y_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    X_train['z_IQR'] = pd.Series(z_list).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
    
    # skewness
    X_train['x_skewness'] = pd.Series(x_list).apply(lambda x: stats.skew(x))
    X_train['y_skewness'] = pd.Series(y_list).apply(lambda x: stats.skew(x))
    X_train['z_skewness'] = pd.Series(z_list).apply(lambda x: stats.skew(x))

    # kurtosis
    X_train['x_kurtosis'] = pd.Series(x_list).apply(lambda x: stats.kurtosis(x))
    X_train['y_kurtosis'] = pd.Series(y_list).apply(lambda x: stats.kurtosis(x))
    X_train['z_kurtosis'] = pd.Series(z_list).apply(lambda x: stats.kurtosis(x))

    # energy
    X_train['x_energy'] = pd.Series(x_list).apply(lambda x: np.sum(x**2)/100)
    X_train['y_energy'] = pd.Series(y_list).apply(lambda x: np.sum(x**2)/100)
    X_train['z_energy'] = pd.Series(z_list).apply(lambda x: np.sum(x**2/100))
    
    # negtive count
    X_train['x_neg_count'] = pd.Series(x_list).apply(lambda x: np.sum(x < 0))
    X_train['y_neg_count'] = pd.Series(y_list).apply(lambda x: np.sum(x < 0))
    X_train['z_neg_count'] = pd.Series(z_list).apply(lambda x: np.sum(x < 0))

    # positive count
    X_train['x_pos_count'] = pd.Series(x_list).apply(lambda x: np.sum(x > 0))
    X_train['y_pos_count'] = pd.Series(y_list).apply(lambda x: np.sum(x > 0))
    X_train['z_pos_count'] = pd.Series(z_list).apply(lambda x: np.sum(x > 0))

    # values above mean
    X_train['x_above_mean'] = pd.Series(x_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['y_above_mean'] = pd.Series(y_list).apply(lambda x: np.sum(x > x.mean()))
    X_train['z_above_mean'] = pd.Series(z_list).apply(lambda x: np.sum(x > x.mean()))

    # number of peaks
    X_train['x_peak_count'] = pd.Series(x_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['y_peak_count'] = pd.Series(y_list).apply(lambda x: len(find_peaks(x)[0]))
    X_train['z_peak_count'] = pd.Series(z_list).apply(lambda x: len(find_peaks(x)[0]))
    
     # avg absolute diff
    X_train['x_aad'] = pd.Series(x_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['y_aad'] = pd.Series(y_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
    X_train['z_aad'] = pd.Series(z_list).apply(lambda x: np.mean(np.absolute(x - np.mean(x))))

    # min
    X_train['x_min'] = pd.Series(x_list).apply(lambda x: x.min())
    X_train['y_min'] = pd.Series(y_list).apply(lambda x: x.min())
    X_train['z_min'] = pd.Series(z_list).apply(lambda x: x.min())

    # max
    X_train['x_max'] = pd.Series(x_list).apply(lambda x: x.max())
    X_train['y_max'] = pd.Series(y_list).apply(lambda x: x.max())
    X_train['z_max'] = pd.Series(z_list).apply(lambda x: x.max())

    # max-min diff
    X_train['x_maxmin_diff'] = X_train['x_max'] - X_train['x_min']
    X_train['y_maxmin_diff'] = X_train['y_max'] - X_train['y_min']
    X_train['z_maxmin_diff'] = X_train['z_max'] - X_train['z_min']

    # avg resultant
    X_train['avg_result_accl'] = [i.mean() for i in ((pd.Series(x_list)**2 + pd.Series(y_list)**2 + pd.Series(z_list)**2)**0.5)]

    # signal magnitude area
    X_train['sma'] =    pd.Series(x_list).apply(lambda x: np.sum(abs(x)/100)) + pd.Series(y_list).apply(lambda x: np.sum(abs(x)/100)) \
                      + pd.Series(z_list).apply(lambda x: np.sum(abs(x)/100))
    
    X_train['Status'] = state
    X_train['zone'] = zone
    X_train['avg_dis'] = avg_dis
    X_train['avg_speed'] = avg_speed
    X_train['UCLA'] = ucla
    X_train['Lubben'] = lubben
    X_train['label'] = label
    return X_train


#Calling the function

from scipy.signal import find_peaks
from scipy import stats

X11 = acc_data_feature_extraction(X1,1)
X12 = acc_data_feature_extraction(X2,2)
X13 = acc_data_feature_extraction(X3,3)
X14 = acc_data_feature_extraction(X4,4)
X17 = acc_data_feature_extraction(X7,7)
frames = [X11, X12, X13, X14,X17]
Complete_dataset = pd.concat(frames)
Complete_dataset['UCLA'] = pd.Series(Complete_dataset['UCLA'], dtype="int64")
Complete_dataset['Lubben'] = pd.Series(Complete_dataset['Lubben'], dtype="int64")
Complete_dataset = Complete_dataset.reset_index()
del Complete_dataset['index']
Complete_dataset



X = Complete_dataset.drop(['label','UCLA','Lubben'], axis = 1)
y = Complete_dataset['label']
print(X)
print(y)


#Splitting the dataset into training and testing datasets

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Machine learning based User classification (Logistic regression)

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# logistic regression model
lr = LogisticRegression(random_state = 21)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
feature_importance=pd.DataFrame({'feature':list(X.columns),'feature_importance':[abs(i) for i in lr.coef_[0]]})
feature_importance.sort_values('feature_importance',ascending=False)
print(feature_importance)
feature_importance = feature_importance.set_index('feature', drop=True)
feature_importance.plot.barh(title='Feature importance curve', figsize=(10,8))
plt.xlabel('Feature Importance Score')
plt.show()

print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
labels = ['1','2','3','4','7']
confusion_matrix = confusion_matrix(y_test, y_pred)
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()



#Machine learning based User classification (Random Forest)

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# Random Forest model
lr = RandomForestClassifier(n_estimators=10)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
feature_importance=pd.DataFrame({'feature':list(X.columns),'feature_importance':[abs(i) for i in lr.feature_importances_]})
feature_importance.sort_values('feature_importance',ascending=False)
print(feature_importance)
feature_importance = feature_importance.set_index('feature', drop=True)
feature_importance.plot.barh(title='Feature importance curve', figsize=(10,8))
plt.xlabel('Feature Importance Score')
plt.show()


print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
labels = ['1','2','3','4','7']
confusion_matrix = confusion_matrix(y_test, y_pred)
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()


#Machine learning based User classification (Support vector Machines)

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# SVM model
lr = SVC(C=15,kernel='rbf')
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(lr, X_test_data_lr, y_test)
feature_names = X.columns
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
labels = ['1','2','3','4','7']
confusion_matrix = confusion_matrix(y_test, y_pred)
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()


#Machine learning based  User classification (XGBoost)

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# Random Forest model
lr = xgb.XGBClassifier(random_state=1)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
feature_importance=pd.DataFrame({'feature':list(X.columns),'feature_importance':[abs(i) for i in lr.feature_importances_]})
feature_importance.sort_values('feature_importance',ascending=False)
print(feature_importance)
feature_importance = feature_importance.set_index('feature', drop=True)
feature_importance.plot.barh(title='Feature importance curve', figsize=(10,8))
plt.xlabel('Feature Importance Score')
plt.show()


print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
sns.set(font_scale=1.5)
from sklearn.metrics import confusion_matrix
labels = ['1','2','3','4','7']
confusion_matrix = confusion_matrix(y_test, y_pred)
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,annot_kws={"size": 20},linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 20)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()


#based on UCLA score
from scipy.signal import find_peaks
from scipy import stats

X11 = acc_data_feature_extraction(X1,0)
X12 = acc_data_feature_extraction(X2,0)
X13 = acc_data_feature_extraction(X3,0)
X14 = acc_data_feature_extraction(X4,1)
X17 = acc_data_feature_extraction(X7,2)
frames = [X11, X12, X13, X14, X17]
Complete_dataset = pd.concat(frames)
Complete_dataset = Complete_dataset.reset_index()
del Complete_dataset['index']
Complete_dataset


X = Complete_dataset.drop(['label','UCLA','Lubben'], axis = 1)
y = Complete_dataset['label']
print(X)
print(y)

print(Complete_dataset['label'].value_counts())
print(Complete_dataset['label'].value_counts()/np.float(len(Complete_dataset)))
figure(figsize=(8, 6), dpi=80)
sns.countplot(Complete_dataset['label'],palette="bright")
plt.show()


#Splitting the dataset into training and testing datasets

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Machine learning based classification (Logistic regression)


# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# logistic regression model
lr = LogisticRegression(random_state = 21)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)


#feature importance
feature_importance=pd.DataFrame({'feature':list(X.columns),'feature_importance':[abs(i) for i in lr.coef_[0]]})
feature_importance.sort_values('feature_importance',ascending=False)
print(feature_importance)
feature_importance = feature_importance.set_index('feature', drop=True)
feature_importance.plot.barh(title='Feature importance curve', figsize=(10,8))
plt.xlabel('Feature Importance Score')
plt.show()

print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()




#Machine learning based classification (Random Forest)


# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# Random Forest model
lr = RandomForestClassifier(n_estimators=10)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
feature_importance=pd.DataFrame({'feature':list(X.columns),'feature_importance':[abs(i) for i in lr.feature_importances_]})
feature_importance.sort_values('feature_importance',ascending=False)
print(feature_importance)
feature_importance = feature_importance.set_index('feature', drop=True)
feature_importance.plot.barh(title='Feature importance curve', figsize=(10,8))
plt.xlabel('Feature Importance Score')
plt.show()


print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1))*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()





#Machine learning based classification (Support vector Machines)


# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# SVM model
lr = SVC(C=15,kernel='rbf',probability=True)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(lr, X_test_data_lr, y_test)
feature_names = X.columns
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()



#Machine learning based classification (XGBoost)


# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# Random Forest model
lr = xgb.XGBClassifier(random_state=1)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
feature_importance=pd.DataFrame({'feature':list(X.columns),'feature_importance':[abs(i) for i in lr.feature_importances_]})
feature_importance.sort_values('feature_importance',ascending=False)
print(feature_importance)
feature_importance = feature_importance.set_index('feature', drop=True)
feature_importance.plot.barh(title='Feature importance curve', figsize=(10,8))
plt.xlabel('Feature Importance Score')
plt.show()


print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
sns.set(font_scale=1.5)
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,annot_kws={"size": 25},linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 20)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()



#based on Lubben score
from scipy.signal import find_peaks
from scipy import stats

X11 = acc_data_feature_extraction(X1,2)
X12 = acc_data_feature_extraction(X2,1)
X13 = acc_data_feature_extraction(X3,0)
X14 = acc_data_feature_extraction(X4,1)
X17 = acc_data_feature_extraction(X7,2)

frames = [X11, X12, X13, X14, X17]
Complete_dataset = pd.concat(frames)
Complete_dataset




Complete_dataset = Complete_dataset.reset_index()
del Complete_dataset['index']
Complete_dataset





X = Complete_dataset.drop(['label','UCLA','Lubben'], axis = 1)
y = Complete_dataset['label']

print(X)
print(y)

print(Complete_dataset['label'].value_counts())
print(Complete_dataset['label'].value_counts()/np.float(len(Complete_dataset)))
figure(figsize=(8, 6), dpi=80)
sns.countplot(Complete_dataset['label'],palette="bright")
plt.show()





#Splitting the dataset into training and testing datasets

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




#Machine learning based classification (Logistic regression)


# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# logistic regression model
lr = LogisticRegression(random_state = 21)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)


#feature importance
feature_importance=pd.DataFrame({'feature':list(X.columns),'feature_importance':[abs(i) for i in lr.coef_[0]]})
feature_importance.sort_values('feature_importance',ascending=False)
print(feature_importance)
feature_importance = feature_importance.set_index('feature', drop=True)
feature_importance.plot.barh(title='Feature importance curve', figsize=(10,8))
plt.xlabel('Feature Importance Score')
plt.show()

print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(confusion_matrix, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()





#Machine learning based classification (Random Forest)


# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# Random Forest model
lr = RandomForestClassifier(n_estimators=10)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
feature_importance=pd.DataFrame({'feature':list(X.columns),'feature_importance':[abs(i) for i in lr.feature_importances_]})
feature_importance.sort_values('feature_importance',ascending=False)
print(feature_importance)
feature_importance = feature_importance.set_index('feature', drop=True)
feature_importance.plot.barh(title='Feature importance curve', figsize=(10,8))
plt.xlabel('Feature Importance Score')
plt.show()


print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()




#Machine learning based classification (Support vector Machines)


# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# SVM model
lr = SVC(C=15,kernel='rbf',probability=True)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(lr, X_test_data_lr, y_test)
feature_names = X.columns
features = np.array(feature_names)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.show()

print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()





#Machine learning based classification (XGBoost)


# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# Random Forest model
lr = xgb.XGBClassifier(random_state=1)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
feature_importance=pd.DataFrame({'feature':list(X.columns),'feature_importance':[abs(i) for i in lr.feature_importances_]})
feature_importance.sort_values('feature_importance',ascending=False)
print(feature_importance)
feature_importance = feature_importance.set_index('feature', drop=True)
feature_importance.plot.barh(title='Feature importance curve', figsize=(10,8))
plt.xlabel('Feature Importance Score')
plt.show()


print("Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("\n -------------Classification Report-------------\n")
print(classification_report(y_test, y_pred))

#Confusion matrix
sns.set(font_scale=1.5)
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,annot_kws={"size": 25},linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 20)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()





