#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

import re
import pandas as pd
import glob
import datetime
import matplotlib.pyplot as plt
import numpy as np
import gmplot
import os
from pytz import timezone
from PIL import Image, ImageDraw
from datetime import timedelta
outdoor_pos_data = pd.DataFrame()


#Data Loading

path = r'C:\\Users\\chadsi\Downloads\\OneDrive_2022-04-11\\Outdoor training data of participants in Keinupuisto (01.03.22 - 31.03.22)'
dirList = glob.glob(path + "/*")

#Diabetes_subset = []
for dir in dirList:
    all_files = glob.glob(dir + "/*.txt")
    for filename in all_files:
        file = os.path.basename(filename)
        file = file.split(' ')[2]
        a = re.findall(r'#(\d+)', file)[0]
        pos_data = pd.read_csv(filename, index_col=None, header=0)
        pos_data.columns = ['MessageID', 'UTCtime', 'Status', 'Latitude','N/S indicator','Longitude','E/W indicator','Speed','Course','Date','na','na1']
        del pos_data['na']
        del pos_data['na1']
        pos_data = pos_data.assign(tagId=a)
        outdoor_pos_data = outdoor_pos_data.append(pos_data, ignore_index = True)


# Formatting date time 
outdoor_pos_data['UTCtime'] = outdoor_pos_data['UTCtime'].apply(lambda d: format(d, '06'))
outdoor_pos_data['UTCtime'] = outdoor_pos_data['UTCtime'].apply(lambda d: ':'.join(re.findall('..', d)))

def convert_date(date):
    a = re.findall('..', date)
    date = '2022-'+a[1]+'-'+a[0]
    return date

outdoor_pos_data['Date'] = outdoor_pos_data['Date'].apply(lambda d: format(d, '06'))
outdoor_pos_data['Date'] = outdoor_pos_data['Date'].apply(convert_date)
outdoor_pos_data['datetime'] = pd.to_datetime(outdoor_pos_data['Date'] + ' ' + outdoor_pos_data['UTCtime'])
outdoor_pos_data


outdoor_pos_data['datetime'] = pd.to_datetime(outdoor_pos_data['datetime'],format ='%Y-%m-%d %H:%M:%S' )
outdoor_pos_data['datetime'] = outdoor_pos_data['datetime']+ timedelta(hours=3)
del outdoor_pos_data['Date']
del outdoor_pos_data['UTCtime']
outdoor_pos_data


# Converting the latitude longitude units 
def conv(d):
    DD = int(float(d)/100)
    MM = float(d) - DD * 100
    LatDec = DD + MM/60 
    return LatDec

outdoor_pos_data['Latitude'] = pd.to_numeric(outdoor_pos_data['Latitude'], errors='coerce')
outdoor_pos_data['Longitude'] = pd.to_numeric(outdoor_pos_data['Longitude'], errors='coerce')
outdoor_pos_data['Latitude'] = outdoor_pos_data['Latitude'].apply(conv)
outdoor_pos_data['Longitude'] = outdoor_pos_data['Longitude'].apply(conv)
outdoor_pos_data


#Checking for null values

null_values = outdoor_pos_data.isnull().sum()
print(null_values)

# Assigning tag ids

outdoor_pos_data.tagId = outdoor_pos_data.tagId.map( {"8778":1 , "3913":2, "7275":3, "3356":4, "3349":6, "1624":7} )
outdoor_pos_data

# Converting lat/long to cartesian

def get_cartesian(lat=None,lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z

outdoor_pos_data["coordinate"]=outdoor_pos_data[['Latitude','Longitude']].apply(lambda x:get_cartesian(x.Latitude,x.Longitude), axis=1)
outdoor_pos_data[['cor_x','cor_y','cor_z']]= outdoor_pos_data["coordinate"].apply(pd.Series)
del outdoor_pos_data["coordinate"]
outdoor_pos_data


# Applying clustering algorithms to know the frequently visited places and visualize it in the google maps
# User1 
import sklearn
from sklearn.cluster import KMeans
User1 = outdoor_pos_data.loc[outdoor_pos_data["tagId"]==1]

kmeans = KMeans(3)
clusters = kmeans.fit_predict(User1[['Latitude','Longitude']])
User1['cluster'] = kmeans.predict(User1[['Latitude','Longitude']])

import folium
colors = ['red','blue','green']
lat = User1.iloc[0]['Latitude']
lng = User1.iloc[0]['Longitude']
map = folium.Map(location=[lng, lat], zoom_start=5)
for _, row in User1.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4, 
        weight=4, 
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])],
    ).add_to(map)
map
map.save("C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\Doctoral Studies\\Implementation plan\\Ppaer3_AISOLA_Dec21\\Diagrams and charts\\outdoor maps\\User1Cluster.html")


#Based on google maps specifying the tentative names of the location where user is frequently visitinh (Only for Understanding)
User1['cluster_name'] = User1.cluster.map( {0:'Keinupuitso_nearbyarea',1:'PRISMA' ,2:'Lidl_lakalaiva' } )
print(User1)

# Using sklearn data is scaled and visualized in x and y km
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,10)) 
arr_scaled = scaler.fit_transform(User1[['cor_x','cor_y']]) 

columns = ['x','y']
df_scaled = pd.DataFrame(arr_scaled, columns=columns)
print(df_scaled)

figure(figsize=(8, 6), dpi=80)
plt.scatter(df_scaled['x'],df_scaled['y'], c=User1['cluster'],s=50, cmap='viridis')
plt.xlabel("x [Km]")
plt.ylabel("y [Km]")
plt.show()

# Calculating the speed and the distance traveeled by the User outdoors
User1 = User1.loc[outdoor_pos_data["tagId"]==1]
User1 = User1.reset_index()
print(User1)
del User1['index']

User1F = pd.DataFrame()
selected_columns = User1[["datetime","cor_x","cor_y","cor_z",]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

diff = User1S.diff()

diff.datetime = diff.datetime.apply(lambda d: d.total_seconds())

coords = [c for c in User1S.columns if not 'datetime' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['datetime']

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User1F = pd.concat([User1, df, df1], axis=1)
User1F.Estimated_Speed = User1F.Estimated_Speed.round(2)
User1F.Estimated_Speed[User1F.Estimated_Speed > 30] = 0.0
print(User1F)

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
plt.scatter(User1F['datetime'],User1F['Estimated_Speed'],label = 'Estimated_Speed')
plt.scatter(User1F['datetime'],User1F['Speed'], label = 'Speed')
plt.title('Estimated speed Vs. Speed for User1')
plt.xticks(rotation=45)
plt.legend()
plt.show()



# ################################User 2#######################################

User2 = outdoor_pos_data.loc[outdoor_pos_data["tagId"]==2]

kmeans = KMeans(5)
clusters = kmeans.fit_predict(User2[['Latitude','Longitude']])
User2['cluster'] = kmeans.predict(User2[['Latitude','Longitude']])


import folium
colors = ['red','blue','green','yellow','orange']
lat = User2.iloc[0]['Latitude']
lng = User2.iloc[0]['Longitude']
map = folium.Map(location=[lng, lat], zoom_start=5)
for _, row in User2.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4, 
        weight=4, 
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])],
    ).add_to(map)
map
map.save("C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\Doctoral Studies\\Implementation plan\\Ppaer3_AISOLA_Dec21\\Diagrams and charts\\outdoor maps\\User2Cluster.html")


User2['cluster_name'] = User2.cluster.map( {0:'Keinupuitso_nearbyarea' , 1:'Lielahti', 2:'Tesoma', 3:'Peltolammi', 4:'Kaukajärvi'} )
print(User2)

# Using sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,10)) 
arr_scaled = scaler.fit_transform(User2[['cor_x','cor_y']]) 

columns = ['x','y']
df_scaled = pd.DataFrame(arr_scaled, columns=columns)
print(df_scaled)

figure(figsize=(8, 6), dpi=80)
plt.scatter(df_scaled['x'],df_scaled['y'], c=User2['cluster'],s=50, cmap='viridis')


User2 = User2.loc[outdoor_pos_data["tagId"]==2]
User2 = User2.reset_index()
print(User2)
del User2['index']

User2F = pd.DataFrame()
selected_columns = User2[["datetime","cor_x","cor_y","cor_z",]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

diff = User1S.diff()

diff.datetime = diff.datetime.apply(lambda d: d.total_seconds())

coords = [c for c in User1S.columns if not 'datetime' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['datetime']

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User2F = pd.concat([User2, df, df1], axis=1)
User2F.Estimated_Speed = User2F.Estimated_Speed.round(2)
User2F.Estimated_Speed[User2F.Estimated_Speed > 30] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
#plt.scatter(User2F['datetime'],User2F['Estimated_Speed'],label = 'Estimated_Speed')
plt.plot(User2F['Estimated_Speed'],'o',label = 'Estimated_Speed')
plt.plot(User2F['Speed'], 'o',label = 'Speed')
plt.title('Estimated speed Vs. Speed for User2')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# ##################################User 3########################################

User3 = outdoor_pos_data.loc[outdoor_pos_data["tagId"]==3]

kmeans = KMeans(3)
clusters = kmeans.fit_predict(User3[['Latitude','Longitude']])
User3['cluster'] = kmeans.predict(User3[['Latitude','Longitude']])


import folium
colors = ['green','blue','red']
lat = User3.iloc[0]['Latitude']
lng = User3.iloc[0]['Longitude']
map = folium.Map(location=[lng, lat], zoom_start=5)
for _, row in User3.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4, 
        weight=4, 
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])],
    ).add_to(map)
map
map.save("C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\Doctoral Studies\\Implementation plan\\Ppaer3_AISOLA_Dec21\\Diagrams and charts\\outdoor maps\\User3Cluster.html")

User3['cluster_name'] = User3.cluster.map( {0:'Hervanta' , 1:'Kauppi', 2:'Keinupuitso_nearbyarea'} )
print(User3)

# Using sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,3)) 
arr_scaled = scaler.fit_transform(User3[['cor_x','cor_y']]) 

columns = ['x','y']
df_scaled = pd.DataFrame(arr_scaled, columns=columns)
print(df_scaled)

figure(figsize=(8, 6), dpi=80)
plt.scatter(df_scaled['x'],df_scaled['y'], c=User3['cluster'],s=50, cmap='viridis')

User3 = User3.loc[outdoor_pos_data["tagId"]==3]
User3 = User3.reset_index()
print(User3)
del User3['index']

User3F = pd.DataFrame()
selected_columns = User3[["datetime","cor_x","cor_y","cor_z",]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

diff = User1S.diff()

diff.datetime = diff.datetime.apply(lambda d: d.total_seconds())

coords = [c for c in User1S.columns if not 'datetime' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['datetime']

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User3F = pd.concat([User3, df, df1], axis=1)
User3F.Estimated_Speed = User3F.Estimated_Speed.round(2)
#User3F.Estimated_Speed[User3FUser3F.Estimated_Speed > 30] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
plt.scatter(User3F['datetime'],User3F['Estimated_Speed'],label = 'Estimated_Speed')
plt.scatter(User3F['datetime'],User3F['Speed'], label = 'Speed')
plt.title('Estimated speed Vs. Speed for User3')
plt.xticks(rotation=45)
plt.legend()
plt.show()

###########################################user 4###########################################
User4 = outdoor_pos_data.loc[outdoor_pos_data["tagId"]==4]

kmeans = KMeans(3)
clusters = kmeans.fit_predict(User4[['Latitude','Longitude']])
User4['cluster'] = kmeans.predict(User4[['Latitude','Longitude']])

import folium
colors = ['red','blue','green']
lat = User4.iloc[0]['Latitude']
lng = User4.iloc[0]['Longitude']
map = folium.Map(location=[lng, lat], zoom_start=5)
for _, row in User4.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4, 
        weight=4, 
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])],
    ).add_to(map)
map
map.save("C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\Doctoral Studies\\Implementation plan\\Ppaer3_AISOLA_Dec21\\Diagrams and charts\\outdoor maps\\User4Cluster.html")
User4['cluster_name'] = User4.cluster.map( {0:'Keinupuitso_nearbyarea' , 1:'Kauppi', 2:'Hervantakeskus'} )
print(User4)

# Using sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,10)) 
arr_scaled = scaler.fit_transform(User4[['cor_x','cor_y']]) 

columns = ['x','y']
df_scaled = pd.DataFrame(arr_scaled, columns=columns)
print(df_scaled)

figure(figsize=(8, 6), dpi=80)
plt.scatter(df_scaled['x'],df_scaled['y'], c=User4['cluster'],s=50, cmap='viridis')
plt.xlabel("x [Km]")
plt.ylabel("y [Km]")
plt.show()

User4 = User4.loc[outdoor_pos_data["tagId"]==4]
User4 = User4.reset_index()
print(User4)
del User4['index']

User4F = pd.DataFrame()
selected_columns = User4[["datetime","cor_x","cor_y","cor_z",]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

diff = User1S.diff()

diff.datetime = diff.datetime.apply(lambda d: d.total_seconds())

coords = [c for c in User1S.columns if not 'datetime' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['datetime']

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User4F = pd.concat([User4, df, df1], axis=1)
User4F.Estimated_Speed = User4F.Estimated_Speed.round(2)
#User3F.Estimated_Speed[User3FUser3F.Estimated_Speed > 30] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
plt.scatter(User4F['datetime'],User4F['Estimated_Speed'],label = 'Estimated_Speed')
plt.scatter(User4F['datetime'],User4F['Speed'], label = 'Speed')
plt.title('Estimated speed Vs. Speed for User4')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# ############################################User 6#######################################
User6 = outdoor_pos_data.loc[outdoor_pos_data["tagId"]==6]

kmeans = KMeans(16)
clusters = kmeans.fit_predict(User6[['Latitude','Longitude']])
User6['cluster'] = kmeans.predict(User6[['Latitude','Longitude']])
print(User6)

import folium
colors = ['red','blue','green','yellow','orange','purple','black','brown','voilet','cyan','beige','grey','olive','indigo','tan','aquamarine']
lat = User6.iloc[0]['Latitude']
lng = User6.iloc[0]['Longitude']
map = folium.Map(location=[lng, lat], zoom_start=5)
for _, row in User6.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4, 
        weight=4, 
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])],
    ).add_to(map)
map
map.save("C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\Doctoral Studies\\Implementation plan\\Ppaer3_AISOLA_Dec21\\Diagrams and charts\\outdoor maps\\User6Cluster.html")

User6['cluster_name'] = User6.cluster.map( {0:'Keinupuitso_nearbyarea' , 15:'Hervantakeskus',1:'Other_area',2:'Other_area',3:'Other_area',4:'Other_area',7:'Other_area',6:'Other_area',8:'Other_area',5:'Other_area',10:'Other_area',11:'Other_area',12:'Other_area',13:'Other_area',14:'Other_area',9:'Other_area'} )
User6

User6['cluster'] = User6.cluster_name.map( {'Keinupuitso_nearbyarea':0 , 'Hervantakeskus':1 ,'Other_area':2} )

import folium
colors = ['red','blue','green']
lat = User6.iloc[0]['Latitude']
lng = User6.iloc[0]['Longitude']
map = folium.Map(location=[lng, lat], zoom_start=5)
for _, row in User6.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4, 
        weight=4, 
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])],
    ).add_to(map)
map
map.save("C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\Doctoral Studies\\Implementation plan\\Ppaer3_AISOLA_Dec21\\Diagrams and charts\\outdoor maps\\User6Clustertest.html")


# Using sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,10)) 
arr_scaled = scaler.fit_transform(User6[['cor_x','cor_y']]) 

columns = ['x','y']
df_scaled = pd.DataFrame(arr_scaled, columns=columns)
print(df_scaled)

figure(figsize=(8, 6), dpi=80)
plt.scatter(df_scaled['x'],df_scaled['y'], c=User6['cluster'],s=50, cmap='viridis')


User6 = User6.loc[outdoor_pos_data["tagId"]==6]
User6 = User6.reset_index()
print(User6)
del User6['index']

User6F = pd.DataFrame()
selected_columns = User6[["datetime","cor_x","cor_y","cor_z",]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

diff = User1S.diff()

diff.datetime = diff.datetime.apply(lambda d: d.total_seconds())

coords = [c for c in User1S.columns if not 'datetime' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['datetime']

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User6F = pd.concat([User6, df, df1], axis=1)
User6F.Estimated_Speed = User6F.Estimated_Speed.round(2)
User6F.Estimated_Speed[User6F.Estimated_Speed > 30] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
plt.scatter(User6F['datetime'],User6F['Estimated_Speed'],label = 'Estimated_Speed')
plt.scatter(User6F['datetime'],User6F['Speed'], label = 'Speed')
plt.title('Estimated speed Vs. Speed for User6')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# ##########################################User 7##########################################

User7 = outdoor_pos_data.loc[outdoor_pos_data["tagId"]==7]
from sklearn.cluster import KMeans

kmeans = KMeans(8)
clusters = kmeans.fit_predict(User7[['Latitude','Longitude']])
User7['cluster'] = kmeans.predict(User7[['Latitude','Longitude']])
print(User7)

import folium
colors = ['red','blue','green','yellow','orange','purple','magenta','aquamarine']
lat = User7.iloc[0]['Latitude']
lng = User7.iloc[0]['Longitude']
map = folium.Map(location=[lng, lat], zoom_start=5)
for _, row in User7.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4, 
        weight=4, 
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])],
    ).add_to(map)
map
map.save("C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\Doctoral Studies\\Implementation plan\\Ppaer3_AISOLA_Dec21\\Diagrams and charts\\outdoor maps\\User7Cluster.html")

User7['cluster_name'] = User7.cluster.map( {0:'Keinupuitso_nearbyarea' , 4:'Hervantakeskus',5:'Hervantakeskus',1:'Other_area',6:'Other_area',2:'Kaleva',3:'Keskustori',7:'Ratina'})
User7

User7['cluster'] = User7.cluster_name.map( {'Keinupuitso_nearbyarea':0 , 'Hervantakeskus':1 ,'Other_area':2,'Kaleva':3,'Keskustori':4,'Ratina':5})
print(User7)

import folium
colors = ['red','blue','green','yellow','orange','purple']
lat = User7.iloc[0]['Latitude']
lng = User7.iloc[0]['Longitude']
map = folium.Map(location=[lng, lat], zoom_start=5)
for _, row in User7.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=4, 
        weight=4, 
        fill=True, 
        fill_color=colors[int(row["cluster"])],
        color=colors[int(row["cluster"])],
    ).add_to(map)
map
map.save("C:\\Users\\chadsi\\OneDrive - TUNI.fi\\Documents\\Doctoral Studies\\Implementation plan\\Ppaer3_AISOLA_Dec21\\Diagrams and charts\\outdoor maps\\User7Clustertest.html")


# Using sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,6)) 
arr_scaled = scaler.fit_transform(User7[['cor_x','cor_y']]) 

columns = ['x','y']
df_scaled = pd.DataFrame(arr_scaled, columns=columns)
print(df_scaled)

figure(figsize=(8, 6), dpi=80)
plt.scatter(df_scaled['x'],df_scaled['y'], c=User7['cluster'],s=50, cmap='viridis')


User7 = User7.loc[outdoor_pos_data["tagId"]==7]
User7 = User7.reset_index()
print(User7)
del User7['index']

User7F = pd.DataFrame()
selected_columns = User7[["datetime","cor_x","cor_y","cor_z",]]
User1S = selected_columns.copy()
User1S = User1S.reset_index()
del User1S['index']

diff = User1S.diff()

diff.datetime = diff.datetime.apply(lambda d: d.total_seconds())

coords = [c for c in User1S.columns if not 'datetime' in c]
v = np.linalg.norm(diff[coords], axis=1)
velocity1 = np.linalg.norm(diff[coords], axis=1)/diff['datetime']

df = pd.DataFrame()
df = pd.Series(velocity1, name="Estimated_Speed")

df1 = pd.DataFrame()
df1 = pd.Series(np.linalg.norm(diff[coords], axis=1), name="Distance")

User7F = pd.concat([User7, df, df1], axis=1)
User7F.Estimated_Speed = User7F.Estimated_Speed.round(2)
User7F.Estimated_Speed[User7F.Estimated_Speed > 30] = 0.0

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(8,6), dpi=80)
plt.scatter(User7F['datetime'],User7F['Estimated_Speed'],label = 'Estimated_Speed')
plt.scatter(User7F['datetime'],User7F['Speed'], label = 'Speed')
plt.title('Estimated speed Vs. Speed for User7')
plt.xticks(rotation=45)
plt.legend()
plt.show()



print("**********User1************")
print(User1F)
print("**********User2************")
print(User2F)
print("**********User3************")
print(User3F)
print("**********User4************")
print(User4F)
print("**********User6************")
print(User6F)
print("**********User7************")
print(User7F)


# Combining all data frames and assigning labels to it
User1F['UCLA'] = '3'
User1F['Lubben'] = '12'
User2F['UCLA'] = '3'
User2F['Lubben'] = '14'
User3F['UCLA'] = '4'
User3F['Lubben'] = '28'
User4F['UCLA'] = '6'
User4F['Lubben'] = '18'
User6F['UCLA'] = '7'
User6F['Lubben'] = '8'
User7F['UCLA'] = '7'
User7F['Lubben'] = '9'
User_combined = pd.concat([User1F, User2F,User3F,User4F,User6F,User7F], axis=0)
User_combined['UCLA'] = pd.Series(User_combined['UCLA'], dtype="int64")
User_combined['Lubben'] = pd.Series(User_combined['Lubben'], dtype="int64")
User_combined['cluster'] = User_combined.cluster_name.map( {'Keinupuitso_nearbyarea':0 , 'Hervantakeskus':1 ,'Other_area':2,'PRISMA':3,'Lidl_lakalaiva':4,'Peltolammi':5,'Lielahti':6,'Kaukajärvi':7,'Tesoma':8,'Hervanta':9,'Kauppi':10,'Kaleva':11,'Ratina':12,'Keskustori':13})
User_combined


# Chcek for null values
null_values = User_combined.isnull().sum()
print(null_values)


tagIdCount = outdoor_pos_data.groupby(['tagId']).count()
tagIdCount = tagIdCount.reset_index()
dem_data = pd.DataFrame()
dem_data['tagId'] = tagIdCount['tagId']
dem_data['UCLA/9'] = [3,3,4,6,7,7]
dem_data['Lubben/30'] = [12,14,28,18,8,9]
print(dem_data)


# Importing libraries required for data analysis, ignore libraries that are already imported
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


selected_columns = User_combined[["datetime","tagId","Speed","Course","cor_x","cor_y","cor_z","cluster",'Estimated_Speed','UCLA','Lubben']]
User_combined_analysis = selected_columns.copy()
User_combined_analysis


User_combined_analysis = User_combined_analysis.reset_index()
del User_combined_analysis['index']
User_combined_analysis


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
sns.set_style("whitegrid")
print(User_combined_analysis['tagId'].value_counts())
print(User_combined_analysis['tagId'].value_counts()/np.float(len(User_combined_analysis)))
figure(figsize=(8, 6), dpi=80)
sns.countplot(User_combined_analysis['tagId'],palette="bright")
plt.show()


null_values = User_combined_analysis.isnull().sum()
print(null_values)
User_combined_analysis = User_combined_analysis.fillna(method="bfill")
is_NaN = User_combined_analysis.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = User_combined_analysis[row_has_NaN]
print(rows_with_NaN)


X1 = User_combined_analysis[User_combined_analysis.tagId == 1]
X2 = User_combined_analysis[User_combined_analysis.tagId == 2]
X3 = User_combined_analysis[User_combined_analysis.tagId == 3]
X4 = User_combined_analysis[User_combined_analysis.tagId == 4]
X6 = User_combined_analysis[User_combined_analysis.tagId == 6]
X7 = User_combined_analysis[User_combined_analysis.tagId == 7]


# Feature extraction 
import math
def data_feature_extraction(X_train, label):

    spt = []
    cpt = []
    vpt = []
    avg_dis = []
    speed  = []
    course = []
    velocity = []
    cluster = []
    ucla = []
    lubben = []
    window_size = 4
    step_size = 4
    # creating overlaping windows of size window-size 100
    for i in range(0, X_train.shape[0] - window_size,step_size):
        xs = X_train['cor_x'].values[i: i + 4]
        ys = X_train['cor_y'].values[i: i + 4]
        zs = X_train['cor_z'].values[i: i + 4]
        st = X_train['Speed'].values[i: i + 4]
        ct = X_train['Course'].values[i: i + 4]
        vt = X_train['Estimated_Speed'].values[i: i + 4]
        clus = stats.mode(X_train['cluster'][i: i + 4])[0][0]
        uc = stats.mode(X_train['UCLA'][i: i + 4])[0][0]
        lu = stats.mode(X_train['Lubben'][i: i + 4])[0][0]
        
        dis=[]
        m = [x - xs[i - 1] for i, x in enumerate(xs)][1:]
        n = [x - ys[i - 1] for i, x in enumerate(ys)][1:]
        p = [x - zs[i - 1] for i, x in enumerate(zs)][1:]
        for x, y, z in zip(m,n,p):
            distance = math.sqrt(x**2 + y**2 + z**2)
            dis.append(distance)
        avg = sum(dis)/len(dis)
        avg_dis.append(avg)
        
        spt.append(st)
        cpt.append(ct)
        vpt.append(vt)
        cluster.append(clus)
        ucla.append(uc)
        lubben.append(lu)
    
    speed = []
    course = []
    velocity = []
    # Statistical Features on raw x, y and z in time domain
    X_train = pd.DataFrame()
    for i in range(len(spt)):
        sp = sum(spt[i])/len(spt[i])
        speed.append(sp)
        
    for i in range(len(cpt)):
        cp = sum(cpt[i])/len(cpt[i])
        course.append(cp)
        
    for i in range(len(vpt)):
        vp = sum(vpt[i])/len(vpt[i])
        velocity.append(vp)
        
    X_train['speed'] = speed
    X_train['course'] = course
    X_train['Estimated_Speed'] = velocity
    X_train['avg_dis'] = avg_dis
    X_train['cluster'] = cluster
    X_train['UCLA'] = ucla
    X_train['Lubben'] = lubben
    X_train['label'] = label
    return X_train


# Calling fetaure extraction function
from scipy.signal import find_peaks
from scipy import stats

X11 = data_feature_extraction(X1,1)
X12 = data_feature_extraction(X2,2)
X13 = data_feature_extraction(X3,3)
X14 = data_feature_extraction(X4,4)
X16 = data_feature_extraction(X6,6)
X17 = data_feature_extraction(X7,7)


Complete_dataset = Complete_dataset.reset_index()
del Complete_dataset['index']
Complete_dataset


X = Complete_dataset.drop(['label','UCLA','Lubben'], axis = 1)
y = Complete_dataset['label']
print(X)
print(y)


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
labels = ['1','2','3','4','6','7']
confusion_matrix = confusion_matrix(y_test, y_pred)
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()


#Machine learning based classification (Random Forest)
#Approach 1: Without balancing the classes

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
figure(figsize=(8, 6))
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
labels = ['1','2','3','4','6','7']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()



#Machine learning based classification (Support vector Machines)
#Approach 2: With balancing the classes

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# SVM model
lr = SVC(C=15,gamma=2,kernel='rbf')
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
figure(figsize=(8, 6))
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

# Confusin matrix
from sklearn.metrics import confusion_matrix
labels = ['1','2','3','4','6','7']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()


import xgboost as xgb
#Approach 2: With balancing the classes

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# SVM model
lr = xgb.XGBClassifier(random_state=1)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
figure(figsize=(8, 6))
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

# Confusin matrix
sns.set(font_scale=1.5)
from sklearn.metrics import confusion_matrix
labels = ['1','2','3','4','6','7']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1,annot_kws={"size": 20}, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 20)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()


from scipy.signal import find_peaks
from scipy import stats

#Labels based on UCLA scale
X11 = data_feature_extraction(X1,0)
X12 = data_feature_extraction(X2,0)
X13 = data_feature_extraction(X3,0)
X14 = data_feature_extraction(X4,1)
X16 = data_feature_extraction(X6,2)
X17 = data_feature_extraction(X7,2)

Complete_dataset = pd.DataFrame()
frames = [X11, X12, X13, X14, X16, X17]
Complete_dataset = pd.concat(frames)
pd.unique(Complete_dataset.cluster)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
sns.set_style("whitegrid")
print(Complete_dataset['label'].value_counts())
print(Complete_dataset['label'].value_counts()/np.float(len(Complete_dataset)))
figure(figsize=(8, 6), dpi=80)
sns.countplot(Complete_dataset['label'],palette="bright")
plt.show()

Complete_dataset = Complete_dataset.reset_index()
del Complete_dataset['index']
Complete_dataset

X = Complete_dataset.drop(['label','UCLA','Lubben'], axis = 1)
y = Complete_dataset['label']

print(X)
print(y)

figure(figsize=(8, 6), dpi=80)
matrix = X.corr().round(2)
sns.heatmap(matrix, annot=True)
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
figure(figsize=(8, 6))
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
confusion_matrix = confusion_matrix(y_test, y_pred)
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()

#Machine learning based classification (Random Forest)
#Approach 1: Without balancing the classes

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
figure(figsize=(8, 6))
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
#Approach 2: With balancing the classes

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# SVM model
lr = SVC(C=15,gamma=2,kernel='rbf')
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
figure(figsize=(8, 6))
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

# Confusin matrix
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()

import xgboost as xgb
#Approach 2: With balancing the classes

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# SVM model
lr = xgb.XGBClassifier(random_state=1)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
figure(figsize=(8, 6))
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

# Confusin matrix
sns.set(font_scale=1.5)
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,annot_kws={"size": 20},linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 20)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()


from scipy.signal import find_peaks
from scipy import stats

#Labels based on Lubben scale
X11 = data_feature_extraction(X1,2)
X12 = data_feature_extraction(X2,1)
X13 = data_feature_extraction(X3,0)
X14 = data_feature_extraction(X4,1)
X16 = data_feature_extraction(X6,2)
X17 = data_feature_extraction(X7,2)

Complete_dataset = pd.DataFrame()
frames = [X11, X12, X13, X14, X16, X17]
Complete_dataset = pd.concat(frames)
pd.unique(Complete_dataset.cluster)

Complete_dataset = Complete_dataset.reset_index()
del Complete_dataset['index']
Complete_dataset

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
sns.set_style("whitegrid")
print(Complete_dataset['label'].value_counts())
print(Complete_dataset['label'].value_counts()/np.float(len(Complete_dataset)))
figure(figsize=(8, 6), dpi=80)
sns.countplot(Complete_dataset['label'],palette="bright")
plt.show()

X = Complete_dataset.drop(['label','UCLA','Lubben'], axis = 1)
y = Complete_dataset['label']

print(X)
print(y)

figure(figsize=(8, 6), dpi=80)
matrix = X.corr().round(2)
sns.heatmap(matrix, annot=True)
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
figure(figsize=(8, 6))
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
confusion_matrix = confusion_matrix(y_test, y_pred)
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()

#Machine learning based classification (Random Forest)
#Approach 1: Without balancing the classes

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
figure(figsize=(8, 6))
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
#Approach 2: With balancing the classes

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# SVM model
lr = SVC(C=15,gamma=2,kernel='rbf')
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
figure(figsize=(8, 6))
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

# Confusin matrix
from sklearn.metrics import confusion_matrix
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 15)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()

import xgboost as xgb
#Approach 2: With balancing the classes

# standardization
scaler = StandardScaler()
scaler.fit(X_train)
X_train_data_lr = scaler.transform(X_train)
X_test_data_lr = scaler.transform(X_test)

# SVM model
lr = xgb.XGBClassifier(random_state=1)
lr.fit(X_train_data_lr, y_train)
y_pred = lr.predict(X_test_data_lr)

#feature importance
figure(figsize=(8, 6))
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

# Confusin matrix
from sklearn.metrics import confusion_matrix
sns.set(font_scale=1.5)
labels = ['0','1','2']
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
cm = (confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])*100
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True,annot_kws={"size": 20},linewidths = 0.1, fmt='.2f', cmap = 'YlGnBu')
plt.title("“Confusion matrix”", fontsize = 20)
plt.ylabel("‘True label’")
plt.xlabel("‘Predicted label’")
plt.show()





