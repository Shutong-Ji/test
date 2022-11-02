from enum import unique
from turtle import width
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import seaborn as sns
import random


def get_hour(TIME):
    hour = TIME[0:2]
    hour = int(hour)
    return hour
def get_month(Month):
    month = Month[5:7]
    month = int(month)
    return month


plt.style.use('seaborn')
df1 = pd.read_csv('NYC Accidents 2020.csv')

df1.LONGITUDE.fillna(0,inplace=True)
df1.LATITUDE.fillna(0,inplace=True)
df1.BOROUGH.fillna('UNKOWN',inplace=True)
df1['hour'] = df1['CRASH TIME'].apply(get_hour)


df = pd.read_csv('NYC Accidents 2020.csv')
df = df.sample(5000, random_state=100) 
df = df.drop(['ZIP CODE','CROSS STREET NAME','OFF STREET NAME','NUMBER OF MOTORIST INJURED','CONTRIBUTING FACTOR VEHICLE 2','NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED','NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST KILLED','CONTRIBUTING FACTOR VEHICLE 3','CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5','VEHICLE TYPE CODE 3','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5'], axis=1)
df.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"}, inplace=True)


#df processing
df['lat'].fillna(0, inplace = True)
df['lon'].fillna(0, inplace = True)
df['BOROUGH'].fillna('Unknown', inplace = True)
df['ON STREET NAME'].fillna('Unknown', inplace = True)
df['CONTRIBUTING FACTOR VEHICLE 1'].fillna('Unspecified', inplace = True)
df['month'] = df['CRASH DATE'].apply(get_month)
df['hour'] = df['CRASH TIME'].apply(get_hour)
hour_statistics = df['hour'].value_counts()



borough_filter = st.sidebar.multiselect(
    'Borough Selector',
    df1.BOROUGH.unique(),
    df1.BOROUGH.unique()
)
df1 = df1[df1.BOROUGH.isin(borough_filter)]

form = st.sidebar.form("month_form")
month_filter = form.text_input('Enter the number of month(enter ALL to reset)', 'ALL')
form.form_submit_button("Apply")


if month_filter!='ALL':
    df = df[df.month == int(month_filter)]


df = df[df.lon <= -21]
time_filter = st.slider('Crash Time:', 0, 23, 12)
df = df[df.hour == time_filter]





location = df1['LOCATION'].value_counts()
count_loc = pd.DataFrame({'LOCATION':location.index,'ValueCount':location})
count_loc.index = range(len(location))

loc = df1.groupby('LOCATION').first()

new_loc = loc.loc[:,['LATITUDE','LONGITUDE','ON STREET NAME','BOROUGH','hour']]

the_loc = pd.merge(count_loc,new_loc,on='LOCATION')
the_loc.drop(the_loc.index[1],inplace=True)
#map1
nmap = folium.Map(location=[40.721757,-73.930529],zoom_start=13)

for i in range(1000):
    lat = the_loc.iloc[i][2]
    lon = the_loc.iloc[i][3]
    radius = the_loc['ValueCount'].iloc[i] / 3
    if the_loc['ValueCount'].iloc[i] > 15:
        color = '#FF4500'
    else:
        color = '#008080'

    popu_text= """Lat : {}<br>
                Lon : {}<br>
                ON STREET NAME : {}<br>
                BOROUGH : {}<br>
                INCIDENTS : {}<br>
                hour : {}<br>"""
    popu_text = popu_text.format(the_loc['LATITUDE'].iloc[i],the_loc['LONGITUDE'].iloc[i],the_loc['ON STREET NAME'].iloc[i],the_loc['BOROUGH'].iloc[i],the_loc['ValueCount'].iloc[i],the_loc['hour'].iloc[i])
    popup = folium.Popup(popu_text,min_width=200,max_width=300)

    folium.CircleMarker(location=[lat, lon],popup=popup, radius = radius, color = color,fill= True).add_to(nmap)



st.data = st_folium(nmap,width=2000)



#map2
mmap = folium.Map(location=[40.721757,-73.930529],zoom_start=13)

for i in range(len(df)):
    lat = df.iloc[i][3]
    lon = df.iloc[i][4]
    
    color = '#FF4500'
   

    popu_text= """Lat : {}<br>
                Lon : {}<br>
                ON STREET NAME : {}<br>
                BOROUGH : {}<br>"""
    popu_text = popu_text.format(df['lat'].iloc[i],df['lon'].iloc[i],df['ON STREET NAME'].iloc[i],df['BOROUGH'].iloc[i])
    popup = folium.Popup(popu_text,min_width=200,max_width=300)
    folium.CircleMarker(location=[lat, lon],popup=popup, radius = radius, color = color,fill= True).add_to(mmap)

st.data = st_folium(mmap,width=2000)






st.subheader('HeatMap:')
accidents_hour_pt = df.pivot_table(index='BOROUGH', columns='hour', aggfunc='size')
accidents_hour_pt = accidents_hour_pt.apply(lambda x: x / accidents_hour_pt.max(axis=1))
plt.figure(figsize=(15,5))
plt.title('Average Number of Accidents per Race and Hour', fontsize=14)
sns.heatmap(accidents_hour_pt, cbar=True, annot=False, fmt=".0f")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

hour_statistics = df['hour'].value_counts()
sorted_hour_statistics = hour_statistics.sort_index()
max_rate = 0
time = 0
for i in range(0,24):
    if df.hour.value_counts(normalize=True).sort_index()[i] > max_rate:
        max_rate = df.hour.value_counts(normalize=True).sort_index()[i]
        time = i
print(f'The max accident rate is {max_rate:.2%}, which is in {time} clock')


st.subheader('Line Chart Of The Clock And Accidents:')
fig, ax = plt.subplots()
sorted_hour_statistics.plot().set_xticks(range(0,24))
st.pyplot(fig)



st.subheader('Scatter Plot:')

df_1 = df[df.month==1]

plt.figure(figsize=(15,20))
sns.scatterplot(x='lon', y='lat', hue='CONTRIBUTING FACTOR VEHICLE 1', data=df_1)
plt.title("Month Accidents")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()

#1


