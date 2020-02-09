#!/usr/bin/env python
# coding: utf-8

# # Segmenting and Clustering Neighborhoods in Toronto

# In this assignment, you will be required to explore, segment, and cluster the neighborhoods in the city of Toronto. However, unlike New York, the neighborhood data is not readily available on the internet. What is interesting about the field of data science is that each project can be challenging in its unique way, so you need to learn to be agile and refine the skill to learn new libraries and tools quickly depending on the project.
# 
# For the Toronto neighborhood data, a Wikipedia page exists that has all the information we need to explore and cluster the neighborhoods in Toronto. You will be required to scrape the Wikipedia page and wrangle the data, clean it, and then read it into a pandas dataframe so that it is in a structured format like the New York dataset.
# 
# Once the data is in a structured format, you can replicate the analysis that we did to the New York City dataset to explore and cluster the neighborhoods in the city of Toronto.

# In[42]:


import pandas as pd # library for data analsysis
import numpy as np


# # Web scraping

# In[43]:


get_ipython().system('pip install lxml')


# In[44]:


url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'


# In[45]:


pdf=pd.read_html(url)
pdf


# In[46]:


df_can=pdf[0]
df_can


# In[47]:


df_can.shape
df_can['Borough'][0:10]


# # Data wrangling

# In[48]:


#for i in range(df_can.shape):
    #if df_can['Borough'][i]=='Not assigned':
        #df_can.drop(index=i, inplace=True)
indexNames = df_can[ (df_can['Borough']=='Not assigned')].index
df_can.drop(indexNames , inplace=True)


# In[49]:


df_can.head()


# In[50]:


df_can=df_can.reset_index(drop=True)
df_can.head()


# # Group by Postcode and Borough

# In[51]:


df_can=df_can.groupby(['Postcode','Borough'], as_index=False, sort=False).agg(lambda x: ','.join(x))
df_can


# In[52]:


df_can.shape


# In[53]:


change = df_can.loc[df_can['Neighbourhood'] == "Not assigned"].index
change


# In[54]:


for i in change:
    df_can.iloc[i, 2] = df_can.iloc[i, 1]
df_can


# In[55]:


df_can.shape


# In[56]:


csv='http://cocl.us/Geospatial_data'


# In[57]:


pdf2=pd.read_csv(csv)
pdf2


# In[58]:


df_can.rename(columns={'Postcode':'Postal Code'}, inplace=True)
df_can.head()


# In[59]:


df_newcan=df_can.join(pdf2.set_index('Postal Code'), on='Postal Code')
df_newcan


# In[60]:


print('There are {} rows in this dataframe'.format(df_newcan.shape[0]))


# # Segmenting and Clustering Neighbourhoods in Toronto

# ## Clustering Borough

# In[61]:


get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')
import folium # map rendering library


# In[62]:


get_ipython().system('conda install -c conda-forge geopy --yes ')
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values


# In[63]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="Toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# In[64]:


# create map of New York using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_newcan['Latitude'], df_newcan['Longitude'], df_newcan['Borough'], df_newcan['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[65]:


groupedtoronto=df_newcan
groupedtoronto


# In[66]:


for i in range(groupedtoronto.shape[0]):
    if 'Toronto' not in groupedtoronto['Neighbourhood'][i]:
        groupedtoronto.drop(index=i, inplace=True)
        
groupedtoronto


# In[67]:


torontogrouped_c = groupedtoronto.groupby('Borough', as_index=False).mean()
torontogrouped_c


# In[69]:


# import k-means from clustering stage
from sklearn.cluster import KMeans

# set number of clusters
kclusters = 5

toronto_grouped_clustering = torontogrouped_c.drop('Borough', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:5]


# In[71]:


# add clustering labels
torontogrouped_c.insert(0, 'Cluster Labels', kmeans.labels_)
torontogrouped_c


# ## Map out Clustered bouroughs

# In[73]:


import matplotlib.cm as cm
import matplotlib.colors as colors

# create map
map_clusters1 = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(torontogrouped_c['Latitude'], torontogrouped_c['Longitude'], torontogrouped_c['Borough'], torontogrouped_c['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters1)
    
    
       
map_clusters1


# In[ ]:




