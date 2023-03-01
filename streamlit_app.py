from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth

st.title('Segmentación de Nubes de Puntos LIDAR con técnicas de clustering')
st.text('Nicolás Urbano Pintos')
st.text('urbano.nicolas@gmail.com')
df_dataset = pd.DataFrame({
    'first column': ['aviones.xyz','tabla.las', 'mesa.las', 'maceta.las',  'autos.xyz'],
#     'second column': [10, 20, 30, 40]
    })

df_clustering = pd.DataFrame({
    'first column': ['K-means', 'DBSCAN', 'MeanShift','BIRD', 'OPTICS'],
#     'second column': [10, 20, 30, 40]
    })

with st.sidebar:
    option = st.selectbox(
        'Dataset',
         df_dataset['first column'])
   
    option_clustering = st.selectbox(
        'Dataset',
         df_clustering['first column'])
   

'Dataset: ', option
def filtro_aleatorio(X_in, porc): 

  N = int(296910*porc/100)
  idx = np.random.choice(np.arange(X_in.shape[0]), size=N)

  return X[idx]
with st.sidebar:
        porc_puntos = st.slider('% puntos', 0, 100, 5)
if option=='aviones.xyz':
        x,y,z,ilum,refle,inte,nb= np.loadtxt('./aviones.xyz',skiprows=1, delimiter=";", unpack=True)
        X=np.column_stack((x, y, z, inte))
        X_filtrada= filtro_aleatorio(X, porc_puntos)
#         fig, ax = plt.subplots()
#         ax.scatter(X_filtrada[:, 0], X_filtrada[:,1], s= 0.01, c= X_filtrada[:,3])
#         st.pyplot(fig)
        df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'y': X_filtrada[:,2]})

        fig = px.scatter(data_frame=df, x="x", y="y", title="Dataset")
        st.plotly_chart(fig)
        
if option_clustering== "DBSCAN":
    with st.sidebar:
        eps_ = st.slider('Epsilon', 0.01, 0.10, 0.10)
    with st.spinner('Agrupando...'):
        cluster_db = DBSCAN( eps=eps_).fit(X_filtrada)
        st.write("Cantidad de Cluster: ")
        st.write(len(set(cluster_db.labels_)) - (1 if -1 in cluster_db.labels_ else 0))

    st.success('Listo!')
    df_out=pd.DataFrame(data={"cat": cluster_db.labels_})

    
    
if option_clustering== "K-means":
    with st.sidebar:
        n_clus = st.slider('Cantidad de Cluster', 1, 20, 2)
    with st.spinner('Agrupando...'):
        kmeans = KMeans(n_clusters=n_clus).fit(X_filtrada)
    st.success('Listo!')
    
    df_out=pd.DataFrame(data={"cat": kmeans.labels_})
    
if option_clustering== "MeanShit":
#     with st.sidebar:
#         n_clus = st.slider('Cantidad de Cluster', 1, 20, 2)
    with st.spinner('Agrupando...'):
        bandwidth = estimate_bandwidth(X_filtrada, quantile=0.2, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X_filtrada)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
    st.success('Listo!')
    st.write("Cantidad de Cluster: ")
    st.write(n_clusters_)
    df_out=pd.DataFrame(data={"cat": labels})
    
    
df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'y': X_filtrada[:,2], 'inten': X_filtrada[:,3]})
fig = px.scatter(data_frame=df, x="x", y="y", color= df_out["cat"].astype("category"), title= option_clustering, color_discrete_sequence=px.colors.qualitative.G10)
st.plotly_chart(fig)
    
    
df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'y': X_filtrada[:,1], 'z': X_filtrada[:,2],'inten': X_filtrada[:,3]})
fig = px.scatter_3d(data_frame=df, x="x", y="y",z="z", color= df_out["cat"].astype("category"),   title= option_clustering, color_discrete_sequence=px.colors.qualitative.G10)
fig.update_traces(marker_size = 1)
fig.update_layout(legend_itemsizing ='trace')
st.plotly_chart(fig)

