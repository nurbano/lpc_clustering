from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df_dataset = pd.DataFrame({
    'first column': ['aviones.xyz','tabla.las', 'mesa.las', 'maceta.las',  'autos.xyz'],
#     'second column': [10, 20, 30, 40]
    })

df_clustering = pd.DataFrame({
    'first column': ['K-means', 'DBSCAN', 'BIRD', 'OPTICS'],
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

if option=='aviones.xyz':
        x,y,z,ilum,refle,inte,nb= np.loadtxt('./aviones.xyz',skiprows=1, delimiter=";", unpack=True)
        X=np.column_stack((x, y, z, inte))
        X_filtrada= filtro_aleatorio(X, 10)
#         fig, ax = plt.subplots()
#         ax.scatter(X_filtrada[:, 0], X_filtrada[:,1], s= 0.01, c= X_filtrada[:,3])
#         st.pyplot(fig)
        df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'y': X_filtrada[:,2]})

        fig = px.scatter(data_frame=df, x="x", y="y", title="Dataset")
        st.plotly_chart(fig)
        
if option_clustering== "K-means":
    n_clus = st.slider('Cantidad de Cluster', 1, 20, 2)
    with st.spinner('Agrupando...'):
        kmeans = KMeans(n_clusters=n_clus).fit(X_filtrada)
    st.success('Listo!')
    
    clase_pred=kmeans.labels_
    df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'y': X_filtrada[:,2], 'inten': X_filtrada[:,3]})
    fig = px.scatter(data_frame=df, x="x", y="y", color= clase_pred,  color_discrete_sequence=px.colors.qualitative.G10,)
    st.plotly_chart(fig)
    
    clase_pred=kmeans.labels_
    df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'y': X_filtrada[:,1], 'z': X_filtrada[:,2],'inten': X_filtrada[:,3]})
    fig = px.scatter_3d(data_frame=df, x="x", y="y",z="z", color= clase_pred,  color_discrete_sequence=px.colors.qualitative.G10,)
    fig.update_traces(marker_size = 1)
    st.plotly_chart(fig)

