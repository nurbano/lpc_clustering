from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import Birch
import laspy

#Funciones

def filtro_aleatorio(X_in, porc): 

  N = int(len(X_in)*porc/100)
  idx = np.random.choice(np.arange(X_in.shape[0]), size=N)

  return X[idx]



def scaled_x_dimension(las_file):
    x_dimension = las_file.X
    scale = las_file.header.scales[0]
    offset = las_file.header.offsets[0]
    return (x_dimension * scale) + offset
def scaled_z_dimension(las_file):
    z_dimension = las_file.Z
    scale = las_file.header.scales[0]
    offset = las_file.header.offsets[0]
    return (z_dimension * scale) + offset
def scaled_y_dimension(las_file):
    y_dimension = las_file.Y
    scale = las_file.header.scales[0]
    offset = las_file.header.offsets[0]
    return (y_dimension * scale) + offset

def leer_las(file):
    with laspy.open(file) as fh:
        las = fh.read()
        ground_pts = las.classification == 2
        bins, counts = np.unique(las.return_number[ground_pts], return_counts=True)
    scaled_x = scaled_x_dimension(las)
    scaled_y = scaled_y_dimension(las)
    scaled_z = scaled_z_dimension(las)
    return np.array(scaled_x), np.array(scaled_y), np.array(scaled_z), np.array(las.intensity)

#Pantalla Princial
st.title('Segmentación de Nubes de Puntos LIDAR con técnicas de clustering')
st.text('Nicolás Urbano Pintos')
st.text('urbano.nicolas@gmail.com')

df_dataset = pd.DataFrame({
    'first column': ['aviones.xyz',
                     'escalera.las', 
                     'mesa.las', 'maceta.las',  
                     'autos.xyz'],

    })

df_clustering = pd.DataFrame({
    'first column': ['K-means', 
                     'DBSCAN', 
                     'MeanShift',
                     'BIRD'],

    })

with st.sidebar:
        option = st.selectbox(
        'Dataset',
         df_dataset['first column'])

        porc_puntos = st.slider('% puntos', 0, 100, 5)
        tam_punto= st.slider("Tamaño de puntos",0,10,2)   
        piso_preg = st.checkbox('Sacar Piso')

        option_clustering = st.selectbox('Metodo',
         df_clustering['first column'])
   

'Dataset: ', option


      
#Apertura de Dataset

if ".las" in option:
    x,y,z,inte= leer_las(option)
    X= np.column_stack((x, y, z, inte))
    X_filtrada= filtro_aleatorio(X, porc_puntos)#Filtro la cantidad de puntos
    df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'z': X_filtrada[:,2], 'inte': X_filtrada[:,3]})
    fig = px.scatter(data_frame=df, x="x", y="z", title="Dataset", color= "inte")
    fig.update_traces(marker_size = tam_punto)
    st.plotly_chart(fig)
    X=np.column_stack((x, y, z))
    if option=='escalera.las':
        if piso_preg:
            piso= 0.35
            mascara_piso= z>piso
            X=np.column_stack((x[mascara_piso], y[mascara_piso], z[mascara_piso]))
            X_filtrada= filtro_aleatorio(X, porc_puntos)
    
if ".xyz" in option:

        if option=='aviones.xyz':
            x,y,z,ilum,refle,inte,nb= np.loadtxt('./aviones.xyz',skiprows=1, delimiter=";", unpack=True)
            piso= 5.1
            mascara_piso= z>piso
            X=np.column_stack((x, y, z, inte))
            if piso_preg:
                X=np.column_stack((x[mascara_piso], y[mascara_piso], z[mascara_piso], inte[mascara_piso]))
        if option=='autos.xyz':
            x,y,z,ilum,refle,inte= np.loadtxt("./autos.xyz",skiprows=1, delimiter=";",unpack=True)
            X=np.column_stack((x, y, z, inte))

        
        
        X_filtrada= filtro_aleatorio(X, porc_puntos)
        df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'z': X_filtrada[:,2], 'inte': X_filtrada[:,3]})
        fig = px.scatter(data_frame=df, x="x", y="z", title="Dataset- Vista XZ", color= "inte", color_continuous_scale='Bluered_r')
        fig.update_traces(marker_size = tam_punto)
        st.plotly_chart(fig)
        X=np.column_stack((x, y, z))
        if piso_preg:
            X=np.column_stack((x[mascara_piso], y[mascara_piso], z[mascara_piso]))
        X_filtrada= filtro_aleatorio(X, porc_puntos)
        
if option_clustering== "DBSCAN":
    with st.sidebar:
        eps_ = st.slider('Epsilon', 0.01, 0.9, 0.50)
    with st.spinner('Agrupando...'):
        cluster_db = DBSCAN( eps=eps_).fit(X_filtrada)
    st.success('Listo!')
    df_out=pd.DataFrame(data={"cat": cluster_db.labels_})
    cantidad_cluster= len(np.unique(cluster_db.labels_))


    
if option_clustering== "K-means":
    with st.sidebar:
        n_clus = st.slider('Cantidad de Cluster', 1, 20, 2)
    with st.spinner('Agrupando...'):
        kmeans = KMeans(n_clusters=n_clus).fit(X_filtrada)
    st.success('Listo!')
    df_out=pd.DataFrame(data={"cat": kmeans.labels_})
    cantidad_cluster= n_clus
    
if option_clustering== "MeanShift":
    with st.sidebar:
        cuantile= st.slider("Cuantil del BW",0.1,1.0, 0.2 )
    with st.spinner('Agrupando...'):
        
        bandwidth = estimate_bandwidth(X_filtrada, quantile=cuantile, n_samples=500)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(X_filtrada)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        cantidad_cluster=n_clusters_
        df_out=pd.DataFrame(data={"cat": labels})
    st.success('Listo!')

    
if option_clustering== 'BIRD':
    with st.sidebar:
        n_clus = st.slider('Cantidad de Cluster', 1, 20, 2)
    clustering_birch = Birch(n_clusters=n_clus)
    clustering_birch.fit(X_filtrada)
    labels=clustering_birch.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cantidad_cluster=n_clusters_
    df_out=pd.DataFrame(data={"cat": labels})



st.metric(label="Clusters", value=cantidad_cluster)   
    
df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'z': X_filtrada[:,2]})
fig = px.scatter(data_frame=df, x="x", y="z", color= df_out["cat"].astype("category"), 
                 title= option_clustering + " - Vista XZ" ,
                 color_discrete_sequence=px.colors.qualitative.G10)
fig.update_traces(marker_size = tam_punto)
st.plotly_chart(fig)
    
    
df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'y': X_filtrada[:,1], 'z': X_filtrada[:,2]})
fig = px.scatter_3d(data_frame=df, x="x", y="y",z="z", color= df_out["cat"].astype("category"),
                    title= option_clustering + " - Vista 3D" , 
                    color_discrete_sequence=px.colors.qualitative.G10)
fig.update_traces(marker_size = tam_punto)
fig.update_layout(legend_itemsizing ='trace')
st.plotly_chart(fig)

