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
    'first column': ['tabla.las', 'mesa.las', 'maceta.las', 'aviones.xyz', 'autos.xyz'],
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
        X_filtrada= filtro_aleatorio(X, 5)
#         fig, ax = plt.subplots()
#         ax.scatter(X_filtrada[:, 0], X_filtrada[:,1], s= 0.01, c= X_filtrada[:,3])
#         st.pyplot(fig)
        df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'y': X_filtrada[:,2]})

        fig = px.scatter(data_frame=df, x="x", y="y", title="Dataset")
        st.plotly_chart(fig)
        
 if clustering== "K-means":
    kmeans = KMeans(n_clusters=5).fit(X_filtrada)
    clase_pred=kmeans.labels_
    df = pd.DataFrame(data= {'x': X_filtrada[:, 0], 'y': X_filtrada[:,2], 'inten': X_filtrada[:,3]})
    fig = px.scatter(data_frame=df, x="x", y="y", color= "inten", title=clustering)
    st.plotly_chart(fig)
    
    
#         st.altair_chart(alt.Chart(pd.DataFrame([X_filtrada[:, 0], X_filtrada[:,1]]), height=500, width=500)
#         .mark_circle(color='#0068c9', opacity=0.5)
#         .encode(x='x:Q', y='y:Q')) 
        
#         df = pd.DataFrame(data= {'x': X_filtrada[:, 0], "y": X_filtrada[:,1]})
        
#         fig = px.scatter(
# #             df.query("year==2007"),
#             x="x",
#             y="y",
#             size="pop",
#             color="continent",
#             hover_name="country",
#             log_x=True,
#             size_max=60,
#         )

#         tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
#         with tab1:
#             # Use the Streamlit theme.
#             # This is the default. So you can also omit the theme argument.
#             st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#         with tab2:
#             # Use the native Plotly theme.
#             st.plotly_chart(fig, theme=None, use_container_width=True)
        
