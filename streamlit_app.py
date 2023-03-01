from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))

df = pd.DataFrame({
    'first column': ['tabla.las', 'mesa.las', 'maceta.las', 'aviones.xyz', 'autos.xyz'],
#     'second column': [10, 20, 30, 40]
    })



with st.sidebar:
    option = st.selectbox(
        'Eliga el dataset que desea visualizar',
         df['first column'])

'Seleccion: ', option
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

        fig = px.scatter(data_frame=df, x="x", y="y", title="Sample Data")
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
        
