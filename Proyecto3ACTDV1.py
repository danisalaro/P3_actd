#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
from pgmpy.sampling import BayesianModelSampling


# In[3]:


samples = pd.read_csv('C:\\Users\\juand\\OneDrive\\Documents\\Andes\\10. Semestre\\Analítica computacional\\Proyectos\\Proyecto 3\\DatosFinalP3.csv')


# In[4]:


from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
scoring_method = K2Score(data=samples)
esth = HillClimbSearch(data=samples)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=7, max_iter=int(1e4))
#
print(estimated_modelh)





print(estimated_modelh.nodes())
print(estimated_modelh.edges())
import networkx as nx
import matplotlib.pyplot as plt

# Crear el gráfico dirigido del modelo
G = nx.DiGraph()
for parent, child in estimated_modelh.edges():
    G.add_edge(parent, child)

# Dibujar el gráfico del modelo
pos = nx.spring_layout(G, seed=42) # Asignar posiciones a los nodos
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500) # Dibujar nodos
nx.draw_networkx_edges(G, pos, arrows=True) # Dibujar arcos con flechas
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif") # Agregar etiquetas de nodos

plt.axis("off") # Ocultar los ejes
plt.title("Modelo Estimado")
plt.show() # Mostrar el gráfico


# In[5]:


print(scoring_method.score(estimated_modelh))

