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


# In[2]:



df_Raw = pd.read_csv('C:\\Users\\juand\\OneDrive\\Documents\\Andes\\10. Semestre\\Analítica computacional\\Proyectos\\Proyecto 3\\Datos_Raw.csv')
                           


# In[3]:


df_Raw


# In[4]:


df_Raw = df_Raw.dropna()


# In[5]:


df_Raw


# In[6]:


df_Usar = df_Raw.copy()


# In[7]:


#para valores entre 20 y 30, 2
#para valores entre 30 y 40, 3
#Puntaje Ingles
df_Usar["punt_ingles"] = np.where((df_Usar["punt_ingles"]<30)&(df_Usar["punt_ingles"]>=20),2,df_Usar["punt_ingles"])
df_Usar["punt_ingles"] = np.where((df_Usar["punt_ingles"]<40)&(df_Usar["punt_ingles"]>=30),3,df_Usar["punt_ingles"])
df_Usar["punt_ingles"] = np.where((df_Usar["punt_ingles"]<50)&(df_Usar["punt_ingles"]>=40),4,df_Usar["punt_ingles"])
df_Usar["punt_ingles"] = np.where((df_Usar["punt_ingles"]<60)&(df_Usar["punt_ingles"]>=50),5,df_Usar["punt_ingles"])
df_Usar["punt_ingles"] = np.where((df_Usar["punt_ingles"]<70)&(df_Usar["punt_ingles"]>=60),6,df_Usar["punt_ingles"])
df_Usar["punt_ingles"] = np.where((df_Usar["punt_ingles"]<80)&(df_Usar["punt_ingles"]>=70),7,df_Usar["punt_ingles"])
df_Usar["punt_ingles"] = np.where((df_Usar["punt_ingles"]<90)&(df_Usar["punt_ingles"]>=80),8,df_Usar["punt_ingles"])
df_Usar["punt_ingles"] = np.where((df_Usar["punt_ingles"]<=100)&(df_Usar["punt_ingles"]>=90),9,df_Usar["punt_ingles"])

#Puntaje Mate
df_Usar["punt_matematicas"] = np.where((df_Usar["punt_matematicas"]<30)&(df_Usar["punt_matematicas"]>=20),2,df_Usar["punt_matematicas"])
df_Usar["punt_matematicas"] = np.where((df_Usar["punt_matematicas"]<40)&(df_Usar["punt_matematicas"]>=30),3,df_Usar["punt_matematicas"])
df_Usar["punt_matematicas"] = np.where((df_Usar["punt_matematicas"]<50)&(df_Usar["punt_matematicas"]>=40),4,df_Usar["punt_matematicas"])
df_Usar["punt_matematicas"] = np.where((df_Usar["punt_matematicas"]<60)&(df_Usar["punt_matematicas"]>=50),5,df_Usar["punt_matematicas"])
df_Usar["punt_matematicas"] = np.where((df_Usar["punt_matematicas"]<70)&(df_Usar["punt_matematicas"]>=60),6,df_Usar["punt_matematicas"])
df_Usar["punt_matematicas"] = np.where((df_Usar["punt_matematicas"]<80)&(df_Usar["punt_matematicas"]>=70),7,df_Usar["punt_matematicas"])
df_Usar["punt_matematicas"] = np.where((df_Usar["punt_matematicas"]<90)&(df_Usar["punt_matematicas"]>=80),8,df_Usar["punt_matematicas"])
df_Usar["punt_matematicas"] = np.where((df_Usar["punt_matematicas"]<=100)&(df_Usar["punt_matematicas"]>=90),9,df_Usar["punt_matematicas"])


#Puntaje Sociales_ciudadanas
df_Usar["punt_sociales_ciudadanas"] = np.where((df_Usar["punt_sociales_ciudadanas"]<30)&(df_Usar["punt_sociales_ciudadanas"]>=20),2,df_Usar["punt_sociales_ciudadanas"])
df_Usar["punt_sociales_ciudadanas"] = np.where((df_Usar["punt_sociales_ciudadanas"]<40)&(df_Usar["punt_sociales_ciudadanas"]>=30),3,df_Usar["punt_sociales_ciudadanas"])
df_Usar["punt_sociales_ciudadanas"] = np.where((df_Usar["punt_sociales_ciudadanas"]<50)&(df_Usar["punt_sociales_ciudadanas"]>=40),4,df_Usar["punt_sociales_ciudadanas"])
df_Usar["punt_sociales_ciudadanas"] = np.where((df_Usar["punt_sociales_ciudadanas"]<60)&(df_Usar["punt_sociales_ciudadanas"]>=50),5,df_Usar["punt_sociales_ciudadanas"])
df_Usar["punt_sociales_ciudadanas"] = np.where((df_Usar["punt_sociales_ciudadanas"]<70)&(df_Usar["punt_sociales_ciudadanas"]>=60),6,df_Usar["punt_sociales_ciudadanas"])
df_Usar["punt_sociales_ciudadanas"] = np.where((df_Usar["punt_sociales_ciudadanas"]<80)&(df_Usar["punt_sociales_ciudadanas"]>=70),7,df_Usar["punt_sociales_ciudadanas"])
df_Usar["punt_sociales_ciudadanas"] = np.where((df_Usar["punt_sociales_ciudadanas"]<90)&(df_Usar["punt_sociales_ciudadanas"]>=80),8,df_Usar["punt_sociales_ciudadanas"])
df_Usar["punt_sociales_ciudadanas"] = np.where((df_Usar["punt_sociales_ciudadanas"]<=100)&(df_Usar["punt_sociales_ciudadanas"]>=90),9,df_Usar["punt_sociales_ciudadanas"])


#Puntaje Ciencias
df_Usar["punt_c_naturales"] = np.where((df_Usar["punt_c_naturales"]<30)&(df_Usar["punt_c_naturales"]>=20),2,df_Usar["punt_c_naturales"])
df_Usar["punt_c_naturales"] = np.where((df_Usar["punt_c_naturales"]<40)&(df_Usar["punt_c_naturales"]>=30),3,df_Usar["punt_c_naturales"])
df_Usar["punt_c_naturales"] = np.where((df_Usar["punt_c_naturales"]<50)&(df_Usar["punt_c_naturales"]>=40),4,df_Usar["punt_c_naturales"])
df_Usar["punt_c_naturales"] = np.where((df_Usar["punt_c_naturales"]<60)&(df_Usar["punt_c_naturales"]>=50),5,df_Usar["punt_c_naturales"])
df_Usar["punt_c_naturales"] = np.where((df_Usar["punt_c_naturales"]<70)&(df_Usar["punt_c_naturales"]>=60),6,df_Usar["punt_c_naturales"])
df_Usar["punt_c_naturales"] = np.where((df_Usar["punt_c_naturales"]<80)&(df_Usar["punt_c_naturales"]>=70),7,df_Usar["punt_c_naturales"])
df_Usar["punt_c_naturales"] = np.where((df_Usar["punt_c_naturales"]<90)&(df_Usar["punt_c_naturales"]>=80),8,df_Usar["punt_c_naturales"])
df_Usar["punt_c_naturales"] = np.where((df_Usar["punt_c_naturales"]<=100)&(df_Usar["punt_c_naturales"]>=90),9,df_Usar["punt_c_naturales"])



#Puntaje lectura_critica
df_Usar["punt_lectura_critica"] = np.where((df_Usar["punt_lectura_critica"]<30)&(df_Usar["punt_lectura_critica"]>=20),2,df_Usar["punt_lectura_critica"])
df_Usar["punt_lectura_critica"] = np.where((df_Usar["punt_lectura_critica"]<40)&(df_Usar["punt_lectura_critica"]>=30),3,df_Usar["punt_lectura_critica"])
df_Usar["punt_lectura_critica"] = np.where((df_Usar["punt_lectura_critica"]<50)&(df_Usar["punt_lectura_critica"]>=40),4,df_Usar["punt_lectura_critica"])
df_Usar["punt_lectura_critica"] = np.where((df_Usar["punt_lectura_critica"]<60)&(df_Usar["punt_lectura_critica"]>=50),5,df_Usar["punt_lectura_critica"])
df_Usar["punt_lectura_critica"] = np.where((df_Usar["punt_lectura_critica"]<70)&(df_Usar["punt_lectura_critica"]>=60),6,df_Usar["punt_lectura_critica"])
df_Usar["punt_lectura_critica"] = np.where((df_Usar["punt_lectura_critica"]<80)&(df_Usar["punt_lectura_critica"]>=70),7,df_Usar["punt_lectura_critica"])
df_Usar["punt_lectura_critica"] = np.where((df_Usar["punt_lectura_critica"]<90)&(df_Usar["punt_lectura_critica"]>=80),8,df_Usar["punt_lectura_critica"])
df_Usar["punt_lectura_critica"] = np.where((df_Usar["punt_lectura_critica"]<=100)&(df_Usar["punt_lectura_critica"]>=90),9,df_Usar["punt_lectura_critica"])





#Cole Urbano
df_Usar["cole_area_ubicacion"] = np.where((df_Usar["cole_area_ubicacion"]=="RURAL"),0,df_Usar["cole_area_ubicacion"])
df_Usar["cole_area_ubicacion"] = np.where((df_Usar["cole_area_ubicacion"]=="URBANO"),1,df_Usar["cole_area_ubicacion"])



#Cole Bilingue
df_Usar["cole_bilingue"] = np.where((df_Usar["cole_bilingue"]=="N"),0,df_Usar["cole_bilingue"])
df_Usar["cole_bilingue"] = np.where((df_Usar["cole_bilingue"]=="S"),1,df_Usar["cole_bilingue"])



#Cole Calendario
df_Usar["cole_calendario"] = np.where((df_Usar["cole_calendario"]=="A"),0,df_Usar["cole_calendario"])
df_Usar["cole_calendario"] = np.where((df_Usar["cole_calendario"]=="B"),1,df_Usar["cole_calendario"])



#Cole Caracter
df_Usar["cole_caracter"] = np.where((df_Usar["cole_caracter"]=="TÉCNICO/ACADÉMICO"),0,df_Usar["cole_caracter"])
df_Usar["cole_caracter"] = np.where((df_Usar["cole_caracter"]=="ACADÉMICO"),1,df_Usar["cole_caracter"])
df_Usar["cole_caracter"] = np.where((df_Usar["cole_caracter"]=="TÉCNICO"),2,df_Usar["cole_caracter"])
df_Usar["cole_caracter"] = np.where((df_Usar["cole_caracter"]=="NO APLICA"),3,df_Usar["cole_caracter"])




#Cole Genero
df_Usar["cole_genero"] = np.where((df_Usar["cole_genero"]=="MIXTO"),0,df_Usar["cole_genero"])
df_Usar["cole_genero"] = np.where((df_Usar["cole_genero"]=="FEMENINO"),1,df_Usar["cole_genero"])
df_Usar["cole_genero"] = np.where((df_Usar["cole_genero"]=="MASCULINO"),2,df_Usar["cole_genero"])


#Cole naturaleza
df_Usar["cole_naturaleza"] = np.where((df_Usar["cole_naturaleza"]=="OFICIAL"),0,df_Usar["cole_naturaleza"])
df_Usar["cole_naturaleza"] = np.where((df_Usar["cole_naturaleza"]=="NO OFICIAL"),1,df_Usar["cole_naturaleza"])



#Cole Estrato
df_Usar["fami_estratovivienda"] = np.where((df_Usar["fami_estratovivienda"]=="Estrato 1"),1,df_Usar["fami_estratovivienda"])
df_Usar["fami_estratovivienda"] = np.where((df_Usar["fami_estratovivienda"]=="Estrato 2"),2,df_Usar["fami_estratovivienda"])
df_Usar["fami_estratovivienda"] = np.where((df_Usar["fami_estratovivienda"]=="Estrato 3"),3,df_Usar["fami_estratovivienda"])
df_Usar["fami_estratovivienda"] = np.where((df_Usar["fami_estratovivienda"]=="Estrato 4"),4,df_Usar["fami_estratovivienda"])
df_Usar["fami_estratovivienda"] = np.where((df_Usar["fami_estratovivienda"]=="Estrato 5"),5,df_Usar["fami_estratovivienda"])
df_Usar["fami_estratovivienda"] = np.where((df_Usar["fami_estratovivienda"]=="Estrato 6"),6,df_Usar["fami_estratovivienda"])
df_Usar["fami_estratovivienda"] = np.where((df_Usar["fami_estratovivienda"]=="Sin Estrato"),0,df_Usar["fami_estratovivienda"])






#Puntaje Global
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<100)&(df_Usar["punt_global"]>=0),100,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<120)&(df_Usar["punt_global"]>=100),110,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<140)&(df_Usar["punt_global"]>=120),130,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<160)&(df_Usar["punt_global"]>=140),150,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<180)&(df_Usar["punt_global"]>=160),170,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<200)&(df_Usar["punt_global"]>=180),190,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<220)&(df_Usar["punt_global"]>=200),210,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<240)&(df_Usar["punt_global"]>=220),230,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<260)&(df_Usar["punt_global"]>=240),250,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<280)&(df_Usar["punt_global"]>=260),270,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<300)&(df_Usar["punt_global"]>=280),280,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<320)&(df_Usar["punt_global"]>=300),310,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<340)&(df_Usar["punt_global"]>=320),330,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<360)&(df_Usar["punt_global"]>=340),350,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<380)&(df_Usar["punt_global"]>=360),370,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<400)&(df_Usar["punt_global"]>=380),390,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<420)&(df_Usar["punt_global"]>=400),410,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<440)&(df_Usar["punt_global"]>=420),430,df_Usar["punt_global"])
df_Usar["punt_global"] = np.where((df_Usar["punt_global"]<500)&(df_Usar["punt_global"]>=440),450,df_Usar["punt_global"])




# fami _ tieneinternet
df_Usar["fami_tieneinternet"] = np.where((df_Usar["fami_tieneinternet"]=="No"),0,df_Usar["fami_tieneinternet"])
df_Usar["fami_tieneinternet"] = np.where((df_Usar["fami_tieneinternet"]=="Si"),1,df_Usar["fami_tieneinternet"])


# fami_tienecomputador
df_Usar["fami_tienecomputador"] = np.where((df_Usar["fami_tienecomputador"]=="No"),0,df_Usar["fami_tienecomputador"])
df_Usar["fami_tienecomputador"] = np.where((df_Usar["fami_tienecomputador"]=="Si"),1,df_Usar["fami_tienecomputador"])


# cole_jornada
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="MAÑANA"),0,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="TARDE"),1,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="NOCHE"),2,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="COMPLETA"),3,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="UNICA"),4,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="SABATINA"),5,df_Usar["cole_jornada"])



# cole_jornada
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="MAÑANA"),0,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="TARDE"),1,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="NOCHE"),2,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="COMPLETA"),3,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="UNICA"),4,df_Usar["cole_jornada"])
df_Usar["cole_jornada"] = np.where((df_Usar["cole_jornada"]=="SABATINA"),5,df_Usar["cole_jornada"])



#
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="1 a 2"),1,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="3 a 4"),3,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="5 a 6"),5,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="7 a 8"),7,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="9 o más"),9,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Cinco"),5,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Cuatro"),4,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Diez"),10,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Doce o más"),12,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Dos"),2,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Nueve"),9,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Once"),11,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Ocho"),8,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Seis"),6,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Siete"),7,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Tres"),3,df_Usar["fami_personashogar"])
df_Usar["fami_personashogar"] = np.where((df_Usar["fami_personashogar"]=="Una"),1,df_Usar["fami_personashogar"])


# In[8]:


df_Usar1 = df_Usar[['cole_area_ubicacion','cole_bilingue','cole_calendario','cole_caracter','cole_jornada',
                   'cole_naturaleza','fami_estratovivienda','fami_personashogar','fami_tienecomputador','fami_tieneinternet',
                  'punt_global']]


# In[11]:


samples = df_Usar1
samples


# In[12]:


from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
scoring_method = K2Score(data=samples)
esth = HillClimbSearch(data=samples)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=7, max_iter=int(1e4),fixed_edges = {("cole_area_ubicacion",""),("",""),
                                                                                    ("",""),("",""),
                                                                                   ("",""),
                                                                                   ("",""),
                                                                                   ("",""),
                                                                                   ("",""),
                                                                                    ("",""),
                                                                                    ("",""),
                                                                                    ("",""),
                                                                                    ("","")
                                                                                   },
                                                                                    black_list = {("",""),
                                                                                                 ("",""),
                                                                                                 ("",""),
                                                                                                 ("",""),
                                                                                                 ("",""),
                                                                                                 ("",""),
                                                                                                  ("",""),
                                                                                                  ("",""),
                                                                                                  ("",""),
                                                                                                  ("",""),
                                                                                                 ("",""),
                                                                                                 ("",""),
                                                                                                 ("",""),
                                                                                                 ("",""),
                                                                                                 ("",""),
                                                                                                  ("",""),
                                                                                                  ("",""),
                                                                                                  ("",""),
                                                                                                  ("",""),
                                                                                                  ("","")
                                                                                                 }
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


# In[13]:


print(scoring_method.score(estimated_modelh))


# In[ ]:





# In[25]:



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
plt.figure(figsize=(180, 160))
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800)

plt.show() # Mostrar el gráfico


# In[18]:


plt.show()

