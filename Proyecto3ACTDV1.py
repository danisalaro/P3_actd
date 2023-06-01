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


samples = pd.read_csv('DatosFinalP3.csv')


# In[4]:

'''
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


print(scoring_method.score(estimated_modelh))
'''






# In[5]:

################################################################################################
#-----------------------------------FRONT DE LA APLICACIÓN-------------------------------------#
###############################################################################################
uniandes = 'https://uniandes.edu.co/sites/default/files/logo-uniandes.png'
costa = 'https://web.comisiondelaverdad.co/images/zoo/territorios/caribe-insularnuevo.jpg'
ifex = 'https://www.icfes.gov.co/documents/39286/0/logo-prueba-2022_11.png/b3f43026-92c4-09c5-c034-527faf7952ca?t=1652155271251&download=true'
mined = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/MinEducaci%C3%B3n_%28Colombia%29_logo.png/1200px-MinEducaci%C3%B3n_%28Colombia%29_logo.png'
nenes = 'https://laguajirahoy.com/wp-content/uploads/2015/09/Cien-estudiantes-de-los-grados-noveno-d%C3%A9cimo-y-once-de-cuatro-instituciones.jpg'
import dash_bootstrap_components as dbc
app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])

app.css.append_css({
    'external_url': 'https://bootswatch.com/4/darkly/bootstrap.css'
})

#Creación de cada uno de los tabs

##df_Usar1 = df_Usar[['cole_area_ubicacion','cole_bilingue','cole_calendario','cole_caracter','cole_jornada',
##                   'cole_naturaleza','fami_estratovivienda','fami_personashogar','fami_tienecomputador','fami_tieneinternet',
##                  'punt_global']]


bienvenida = dbc.Card(
    dbc.CardBody(
        [
            html.Div([
                html.H1("PROYECTO ANALÍTICA COMPUTACIONAL PARA LA TOMA DE DECISIONES", className="text-center"),
                html.Br(),
                html.Br(),
                html.P("El objetivo del presente proyecto consiste en generar una herramienta de analítica de datos para las gobernaciones de Atlántico, Bolívar, Magdalena, La Guajira, Cesar, Córdoba y Sucre. Con el fin de predecir el comportamiento de las pruebas Saber 11® de un individuo según sus caracterizaciones socioeconómicas con el fin de evaluar posibles planes de mejora en conjunto con el Gobierno Nacional de Colombia", style={'text-align':'justify', 'font-size':'1.4em'}),
                html.Div([
                        html.Img(src=costa, style={"width": "1000px", "height": "510px"}),
                        html.Div([html.Img(src=uniandes, style={"width": "300px", "height": "160px"}),
                        html.Img(src=ifex, style={"width": "310px", "height": "120px"}),html.Img(src=mined, style={"width": "400px", "height": "80px"}),])
                    ], style={'text-align': 'center'}),
                html.Br(),
                html.P("Proyecto realizado por: Juan Diego Prada y Daniel Felipe Salazar"),
                html.P("DISCLAIMER: El presente proyecto no representa a la Universidad de los Andes ni sus intereses.", style={'color': 'red', 'text-align':'center', 'font-size':'150%'}, className="fst-italic")
        ], style={'margin':'30px'})
            
        ]
    ),
    className="mt-3", 
    style={
        'border': 'none', 
        'background-color': 'transparent',
        'width': '100%', 
        'height': '100%', 
        'display': 'flex', 
        'justify-content': 'center', 
        'align-items': 'center',
        'flex-direction': 'column'
    }
)

tab1_content = dbc.Card(
    dbc.CardBody(
        [       html.Div([
                        html.Br(),
                        dbc.Row([
                            dbc.Row(
                            
                                dbc.Col(html.H1("PREDICCIÓN DEL RESULTADO DE PRUEBAS SABER 11 ® PARA UN ESTUDIANTE DE LA COSTA", className="text-center", style={'margin-top':'-20px'}))),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            dbc.Col(html.Div([ 
                                html.P("Por favor seleccione los datos del estudiante:"),
                                dcc.Dropdown(
                                id="cole_area_ubicacion",
                                options=[
                                        {'label': 'URBANO', 'value': 1},
                                        {'label': 'RURAL', 'value': 0},
                                        ],
                                placeholder = "Área donde se encuentra el colegio"
                            ),
                                dcc.Dropdown(
                                id="cole_jornada",
                                options=[
                                        {'label': 'MAÑANA', 'value': 0},
                                        {'label': 'TARDE', 'value': 1},
                                        {'label': 'NOCHE', 'value': 2},
                                        {'label': 'COMPLETA', 'value': 3},
                                        {'label': 'ÚNICA', 'value': 4},
                                        {'label': 'SABATINA', 'value': 5},
                                        ],
                                placeholder="Jornada del colegio",
                            
                            ),
                                dcc.Dropdown(
                                id="cole_caracter",
                                options=[
                                        {'label': 'TÉCNICO/ACADÉMICO', 'value': 0},
                                        {'label': 'ACADÉMICO', 'value': 1},
                                        {'label': 'TÉCNICO', 'value': 2},
                                        {'label': 'NO APLICA', 'value': 3},
                                        ],
                                placeholder="Caracter del colegio",
                                
                            ),
                                dcc.Dropdown(
                                id="cole_calendario",
                                options=[
                                        {'label': 'CALENDARIO A', 'value': 0},
                                        {'label': 'CALENDARIO B', 'value': 1},
                                        ],
                                placeholder="Calendario del colegio",
                                
                            ),
                                dcc.Dropdown(
                                id="cole_bilingue",
                                    options=[
                                        {'label': 'NO', 'value': 0},
                                        {'label': 'SI', 'value': 1},
                                        ],
                                placeholder="¿Colegio es bilingüe?",
                            
                            ),
                                dcc.Dropdown(
                                id="cole_naturaleza",
                                options=[
                                        {'label': 'PÚBLICO', 'value': 0},
                                        {'label': 'PRIVADO', 'value': 1},
                                        ],
                                placeholder="¿Colegio público o privado?",
                            
                            ),
                                dcc.Dropdown(
                                id="fami_estratovivienda",
                                options=[
                                        {'label': '1', 'value': 1},
                                        {'label': '2', 'value': 2},
                                        {'label': '3', 'value': 3},
                                        {'label': '4', 'value': 4},
                                        {'label': '5', 'value': 5},
                                        {'label': '6', 'value': 6},
                                        {'label': 'Sin estrato', 'value': 0},
                                        ],
                                placeholder="Estrato en el que vive",

                            ),
                            dcc.Dropdown(
                                id="fami_personashogar",
                                options=[
                                        {'label': '1', 'value': 1},
                                        {'label': '2', 'value': 2},
                                        {'label': '3', 'value': 3},
                                        {'label': '4', 'value': 4},
                                        {'label': '5', 'value': 5},
                                        {'label': '6', 'value': 6},
                                        {'label': '7', 'value': 7},
                                        {'label': '8', 'value': 8},
                                        {'label': '9', 'value': 9},
                                        {'label': '10', 'value': 10},
                                        {'label': '11', 'value': 11},
                                        {'label': '12 O MÁS', 'value': 12},
                                        ],
                                placeholder="Número de personas en el hogar",
                                
                            ),
                            dcc.Dropdown(
                                id="fami_tienecomputador",
                                options=[
                                        {'label': 'SI', 'value': 1},
                                        {'label': 'NO', 'value': 0},
                                        ],
                                placeholder="¿Tiene computador?",
                                
                            ),
                             dcc.Dropdown(
                                id="fami_tieneinternet",
                                options=[
                                        {'label': 'SI', 'value': 1},
                                        {'label': 'NO', 'value': 0},
                                        ],
                                placeholder="¿Tiene acceso a internet?",
                                
                            )]), width=6),
                            
                            dbc.Col(html.Img(src=nenes, style={"width": "350", "height": "455px"}), style={'margin-left':'120px','margin-top':'30px'}),
                        ]),
                        html.Div([
                            html.Br(),
                            dbc.Button("Consultar", color="dark", className="me-1", id='btn')
                        ]),
                        html.Div([
                            html.Br(),
                            html.H4(id="output")
                        , ]),
                        html.Div([
                          html.Div([html.Img(src=uniandes, style={"width": "300px", "height": "160px"}),
                            html.Img(src=ifex, style={"width": "310px", "height": "120px"}),html.Img(src=mined, style={"width": "400px", "height": "80px"}),])
                        , ], style={'text-align': 'center'})
                ], style={'margin':'30px'})
        ]
    ),
    className="mt-3", style={'border': 'none', 'background-color': 'transparent','width': '100%', 'height': '100%'}
)




tab2_content = dbc.Card(
    dbc.CardBody(
        [
        html.Div([
            html.Div([
                dcc.Graph(
                        id='heatmap',
                        
                    )
            ], style={'margin':'30px'}),
            html.Br(),
            
            html.Div([
                
                dcc.Graph(
                        id='heatmap',
                        
                    )
            ],style={'margin':'30px'}),
            
            html.Br(),
            
             html.Div([
                
                dcc.Graph(
                        id='heatmap',
                        
                    )
            ],style={'margin':'30px'})
            
           ], style={'margin-left':'75px'}) 
        ]
    ),
    className="mt-3", style={'border': 'none', 'background-color': 'transparent','width': '100%', 'height': '100%'}
)


tabs = dbc.Tabs(
    [
        dbc.Tab(bienvenida, label="Bienvenida"),
        dbc.Tab(tab1_content, label="Interfaz"),
        dbc.Tab(tab2_content, label="Gráficos de Interés"),
        
    ]
)

app.layout = html.Div([
    tabs,    
],style={"margin-top": "30px", "margin-left":"60px"})

@app.callback(Output("output", "children"), Input("btn", "n_clicks"),
              Input("cole_area_ubicacion", "value"), Input("cole_bilingue", "value"),
              Input("cole_calendario", "value"), Input("cole_caracter", "value"),
              Input("cole_jornada","value"),Input("cole_naturaleza","value"),Input("fami_estratovivienda","value"),
              Input("fami_personashogar","value"),Input("fami_tienecomputador","value"),Input("fami_tieneinternet","value"))


def run_query(n_clicks, age, trestbps, chol, fbs, sex, restecg, thalach,cp,exang):
    if n_clicks is not None:
        posterior_p2 = infer.query(["Num"], evidence={"Age": age, "Trestbps": trestbps, "Chol": chol, "Fbs": fbs, "Sex": sex,"Restecg": restecg,"Thalach": thalach,"CP": cp,"Exang": exang})
        suma = posterior_p2.values[1]+posterior_p2.values[2]+posterior_p2.values[3]+posterior_p2.values[4]
        return f"La probabilidad del paciente de tener la enfermedad es de: {round(suma*100,2)}%"
if __name__ == '__main__':
    app.run_server(debug=False)
    





