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


from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# Realizar la búsqueda de HillClimb
scoring_method = K2Score(data=samples)
esth = HillClimbSearch(data=samples)
estimated_modelh = esth.estimate(
    scoring_method=scoring_method, max_indegree=2, max_iter=int(1e4))

# Imprimir el modelo estimado
print(estimated_modelh)

# Imprimir los nodos y las aristas del modelo estimado
print(estimated_modelh.nodes())
print(estimated_modelh.edges())

# Crear el gráfico dirigido del modelo
G = nx.DiGraph()
for parent, child in estimated_modelh.edges():
    G.add_edge(parent, child)

# Dibujar el gráfico del modelo
pos = nx.spring_layout(G, seed=42)  # Asignar posiciones a los nodos
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)  # Dibujar nodos
nx.draw_networkx_edges(G, pos, arrows=True)  # Dibujar arcos con flechas
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")  # Agregar etiquetas de nodos

plt.axis("off")  # Ocultar los ejes
plt.title("Modelo Estimado")
plt.show()  # Mostrar el gráfico

# Calcular el puntaje del modelo estimado
print(scoring_method.score(estimated_modelh))

# Crear el modelo bayesiano y ajustarlo a los datos
estimated_model = BayesianNetwork(estimated_modelh)
estimated_model.fit(data=samples, estimator=MaximumLikelihoodEstimator)
for i in estimated_model.nodes():
    print(estimated_model.get_cpds(i))

# Realizar inferencia en el modelo
infer = VariableElimination(estimated_model)




################################################################################################
#--------------------------------------VISUALIZACIONES-----------------------------------------#
###############################################################################################

# Crear las visualizaciones
df = pd.read_csv('Datos_Raw.csv')

# Verificar una figura para comparar el colegio con el promedio de pruebas icfes
df_col1 = df.groupby(["cole_jornada"], as_index=False)["punt_global"].mean()
print(df_col1)
fig2 = px.bar(df_col1, x = "cole_jornada", y = "punt_global", barmode='group')
fig2.update_layout(
    xaxis_title="Jornada del colegio",
    yaxis_title="Promedio puntaje prueba",
    font=dict(size=12, color='black'),
    legend=dict(title='', font=dict(size=12)),
    margin=dict(t=80),
    plot_bgcolor='white',
    coloraxis=dict(colorbar=dict(title='', tickfont=dict(size=12)), colorbar_len=0.3),
)
fig2.update_layout(
    width=1000,
    height=500
)



# Obtener los datos agrupados
df_col2 = df.groupby(["fami_estratovivienda", "cole_area_ubicacion"], as_index=False)["punt_global"].mean()

# Agregar columna de orden de estrato
df_col2['orden_estrato'] = np.where(df_col2['fami_estratovivienda'] == 'Sin Estrato', 0, 1)

# Ordenar el DataFrame por orden de estrato y fami_estratovivienda
df_col2 = df_col2.sort_values(by=['orden_estrato', 'fami_estratovivienda'])

# Crear la gráfica
fig3 = px.line(df_col2, x='fami_estratovivienda', y='punt_global', color='cole_area_ubicacion')

fig3.update_layout(
    xaxis_title="Estrato",
    yaxis_title="Puntaje promedio prueba",
    font=dict(size=12, color='black'),
    legend=dict(title='', font=dict(size=12)),
    margin=dict(t=80),
    plot_bgcolor='white',
    coloraxis=dict(colorbar=dict(title='', tickfont=dict(size=12)), colorbar_len=0.3)
)
fig3.update_layout(
    width=1000,
    height=500
)

# Mostrar la gráfica



# Definir el orden personalizado de las categorías
order = ["Una", "Dos", "Tres", "Cuatro", "Cinco", "Seis", "Siete", "Ocho", "Nueve", "Diez", "Once", "Doce o más"]

# Filtrar el DataFrame para las categorías seleccionadas
selected_categories = ["Una", "Dos", "Tres", "Cuatro", "Cinco", "Seis", "Siete", "Ocho", "Nueve", "Diez", "Once", "Doce o más"]
df_filtered = df[df['fami_personashogar'].isin(selected_categories)]

# Obtener los datos agrupados
df_col3 = df_filtered.groupby(["fami_estratovivienda", "fami_personashogar"], as_index=False)["punt_global"].mean()

# Crear una columna para el orden personalizado
df_col3['order'] = pd.Categorical(df_col3['fami_personashogar'], categories=order, ordered=True)

# Ordenar el DataFrame por el orden personalizado
df_col3 = df_col3.sort_values(by=['fami_estratovivienda', 'order'])

# Crear la gráfica
fig4 = px.line(df_col3, x='fami_personashogar', y='punt_global', color='fami_estratovivienda')

# Mostrar la gráfica

fig4.update_layout(
    xaxis_title="Personas en el hogar",
    yaxis_title="Puntaje promedio prueba",
    font=dict(size=12, color='black'),
    legend=dict(title='', font=dict(size=12)),
    margin=dict(t=80),
    plot_bgcolor='white',
    coloraxis=dict(colorbar=dict(title='', tickfont=dict(size=12)), colorbar_len=0.3)
)
fig4.update_layout(
    width=1000,
    height=500
)




################################################################################################
#-----------------------------------FRONT DE LA APLICACIÓN-------------------------------------#
###############################################################################################
uniandes = 'https://uniandes.edu.co/sites/default/files/logo-uniandes.png'
costa = 'https://web.comisiondelaverdad.co/images/zoo/territorios/caribe-insularnuevo.jpg'
ifex = 'https://www.icfes.gov.co/documents/39286/0/logo-prueba-2022_11.png/b3f43026-92c4-09c5-c034-527faf7952ca?t=1652155271251&download=true'
mined = 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/MinEducaci%C3%B3n_%28Colombia%29_logo.png/1200px-MinEducaci%C3%B3n_%28Colombia%29_logo.png'
nenes = 'https://www.elheraldo.co/sites/default/files/articulo/2022/09/05/whatsapp_image_2022-09-05_at_05.51.43.jpeg'
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
                        figure = fig2
                    )
            ], style={'text-align':'center'}),
            html.Br(),
            
            html.Div([
                
                dcc.Graph(
                        id='heatmap',
                        figure = fig3
                    )
            ],style={'text-align':'center'}),
            
            html.Br(),
            
             html.Div([
                
                dcc.Graph(
                        id='heatmap',
                        figure = fig4
                    )
            ],style={'text-align':'center'})
            
           ], style={'margin-left':'375px'}) 
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


def run_query(n_clicks,cole_area_ubicacion, cole_bilingue,cole_calendario,
              cole_caracter,cole_jornada,cole_naturaleza,fami_estratovivienda,
              fami_personashogar,fami_tienecomputador,fami_tieneinternet):
    if n_clicks is not None:
        posterior_p2 = infer.query(["punt_global"], evidence={"cole_area_ubicacion": cole_area_ubicacion,
                                                              "cole_bilingue": cole_bilingue, "cole_calendario": cole_calendario,
                                                              "cole_caracter": cole_caracter, "cole_jornada": cole_jornada,"cole_naturaleza": cole_naturaleza,
                                                              "fami_estratovivienda": fami_estratovivienda,"fami_personashogar": fami_personashogar,
                                                              "fami_tienecomputador": fami_tienecomputador,"fami_tieneinternet":fami_tieneinternet})
        
        # Obtener los nombres de los estados y los valores de probabilidad
        estados_punt_global = posterior_p2.state_names["punt_global"]
        probabilidades_punt_global = posterior_p2.values
        
        # Obtener el índice del valor con mayor probabilidad
        indice_max_probabilidad = np.argmax(probabilidades_punt_global)
        
        # Obtener el valor de "punt_global" con mayor probabilidad
        valor_max_probabilidad = estados_punt_global[indice_max_probabilidad]

        return f"El valor de 'punt_global' con mayor probabilidad es:, {valor_max_probabilidad:}"
if __name__ == '__main__':
    app.run_server(debug=False)
    






# %%
