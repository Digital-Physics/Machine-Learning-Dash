# -*- coding: utf-8 -*-
"""
Created 2019
@author: jon khanlian
"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np
from keras.models import Model, load_model, model_from_json
from keras.layers import Dense, Input, Reshape 
from keras.layers import BatchNormalization

#from dash.dependencies import Input, Output
#import pandas as pd
#import plotly.plotly as py
#import dash_table
#import random
#from keras.optimizers import Adam, SGD
#from keras.initializers import Zeros
#import keras.backend as K
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from IPython.display import HTML
#from tqdm import tqdm

#this defines how your cellular automata should be updated
def life_step(life_matrix, on_numbers, n_neigh, s_neigh, e_neigh, w_neigh):
    life_matrix_out = np.zeros(life_matrix.shape)
    for i in range(0, life_matrix.shape[0]):
        for j in range(0, life_matrix.shape[1]):
            if (life_matrix[i,j]+life_matrix[i-s_neigh,j]+life_matrix[(i+n_neigh) % life_matrix.shape[0],j]+life_matrix[i,j-w_neigh]+life_matrix[i,(j+e_neigh) % life_matrix.shape[1]]) in on_numbers:
                life_matrix_out[i,j]=1 
            else:
                life_matrix_out[i,j]=0  
    return life_matrix_out

#this defines the Keras Machine Learning architecture for training
#it also generates the trainining examples    
def train_model(on_prob, training_number, epoch_num, on_states, n_off, s_off, e_off, w_off):
    one_dimension = 10 
    
    #generate training examples
    x = np.random.choice(a=[0,1], p=[1-on_prob, on_prob], size=(training_number, one_dimension, one_dimension))
    y = np.zeros((x.shape[0], one_dimension*one_dimension))
    for i in range(x.shape[0]-1):
        y[i] = life_step(x[i], on_states, n_off, s_off, e_off, w_off).ravel()
    
    #define the machine leanrning architecture
    X_input = Input(shape = (10,10))
    X = Reshape((100,))(X_input)
    X = Dense(100, activation = "sigmoid")(X)
    X = BatchNormalization()(X)
    X = Dense(100, activation = "sigmoid")(X)
    X = BatchNormalization()(X)
    X = Dense(100, activation = "sigmoid")(X)
    X = BatchNormalization()(X)
    X = Dense(100, activation = "sigmoid")(X)
    X = BatchNormalization()(X)
    X = Dense(100, activation = "sigmoid")(X)
    model = Model(inputs = X_input, outputs = X)
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['accuracy'])
    #model.summary()
    #for e in range(1,4): #epochs
    #    print("Epoch %d" %e)
    #    for _ in tqdm(range(50000)): #batch size
    #        model.train_on_batch(x, y)

    model.fit(x, y, validation_split=0.20, batch_size = 1000, epochs=epoch_num)
    
    return model

#app = dash.Dash()
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
        #line one
        html.Div([
            html.Div([    
                    html.Img(src=app.get_asset_url('conway2.png')), #image is located in 'assets' folder in H: drive, same directory where AV600.py is located
                    ], style={'display': 'inline-block','marginLeft':25}), 
            html.Div([   
                    html.H2('Predicting the Next State in (a Generalize Version of) "Conway\'s Game of Life"'),
                    html.H3('...Using Neural Networks')
                    ], style={'display': 'inline-block','marginLeft':25, 'vertical-align': 'top'}) 
        ], style={'width': '100%', 'display': 'inline-block'}),
        #line two
        html.Div([  
            html.Div([
                    dcc.Markdown('''Training Set Parameters'''),
                    ], style={'width': '45%', 'display': 'inline-block','marginLeft':25, 'color': 'blue', 'fontSize': 24}),
            html.Div([
                    dcc.Markdown('''Update Rules (For Generating Training & Test Set Answers)'''),
                    ], style={'width': '45%', 'display': 'inline-block', 'marginLeft':45, 'color': 'orange', 'fontSize': 24}),
        ], style={'width': '100%', 'display': 'inline-block'}),
        #line three
        html.Div([   
            #line three, left side    
            html.Div([
                html.Button('Train Your Neural Net', id='button'),
                html.Div(id='text-div'),
                html.Div([],style={'width': '100%', 'display': 'inline-block'}), 
                html.Div(
                        [
                                dbc.Progress(id="progress", value=0, striped=True, animated=True),
                                dcc.Interval(id="interval", interval=250, n_intervals=0),
                         ]),
                html.Div([],style={'width': '100%', 'display': 'inline-block'}), 
                dcc.Markdown('''Number of Training Examples:'''),
                dcc.Slider(id='train_num_slider', min=10, max=100000, step=10, value=10000, 
                           marks={ 10: '10', 50000: '50,000', 100000: '100,000'}),
                html.Div([],style={'width': '100%', 'display': 'inline-block'}),
                dcc.Markdown('''Number of Epochs:'''),
                dcc.Slider(id='epoch_num_slider', min=1, max=50, step=1, value=10, 
                           marks={ 1: '1', 10: '10', 20: '20', 30: '30', 40: '40', 50: '50'}),           
                html.Div([],style={'width': '100%', 'display': 'inline-block'}),
                dcc.Markdown('''Probability a Given Cell in Initial State is On:'''),
                dcc.Slider(id='probability_slider_train', min=0, max=10, step=1, value=2, #marks={i: format(i) for i in np.round(np.linspace(0,1,11),1)}),
                    marks={ 0: '0%', 1: '10%', 2: '20%', 3: '30%', 4: '40%', 5: '50%', 6: '60%', 7: '70%', 8: '80%', 9: '90%',  10: '100%'})
            ], style={'width': '45%', 'display': 'inline-block','marginLeft':25}),
            #line three, right side    
            html.Div([ 
                dcc.Markdown('''North, South, East, and West Neighbor Distance (Normally = 1):'''),    
                html.Div([
                        html.Div([
                                dcc.Slider(id='north_offset', min=1, max=9, step=1, value=1, 
                                           marks={ 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'})],
                                           style={'width': '18%', 'display': 'inline-block'}),
                        html.Div([
                                dcc.Slider(id='south_offset', min=1, max=9, step=1, value=1, 
                                           marks={ 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'})],
                                           style={'width': '18%', 'marginLeft':20, 'display': 'inline-block'}),
                        html.Div([
                                dcc.Slider(id='east_offset', min=1, max=9, step=1, value=1, 
                                           marks={ 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'})],
                                           style={'width': '18%', 'marginLeft':20, 'display': 'inline-block'}),
                        html.Div([
                                dcc.Slider(id='west_offset', min=1, max=9, step=1, value=1, 
                                           marks={ 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'})],
                                           style={'width': '18%', 'marginLeft':20, 'display': 'inline-block'}),
                        ],style={'width': '100%', 'display': 'inline-block'}),
                html.Div([],style={'width': '100%', 'display': 'inline-block'}),                           
                dcc.Markdown('''Number of "On Cells" in Von Neumann Neighborhood That Lead to "On Cell" in Next Step:'''),
                html.Div([
                        dcc.Checklist(id= 'on_states', 
                                options=[
                                {'label': '0', 'value': 0},
                                {'label': '1', 'value': 1},
                                {'label': '2', 'value': 2},
                                {'label': '3', 'value': 3},
                                {'label': '4', 'value': 4},
                                {'label': '5', 'value': 5}], 
                                values=[2, 3], 
                                inputStyle={"margin-left": "20px"}),
                        ],style={'width': '100%', 'display': 'inline-block'}),
                html.Div([
                dcc.Markdown('''Test Set Parameters'''),
                    ], style={'width': '45%', 'color': 'green', 'fontSize': 24}),    
                html.Button('Test Your Neural Net', id='button2'),    
                dcc.Markdown('''Probability a Given Cell in Initial State is On:'''),
                #dcc.Store(id='store',storage_type='local'),
                dcc.Slider(id='probability_slider_test', min=0, max=10, step=1, value=2, #marks={i: format(i) for i in np.round(np.linspace(0,1,11),1)}),
                    marks={ 0: '0%', 1: '10%', 2: '20%', 3: '30%', 4: '40%', 5: '50%', 6: '60%', 7: '70%', 8: '80%', 9: '90%',  10: '100%'}),       
                html.Div(id='intermediate-value', style={'display': 'none'}), #need to figure out how to do a proper storing of data
            ], style={'width': '45%', 'display': 'inline-block', 'marginLeft':45}),
        ], style={'width': '100%', 'display': 'inline-block'}), 
        
        #line four
        html.Div([   
            html.Div([
                html.Div([],style={'width': '100%', 'display': 'inline-block'}),     
                dcc.Graph(
                id='accuracy',
                figure={'data': [go.Scatter(x=[], y=[])],
                'layout': {'title': 'Training Accuracy'}}#, 'xaxis': {'title':'Epoch'}}
                ), 
                dcc.Graph(
                id = 'training_x',
                figure={'data': [go.Heatmap(z=[np.random.choice(a=[0,1], p=[0.6,0.4], size=(10,10))])],
                'layout': {'title':'Training Set Example: Initial State/Input/X'}}
                ),
                dcc.Graph(
                id = 'training_y',
                figure={'data': [go.Heatmap(z=[np.random.choice(a=[0,1], p=[0.6,0.4], size=(10,10))])],
                'layout': {'title':'Training Set Example: Next State/Output/Y'}},
                ),
                    ], style={'width': '45%', 'display': 'inline-block','marginLeft':25, 'vertical-align': 'top'}),
            html.Div([
                html.Div([],style={'width': '100%', 'display': 'inline-block'}),
                dcc.Graph(
                id = 'initial_state',
                figure={'data': [go.Heatmap(z=[np.random.choice(a=[0,1], p=[0.6,0.4], size=(10,10))])],
                'layout': {'title':'Test Example: Initial State'}}
                ),
                dcc.Graph(
                id = 'actual_state',
                figure={'data': [go.Heatmap(z=[np.random.choice(a=[0,1], p=[0.6,0.4], size=(10,10))])],
                'layout': {'title':'Test Example: Next State'}}
                ),
                dcc.Graph(
                id = 'predicted_state',
                figure={'data': [go.Heatmap(z=[np.random.choice(a=[0,1], p=[0.6,0.4], size=(10,10))])],
                'layout': {'title':'Test Example: Predicted Next State'}}
                ),
                    ], style={'width': '45%', 'display': 'inline-block','marginLeft':25}),
        ], style={'width': '100%', 'display': 'inline-block'})      

], style={'width': '100%', 'display': 'inline-block'})                

#@app.callback(
#        dash.dependencies.Output('progress', 'value'),
#        [dash.dependencies.Input('button', 'n_clicks')],
#         #dash.dependencies.Input('interval', 'n_intervals')],
#         )
#def update_progress_bar(n_clicks): 
#    if n_clicks is None:
#        raise dash.exceptions.PreventUpdate
#    #return min(n_intervals % 110, 100)
#    return 100

#@app.callback(dash.dependencies.Output('progress', 'value'),
#              [dash.dependencies.Input('button', 'n_clicks_timestamp'), 
#               dash.dependencies.Input('button2', 'n_clicks_timestamp')])
#def set_progress(ts1, ts2):
#    if ts1 < 0 and ts2 < 0:
#        # before any click, timestamp is -1 by convention
#        raise dash.exception.PreventUpdate
#    if ts1 < 0:
#        return 0
#    if ts2 < 0:
#        return 100
#    if ts1 > ts2:
#        return 100
#    else:
#        return 0

@app.callback([
        #dash.dependencies.Output('store', 'data'),
        dash.dependencies.Output('intermediate-value', 'children'),
        dash.dependencies.Output('accuracy', 'figure'),
        dash.dependencies.Output('training_x', 'figure'),
        dash.dependencies.Output('training_y', 'figure')],
        [dash.dependencies.Input('button', 'n_clicks')],
        [dash.dependencies.State('on_states', 'values'),
         dash.dependencies.State('north_offset', 'value'),
         dash.dependencies.State('south_offset', 'value'),
         dash.dependencies.State('east_offset', 'value'),
         dash.dependencies.State('west_offset', 'value'),
         dash.dependencies.State('train_num_slider', 'value'),
         dash.dependencies.State('epoch_num_slider', 'value'),
         dash.dependencies.State('probability_slider_train', 'value')]
        #[dash.dependencies.State('store', 'data')]
         )
def update_on_action(n_clicks, on_states, n_off, s_off, e_off, w_off, slider_train, slider_epoch, slider_prob): 
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    #give an example of what a training pair looks like
    x_train = np.random.choice(a=[0,1], p=[(10-slider_prob)/10,slider_prob/10], size=(10, 10))
    y_train = life_step(x_train, on_states, n_off, s_off, e_off, w_off)
    figure1 = {'data': [go.Heatmap(z=x_train, colorscale=[[0.0, 'rgb(00,00,00)'],[1.0, 'rgb(25,255,03)']])], 'layout': {'title':'Training Set Example: Initial State/Input/X'}}
    figure2 = {'data': [go.Heatmap(z=y_train, colorscale=[[0.0, 'rgb(00,00,00)'],[1.0, 'rgb(25,255,03)']])], 'layout': {'title':'Training Set Example: Next State/Output/Y'}}
    
    #train model... this takes time    
    model = train_model(slider_prob/10, slider_train, slider_epoch, on_states, n_off, s_off, e_off, w_off) 
    
    figure3 = {'data': [go.Scatter(x=np.arange(slider_epoch)+1, y=model.history.history['val_acc'], name = 'Validation Set'),
                        go.Scatter(x=np.arange(slider_epoch)+1, y=model.history.history['acc'], name = 'Training Set')], 
                        'layout': {'title': 'Training Accuracy'}}#, 'xaxis': {'title':'Epoch'}}}
    
    #save model
    model_json = model.to_json()
    #with open("model.json","w") as json_file:
    #    json_file.write(model_json)
    model.save_weights("model.h5")

    return model_json, figure3, figure1, figure2

#@app.callback(dash.dependencies.Output('button','n_clicks'),
#             [dash.dependencies.Input('progress','value')])
#def update(value):
#    return 0

@app.callback([
        dash.dependencies.Output('initial_state', 'figure'),
        dash.dependencies.Output('actual_state', 'figure'),
        dash.dependencies.Output('predicted_state', 'figure')],
        [dash.dependencies.Input('button2', 'n_clicks')],
        [dash.dependencies.State('on_states', 'values'),
         dash.dependencies.State('north_offset', 'value'),
         dash.dependencies.State('south_offset', 'value'),
         dash.dependencies.State('east_offset', 'value'),
         dash.dependencies.State('west_offset', 'value'),
         dash.dependencies.State('probability_slider_test', 'value'),
         dash.dependencies.State('intermediate-value', 'children')])
def any_func_name(n_clicks, on_states, n_off, s_off, e_off, w_off, slider_prob, model_json):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
        
    test_validation = 100
    x_test = np.random.choice(a=[0,1],  p=[(10-slider_prob)/10, slider_prob/10], size=(test_validation, 10, 10))
    
    #show first test input and correct answer
    figure1 = {'data': [go.Heatmap(z=x_test[0], colorscale=[[0.0, 'rgb(00,00,00)'],[1.0, 'rgb(25,255,03)']])], 'layout': {'title':'Test Example: Initial State'}}
    y_test = life_step(x_test[0], on_states, n_off, s_off, e_off, w_off)
    figure2 = {'data': [go.Heatmap(z=y_test, colorscale=[[0.0, 'rgb(00,00,00)'],[1.0, 'rgb(25,255,03)']])], 'layout': {'title':'Test Example: Next State'}}
    
    #load trained model
    model = model_from_json(model_json) 
    model.load_weights("model.h5")
    
    #predict
    predictions = model.predict(x_test)
    figure3 = {'data': [go.Heatmap(z=np.round(predictions[0].reshape(10,10),2), colorscale=[[0.0, 'rgb(00,00,00)'],[1.0, 'rgb(25,255,03)']])], 'layout': {'title':'Test Example: Predicted Next State'}}
    
    return figure1, figure2, figure3

if __name__ == '__main__':
    #app.run_server(debug=False, threaded=False)
    app.run_server(debug=True, threaded=False, port=10888)
