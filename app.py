
from datetime import date
import math
from yahoo_fin import news
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output,State
import pandas as pd
import numpy as np
import yfinance as yf
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
## from sklearn.preprocessing import MinMaxScaler
## import tensorflow as tf
from alpha_vantage.timeseries import TimeSeries



#Constants to download data
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
API_KEY = 'T6X62ZNC78VB55YK'
IMGE = 'assets/rss.png'

#Grabbing all tick symbols from a csv
tick_data = pd.read_csv('ticks.csv')
tick_df = pd.DataFrame(tick_data)
tick_df = tick_df.drop('Company Name',axis=1)
ticks = tick_df['Symbol'].to_list()

#Helper function to load historical daily data
def load_daily(ticker):
    """
    Loads the dataset for the selected ticker
    """
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data 

#Helper function to load intraday data
def load_intraday(ticker):
    """
    Loads the dataset for the selected ticker
    """
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, metadata = ts.get_intraday(symbol=ticker, interval='1min', outputsize='compact')
    data = data.rename(columns={'1. open': 'open',
                        '2. high': 'high',
                        '3. low': 'low', 
                        '4. close': 'close',
                        '5. volume': 'volume'})
    return data

#Initializing dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(children=[
                html.Div(children=[
                    html.H2(children='Stock data dashboard'),
                    html.H6(children='Analyze stock data from Jan 2018 till date', style={'marginTop': '-15px', }),
                    html.H6(children='Using hourly and daily data', style={'marginTop': '-15px','marginBottom': '30px'})
                ], style={'textAlign': 'center'}),

                html.Div(children=[
                    html.Div(children=[
                        html.Label('Select ticker:', style={'paddingTop': '2rem'}),
                        dcc.Dropdown(
                                id = 'ticker-drop',
                                options = [{'label': i,'value':i} for i in ticks ],
                                multi = False,
                                value = 'AAPL',
                                className= "mb-3"
                            ),
                    ],),
                    html.Hr(),
                    html.Div(children=[
                        html.Label('RSS feeds ',style={'paddingTop': '2rem'}),
                        html.Img(src= IMGE, height="30px")
                        ], style={'display':'inline'}        
                    ),
                    html.Hr(),
                    html.Div(children=[
                        html.B(id='one-title',style={}),
                        html.Br(),
                        html.A(id='one-link', children='Click to view...'),
                        html.P(id='one-published')
                    ]),
                    html.Div(children=[
                        html.B(id='two-title',style={}),
                        html.Br(),
                        html.A(id='two-link', children='Click to view...'),
                        html.P(id='two-published')
                    ]),
                    html.Div(children=[
                        html.B(id='three-title',style={}),
                        html.Br(),
                        html.A(id='three-link', children='Click to view...'),
                        html.P(id='three-published')
                    ])
                ],className="three columns",
                style={'padding':'2rem', 'margin':'1rem', 'boxShadow': '#e3e3e3 4px 4px 2px', 'border-radius': '10px', 'marginTop': '2rem'} ),
                               
# Number statistics & OHLC each day
            html.Div(children= [ 
                html.Div(children=[
                    #Stats box 
                    html.Div(children=[
                        html.Div(children=[
                                    html.H3(id='open-sb', style={'fontWeight': 'bold'}),
                                    html.Label('Opening price in USD($)', style={'paddingTop': '.3rem'}),
                        ], className="two columns number-stat-box"),
                    
                        html.Div(children=[
                                    html.H3(id='high-sb', style={'fontWeight': 'bold', 'color': '#f73600'}),
                                    html.Label('Highest price in USD($)', style={'paddingTop': '.3rem'}),
                        ], className="two columns number-stat-box"),

                        html.Div(children=[
                                    html.H3(id='low-sb', style={'fontWeight': 'bold', 'color': '#00aeef'}),
                                    html.Label('Lowest price in USD($)', style={'paddingTop': '.3rem'}),
                        ], className="two columns number-stat-box"),
                        
                        html.Div(children=[
                                    html.H3(id='close-sb', style={'fontWeight': 'bold', 'color': '#a0aec0'}),
                                    html.Label('Closing price in USD($)', style={'paddingTop': '.3rem'}),
                        ], className="two columns number-stat-box"),
                        
                        html.Div(children=[
                                    html.H3(id='volume-sb', style={'fontWeight': 'bold', 'color': '#0fa'}),
                                    html.Label('Volume of shares', style={'paddingTop': '.3rem'}),
                        ], className="two columns number-stat-box")],
                        style= {'margin':'1rem', 'display': 'flex', 'justify-content': 'space-between', 'width': '100%', 'flex-wrap': 'wrap'}),

                    #OH & LC Graphs
                    html.Div(children=[
                        dcc.Graph(id='open-close-fig')
                    ], className="six columns", style={'padding':'.3rem', 'marginTop':'1rem', 'marginLeft':'1rem', 'boxShadow': '#e3e3e3 4px 4px 2px', 'border-radius': '10px', 'backgroundColor': 'white', }),
                    
                    html.Div(children=[
                        dcc.Graph(id='high-low-fig')
                    ], className="six columns", style={'padding':'.3rem', 'marginTop':'1rem', 'marginLeft':'3rem', 'boxShadow': '#e3e3e3 4px 4px 2px', 'border-radius': '10px', 'backgroundColor': 'white', }),
                   
                ], className="twelve columns", style={'backgroundColor': '#f2f2f2', 'margin': '1rem'})
            ], style={'display': 'flex', 'flex-wrap': 'wrap'}),

        html.Div(children=[
                    #Forecast button 
                    html.Div(children=[
                        html.Label('Forecast using Daily data  ', style={'paddingTop': '2rem'}),
                        html.Br(),
                        dbc.Button("View", id='forecast-button', color="primary", className="button", type='button')
                    ],)
                ],className="two columns",
                style={'padding':'2rem', 'margin':'1rem', 'boxShadow': '#e3e3e3 4px 4px 2px', 'border-radius': '10px', 'marginTop': '2rem'} ),
       
        html.Div(children=[
        # scatter chart for forecasting
            html.Div(children=[
                html.Div(children=[
                    dcc.Loading(id="loading-2",
                            children=[html.Div(dcc.Graph(id= 'Open-forecast'))],
                            type='circle',
                            fullscreen=False,
                            color='#119DFF')
                    ], className="six columns widget-box", style={'padding':'.3rem', 'marginTop':'1rem', 'marginLeft':'1rem', 'boxShadow': '#e3e3e3 4px 4px 2px', 'border-radius': '10px', 'backgroundColor': 'white', }),
                    
                    html.Div(children=[
                        dcc.Loading(id="loading-3",
                            children=[html.Div(dcc.Graph(id= 'Close-forecast'))],
                            type='circle',
                            fullscreen=False,
                            color='#119DFF')
                    ], className="six columns widget-box", style={'padding':'.3rem', 'marginTop':'1rem', 'marginLeft':'3rem', 'boxShadow': '#e3e3e3 4px 4px 2px', 'border-radius': '10px', 'backgroundColor': 'white', }),
                    
                ],className="twelve columns", style={'backgroundColor': '#f2f2f2', 'margin': '1rem'}),
            ], style={'display': 'flex', 'flex-wrap': 'wrap'}),
        html.Div(children=[ 
        # OHLC
            html.Div(children=[
                dcc.Graph(id='ohlc')
            ], className="twleve columns", style={'padding':'2rem',  'boxShadow': '#e3e3e3 4px 4px 2px', 'border-radius': '10px', 'backgroundColor': '#31353b'})        
        ], style={'margin': '1rem', })            
                    
], style={'padding': '2rem'})

#--------------------------
#Callbacks

#Callbacks for news 
@app.callback(
    Output('one-title','children'),
    Output('one-link', 'href'),
    Output('one-published', 'children'),
    [Input('ticker-drop','value')]
)
def get_news(value):
    """
    This callback grabs latest news feeds on selected ticker from yahoo_fin
    """
    raw_feeds = news.get_yf_rss(value)

    title = raw_feeds[0]['title']
    link = raw_feeds[0]['link']
    published = raw_feeds[0]['published']
    
    return title, link, published

@app.callback(
    Output('two-title','children'),
    Output('two-link', 'href'),
    Output('two-published', 'children'),
    [Input('ticker-drop','value')]
)
def get_news(value):
    """
    This callback grabs latest news feeds on selected ticker from yahoo_fin
    """
    raw_feeds = news.get_yf_rss(value)

    title = raw_feeds[1]['title']
    link = raw_feeds[1]['link']
    published = raw_feeds[1]['published']
    
    return title, link, published

@app.callback(
    Output('three-title','children'),
    Output('three-link', 'href'),
    Output('three-published', 'children'),
    [Input('ticker-drop','value')]
)
def get_news(value):
    """
    This callback grabs latest news feeds on selected ticker from yahoo_fin
    """
    raw_feeds = news.get_yf_rss(value)

    title = raw_feeds[2]['title']
    link = raw_feeds[2]['link']
    published = raw_feeds[2]['published']
    
    return title, link, published

#callbacks for graphs
#callback for open/close graph
@app.callback(
    Output('open-close-fig','figure'),
    Output('high-low-fig','figure'),
    Output('open-sb','children'),
    Output('high-sb','children'),
    Output('low-sb','children'),
    Output('close-sb','children'),
    Output('volume-sb','children'),
    [Input('ticker-drop','value')]
)
def open_close_graph(value):
    """
    This callback returns a open/close & high/low graph along with the stats for the statbox 
    based on the selected ticker
    """
    data = load_intraday(value)
    stats = data.tail(1)
    op = stats['open']
    cl = stats['close']
    hg = stats['high']
    lw = stats['low']
    vol = stats['volume']

    df_oc = data.filter(['open','close'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_oc.index, y=df_oc['open'], name= "Open"))
    fig.add_trace(go.Scatter(x=df_oc.index, y=df_oc['close'], name= "High"))
    fig.layout.update(title =f"Time series Data on Opening and Closing prices for {value}",
                    xaxis_title="Date/Time", 
                    yaxis_title="Price in USD($)",
                    legend_title="Legend",
                    xaxis_rangeslider_visible=True
                )
    
    df_hl = data.filter(['high','low'])

    high_low_fig = go.Figure()
    high_low_fig.add_trace(go.Scatter(x=df_hl.index, y=df_hl['high'], name= "High"))
    high_low_fig.add_trace(go.Scatter(x=df_hl.index, y=df_hl['low'], name= "Low"))
    high_low_fig.layout.update(title =f"Time series Data on High and low prices for {value}",
                    xaxis_title="Date/Time", 
                    yaxis_title="Price in USD($)",
                    legend_title="Legend",
                    xaxis_rangeslider_visible=True
                )
    return fig, high_low_fig, op, hg, lw, cl, vol

#Callback for foreacast
@app.callback(
    Output('Open-forecast', 'figure'),
    [Input('forecast-button', 'n_clicks')],
    [State('ticker-drop', 'value')]
)
def fore_open(n,value):
    """
    This callback generates a Time series Graph with 
    forecasting on Opening prices of selected tick using 
    LSTM Neural network model with dataframe.
    
    """
    if n is None:
        raise PreventUpdate    
    else:
        #Load the datasets
        df = load_daily(value)

        #Filter datasets
        df_open = df.filter(['Open'])
            
        #Save values in an array
        data = df_open.values

        #assign test size
        test_data_size = math.ceil(len(data)*.2) 

        #SPlit datasets into Train and test
        test_data = data[-(test_data_size + 60):,:]

        #Scale data sets
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_test = scaler.fit_transform(test_data)

        #Split the labels and target and assign test data
        X_test = []
        y_test = df[-(test_data_size):]
            
        for i in range(60,len(scaled_test)):
            X_test.append(scaled_test[i-60:i, 0])

        #Converting to numpy array
        X_test = np.array(X_test)

        #Rehsape to (batch_size,time_steps,features)
        X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

        #Import & load the pickled model
        model = tf.keras.models.load_model('LSTM2')

        #Test/validate the model & inverse transoform to get the actual predictions
        pred = model.predict(X_test)
        predictions = scaler.inverse_transform(pred)

        #Caculate the rmse value for the predictions
        #rmse = np.sqrt(np.mean(predictions - y_test)**2)

        #Plot the results 
        train = df_open[:len(df_open)-test_data_size]
        valid = df_open[len(train):]
        valid['Predictions'] = predictions
        #co_table = dbc.Table.from_dataframe(valid, striped=True, bordered=True, hover=True)
        Open_forecast = go.Figure()
        Open_forecast.add_trace(go.Scatter(x= df_open.index.values, y=df_open['Open'],name= "Train"))
        Open_forecast.add_trace(go.Scatter(x= valid.index.values, y=valid['Open'],name= "Val"))
        Open_forecast.add_trace(go.Scatter(x= valid.index.values, y=valid['Predictions'],name= "Predict"))
        Open_forecast.layout.update(title =f"Forecasting on Opening prices for {value}", 
                            yaxis_title="Opening Price in USD($)",
                            legend_title="Legend",
                            xaxis_rangeslider_visible=True)

        return Open_forecast

@app.callback(
    Output('Close-forecast', 'figure'),
    [Input('forecast-button', 'n_clicks')],
    [State('ticker-drop', 'value')]
)
def fore_close(n, value):
    """
    This callback generates a Time series Graph with 
    forecasting on Closing prices of selected tick using 
    LSTM Neural network model with dataframe.
    
    """
    if n is None:
        raise PreventUpdate    
    else:
        #load the datasets
        df_new = load_daily(value)

        #Filter the datasets 
        df_close = df_new.filter(['Close'])

        #Save values in an array
        data_close = df_close.values

        #assign test size
        cl_test_data_size = math.ceil(len(data_close)*.2)  
        
        #SPlit datasets into Train and test
        cl_test_data = data_close[-(cl_test_data_size + 60):,:]

        #Scale data sets
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_cl_test = scaler.fit_transform(cl_test_data)

        #Split the labels and target and assign test data
        cl_X_test = []
        #cl_y_test = df_close[-(cl_test_data_size):]
        
        for i in range(60,len(scaled_cl_test)):
            cl_X_test.append(scaled_cl_test[i-60:i, 0])

        #Converting to numpy array
        cl_X_test = np.array(cl_X_test)

        #Rehsape to (batch_size,time_steps,features)
        cl_X_test = np.reshape(cl_X_test, (cl_X_test.shape[0], cl_X_test.shape[1], 1))

        #Import & load the pickled model
        model1 = tf.keras.models.load_model('LSTM2.h5')

        #Test/validate the model & inverse transoform to get the actual predictions
        fore = model1.predict(cl_X_test)
        forecast = scaler.inverse_transform(fore)

        #Caculate the rmse value for the predeictions
        #cl_rmse = np.sqrt(np.mean(forecast - cl_y_test)**2)

        #Plot the results 
        cl_train = df_close[:len(df_close)-cl_test_data_size]
        cl_valid = df_close[len(cl_train):]
        cl_valid['predictions'] = forecast
        #cl_table = dbc.Table.from_dataframe(cl_valid, striped=True, bordered=True, hover=True)
        Close_forecast = go.Figure()
        Close_forecast.add_trace(go.Scatter(x= df_close.index.values, y=df_close['Close'],name= "Train"))
        Close_forecast.add_trace(go.Scatter(x= cl_valid.index.values, y=cl_valid['Close'],name= "Val"))
        Close_forecast.add_trace(go.Scatter(x= cl_valid.index.values, y=cl_valid['predictions'],name= "predictions"))
        Close_forecast.layout.update(title =f"Forecasting on Closing prices for {value}", 
                            yaxis_title="Closing Price in USD($)",
                            legend_title="Legend",
                            xaxis_rangeslider_visible=True)

        return Close_forecast

@app.callback(
    Output('ohlc','figure'),
    Input('ticker-drop','value')
)
def ohlc_graph(value):
    """
    This callback generates an OHLC chart based on the selected tick 
    """
    df = load_daily(value)

    fig = go.Figure(data=[go.Ohlc(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color= 'cyan', decreasing_line_color= '#33ff00'
    )])
    fig.layout.update(title=f'OHLC chart for {value} from Jan 2018 till date ', 
        title_font_size= 23,
        xaxis_title='Date',
        yaxis_title= 'Price in USD($)')
    fig.layout.plot_bgcolor = '#31353b'
    fig.layout.paper_bgcolor = '#31353b'
    fig.update_layout(
        font_color= '#FFFFFF'
    )
    return fig



if __name__ == '__main__':
    app.run_server(debug=True, )