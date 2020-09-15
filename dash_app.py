import os
import sys
import importlib

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_table.FormatTemplate as FormatTemplate
from dash_table import Format
from dash.dependencies import Input, Output

import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

from utility.data_logger import DataLogger

# set path and import parameters
exp_name = 'pendulum'

# import __file__ as gps_filepath
gps_filepath = os.path.abspath(__file__)
gps_dir = '/'.join(str.split(gps_filepath, '/')[:-1]) + '/'
exp_dir = gps_dir + 'experiments/' + exp_name + '/'
sys.path.append(exp_dir)

hyperparams_file = exp_dir + 'hyperparams.py'
if not os.path.exists(hyperparams_file):
    sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))

hyperparams = importlib.import_module('hyperparams')

data_files_dir = hyperparams.config['common']['data_files_dir']
M = len(hyperparams.config['common']['train_conditions'])

data_logger = DataLogger()


def find_available_files(dir_, file_name):
    all_files = os.listdir(dir_)
    available_files = []

    for file_path in all_files:
        if file_name in file_path:
            available_files.append(dir_ + file_path)

    available_files.sort()
    return available_files


"""
Begin Dash Application

The Dash Application is used to show the training progress. 
"""
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    # first row
    html.Div([
        html.Div([
            dcc.Graph(id='global_costs'),
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # in milliseconds
                n_intervals=0
            )
        ], style={'display': 'inline-block', 'width': '58%'}),
        html.Div([
            html.Div([
                html.H4(
                    children='Average costs',
                    style={
                        'textAlign': 'center',
                        'color': 'rgb(37,37,37)',
                        'fontFamily': 'Courier New',
                        'fontSize': 24,
                    }
                )
            ]),
            html.Div(id='global_itr_data')
            ], style={
                     'display': 'inline-block',
                     'vertical-align': 'top',
                     'margin-left': '3vw',
                     'margin-top': '3vw',
                     'width': '38%'
        }),
    ]),

    # second row
    html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='01_local_sample_plot')
            ]),
            html.Div(id='01_local_itr_data'),
        ], style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '3vw',
            'margin-top': '3vw',
            'width': '45%'
        }),

        html.Div([
            html.Div([
                dcc.Graph(id='02_local_sample_plot'),
            ], style={
                'horizontal-align': 'center',
            }),
            html.Div(id='02_local_itr_data'),
        ], style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '3vw',
            'margin-top': '3vw',
            'width': '45%'
        }),
    ]),

    # third row
    html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='03_local_sample_plot')
            ]),
            html.Div(id='03_local_itr_data'),
        ], style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '3vw',
            'margin-top': '3vw',
            'width': '45%'
        }),

        html.Div([
            html.Div([
                dcc.Graph(id='04_local_sample_plot')
            ]),
            html.Div(id='04_local_itr_data'),
        ], style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'margin-left': '3vw',
            'margin-top': '3vw',
            'width': '45%'
        }),
    ]),

])


def create_global_costs(data):

    fig = go.Figure()

    if data:
        for n in range(data['data'].shape[0]):
            fig.add_trace(go.Scatter(
                x=np.squeeze(data['ts']),
                y=data['data'][n, :],
                mode='markers',
                name='cond. %02d' % (n+1)
            ))

        fig.add_trace(go.Scatter(
            x=np.squeeze(data['ts']),
            y=np.squeeze(data['data_mean']),
            mode='lines+markers',
            name='mean'
        ))

    fig.update_xaxes(showgrid=True)

    fig.update_layout(
        title=go.layout.Title(
            text='Global cost',
            font=dict(
                family='Courier New',
                size=24,
                color='rgb(37,37,37)'
            ),
        ),
        title_x=0.5,
        yaxis_title='cost',
        xaxis_title='iteration',
        autosize=False,
        height=500,
        margin={'l': 40, 'b': 30, 'r': 20, 't': 50})

    return fig


@app.callback(
    Output(component_id='global_costs', component_property='figure'),
    [Input(component_id='interval-component', component_property='n_intervals')]
)
def update_global_costs(n):
    available_files = find_available_files(data_files_dir, 'global_cost_itr')

    if available_files:
        data = data_logger.unpickle(available_files[-1])
        return create_global_costs(data)
    else:
        return create_global_costs(data=None)


def create_global_itr_table(df):
    if not df.empty:
        table = dt.DataTable(
            id='global_itr_table',
            columns=[{"name": i, "id": i, 'type': 'numeric', 'format': {'specifier': '.2f'}} for i in df.columns],
            data=df.to_dict('records'),
            style_data_conditional=[
                {
                    'if': {
                        'column_id': 'avg_cost',
                        'filter_query': '{{avg_cost}} = {}'.format(df['avg_cost'].min()),
                    },
                    'backgroundColor': '#62C54E',
                    'color': 'black'
                },
                {
                    'if': {
                        'column_id': 'avg_pol_cost',
                        'filter_query': '{{avg_pol_cost}} = {}'.format(df['avg_pol_cost'].min()),
                    },
                    'backgroundColor': '#2DA428',
                    'color': 'black'
                },
            ],
            page_size=16,
            page_current=0,
        )
    else:
        table = dt.DataTable(
            id='global_itr_table',
        )

    return table


@app.callback(
    Output(component_id='global_itr_data', component_property='children'),
    [Input(component_id='interval-component', component_property='n_intervals')]
)
def update_global_itr_data(n):
    available_itr_files = find_available_files(data_files_dir, 'global_itr_data')

    if available_itr_files:
        data_frame = data_logger.unpickle(available_itr_files[-1])
        # data_frame['avg_cost'] = data_frame['avg_cost'].map('{:.2f}'.format)
        # data_frame['avg_pol_cost'] = data_frame['avg_pol_cost'].map('{:.2f}'.format)
        return create_global_itr_table(data_frame)
    else:
        return create_global_itr_table(df=pd.DataFrame())


def create_local_sample_fig(traj_samples, pol_samples, m):
    fig = go.Figure()

    if traj_samples is not None:
        N = len(traj_samples.columns)
        for n in range(N):
            fig.add_trace(go.Scatter(
                # x=traj_samples.iloc[:, 2*n],
                y=traj_samples.iloc[:, n],
                mode='lines',
                name='traj. rollout %d' % (n + 1)
            ))

    if pol_samples is not None:
        N = len(pol_samples.columns)
        for n in range(N):
            fig.add_trace(go.Scatter(
                y=pol_samples.iloc[:, n],
                mode='lines',
                name='pol. rollout %d' % (n + 1)
            ))

    fig.update_layout(
        title=go.layout.Title(
            text=('condition %d: samples' % m),
            font=dict(
                family='Courier New',
                size=24,
                color='rgb(37,37,37)'
            ),
        ),
        title_x=0.5,
        yaxis_title='theta',
        xaxis_title='time step',
        autosize=False,
        height=500,
        width=800,
        margin={'l': 150, 'b': 50, 'r': 50, 't': 50})

    return fig


@app.callback(
    [Output(component_id='01_local_sample_plot', component_property='figure'),
     Output(component_id='02_local_sample_plot', component_property='figure'),
     Output(component_id='03_local_sample_plot', component_property='figure'),
     Output(component_id='04_local_sample_plot', component_property='figure')],
    [Input(component_id='interval-component', component_property='n_intervals')]
)
def update_local_plots(n):
    files = True
    traj_pos_list = []
    pol_pos_list = []

    for m in range(1, M + 1):
        available_traj_files_pos = find_available_files(data_files_dir, ('%02d_traj_samples_pos' % m))
        available_pol_files_pos = find_available_files(data_files_dir, ('%02d_pol_samples_pos' % m))
        if available_traj_files_pos:
            traj_samples = data_logger.unpickle(available_traj_files_pos[-1])
            traj_pos_list.append(traj_samples)
            pol_samples = data_logger.unpickle(available_pol_files_pos[-1])
            pol_pos_list.append(pol_samples)
        else:
            files = False

    if files:
        return [create_local_sample_fig(traj_, pol_, m + 1) for traj_, pol_, m in
                zip(traj_pos_list, pol_pos_list, range(M))]
    else:
        return [create_local_sample_fig(None, None, m + 1) for m in
                range(M)]


def create_local_itr_table(df, n):
    if not df.empty:
        table = dt.DataTable(
            id=('%02d_local_itr_table' % n),
            columns=[{"name": i, "id": i, 'type': 'numeric', 'format': {'specifier': '.2f'}} for i in df.columns],
            data=df.to_dict('records'),
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{{cost}} = {}'.format(df['cost'].min()),
                        # 'column_id': 'cost'
                    },
                    'backgroundColor': 'tomato',
                    'color': 'black'
                },
            ],
            page_size=16,
            page_current=0,
        )
    else:
        table = dt.DataTable(
            id=('%02d_local_itr_table' % n),
        )

    return table


@app.callback(
    [Output(component_id='01_local_itr_data', component_property='children'),
     Output(component_id='02_local_itr_data', component_property='children'),
     Output(component_id='03_local_itr_data', component_property='children'),
     Output(component_id='04_local_itr_data', component_property='children')],
    [Input(component_id='interval-component', component_property='n_intervals')]
)
def update_local_itr_data(n):
    files = True
    df_list = []

    for m in range(1, M + 1):
        available_itr_files = find_available_files(data_files_dir, ('%02d_local_itr_data' % m))
        if available_itr_files:
            data_frame = data_logger.unpickle(available_itr_files[-1])
            df_list.append(data_frame)
        else:
            files = False

    if files:
        return [create_local_itr_table(df, (m+1)) for df, m in zip(df_list, range(M))]
    else:
        return [create_local_itr_table(pd.DataFrame(), (m+1)) for m in range(M)]


if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='10.27.194.33')
