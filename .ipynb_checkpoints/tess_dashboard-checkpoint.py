import dash 
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import requests as r
import plotly.graph_objects as go

import astropy.coordinates as coord
from astropy import units as u
import matplotlib.pyplot as plt

def load_lc(tic):
    url = "http://tessebs.villanova.edu/static/catalog/lcs_ascii/tic"+str(int(tic)).zfill(10)+".01.norm.lc"
    lc = r.get(url)
    lc_data = np.fromstring(lc.text, sep=' ')
    lc_data = lc_data.reshape(int(len(lc_data)/4), 4)
    return pd.DataFrame.from_dict({
        'times': lc_data[:,0],
        'phases': lc_data[:,1],
        'fluxes': lc_data[:,2],
        'sigmas': lc_data[:,3]
    })
  
def load_model(tic, model='2g'):
    if model == '2g':
        filename = 'data/twog_lcs/tic'+str(int(tic)).zfill(10)+'.2g.lc'
    elif model == 'pf':
        filename = 'data/pf_lcs/tic'+str(int(tic)).zfill(10)+'.pf.lc'
    else:
        raise NotImplementedError
    lc = np.loadtxt(filename)
    phases, fluxes = lc[:,0], lc[:,1]
    phases[phases < 0] = phases[phases < 0] + 1
    phases[phases > 1] = phases[phases > 1] - 1
    s = np.argsort(phases)
    return phases[s], fluxes[s]

  
external_stylesheets = [dbc.themes.SPACELAB]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# df_ephem = pd.read_csv('data/ephemerides_clean_morph_09132021.csv')
# df_eclipse = pd.read_csv('data/eclipse_params_clean_09132021.csv')
# catalog = pd.read_csv('data/catalog.csv', delimiter=',')

# df = pd.concat([df_ephem, df_eclipse.drop('TIC', axis=1)], axis=1)
df = pd.read_csv('data/server_df.csv')

fig_lc = go.Figure()

app.layout = dbc.Container([

    dbc.Container([
        # title and information
        html.H3(children='TESS Eclipsing Binary Catalog Explorer'),
        html.Div(children='A visual explorer of the TESS EBs catalog. You can browse or download the catalog at http://tessebs.villanova.edu.')
    ], style={'margin': '0em', 'padding': '2.5em', 'backgroundColor': '#baddff'}),
    
    dbc.Container([
        # x-y explorer
        dbc.Row([
            dbc.Col([
            html.H4(children='Light Curve Explorer'),
            html.Div(children='Click on a point in the scatter plot to view its corresponding light curve and info.')
            ])
        ], style={'marginBottom': '1em'}),
        dbc.Row([
            dbc.Col(
                [
                    dbc.Row(
                        [ 
                        dbc.Col([
                             # options for x axis
                            html.Div(children='x-axis'),
                            dcc.Dropdown(
                                id='scatter-xaxis-column',
                                options=[{'label': i, 'value': i} for i in df.columns],
                                value = 'tsne_x'
                            ),
                            dcc.RadioItems(
                                id='scatter-xaxis-linlog',
                                options=[
                                    {'label': ' linear', 'value': 'lin'},
                                    {'label': ' log', 'value': 'log'}
                                ],
                                value='lin',
                                labelStyle={'display': 'inline-block', 'width': '50%'}
                            )
                             
                         ]),
                         
                        dbc.Col([
                            # options for y axis
                            html.Div(children='y-axis'),
                            dcc.Dropdown(
                                id='scatter-yaxis-column',
                                options=[{'label': i, 'value': i} for i in df.columns],
                                value = 'tsne_y'
                            ),
                            dcc.RadioItems(
                                id='scatter-yaxis-linlog',
                                options=[
                                    {'label': ' linear', 'value': 'lin'},
                                    {'label': ' log', 'value': 'log'}
                                ],
                                value='lin',
                                labelStyle={'display': 'inline-block', 'width': '50%'}
                            )
                        ]),
                        
                        dbc.Col([
                             # options for x axis
                            html.Div(children='color'),
                            dcc.Dropdown(
                                id='scatter-caxis-column',
                                options=[{'label': i, 'value': i} for i in df.columns],
                                value='morph'
                            ),
                            dcc.RadioItems(
                                id='scatter-caxis-linlog',
                                options=[
                                    {'label': ' linear', 'value': 'lin'},
                                    {'label': ' log', 'value': 'log'}
                                ],
                                value='lin',
                                labelStyle={'display': 'inline-block', 'width': '50%'}
                            )
                             
                         ]),
                        ]
                    ),
                    
                    dcc.Graph(
                        id='scatter-graph',
                        # figure=fig_s
                    )
                
                ], style={'marginRight': '1em'}
            ),
            dbc.Col(
                [
                    dbc.Row(
                        [ 
                        dbc.Col([
                            dcc.Checklist(
                                id='lc-data-checklist',
                                options=[
                                    {'label': ' two-Gaussian model', 'value': '2g'},
                                    {'label': ' polyfit model', 'value': 'pf'}
                                ],
                                value=[],
                                labelStyle={'display': 'block'}
                            ),
                            
                         ]),
                         dbc.Col([
                            # checkboxes for eclipse edges
                            dcc.Checklist(
                                id='lc-eclipses-checklist',
                                options=[
                                    {'label': ' eclipses (two-Gaussian)', 'value': '2g'},
                                    {'label': ' eclipses (polyfit)', 'value': 'pf'}
                                    
                                ],
                                value=[],
                                labelStyle={'display': 'block'}
                            )
                        ])
                        ]
                    ),

                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(
                                id='lc-graph',
                                figure=fig_lc
                            ),
                            dcc.RadioItems(
                                        id='lc-phase-options',
                                        options=[
                                            {'label': ' phase-folded', 'value': 'phase'},
                                            {'label': ' time series', 'value': 'time'}
                                        ],
                                        value='phase',
                                        labelStyle={'display': 'inline-block', 'width': '50%'}
                                    )
                            
                        ], align='center')
                    ])
                    
                    
                ], style={'marginLeft': '1em'}
            )
        ])
    ], style={'margin': '1em'}),
    
    dbc.Container([ 
        dbc.Row([ 
            html.H4(children='Parameter Distribution Explorer')
        ]),
        
        dbc.Row([ 
            dbc.Col([ 
                html.Div(children = 'Choose one or multiple parameters to visualize their distributions in the catalog.')
            ]),
            dbc.Col([
                dcc.Dropdown(
                    id='histogram-column',
                    options=[{'label': i, 'value': i} for i in df.columns],
                    value=['period'],
                    multi=True,
                ),
            ]),
            
            dbc.Col([
                dcc.RadioItems(
                    id='histogram-linlog',
                    options=[
                        {'label': ' linear', 'value': 'lin'},
                        {'label': ' log', 'value': 'log'}
                    ],
                    value='lin',
                    labelStyle={'display': 'inline-block', 'width': '50%'}
                )
            ]),
        ], style={'margin': '1em'}),
            
        dbc.Row([            
            dbc.Col([
                
                dbc.Row([
                    dbc.Col([
                    html.Div(children='Bins:'),
                    dcc.Slider(
                        id='bins-slider',
                        min=10,
                        max=500,
                        value=100,
                        step=5,
                        marks={str(i): str(i) for i in [0,100,200,300,400,500]},
                    )
                    ])
                
                ]),
                
                dcc.Graph(
                    id='histogram-graph',
                    # figure=fig_h
                ),
                dbc.Row([
                    dbc.Col([
                        html.Div(children='Counts format: '),
                    ]),
                    dbc.Col([
                        dcc.RadioItems(
                            id='counts-linlog',
                            options=[
                                {'label': ' linear', 'value': 'lin'},
                                {'label': ' log', 'value': 'log'}
                            ],
                            value='lin',
                            labelStyle={'display': 'inline-block', 'width': '40%'}
                        )   
                    ])
                ])
                  
            ]),
            
            dbc.Col([
                # html.Img(id='cs-graph'), # img element
                dbc.Row(
                    dbc.Col([
                    html.H5(children='Distribution on the sky'),
                    html.Div(children='Note: only shows the distribution of the first selected parameter.', 
                             style={'fontSize': '0.75em'})
                    ], align='center')
                ),
                # dcc.Graph(
                #     id='cs-graph',
                #     # figure=fig_c
                # ),
                html.Img(id='cs-graph'),
                
                dbc.Row([
                    dbc.Col([
                        html.Div(children='Projection: '),
                    ]),
                    dbc.Col([
                        dcc.RadioItems(
                            id='cs-coords',
                            options=[
                                {'label': ' Equatorial', 'value': 'radec'},
                                {'label': ' Galactic', 'value': 'gal'}
                            ],
                            value='radec',
                            labelStyle={'display': 'inline-block', 'width': '50%'}
                        )
                            ]),
                ]),
            ], style={'border': 'solid gray 0.5px', 'justifyContent': 'center', 'alignItems': 'center',
                      'paddingTop': '2em'})
        ])
        ])
        # histogram and lat-long plot
    
])

### CALLBACKS

@app.callback(
    Output('scatter-graph', 'figure'),
    [Input('scatter-xaxis-column', 'value'),
     Input('scatter-xaxis-linlog', 'value'),
     Input('scatter-yaxis-column', 'value'),
     Input('scatter-yaxis-linlog', 'value'),
     Input('scatter-caxis-column', 'value'),
     Input('scatter-caxis-linlog', 'value')
    ]
)
def update_scatter_plot(xcolumn, xtype, ycolumn, ytype, ccolumn, ctype):
    if ctype=='log':
        colors = np.log10(df[ccolumn])
        color_title = 'log_' + ccolumn
    else:
        colors = df[ccolumn]
        color_title = ccolumn
    fig_s = px.scatter(df, x=xcolumn, y=ycolumn, color=colors,
                       custom_data = ['TIC'],
                       log_x = True if xtype == 'log' else False,
                       log_y = True if ytype == 'log' else False
                       )
    fig_s.update_layout(coloraxis_colorbar=dict(
        title=color_title), clickmode='event+select')
    return fig_s
    

@app.callback(
    Output('lc-graph', 'figure'),
    [Input('lc-graph', 'figure'),
     Input('scatter-graph', 'clickData'),
     Input('lc-phase-options', 'value'),
     Input('lc-data-checklist', 'value'),
     Input('lc-eclipses-checklist', 'value')
     ]
)
def update_lc(fig_lc, click_data, phasevstime, models, eclipses):
    twog_color = 'lightsalmon'
    pf_color = 'lightgreen'
    if click_data is None:
        return {}
    else:
        
        tic = click_data['points'][0]['customdata'][0]
        layout = {'title': {'text':'TIC '+str(int(tic))}}
        fig_lc = go.Figure(layout=layout)
        lc_df = load_lc(tic)
        
        if phasevstime == 'phase':
            xs = ['phases']
            ys = ['fluxes']

        else:
            xs = ['times']
            ys = ['fluxes']

        fig_lc.add_trace(go.Scatter(mode='markers', x=lc_df[xs[0]], y=lc_df[ys[0]], name='data'))
        if '2g' in models and phasevstime == 'phase':
            phases_2g, fluxes_2g = load_model(tic, model='2g')
            fig_lc.add_trace(go.Scatter(mode='lines', x=phases_2g, y=fluxes_2g, name='2g', 
                                        line=dict(
                                            color=twog_color,
                                            width=3)))
        
        if '2g' in eclipses and phasevstime == 'phase':
            row = df[df['TIC'] == tic]
            pos1 = row['pos1_2g'].values[0]
            pos2 = row['pos2_2g'].values[0]
            w1 = row['width1_2g'].values[0]
            w2 = row['width2_2g'].values[0]
            edges = np.array([pos1-0.5*w1, pos1+0.5*w1, pos2-0.5*w2, pos2+0.5*w2])
            edges[edges < 0] = edges[edges < 0] + 1
            edges[edges > 1] = edges[edges > 1] - 1
            if ~np.isnan(pos1):
                fig_lc.add_vline(x=pos1, line=dict(
                                            color=twog_color,
                                            width=3,
                                        ))
            if ~np.isnan(pos2):
                fig_lc.add_vline(x=pos2, line=dict(
                                            color=twog_color,
                                            width=3,
                                        ))
            for edge in edges:
                if ~np.isnan(edge):
                    fig_lc.add_vline(x=edge, line=dict(
                                            color=twog_color,
                                            width=2,
                                            dash="dot",
                                        ))

                

        if 'pf' in models and phasevstime == 'phase':
            phases_pf, fluxes_pf = load_model(tic, model='pf')
            fig_lc.add_trace(go.Scatter(mode='lines', x=phases_pf, y=fluxes_pf, name='pf', 
                                        line=dict(
                                            color=pf_color,
                                            width=3,
                                        )))

        if 'pf' in eclipses and phasevstime == 'phase':
            row = df[df['TIC'] == tic]
            pos1 = row['pos1_pf'].values[0]
            pos2 = row['pos2_pf'].values[0]
            w1 = row['width1_pf'].values[0]
            w2 = row['width2_pf'].values[0]
            edges = np.array([pos1-0.5*w1, pos1+0.5*w1, pos2-0.5*w2, pos2+0.5*w2])
            edges[edges < 0] = edges[edges < 0] + 1
            edges[edges > 1] = edges[edges > 1] - 1
            if ~np.isnan(pos1):
                fig_lc.add_vline(x=pos1, line=dict(
                                            color=pf_color,
                                            width=3,
                                        ))
            if ~np.isnan(pos2):
                fig_lc.add_vline(x=pos2, line=dict(
                                            color=pf_color,
                                            width=3,
                                        ))
            for edge in edges:
                if ~np.isnan(edge):
                    fig_lc.add_vline(x=edge, line=dict(
                                            color=pf_color,
                                            width=2,
                                            dash="dot",
                                        ))    
        
        return fig_lc


@app.callback(
    Output('histogram-graph', 'figure'),
    [Input('histogram-column', 'value'),
     Input('histogram-linlog', 'value'),
     Input('counts-linlog', 'value'),
     Input('bins-slider', 'value')]
)
def update_histogram(columns, xtype, ytype, nbins):
    if xtype == 'log':
        columns_log = []
        for column in columns:
            if 'log_'+column not in df.columns:
                df['log_'+column] = np.log10(df[column])
            columns_log.append('log_'+column)
        columns = columns_log
    fig_h = px.histogram(df, x=columns, nbins=nbins,
                         log_y = True if ytype == 'log' else False)
    return fig_h


@app.callback(
    Output('cs-graph', 'src'),
    [Input('histogram-column', 'value'),
     Input('histogram-linlog', 'value'),
     Input('cs-coords', 'value')]
)
def update_coordinates(xcolumns, xtype, coords_type):

    import astropy.coordinates as coord
    from astropy import units as u
    import io
    import base64
    
    if coords_type == 'radec':
        x = 'ra'
        y = 'dec'
    else:
        x = 'glon'
        y = 'glat'
        
    if xtype == 'log':
        colors = np.log10(df[xcolumns[0]])
        label = 'log_' + xcolumns[0]
    else:
        colors = df[xcolumns[0]]
        label=xcolumns[0]
        
    lon = coord.Angle(df[x].fillna(np.nan)*u.degree)
    lon = lon.wrap_at(180*u.degree)
    lat = coord.Angle(df[y].fillna(np.nan)*u.degree)
    
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection="mollweide")
    col = ax.scatter(lon.radian, lat.radian, c=colors, s=3)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid()
    fig.colorbar(col,label=label,location='bottom')
    fig.tight_layout()
    
    buf = io.BytesIO() # in-memory files
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    figdata = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f'data:image/png;base64,{figdata}'



if __name__ == '__main__':
    app.run_server(debug=True)


