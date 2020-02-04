from plotly.offline import plot as offpy
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import plotly.graph_objs as go
import math


def visualize(config, G, node_size):
    keys = G.nodes()
    values = range(len(G.nodes()))
    dictionary = dict(zip(keys, values))
    inv_map = {v: k for k, v in dictionary.items()}
    G = nx.relabel_nodes(G, dictionary)
    try:
        pos = graphviz_layout(G)
    except:
        raise Exception("there is something wrong with graphviz")

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.8, color='rgba(136, 136, 136, .9)'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # for weight in weights:
    #     print(weight)
    #     edge_trace['line']['width'].append(weight)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textfont=dict(family='Calibri (Body)', size=14, color='black'),
        opacity=1,
        # hoverinfo='text',
        marker=go.Marker(
            showscale=True,
            # colorscale options
            # 'aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
            #              'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
            #              'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
            #              'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
            #              'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
            #              'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
            #              'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
            #              'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
            #              'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor',
            #              'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy',
            #              'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar', 'spectral',
            #              'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn', 'tealrose',
            #              'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'twilight',
            #              'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd'
            colorscale=config.color_scale,
            reversescale=True,
            color=[],
            size=[],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=1.7)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)

    for i, adjacencies in G.adjacency():
        node_trace['marker']['color'] += (len(adjacencies),)

    for node in G.nodes():
        node_trace['text'] += (inv_map[node],)

    for size in node_size:
        node_trace['marker']['size'] += (abs(math.log(size))*10,)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>' + config.title,
                        titlefont=dict(size=12),
                        showlegend=False,
                        width=1400,
                        height=800,
                        hovermode='closest',
                        margin=dict(b=20, l=350, r=5, t=200),
                        # family='Courier New, monospace', size=18, color='#7f7f7f',
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=go.layout.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=go.layout.YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    offpy(fig, filename=config.out_file_name, auto_open=True, show_link=False)


def visualize_3d(config, G, node_sizes):
    keys = G.nodes()
    values = range(len(G.nodes()))
    dictionary = dict(zip(keys, values))
    inv_map = {v: k for k, v in dictionary.items()}
    G = nx.relabel_nodes(G, dictionary)

    edge_trace = go.Scatter3d(x=[],
                       y=[],
                       z=[],
                       mode='lines',
                       line=go.scatter3d.Line(color='rgba(136, 136, 136, .8)', width=1),
                       hoverinfo='none'
                       )


    node_trace = go.Scatter3d(x=[],
                       y=[],
                       z=[],
                       mode='markers',
                       #name='actors',
                       marker=go.scatter3d.Marker(symbol='circle',
                                     size=[],
                                     color=[],
                                     colorscale=config.color_scale,#'Viridis',
                                     colorbar=dict(
                                         thickness=15,
                                         title='Node Connections',
                                         xanchor='left',
                                         titleside='right'
                                     ),
                                     #line=go.Line(color='rgb(50,50,50)', width=0.5)
                                     ),
                       text=[],
                       hoverinfo='text'
                       )

    positions = nx.fruchterman_reingold_layout(G, dim=3, k=0.5, iterations=1000)



    for edge in G.edges():
        x0, y0, z0 = positions[edge[0]]
        x1, y1, z1 = positions[edge[1]]
        edge_trace['x'] += (x0, x1,)
        edge_trace['y'] += (y0, y1,)
        edge_trace['z'] += (z0, z1,)


    for node in G.nodes():
        x, y, z = positions[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['z'] += (z,)


    for i, adjacencies in G.adjacency():
        node_trace['marker']['color'] += (len(adjacencies),)

    for size in node_sizes:
        node_trace['marker']['size'] += (abs(math.log(size))*10,)



    for node in G.nodes():
        node_trace['text'] += (inv_map[node],)


    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title=''
                )

    layout = go.Layout(
        title=config.title,
        width=1000,
        height=1000,
        showlegend=False,
        scene=go.Scene(
            xaxis=go.XAxis(axis),
            yaxis=go.YAxis(axis),
            zaxis=go.ZAxis(axis),
        ),
        margin=go.Margin(
            t=100
        ),
        hovermode='closest',
        )


    fig = go.Figure(data=[node_trace, edge_trace], layout=layout)

    offpy(fig, filename=config.out_file_name, auto_open=True, show_link=False)





# def three_d_nework_graph(x_positions, y_positions, z_positions, colors, labels="unknown"):
#     trace2 = go.Scatter3d(x=x_positions,
#                        y=y_positions,
#                        z=z_positions,
#                        mode='markers',
#                        name='actors',
#                        marker=go.scatter.Marker(symbol='dot',
#                                      size=6,
#                                      color=colors,
#                                      colorscale='Viridis',
#                                      line=go.Line(color='rgb(50,50,50)', width=0.5)
#                                      ),
#                        text=labels,
#                        hoverinfo='text'
#                        )
#     axis = dict(showbackground=True,
#                 showline=True,
#                 zeroline=True,
#                 showgrid=True,
#                 showticklabels=True,
#                 title=''
#                 )
#     layout = go.Layout(
#         title="Network of coappearances of documents in the whole repository(3D visualization)",
#         width=1000,
#         height=1000,
#         showlegend=False,
#         scene=go.Scene(
#             xaxis=go.XAxis(axis),
#             yaxis=go.YAxis(axis),
#             zaxis=go.ZAxis(axis),
#         ),
#         margin=go.Margin(
#             t=100
#         ),
#         hovermode='closest',
#         annotations=go.layout.Annotations([
#             go.layout.Annotation(
#                 showarrow=False,
#                 text="",
#                 xref='paper',
#                 yref='paper',
#                 x=0,
#                 y=0.1,
#                 xanchor='left',
#                 yanchor='bottom',
#                 font=go.Font(
#                     size=14
#                 )
#             )
#         ]), )
#     data = go.Data([trace2])
#     fig = go.Figure(data=data, layout=layout)
#
#     offpy(fig, filename="dd", auto_open=True, show_link=False)
