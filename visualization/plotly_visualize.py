from plotly.offline import plot as offpy
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import plotly.graph_objs as go
import math
from matplotlib import colors
import matplotlib.cm as cm


def get_color_scale_range(n):
    lst = list(range(n))

    minima = min(lst)
    maxima = max(lst)

    norm = colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.tab10)

    color_scale = []
    for v in lst:
        color_scale.append([v / n, str("rgb" + str(mapper.to_rgba(v)[:-1]))])
        color_scale.append([(v+1) / n, str("rgb" + str(mapper.to_rgba(v)[:-1]))])

    return color_scale


def visualize(config, G, node_size, auto_open=True):
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

    num_topics = len(set([x[1]['color_code'] for x in G.nodes(data=True)]))
    color_scale = get_color_scale_range(num_topics)

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textfont=dict(family='Calibri (Body)', size=18, color='black'),
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
            #colorscale=config.color_scale,
            colorscale=color_scale,
            reversescale=True,
            color=[],
            size=[],
            colorbar=dict(
                thickness=15,
                title='Topics',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=1.7)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)

    # for i, adjacencies in G.adjacency():
    #     node_trace['marker']['color'] += (len(adjacencies),)

    for node in G.nodes(data=True):
        node_trace['marker']['color'] += (node[1]['color_code'],)


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
                        margin=dict(b=10, l=0, r=150, t=10),
                        # family='Courier New, monospace', size=18, color='#7f7f7f',
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=go.layout.XAxis(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=go.layout.YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    offpy(fig, filename=config.out_file_name, auto_open=auto_open, show_link=False)

    return fig


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

    return fig



def barchart(x, y):
    assert len(x) == len(y), "the x and y should be the same length"
    x = [str(i)+" "+str(xi) for i, xi in enumerate(x)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
    ))

    offpy(fig, filename="barchart.html", auto_open=True, show_link=False)


def multi_barchart(x, y, z, names=[]):
    assert len(x) == len(y) == len(z), "the lengths should be the same"
    x = [str(i) + " " + str(xi) for i, xi in enumerate(x)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        name=names[0]
    ))

    fig.add_trace(go.Bar(
        x=x,
        y=z,
        name=names[1]
    ))


    offpy(fig, filename="multi_barchart.html", auto_open=True, show_link=False)




if __name__ == "__main__":
    barchart(["this", "is", "this"], [0.1, 0.5, 0.8])