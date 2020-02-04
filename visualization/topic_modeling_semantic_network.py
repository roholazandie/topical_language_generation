import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout



from visualization import plotly_visualize as plotly


def visualize_semantic_netwrok(config, topics, visualize_method='plotly'):
    graph = get_semantic_network(config, topics)
    weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    node_size = nx.get_node_attributes(graph, 'importance').values()

    if visualize_method == "networkx":
        # pos = nx.graphviz_layout(graph)
        try:
            pos = graphviz_layout(graph)
        except:
            pos = nx.random_layout(graph)

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=graph.nodes(),
                               # node_color=node_color,
                               node_size=node_size,
                               alpha=1)
        # edges
        nx.draw_networkx_edges(graph, pos, width=weights, alpha=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=12, font_color='w')

        plt.axis('off')
        plt.show()
    elif visualize_method == "plotly":
        plotly.visualize(config, graph, node_size)
    elif visualize_method == "plotly3d":
        plotly.visualize_3d(config, graph, node_size)
    else:
        raise visualize_method + " not defined for visualize method. use networkx or plotly as visualize_method"


def get_semantic_network(config, topics):
    graph = nx.Graph()
    for topic in topics:
        for (word1, prob1) in topic:
            for (word2, prob2) in topic:
                if word1 != word2 and abs(prob1) * abs(prob2) > config.threshold:
                    graph.add_edge(word1, word2, weight=abs(prob1) * abs(prob2) * 1000)
                    graph.add_node(word1, importance=abs(prob1) * config.node_size)
                    graph.add_node(word2, importance=abs(prob2) * config.node_size)

    return graph
