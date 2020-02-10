from collections import namedtuple
from dataclasses import dataclass
from visualization.topic_modeling_semantic_network import visualize_semantic_netwrok
from lsi_model import LSIModel


#config_file = "configs/alexa_lsi_config.json"
#config_file = "configs/nytimes_lsi_config.json"
#config_file = "configs/anes_lsi_config.json"
config_file = "configs/newsgroup_lsi_config.json"

lsi = LSIModel(config_file=config_file, build=True)

#lsi._start()

tw = lsi.get_topic_words(num_words=15)
topic_words = [t[1] for t in tw]
#clean up words
topic_words = [[(t[0].strip('Ġ'), t[1]) for t in tw] for tw in topic_words]
for topic in topic_words:
    print(topic)



#todo remove dataclass and replace it with VisualizationConfig class
@dataclass
class config:
    dimension: int = 2
    threshold: float = 0.001
    node_size: float = 30
    color_scale: str = "Viridis"
    title: str = "LSI"
    out_file_name: str = lsi.config.cached_dir+"/lsi_viz.html"


visualize_method = ""
if config.dimension == 2:
    visualize_method = 'plotly'
elif config.dimension == 3:
    visualize_method = 'plotly3d'
else:
    raise("Wrong dimension, can accept only 2 or 3")

visualize_semantic_netwrok(config, topic_words, visualize_method=visualize_method)