from collections import namedtuple
from dataclasses import dataclass

from configs import LSIConfig
from visualization.topic_modeling_semantic_network import visualize_semantic_netwrok
from lsi_model import LSIModel


config_file = "configs/alexa_lsi_config.json"
#config_file = "configs/nytimes_lsi_config.json"
#config_file = "configs/anes_lsi_config.json"
#config_file = "configs/newsgroup_lsi_config.json"

config = LSIConfig.from_json_file(config_file)
lsi = LSIModel(config)

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
    node_size: float = 150
    color_scale: str = "Viridis"
    title: str = "LSI"
    out_file_name: str = lsi.config.cached_dir+"/lsi_viz.html"

visualize_semantic_netwrok(config, topic_words)