from collections import namedtuple
from dataclasses import dataclass
from visualization.topic_modeling_semantic_network import visualize_semantic_netwrok
from lda_model import LDAModel


#config_file = "configs/alexa_lda_config.json"
#config_file = "configs/nytimes_lda_config.json"
#config_file = "configs/anes_lda_config.json"
config_file = "configs/congress_lda_config.json"

lda = LDAModel(config_file=config_file)

lda._start()

all_topic_tokens = lda.get_all_topic_tokens(num_words=15)

#clean up words
topic_words = [[(t[0].strip('Ġ'), t[1]) for t in tw] for tw in all_topic_tokens]
for topic in topic_words:
    print(topic)



#todo remove dataclass and replace it with VisualizationConfig class
@dataclass
class config:
    dimension: int = 2
    threshold: float = 0.00001
    node_size: float = 30
    color_scale: str = "Viridis"
    title: str = "LDA"
    out_file_name: str = lda.config.cached_dir + "/lda_viz.html"



visualize_semantic_netwrok(config, topic_words)
