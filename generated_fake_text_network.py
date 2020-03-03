from configs import LDAConfig, GenerationConfig, PlotConfig
from lda_model import LDAModel
from topical_generation import generate_document_like_text
from visualization.topic_modeling_semantic_network import visualize_semantic_netwrok
import os


lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"

config = LDAConfig.from_json_file(lda_config_file)
generation_config = GenerationConfig.from_json_file(generation_config_file)

doc_id = 320

texts = []
for i in range(100):
    generation_config.seed = i
    text, doc = generate_document_like_text(prompt_text="This is a",
                                                doc_id=doc_id,  #173,
                                                lda_config=config,
                                                generation_config=generation_config)
    texts.append(text)
    print("original: ", doc)
    print("generated: ", text)


all_text = " ".join(texts)

lda = LDAModel(config)
num_topics = sum(lda.get_theta_matrix()[doc_id, :] != 0)
###########visualize

lda_config_file = "/home/rohola/codes/topical_language_generation/configs/generated_fake_alexa_lda_config.json"

config = LDAConfig.from_json_file(lda_config_file)
config.num_topics = num_topics

##save the generate text to disk
if not os.path.isdir(config.dataset_dir):
    os.mkdir(config.dataset_dir)
with open(os.path.join(config.dataset_dir, "generated_text.txt"), 'w') as file_writer:
    file_writer.write(all_text)


lda = LDAModel(config, build=True)
all_topic_tokens = lda.get_all_topic_tokens(num_words=15)


# clean up words
topic_words = [[(t[0].strip('Ġ'), t[1]) for t in tw] for tw in all_topic_tokens]

for tw in topic_words:
    print(topic_words)

plot_config = PlotConfig.from_json_file("configs/lda_plot_config.json")

fig = visualize_semantic_netwrok(plot_config,
                                 topic_words)