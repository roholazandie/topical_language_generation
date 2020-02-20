from gensim.models import CoherenceModel
from gensim.topic_coherence import segmentation
from configs import LDAConfig, LSIConfig, GenerationConfig
from lda_model import LDAModel
import numpy as np
from evaluation.metrics import TopicCoherence
from lsi_model import LSIModel
from topical_generation import generate_lsi_text, generate_lda_text, ctrl_text


def eval_topic_coherence(config, generation_config, topic_index, prompt_file, out_file):
    num_prompt_words = 4
    text_length = 10

    topic_coherence = TopicCoherence(config)
    coherences = []
    with open(prompt_file) as fr:
        for i, line in enumerate(fr):
            if len(coherences) > 200:
                break
            prompt_text = " ".join(line.split()[:num_prompt_words])
            if type(config) == LSIConfig:
                text = generate_lsi_text(prompt_text=prompt_text,
                                         selected_topic_index=topic_index,
                                         lsi_config=config,
                                         generation_config=generation_config)
            elif type(config) == LDAConfig:
                text = generate_lda_text(prompt_text=prompt_text,
                                         selected_topic_index=topic_index,
                                         lda_config=config,
                                         generation_config=generation_config
                                         )
            else:
                #ctrl
                text = ctrl_text(prompt_text=prompt_text,
                                 topic="Opinion",
                                 generation_config=generation_config)

            if len(text.split()) > text_length:
                coherence = topic_coherence.get_coherence(text)
                coherences.append(coherence)

    with open(out_file, 'w') as file_writer:
        file_writer.write(str(coherences) + "\n")
        file_writer.write("mean coherence: " + str(np.mean(coherences)) + "\n")

    print(coherences)
    print("mean coherence: ", np.mean(coherences))


if __name__ == "__main__":
    lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
    lda_config = LDAConfig.from_json_file(lda_config_file)

    #prompt_file = "/media/rohola/data/sample_texts/films/film_reviews.txt"

    #generation_config_file = "/home/rohola/codes/topical_language_generation/configs/ctrl_generation_config.json"
    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    generation_config = GenerationConfig.from_json_file(generation_config_file)
    prompt_file = "/media/data2/rohola_data/film_reviews.txt"
    out_file = "/home/rohola/codes/topical_language_generation/results/topic_coherence/topic_coherence_gpt_lda_result.txt"

    eval_topic_coherence(config=lda_config,
                         generation_config=generation_config,
                         topic_index=2,
                         prompt_file=prompt_file,
                         out_file=out_file)
