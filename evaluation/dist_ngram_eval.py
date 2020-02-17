from configs import LSIConfig, GenerationConfig, LDAConfig
from topical_generation import generate_lsi_text, generate_lda_text
from evaluation.metrics import Metrics
import numpy as np

def eval_ngram(model, prompt_file):
    num_prompt_words = 4
    text_length = 50
    metric = Metrics()
    all_texts = []

    dists1 = []
    dists2 = []
    dists3 = []

    if model == "lsi":
        lsi_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lsi_config.json"
        generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
        lsi_config = LSIConfig.from_json_file(lsi_config_file)
        generation_config = GenerationConfig.from_json_file(generation_config_file)
    elif model == "lda":
        lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
        generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"

        lda_config = LDAConfig.from_json_file(lda_config_file)
        generation_config = GenerationConfig.from_json_file(generation_config_file)

    else:
        raise ValueError("Unknown model")

    with open(prompt_file) as fr:
        for i, line in enumerate(fr):
            prompt_text = " ".join(line.split()[:num_prompt_words])
            if model == "lsi":
                text = generate_lsi_text(prompt_text=prompt_text,
                                         selected_topic_index=2,
                                         lsi_config=lsi_config,
                                         generation_config=generation_config)
            elif model == "lda":
                text = generate_lda_text(prompt_text="The issue is",
                                         selected_topic_index=2,
                                         lda_config=lda_config,
                                         generation_config=generation_config
                                         )



            if len(text.split()) > text_length:
                print(text)
                print("###########################")
                text = " ".join(text.split()[1:])
                all_texts.append(text)
                dist1 = metric.distinct_1(text)
                print("dist1: ", dist1)
                dists1.append(dist1)

                dist2 = metric.distinct_2(text)
                print("dist2: ", dist2)
                dists2.append(dist2)

                dist3 = metric.distinct_3(text)
                print("dist3: ", dist3)
                dists3.append(dist3)

    print(dists1)
    print(dists2)
    print(dists3)

    print("dist1: ", np.mean(dists1))
    print("dist3: ", np.mean(dists2))
    print("dist3: ", np.mean(dists3))


if __name__ == "__main__":
    prompt_file = "/media/rohola/data/sample_texts/films/film_reviews.txt"
    #prompt_file = "/media/data2/rohola_data/film_reviews.txt"
    eval_ngram("lda", prompt_file)