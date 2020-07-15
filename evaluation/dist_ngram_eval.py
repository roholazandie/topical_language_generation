from configs import LSIConfig, GenerationConfig, LDAConfig
from run_generation import generate_unconditional_text
from topical_generation import generate_lsi_text, generate_lda_text, pplm_text, ctrl_text
from evaluation.metrics import Distinct
import numpy as np

def eval_ngram(model, config, generation_config, prompt_file, out_file, topic=0):
    num_prompt_words = 4
    text_length = 50
    metric = Distinct()
    all_texts = []

    dists1 = []
    dists2 = []
    dists3 = []

    # if model == "lsi":
    #     lsi_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lsi_config.json"
    #     generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    #     lsi_config = LSIConfig.from_json_file(lsi_config_file)
    #     generation_config = GenerationConfig.from_json_file(generation_config_file)
    # elif model == "lda":
    #     lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
    #     generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    #
    #     lda_config = LDAConfig.from_json_file(lda_config_file)
    #     generation_config = GenerationConfig.from_json_file(generation_config_file)
    #
    # else:
    #     raise ValueError("Unknown model")

    with open(prompt_file) as fr:
        for i, line in enumerate(fr):
            prompt_text = " ".join(line.split()[:num_prompt_words])
            if model == "gpt2":
                text = generate_unconditional_text(prompt_text=prompt_text,
                                                   generation_config=generation_config)

            if model == "lsi":
                text, _, _ = generate_lsi_text(prompt_text=prompt_text,
                                         selected_topic_index=topic,
                                         lsi_config=config,
                                         generation_config=generation_config)
            elif model == "lda":
                text, _, _ = generate_lda_text(prompt_text=prompt_text,
                                         selected_topic_index=topic,
                                         lda_config=config,
                                         generation_config=generation_config
                                         )

            elif model == "pplm":
                text = pplm_text(prompt_text=prompt_text,
                                 topic=topic,
                                 generation_config=generation_config)

            elif model == "ctrl":
                text = ctrl_text(prompt_text=prompt_text,
                                  topic=topic,
                                  generation_config=generation_config)


            if len(text.split()) > text_length:
                print(text)
                print("###########################")
                text = " ".join(text.split()[1:])
                all_texts.append(text)
                #dist1 = metric.distinct_1(text)
                dist1 = metric.distinct_n(text, 1)
                print("dist1: ", dist1)
                dists1.append(dist1)

                #dist2 = metric.distinct_2(text)
                dist2 = metric.distinct_n(text, 2)
                print("dist2: ", dist2)
                dists2.append(dist2)

                #dist3 = metric.distinct_3(text)
                dist3 = metric.distinct_n(text, 3)
                print("dist3: ", dist3)
                dists3.append(dist3)

    with open(out_file, 'w') as file_writer:
        file_writer.write(str(dists1)+"\n")
        file_writer.write(str(dists2)+"\n")
        file_writer.write(str(dists3)+"\n")
        file_writer.write("dist1: "+str(np.mean(dists1)) + "\n")
        file_writer.write("dist2: "+str(np.mean(dists2)) + "\n")
        file_writer.write("dist3: "+str(np.mean(dists3)) + "\n")

    print(dists1)
    print(dists2)
    print(dists3)

    print("dist1: ", np.mean(dists1))
    print("dist2: ", np.mean(dists2))
    print("dist3: ", np.mean(dists3))


if __name__ == "__main__":
    #prompt_file = "/media/rohola/data/sample_texts/films/film_reviews.txt"
    out_file = "/home/rohola/codes/topical_language_generation/results/distngram/lsi_result.txt"
    prompt_file = "/media/data2/rohola_data/film_reviews.txt"
    #prompt_file = "/media/rohola/data/sample_texts/films/film_reviews.txt"

    ##Unconditional GPT2
    # generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    # generation_config = GenerationConfig.from_json_file(generation_config_file)


    ##LSI
    # lsi_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lsi_config.json"
    # generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    # lsi_config = LSIConfig.from_json_file(lsi_config_file)
    # generation_config = GenerationConfig.from_json_file(generation_config_file)

    ##LDA
    # lda_config_file = "/home/rohola/codes/topical_language_generation/configs/alexa_lda_config.json"
    # generation_config_file = "/home/rohola/codes/topical_language_generation/configs/generation_config.json"
    # lda_config = LDAConfig.from_json_file(lda_config_file)
    # generation_config = GenerationConfig.from_json_file(generation_config_file)


    ##CTRL
    generation_config_file = "/home/rohola/codes/topical_language_generation/configs/ctrl_generation_config.json"
    generation_config = GenerationConfig.from_json_file(generation_config_file)


    # generation_config_file = "/home/rohola/codes/topical_language_generation/configs/pplm_generation_config.json"
    # generation_config = GenerationConfig.from_json_file(generation_config_file)

    eval_ngram(model="ctrl",
               config=None,
               generation_config=generation_config,
               prompt_file=prompt_file,
               out_file=out_file,
               topic="politics")