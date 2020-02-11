from configs import LDAConfig, LSIConfig
import json

def convert_to_dataset_name(dataset_name):
    if dataset_name == "Alexa Topical":
        dataset = "alexa"
    elif dataset_name == "Newsgroup":
        dataset = "newsgroup"
    elif dataset_name == "New York Times":
        dataset = "nytimes"
    elif dataset_name == "ANES":
        dataset = "anes"
    elif dataset_name == "Congress Debates":
        dataset = "congress"

    return dataset


def convert_topical_model_names(topic_model_name):
    if topic_model_name == "Latent Dirichlet Allocation":
        topic_model = "lda"
    elif topic_model_name == "Latent Semantic Indexing":
        topic_model = "lsi"

    return topic_model


def convert_decoding_algorithm_names(decoding_algorithm_name):
    if decoding_algorithm_name == "No algorithm":
        decoding_algorithm = "method0"
    elif decoding_algorithm_name == "prenorm":
        decoding_algorithm = "method3"

    return decoding_algorithm


def find_default_values(dataset, topic_model):
    if dataset == "alexa":
        if topic_model == "lda":
            config_file = "configs/alexa_lda_config.json"
            topic_modeling_config = LDAConfig.from_json_file(config_file)
        elif topic_model == "lsi":
            config_file = "configs/alexa_lsi_config.json"
            topic_modeling_config = LSIConfig.from_json_file(config_file)
    elif dataset == "newsgroup":
        if topic_model == "lsi":
            config_file = "configs/newsgroup_lsi_config.json"
            topic_modeling_config = LSIConfig.from_json_file(config_file)
        elif topic_model == "lda":
            config_file = "configs/newsgroup_lda_config.json"
            topic_modeling_config = LDAConfig.from_json_file(config_file)
    elif dataset == "congress":
        if topic_model == "lsi":
            config_file = "configs/congress_lsi_config.json"
            topic_modeling_config = LSIConfig.from_json_file(config_file)
        elif topic_model == "lda":
            config_file = "configs/congress_lda_config.json"
            topic_modeling_config = LDAConfig.from_json_file(config_file)
    elif dataset == "nytimes":
        if topic_model == "lsi":
            config_file = "configs/nytimes_lsi_config.json"
            topic_modeling_config = LSIConfig.from_json_file(config_file)
        elif topic_model == "lda":
            config_file = "configs/nytimes_lda_config.json"
            topic_modeling_config = LDAConfig.from_json_file(config_file)
    elif dataset == "anes":
        if topic_model == "lsi":
            config_file = "configs/anes_lsi_config.json"
            topic_modeling_config = LSIConfig.from_json_file(config_file)
        elif topic_model == "lda":
            config_file = "configs/anes_lda_config.json"
            topic_modeling_config = LDAConfig.from_json_file(config_file)

    return topic_modeling_config


def create_streamlit_config_file(config_file, topic_model, dataset):
    if topic_model == "lda":
        if dataset == "congress":
            lda_config = LDAConfig.from_json_file("configs/congress_lda_config.json")
            json.dump(lda_config.__dict__, open(config_file, 'w'))
        elif dataset == "nytimes":
            lda_config = LDAConfig.from_json_file("configs/nytimes_lda_config.json")
            json.dump(lda_config.__dict__, open(config_file, 'w'))
        elif dataset == "alexa":
            lda_config = LDAConfig.from_json_file("configs/alexa_lda_config.json")
            json.dump(lda_config.__dict__, open(config_file, 'w'))
        elif dataset == "newsgroup":
            lda_config = LDAConfig.from_json_file("configs/newsgroup_lda_config.json")
            json.dump(lda_config.__dict__, open(config_file, 'w'))
        elif dataset == "anes":
            lda_config = LDAConfig.from_json_file("configs/anes_lda_config.json")
            json.dump(lda_config.__dict__, open(config_file, 'w'))


    elif topic_model == "lsi":
        if dataset == "congress":
            lsi_config = LSIConfig.from_json_file("configs/congress_lsi_config.json")
            json.dump(lsi_config.__dict__, open(config_file, 'w'))
        elif dataset == "nytimes":
            lsi_config = LSIConfig.from_json_file("configs/nytimes_lsi_config.json")
            json.dump(lsi_config.__dict__, open(config_file, 'w'))
        elif dataset == "alexa":
            lsi_config = LSIConfig.from_json_file("configs/alexa_lsi_config.json")
            json.dump(lsi_config.__dict__, open(config_file, 'w'))
        elif dataset == "newsgroup":
            lsi_config = LSIConfig.from_json_file("configs/newsgroup_lsi_config.json")
            json.dump(lsi_config.__dict__, open(config_file, 'w'))
        elif dataset == "anes":
            lsi_config = LSIConfig.from_json_file("configs/anes_lsi_config.json")
            json.dump(lsi_config.__dict__, open(config_file, 'w'))