from gensim.models import CoherenceModel
from gensim.topic_coherence import segmentation
from configs import LDAConfig, LSIConfig
from lda_model import LDAModel
import numpy as np

from lsi_model import LSIModel


def get_coherence(doc):
    config_file = "../configs/alexa_lda_config.json" #0.5320263
    config = LDAConfig.from_json_file(config_file)
    model = LDAModel(config)

    # config_file = "../configs/alexa_lsi_config.json"
    # config = LSIConfig.from_json_file(config_file)
    # model = LSIModel(config=config, build=False)

    dictionary = model.get_dictionary()
    temp = dictionary[0]  # This is only to "load" the dictionary.

    cm = CoherenceModel(model=model.get_model(),
                        texts=model.get_docs(),
                        dictionary=dictionary,
                        coherence="c_w2v")

    tokens = model.tokenizer.tokenize(doc)
    doc_ids = [np.array([dictionary.token2id[token] for token in tokens])]

    # #with truncated dictionary
    # doc_ids = []
    # for token in tokens:
    #     try:
    #         doc_ids.append(dictionary.token2id[token])
    #     except:
    #         pass
    # doc_ids = [np.array(doc_ids)]

    segmented_doc = segmentation.s_one_set(doc_ids)

    doc_coherence = cm.get_coherence_per_topic(segmented_doc)[0]
    return doc_coherence


if __name__ == "__main__":
    coherence = get_coherence("war war war")
    print(coherence)