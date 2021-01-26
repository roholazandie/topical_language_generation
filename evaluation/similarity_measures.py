import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
#from sentence_transformers import SentenceTransformer


class TextSimilarity:

    def __init__(self):
        module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
        self.embed = hub.KerasLayer(module_url)

        #self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def nnlm_sentence_similarity(self, sent1, sent2):
        embeddings = self.embed([sent1, sent2])
        sent1_embed = embeddings[0, :].numpy()
        sent2_embed = embeddings[1, :].numpy()
        similarity = np.dot(sent1_embed, sent2_embed) / (np.linalg.norm(sent1_embed) * np.linalg.norm(sent2_embed))
        return similarity


    def bert_sentence_similarity(self, sent1, sent2):
        embeddings = self.model.encode([sent1, sent2])
        sent1_embed = embeddings[0]
        sent2_embed = embeddings[1]
        similarity = np.dot(sent1_embed, sent2_embed) / (np.linalg.norm(sent1_embed) * np.linalg.norm(sent2_embed))
        return similarity


if __name__ == "__main__":

    text_similarity = TextSimilarity()

    sent1 = "Apple watch is the best product."
    sent2 = "I like to eat apple."
    sent3 = " rises Apple corp the  the wages."
    s1 = text_similarity.nnlm_sentence_similarity(sent1, sent2)
    s2 = text_similarity.nnlm_sentence_similarity(sent1, sent3)
    print(s1)
    print(s2)

    print("###")
    # s1 = text_similarity.bert_sentence_similarity(sent1, sent2)
    # s2 = text_similarity.bert_sentence_similarity(sent1, sent3)
    # print(s1)
    # print(s2)