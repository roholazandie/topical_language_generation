import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def calculate_similarity(sent1, sent2):
    module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
    embed = hub.KerasLayer(module_url)
    embeddings = embed([sent1, sent2])
    sent1_embed = embeddings[0, :].numpy()
    sent2_embed = embeddings[1, :].numpy()
    similarity = np.dot(sent1_embed, sent2_embed) / (np.linalg.norm(sent1_embed) * np.linalg.norm(sent2_embed))
    return similarity

if __name__ == "__main__":
    sent1 = "Apple watch is the best product."
    sent2 = "I like to eat apple."
    sent3 = " rises Apple corp the  the wages."
    s1 = calculate_similarity(sent1, sent2)
    s2 = calculate_similarity(sent1, sent3)
    print(s1)
    print(s2)