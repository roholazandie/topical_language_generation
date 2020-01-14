from datasets.topical_dataset import TopicalDataset
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from tokenizers.tokenizers import TransformerGPT2Tokenizer
from dictionary import TopicalDictionary
import numpy as np
import pickle
from pprint import pprint
from configs import LDAConfig
from tokenizers.tokenizers import SpacyTokenizer
import time

def gpt2_tokenizer_dataset(docs, tokenizer):
    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    topical_dictionary = TopicalDictionary(docs, tokenizer)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    topical_dictionary.filter_extremes(no_below=20, no_above=0.2)

    # Bag-of-words representation of the documents.
    corpus = [topical_dictionary.doc2bow(doc) for doc in docs]
    id2word = topical_dictionary.dictionary

    return corpus, id2word


def spacy_tokinzer_dataset(docs):
    dictionary = Dictionary(docs)
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.2)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    return corpus, id2word


config_file = "configs/alexa_config.json"
config = LDAConfig.from_json_file(config_file)

#tokenizer = TransformerGPT2Tokenizer(config.cached_dir)
tokenizer = SpacyTokenizer(dict_dir=config.cached_dir)

topical_dataset = TopicalDataset(config.dataset_dir, tokenizer)

docs = [doc for doc in topical_dataset]

#corpus, id2word = spacy_tokinzer_dataset(docs)
corpus, id2word = gpt2_tokenizer_dataset(docs, tokenizer)

# topical_dictionary = TopicalDictionary(docs, tokenizer)
#
# topical_dictionary.filter_extremes(no_below=20, no_above=0.2)
#
# corpus = [topical_dictionary.doc2bow(doc) for doc in docs]
# id2word = topical_dictionary.dictionary


# Set training parameters.
# num_topics = 10
# chunksize = 2000
# passes = 20
# iterations = 400
# eval_every = None  # Don't evaluate model perplexity, takes too much time.


model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=config.chunksize,
    alpha=config.alpha,
    eta=config.eta,
    iterations=config.iterations,
    num_topics=config.num_topics,
    passes=config.passes,
    eval_every=config.eval_every
)

top_topics = model.top_topics(corpus)

# np.save(config.topics_file, model.get_topics())
# # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
# avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
# print('Average topic coherence: %.4f.' % avg_topic_coherence)


pprint(top_topics)

pickle.dump(top_topics, open(config.topic_top_words_file, "wb"))