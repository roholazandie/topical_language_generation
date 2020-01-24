from datasets.topical_dataset import TopicalDataset

from topical_tokenizers import TransformerGPT2Tokenizer, SpacyTokenizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from configs import LDAConfig
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from evaluate import calculate_similarity


def get_topic_model(config, dataset):
    dictionary = Dictionary(dataset)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.2)

    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in dataset]

    # Set training parameters.
    # num_topics = 10
    # chunksize = 2000
    # passes = 20
    # iterations = 400
    # eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]
    id2word = dictionary.id2token

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
    print("model alpha: ", model.alpha)

    top_topics = model.top_topics(corpus)
    pprint(top_topics)

    return model, corpus, id2word


def doc_bow_generate(config, dataset, doc_id, original_dictionary):
    '''
    here we try to generate a sample similar to given doc (with doc_id)
    with the same length using LDA. This will give us a similar bow representation
    of the given doc:
    psi ~ Dir(beta)
    theta ~ Dir(alpha)
    z ~ Cat(theta)
    w ~ Cat(psi_z)
    '''
    model, corpus, id2word = get_topic_model(config, dataset)

    selected_bow_doc = corpus[doc_id]
    generated_words = []
    for i in range(len(selected_bow_doc)):
        document_topic = model.get_document_topics(selected_bow_doc)
        theta = np.array([dt[1] for dt in document_topic])
        doc_indices = np.array([dt[0] for dt in document_topic])
        theta /= theta.sum()
        z = np.random.choice(doc_indices, 1, p=theta)[0]
        psi = model.get_topics()[z, :]
        psi /= psi.sum()
        word_index = np.random.choice(list(id2word.keys()), 1, p=psi)[0]
        generated_words.append(id2word[word_index])

    generated_doc = " ".join(generated_words)
    return generated_doc


if __name__ == "__main__":
    config_file = "configs/alexa_lda_config.json"
    config = LDAConfig.from_json_file(config_file)
    #tokenizer = SpacyTokenizer(dict_dir=config.cached_dir)
    tokenizer = TransformerGPT2Tokenizer(cached_dir=config.cached_dir)
    topical_dataset = TopicalDataset(config.dataset_dir, tokenizer)
    topical_dataset = [doc for doc in topical_dataset]
    doc_id = 0
    generated_doc = doc_bow_generate(config,
                                     doc_id=doc_id,
                                     dataset=topical_dataset,
                                     original_dictionary=tokenizer.dictionary)

    print(generated_doc)
    selected_doc = " ".join(topical_dataset[doc_id])
    s = calculate_similarity(generated_doc, selected_doc)
    print(s)