from datasets.topical_dataset import TopicalDataset
from lda_model import LDAModel

from topical_tokenizers import TransformerGPT2Tokenizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from configs import LDAConfig
from pprint import pprint
import numpy as np
from evaluation.similarity_measures import calculate_similarity


# def get_topic_model(config, dataset):
#     dictionary = Dictionary(dataset)
#
#     # Filter out words that occur less than 20 documents, or more than 50% of the documents.
#     dictionary.filter_extremes(no_below=20, no_above=0.2)
#
#     # Bag-of-words representation of the documents.
#     corpus = [dictionary.doc2bow(doc) for doc in dataset]
#
#     # Set training parameters.
#     # num_topics = 10
#     # chunksize = 2000
#     # passes = 20
#     # iterations = 400
#     # eval_every = None  # Don't evaluate model perplexity, takes too much time.
#
#     # Make a index to word dictionary.
#     temp = dictionary[0]
#     id2word = dictionary.id2token
#
#     model = LdaModel(
#         corpus=corpus,
#         id2word=id2word,
#         chunksize=config.chunksize,
#         alpha=config.alpha,
#         eta=config.eta,
#         iterations=config.iterations,
#         num_topics=config.num_topics,
#         passes=config.passes,
#         eval_every=config.eval_every
#     )
#     print("model alpha: ", model.alpha)
#
#     top_topics = model.top_topics(corpus)
#     pprint(top_topics)
#
#     return model, corpus, id2word


def doc_bow_generate(config, doc_id):
    '''
    here we try to generate a sample similar to given doc (with doc_id)
    with the same length using LDA. This will give us a similar bow representation
    of the given doc:
    psi ~ Dir(beta)
    theta ~ Dir(alpha)
    z ~ Cat(theta)
    w ~ Cat(psi_z)
    '''
    #model, corpus, id2word = get_topic_model(config, dataset)

    lda_model = LDAModel(config)
    psi = lda_model.get_psi_matrix()
    theta = lda_model.get_theta_matrix()
    corpus = lda_model.get_corpus()
    docs = lda_model.get_docs()
    print(" ".join([t.strip('Ġ') for t in docs[doc_id]]))
    dictionary = lda_model.get_dictionary()
    temp = dictionary[0]
    num_gpt_vocabs = len(lda_model.tokenizer.tokenizer)
    id2word = [lda_model.tokenizer.tokenizer.convert_ids_to_tokens(i) for i in range(num_gpt_vocabs)]

    doc_theta = theta[doc_id, :]
    num_topics = theta.shape[1]
    selected_bow_doc = corpus[doc_id]

    generated_words = []
    for i in range(len(selected_bow_doc)):
        doc_theta /= doc_theta.sum()
        z = np.random.choice(list(range(num_topics)), 1, p=doc_theta)[0]
        selected_psi = psi[z, :]
        selected_psi /= selected_psi.sum()
        word = np.random.choice(id2word, 1, p=selected_psi)[0]
        generated_words.append(word)

    generated_doc = " ".join(generated_words)
    return generated_doc


if __name__ == "__main__":
    config_file = "configs/alexa_lda_config.json"
    config = LDAConfig.from_json_file(config_file)
    # tokenizer = TransformerGPT2Tokenizer(cached_dir=config.cached_dir)
    # topical_dataset = TopicalDataset(config.dataset_dir, tokenizer)
    # topical_dataset = [doc for doc in topical_dataset]


    doc_id = 10
    generated_doc = doc_bow_generate(config, doc_id=doc_id)
    print(generated_doc)
    # print(generated_doc)
    # selected_doc = " ".join(topical_dataset[doc_id])
    # s = calculate_similarity(generated_doc, selected_doc)
    # print(s)