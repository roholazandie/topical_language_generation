from gensim.utils import ClippedCorpus
from datasets.anes_dataset import AnesDataset
from datasets.congress_dataset import CongressDataset
from datasets.nytimes_dataset import NYTimesDataset
from datasets.topical_dataset import TopicalDataset
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary, MmCorpus
import gensim
import logging
gensim.logger.setLevel(logging.ERROR)

from datasets.wikidata import WikiData, WikiCorpus
from dictionary import TopicalDictionary
import numpy as np
import os
import pickle
from pprint import pprint
from configs import LDAConfig, LDAWikiConfig
from topical_tokenizers import SpacyTokenizer, TransformerGPT2Tokenizer
import time
from collections import defaultdict


class LDAModel:

    def __init__(self, config_file):
        self.config = LDAConfig.from_json_file(config_file)
        if self.config.tokenizer == "spacy":
            self.tokenizer = SpacyTokenizer(self.config.cached_dir)
        elif self.config.tokenizer == "gpt":
            self.tokenizer = TransformerGPT2Tokenizer(self.config.cached_dir)
        else:
            raise ValueError("Tokenizer not recognized")
        self.model_file = os.path.join(self.config.cached_dir, "model.p")
        self.corpus_file = os.path.join(self.config.cached_dir, "corpus.p")
        self.docs_file = os.path.join(self.config.cached_dir, "docs.p")
        self.dictionary_file = os.path.join(self.config.cached_dir, "dictionary.p")
        self.all_topic_tokens_file = os.path.join(self.config.cached_dir, "all_topic_tokens.p")
        self.psi_matrix_file = os.path.join(self.config.cached_dir, "psi_matrix.p")
        self.theta_matrix_file = os.path.join(self.config.cached_dir, "theta_matrix.p")
        # self._start()



    def _start(self):
        self._clear_caches()
        self._prepare_dataset()
        self._run_model()

    def _prepare_dataset(self):
        if self.config.name == "alexa":
            dataset = TopicalDataset(self.config.dataset_dir, self.tokenizer)
        elif self.config.name == "nytimes":
            dataset = NYTimesDataset(self.config.dataset_dir, self.tokenizer)
        elif self.config.name == "anes":
            dataset = AnesDataset(self.config.dataset_dir, self.tokenizer)
        elif self.config.name == "congress":
            dataset = CongressDataset(self.config.dataset_dir, self.tokenizer)
        else:
            raise ValueError("unknown dataset")

        self.docs = [doc for doc in dataset]
        pickle.dump(self.docs, open(self.docs_file, 'wb'))
        self.dictionary = Dictionary(self.docs)
        self.dictionary.filter_extremes(no_below=50, no_above=0.4)
        pickle.dump(self.dictionary, open(self.dictionary_file, 'wb'))
        # Bag-of-words representation of the documents.
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.docs]

        pickle.dump(self.corpus, open(self.corpus_file, 'wb'))

        temp = self.dictionary[0]  # This is only to "load" the dictionary.
        self.id2token = self.dictionary.id2token


    def _run_model(self):
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.id2token,
            chunksize=self.config.chunksize,
            alpha=self.config.alpha,
            eta=self.config.eta,
            iterations=self.config.iterations,
            num_topics=self.config.num_topics,
            passes=self.config.passes,
            eval_every=self.config.eval_every
        )

        self.lda_model.save(self.model_file)


    def get_model(self):
        return LdaModel.load(self.model_file)

    def get_corpus(self):
        return pickle.load(open(self.corpus_file, 'rb'))

    def get_dictionary(self):
        return pickle.load(open(self.dictionary_file, 'rb'))

    def get_docs(self):
        return pickle.load(open(self.docs_file, 'rb'))

    def _clear_caches(self):
        all_cached_files = [self.all_topic_tokens_file,
                            self.psi_matrix_file,
                            self.theta_matrix_file]
        for f in all_cached_files:
            try:
                os.remove(f)
            except:
                pass

    def get_all_topic_tokens(self, num_words=10):
        if not os.path.isfile(self.all_topic_tokens_file):
            try:
                lda_model = self.lda_model
            except:
                lda_model = self.get_model()
            tokenid_probs = [lda_model.get_topic_terms(i, topn=num_words) for i in range(self.config.num_topics)]
            all_topic_tokens = [[(self.id2token[i], p) for i, p in tokenid_probs_topic] for tokenid_probs_topic in
                                tokenid_probs]
            pickle.dump(all_topic_tokens, open(self.all_topic_tokens_file, 'wb'))
        else:
            all_topic_tokens = pickle.load(open(self.all_topic_tokens_file, 'rb'))

        return all_topic_tokens

    def get_psi_matrix(self):
        if type(self.tokenizer) == TransformerGPT2Tokenizer:
            if not os.path.isfile(self.psi_matrix_file):
                #self._start()
                psi_matrix = np.zeros((self.config.num_topics, self.tokenizer.tokenizer.vocab_size))
                matrix = self.lda_model.get_topics()  # a matrix of k x V' (num_topic x selected_vocab_size)
                for i in range(len(self.id2token)):
                    j = self.tokenizer.tokenizer.convert_tokens_to_ids(self.id2token[i])
                    psi_matrix[:, j] = matrix[:, i]

                pickle.dump(psi_matrix, open(self.psi_matrix_file, 'wb'))
            else:
                psi_matrix = pickle.load(open(self.psi_matrix_file, 'rb'))

        elif type(self.tokenizer) == SpacyTokenizer:
            '''
            Converting spacy tokenizer outputs to gpt2tokenizer is not a one-to-one mapping
            To do the mapping we first create a dictionary from spacy ids to gpt ids. one word in spacy can be 
            tokenized to multiple gpt tokens so we have a dictionary of {spacy_id: [gpt_id, gpt_id , gpt_id]}
            after inverting this dictionary to {gpt_id: [spacy_id, spacy_id]}
            we take the mean of the corresponding columns from topic_matrix with spacy ids.
            '''
            if not os.path.isfile(self.psi_matrix_file):
                self._start()
                gpt_tokenizer = TransformerGPT2Tokenizer(self.config.cached_dir)
                spacy_to_gpt = {i: gpt_tokenizer.tokenizer.encode(token) for i, token in
                                enumerate(list(self.dictionary.token2id.keys()))}

                gpt_to_spacy = {}
                for k, v in spacy_to_gpt.items():
                    for x in v:
                        gpt_to_spacy.setdefault(x, []).append(k)

                matrix = self.lda_model.get_topics()

                psi_matrix = np.zeros((self.config.num_topics, len(gpt_tokenizer.tokenizer)))
                # take mean of the tokenized items by spacy
                for gpt_id in gpt_to_spacy:
                    spacy_ids = gpt_to_spacy[gpt_id]
                    mean_value = np.mean([matrix[:, id] for id in spacy_ids], axis=0)
                    psi_matrix[:, gpt_id] = mean_value

                pickle.dump(psi_matrix, open(self.psi_matrix_file, 'wb'))
            else:
                psi_matrix = pickle.load(open(self.psi_matrix_file, 'rb'))
        else:
            raise ValueError("tokenizer not recognized")

        return psi_matrix

    def get_theta_matrix(self):
        if not os.path.isfile(self.theta_matrix_file):
            num_documents = len(self.corpus)
            theta_matrix = np.zeros((num_documents, self.config.num_topics))
            for i, c in enumerate(self.corpus):
                for j, p in self.lda_model.get_document_topics(c):
                    theta_matrix[i, j] = p

            pickle.dump(theta_matrix, open(self.theta_matrix_file, 'wb'))
        else:
            theta_matrix = pickle.load(open(self.theta_matrix_file, 'rb'))

        return theta_matrix

    def get_coherence_score(self, method="u_mass"):
        try:
            model = self.lda_model
        except:
            model = self.get_model()

        try:
            corpus = self.corpus
        except:
            corpus = self.get_corpus()

        try:
            dictionary = self.dictionary
        except:
            dictionary = self.get_dictionary()

        try:
            docs = self.docs
        except:
            docs = self.get_docs()

        if method == "u_mass":
            cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
            coherence = cm.get_coherence()

        elif method == "c_w2v":
            cm = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence=method)
            coherence = cm.get_coherence()
        else:
            raise ValueError("unknown method")

        return coherence


if __name__ == "__main__":
    config_file = "configs/alexa_lda_config.json" #0.5320263
    #config_file = "configs/nytimes_lda_config.json" #0.63788706

    # all_coherences = []
    # for i in range(5, 20):
    #     lda = LDAModel(config_file=config_file)
    #     lda.config.num_topics = i
    #     lda._start()
    #     #lda._start()
    #     # topic_words = lda.model.show_topic(0) # show 0th topic words
    #     # psi = lda.get_psi_matrix()
    #     # theta = lda.get_theta_matrix()
    #     # all_topic_tokens = lda.get_all_topic_tokens()
    #     # for tt in all_topic_tokens:
    #     #     print(tt)
    #
    #     coherence_score = lda.get_coherence_score()
    #     all_coherences.append((i, coherence_score))
    #     print(coherence_score)
    #
    # print(all_coherences)
    all_scores = []
    for i in range(3, 20):
        lda = LDAModel(config_file=config_file)
        lda.config.num_topics = i
        lda._start()
        c_w2v_score = lda.get_coherence_score(method="c_w2v")
        all_scores.append((i, c_w2v_score))

    print(all_scores)