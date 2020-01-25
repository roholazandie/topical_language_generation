from gensim.utils import ClippedCorpus

from datasets.topical_dataset import TopicalDataset
from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus

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
        # self._start()

    def _start(self):
        self._prepare_dataset()
        self._run_model()

    def _prepare_dataset(self):
        topical_dataset = TopicalDataset(self.config.dataset_dir, self.tokenizer)
        docs = [doc for doc in topical_dataset]
        self.dictionary = Dictionary(docs)
        self.dictionary.filter_extremes(no_below=20, no_above=0.2)
        # Bag-of-words representation of the documents.
        self.corpus = [self.dictionary.doc2bow(doc) for doc in docs]

        corpus_file = os.path.join(self.config.cached_dir, "corpus.p")
        pickle.dump(self.corpus, open(corpus_file, 'wb'))

        temp = self.dictionary[0]  # This is only to "load" the dictionary.
        self.id2token = self.dictionary.id2token


    def _run_model(self):
        self.model = LdaModel(
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
        model_file = os.path.join(self.config.cached_dir, "model.p")
        self.model.save(model_file)


    def get_model(self):
        model_file = os.path.join(self.config.cached_dir, "model.p")
        return LdaModel.load(model_file)

    def get_corpus(self):
        corpus_file = os.path.join(self.config.cached_dir, "corpus.p")
        return pickle.load(open(corpus_file, 'rb'))

    def clear_cache(self):
        all_topic_tokens_file = os.path.join(self.config.cached_dir, "all_topic_tokens.p")
        psi_matrix_file = os.path.join(self.config.cached_dir, "psi_matrix.p")
        theta_matrix_file = os.path.join(self.config.cached_dir, "theta_matrix.p")

        all_cached_files = [all_topic_tokens_file, psi_matrix_file, theta_matrix_file]
        for f in all_cached_files:
            try:
                os.remove(f)
            except:
                pass

    def get_all_topic_tokens(self):
        all_topic_tokens_file = os.path.join(self.config.cached_dir, "all_topic_tokens.p")
        if not os.path.isfile(all_topic_tokens_file):
            tokenid_probs = [self.model.get_topic_terms(i) for i in range(self.config.num_topics)]
            all_topic_tokens = [[(self.id2token[i], p) for i, p in tokenid_probs_topic] for tokenid_probs_topic in
                                tokenid_probs]
            pickle.dump(all_topic_tokens, open(all_topic_tokens_file, 'wb'))
        else:
            all_topic_tokens = pickle.load(open(all_topic_tokens_file, 'rb'))

        return all_topic_tokens

    def get_psi_matrix(self):
        psi_matrix_file = os.path.join(self.config.cached_dir, "psi_matrix.p")
        if type(self.tokenizer) == TransformerGPT2Tokenizer:
            if not os.path.isfile(psi_matrix_file):
                self._start()
                psi_matrix = np.zeros((self.config.num_topics, self.tokenizer.tokenizer.vocab_size))
                matrix = self.model.get_topics()  # a matrix of k x V' (num_topic x selected_vocab_size)
                for i in range(len(self.id2token)):
                    j = self.tokenizer.tokenizer.convert_tokens_to_ids(self.id2token[i])
                    psi_matrix[:, j] = matrix[:, i]

                pickle.dump(psi_matrix, open(psi_matrix_file, 'wb'))
            else:
                psi_matrix = pickle.load(open(psi_matrix_file, 'rb'))

        elif type(self.tokenizer) == SpacyTokenizer:
            '''
            Converting spacy tokenizer outputs to gpt2tokenizer is not a one-to-one mapping
            To do the mapping we first create a dictionary from spacy ids to gpt ids. one word in spacy can be 
            tokenized to multiple gpt tokens so we have a dictionary of {spacy_id: [gpt_id, gpt_id , gpt_id]}
            after inverting this dictionary to {gpt_id: [spacy_id, spacy_id]}
            we take the mean of the corresponding columns from topic_matrix with spacy ids.
            '''
            if not os.path.isfile(psi_matrix_file):
                self._start()
                gpt_tokenizer = TransformerGPT2Tokenizer(self.config.cached_dir)
                spacy_to_gpt = {i: gpt_tokenizer.tokenizer.encode(token) for i, token in
                                enumerate(list(self.dictionary.token2id.keys()))}

                gpt_to_spacy = {}
                for k, v in spacy_to_gpt.items():
                    for x in v:
                        gpt_to_spacy.setdefault(x, []).append(k)

                matrix = self.model.get_topics()

                psi_matrix = np.zeros((self.config.num_topics, len(gpt_tokenizer.tokenizer)))
                # take mean of the tokenized items by spacy
                for gpt_id in gpt_to_spacy:
                    spacy_ids = gpt_to_spacy[gpt_id]
                    mean_value = np.mean([matrix[:, id] for id in spacy_ids], axis=0)
                    psi_matrix[:, gpt_id] = mean_value

                pickle.dump(psi_matrix, open(psi_matrix_file, 'wb'))
            else:
                psi_matrix = pickle.load(open(psi_matrix_file, 'rb'))
        else:
            raise ValueError("tokenizer not recognized")

        return psi_matrix

    def get_theta_matrix(self):
        theta_matrix_file = os.path.join(self.config.cached_dir, "theta_matrix.p")
        if not os.path.isfile(theta_matrix_file):
            num_documents = len(self.corpus)
            theta_matrix = np.zeros((num_documents, self.config.num_topics))
            for i, c in enumerate(self.corpus):
                for j, p in self.model.get_document_topics(c):
                    theta_matrix[i, j] = p

            pickle.dump(theta_matrix, open(theta_matrix_file, 'wb'))
        else:
            theta_matrix = pickle.load(open(theta_matrix_file, 'rb'))

        return theta_matrix




if __name__ == "__main__":
    config_file = "configs/alexa_lda_config.json"
    lda = LDAModel(config_file=config_file)

    # topic_words = lda.model.show_topic(0) # show 0th topic words
    psi = lda.get_psi_matrix()
    theta = lda.get_theta_matrix()
    all_topic_tokens = lda.get_all_topic_tokens()
    print(all_topic_tokens)

    # print(theta)
