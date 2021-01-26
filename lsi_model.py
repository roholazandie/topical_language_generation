from gensim import corpora
from collections import defaultdict
from gensim.models import LsiModel, TfidfModel, CoherenceModel
from gensim.corpora import Dictionary
from configs import LSIConfig
from datasets.congress_dataset import CongressDataset
from datasets.newsgroups import NewsgroupDataset
from topical_tokenizers import SpacyTokenizer, TransformerGPT2Tokenizer
from datasets.topical_dataset import TopicalDataset
from datasets.nytimes_dataset import NYTimesDataset
from datasets.anes_dataset import AnesDataset
from spacy.lang.en.stop_words import STOP_WORDS
import numpy as np
import pickle
import os


class LSIModel:

    def __init__(self, config, build=False):
        self.config = config
        if self.config.tokenizer == "spacy":
            self.tokenizer = SpacyTokenizer(self.config.cached_dir)
        elif self.config.tokenizer == "gpt":
            self.tokenizer = TransformerGPT2Tokenizer(self.config.cached_dir)

        self.topic_top_words_file = os.path.join(self.config.cached_dir, "top_word_file.p")
        self.topic_words_matrix_file = os.path.join(self.config.cached_dir, "topic_matrix.p")
        self.model_file = os.path.join(self.config.cached_dir, "lsi_model_file.p")
        self.dictionary_file = os.path.join(self.config.cached_dir, "dictionary.p")
        self.corpus_file = os.path.join(self.config.cached_dir, "corpus.p")
        self.docs_file = os.path.join(self.config.cached_dir, "docs.p")

        if build:
            self._start()

    def _start(self):
        self._clear_cache()
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
        elif self.config.name == "newsgroup":
            dataset = NewsgroupDataset(self.config.dataset_dir, self.tokenizer)
        else:
            raise ValueError("unknown dataset")

        docs = [doc for doc in dataset]
        pickle.dump(docs, open(self.docs_file, 'wb'))
        self.dictionary = Dictionary(docs)

        self.dictionary.filter_extremes(no_below=self.config.no_below,
                                        no_above=self.config.no_above)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in docs]

        pickle.dump(self.corpus, open(self.corpus_file, 'wb'))
        pickle.dump(self.dictionary, open(self.dictionary_file, 'wb'))

    def _run_model(self):
        tfidf = TfidfModel(self.corpus)
        corpus_tfidf = tfidf[self.corpus]
        self.lsi_model = LsiModel(corpus_tfidf,
                                  id2word=self.dictionary,
                                  num_topics=self.config.num_topics)
        self.lsi_model.save(self.model_file)

    def get_docs(self):
        return pickle.load(open(self.docs_file, 'rb'))

    def get_model(self):
        return LsiModel.load(self.model_file)

    def get_dictionary(self):
        return pickle.load(open(self.dictionary_file, 'rb'))

    def get_corpus(self):
        return pickle.load(open(self.corpus_file, 'rb'))

    def get_topic_words(self, num_words=None):
        if not os.path.isfile(self.topic_top_words_file):
            if not num_words:
                num_words = len(self.dictionary)
            try:
                lsi_model = self.lsi_model
            except:
                lsi_model = self.get_model()

            topic_words = lsi_model.show_topics(self.config.num_topics,
                                                num_words=num_words,
                                                formatted=False)
            pickle.dump(topic_words, open(self.topic_top_words_file, 'wb'))
        else:
            topic_words = pickle.load(open(self.topic_top_words_file, 'rb'))
        return topic_words

    def get_topic_words_matrix(self):
        if not os.path.isfile(self.topic_words_matrix_file):
            try:
                lsi_model = self.lsi_model
            except:
                try:
                    lsi_model = self.get_model()
                except:
                    self._start()
                    lsi_model = self.lsi_model

            topic_words = lsi_model.get_topics()  # K X V' (num_topics x selected_vocab_size)
            topic_word_matrix = np.zeros(
                (self.config.num_topics, self.tokenizer.tokenizer.vocab_size))  # K x V (num_topics x vocab_size)
            try:
                dictionary = self.dictionary
            except:
                dictionary = self.get_dictionary()

            for i in range(len(dictionary)):
                j = self.tokenizer.tokenizer.convert_tokens_to_ids(lsi_model.id2word[i])
                topic_word_matrix[:, j] = topic_words[:, i]

            pickle.dump(topic_word_matrix, open(self.topic_words_matrix_file, 'wb'))
        else:
            topic_word_matrix = pickle.load(open(self.topic_words_matrix_file, 'rb'))
        return topic_word_matrix

    def get_topic_words_id(self):
        topic_words_id = self.lsi_model.get_topics()  # K X V (num_topics x num_words)
        pickle.dump(topic_words_id, open(self.config.topic_top_words_file, 'wb'))
        return topic_words_id

    def _clear_cache(self):
        all_cached_files = [self.model_file,
                            self.topic_top_words_file,
                            self.topic_words_matrix_file]
        for f in all_cached_files:
            try:
                os.remove(f)
            except:
                pass

    def get_coherence_score(self, coherence="u_mass"):
        try:
            model = self.lsi_model
        except:
            model = self.get_model()

        try:
            corpus = self.corpus
        except:
            corpus = self.get_corpus()

        if coherence == "u_mass":
            cm = CoherenceModel(model=model, corpus=corpus, coherence='c_w2v')
        elif coherence == "c_w2v":
            cm = CoherenceModel(model=model, texts=self.get_docs(),
                                dictionary=self.get_dictionary(), coherence=coherence)
        coherence = cm.get_coherence()
        # coherence = cm.get_coherence_per_topic()
        return coherence


if __name__ == "__main__":
    config_file = "configs/alexa_lsi_config.json"  # -2.240693249483874
    # config_file = "configs/nytimes_lsi_config.json" #-2.3255072569456896
    # config_file = "configs/anes_lsi_config.json" #-3.35591499048434
    # config_file = "configs/congress_lsi_config.json" #-2.842185966368092
    config = LSIConfig.from_json_file(config_file)
    lsi = LSIModel(config=config, build=False)
    #lsi._start()
    tw = lsi.get_topic_words(num_words=10)
    topic_words = [t[1] for t in tw]
    for topic in topic_words:
        print(topic)
    # sc = lsi.get_coherence_score("c_w2v")
    # print(sc)
    #
