from gensim import corpora
from collections import defaultdict
from gensim.models import LsiModel, TfidfModel
from gensim.corpora import Dictionary
from configs import LSIConfig
from topical_tokenizers import SpacyTokenizer, TransformerGPT2Tokenizer
from datasets.topical_dataset import TopicalDataset
import numpy as np
import pickle
import os


class LSIModel:

    def __init__(self, config_file):
        self.config = LSIConfig.from_json_file(config_file)
        if self.config.tokenizer == "spacy":
            self.tokenizer = SpacyTokenizer(self.config.cached_dir)
        elif self.config.tokenizer == "gpt":
            self.tokenizer = TransformerGPT2Tokenizer(self.config.cached_dir)

        self.topic_top_words_file = os.path.join(self.config.cached_dir, "top_word_file.p")
        self.topic_words_matrix = os.path.join(self.config.cached_dir, "topic_matrix.p")
        self.model_file = os.path.join(self.config.cached_dir, "model_file.p")

        #self._start()

    def _start(self):
        self._prepare_dataset()
        self._run_model()

    def _prepare_dataset(self):
        topical_dataset = TopicalDataset(self.config.dataset_dir, self.tokenizer)
        docs = [doc for doc in topical_dataset]
        self.dictionary = Dictionary(docs)
        self.dictionary.filter_extremes(no_below=20, no_above=0.2)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in docs]

    def _run_model(self):
        tfidf = TfidfModel(self.corpus)
        corpus_tfidf = tfidf[self.corpus]
        self.lsi_model = LsiModel(corpus_tfidf,
                                  id2word=self.dictionary,
                                  num_topics=self.config.num_topics)
        self.lsi_model.save(self.model_file)

    def get_model(self):
        return LsiModel.load(self.model_file)

    def get_topic_words(self, num_words=None):
        if not os.path.isfile(self.topic_top_words_file):
            if not num_words:
                num_words = len(self.dictionary)

            topic_words = self.lsi_model.show_topics(self.config.num_topics,
                                                     num_words=num_words,
                                                     formatted=False)
            pickle.dump(topic_words, open(self.topic_top_words_file, 'wb'))
        else:
            topic_words = pickle.load(open(self.topic_top_words_file, 'rb'))
        return topic_words

    def get_topic_words_matrix(self):
        if not os.path.isfile(self.topic_words_matrix):
            lsi_model = self.get_model()
            topic_words = lsi_model.get_topics()  # K X V' (num_topics x selected_vocab_size)
            topic_word_matrix = np.zeros(
                (self.config.num_topics, self.tokenizer.tokenizer.vocab_size))  # K x V (num_topics x vocab_size)

            for i in range(len(self.dictionary)):
                j = self.tokenizer.tokenizer.convert_tokens_to_ids(lsi_model.id2word[i])
                topic_word_matrix[:, j] = topic_words[:, i]

            pickle.dump(topic_word_matrix, open(self.topic_words_matrix, 'wb'))
        else:
            topic_word_matrix = pickle.load(open(self.topic_words_matrix, 'rb'))
        return topic_word_matrix

    def get_topic_words_id(self):
        topic_words_id = self.lsi_model.get_topics()  # K X V (num_topics x num_words)
        pickle.dump(topic_words_id, open(self.config.topic_top_words_file, 'wb'))
        return topic_words_id


if __name__ == "__main__":
    config_file = "configs/alexa_lsi_config.json"
    lsi = LSIModel(config_file=config_file)
    #lsi._start()
    tw = lsi.get_topic_words(num_words=10)
    print([t[1] for t in tw])

    m = lsi.get_topic_words_matrix()
    print(m.shape)