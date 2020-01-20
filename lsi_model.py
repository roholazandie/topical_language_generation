from gensim import corpora
from collections import defaultdict
from gensim.models import LsiModel, TfidfModel
from gensim.corpora import Dictionary
from configs import LSIConfig
from topical_tokenizers import SpacyTokenizer, TransformerGPT2Tokenizer
from datasets.topical_dataset import TopicalDataset
import pickle


class LSIModel:

    def __init__(self, config_file):
        self.config = LSIConfig.from_json_file(config_file)
        if self.config.tokenizer == "spacy":
            self.tokenizer = SpacyTokenizer(self.config.cached_dir)
        elif self.config.tokenizer == "gpt":
            self.tokenizer = TransformerGPT2Tokenizer(self.config.cached_dir)
        self._start()

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

    def get_topic_words(self):
        topic_words = self.lsi_model.show_topics(self.config.num_topics,
                                                 num_words=len(self.dictionary), formatted=False)
        pickle.dump(topic_words, open(self.config.topic_top_words_file, 'wb'))
        return topic_words

    def get_topic_words_id(self):
        topic_words_id = self.lsi_model.get_topics()  # K X V (num_topics x num_words)
        pickle.dump(topic_words_id, open(self.config.topic_top_words_file, 'wb'))
        return topic_words_id


# config_file = "configs/alexa_lsi_config.json"
# config = LSIConfig.from_json_file(config_file)
#
# #tokenizer = TransformerGPT2Tokenizer(config.cached_dir)
# tokenizer = SpacyTokenizer(dict_dir=config.cached_dir, preprocess=True)
#
# topical_dataset = TopicalDataset(config.dataset_dir, tokenizer)
#
# docs = [doc for doc in topical_dataset]
#
#
# dictionary = Dictionary(docs)
# dictionary.filter_extremes(no_below=20, no_above=0.2)
# corpus = [dictionary.doc2bow(doc) for doc in docs]
#
# tfidf = TfidfModel(corpus)
#
# corpus_tfidf = tfidf[corpus]
#
# lsi_model = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=config.num_topics)
# corpus_lsi = lsi_model[corpus_tfidf]
#
# topic_words = lsi_model.show_topics(config.num_topics, num_words=len(dictionary), formatted=False)
# pickle.dump(topic_words, open(config.topic_top_words_file, 'wb'))
#
#
# topic_words_id = lsi_model.get_topics() # K X V (num_topics x num_words)
# pickle.dump(topic_words_id, open(config.topics_file, 'wb'))

if __name__ == "__main__":
    config_file = "configs/alexa_lsi_config.json"
    lsi = LSIModel(config_file=config_file)
    tw = lsi.get_topic_words()
    twi = lsi.get_topic_words_id()
    print(tw)
    print(twi)
