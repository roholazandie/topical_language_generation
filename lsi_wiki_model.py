from gensim.utils import ClippedCorpus
from gensim.models import LsiModel, TfidfModel
from gensim.corpora import Dictionary, MmCorpus

from datasets.topical_dataset import TopicalDataset
from datasets.wikidata import WikiData, WikiCorpus
from datasets.wiki_database import WikiDatabase
import numpy as np
import os
import pickle
from configs import LSIWikiConfig
from topical_tokenizers import SpacyTokenizer, TransformerGPT2Tokenizer

class LSIModelWiki:

    def __init__(self, config_file):
        self.config = LSIWikiConfig.from_json_file(config_file)
        if self.config.tokenizer == "gpt":
            self.tokenizer = TransformerGPT2Tokenizer(self.config.cached_dir)

        self.wiki_dict_file = os.path.join(self.config.cached_dir, "wiki_dict")
        self.mm_corpus_file = os.path.join(self.config.cached_dir, "wiki_bow.mm")
        self.wiki_tfidf_file = os.path.join(self.config.cached_dir, "wiki_tfidf.mm")
        self.wiki_lsi_file = os.path.join(self.config.cached_dir, "wiki_lsi.mm")
        self.model_file = os.path.join(self.config.cached_dir, "lsi_model.p")
        self.topic_words_matrix_file = os.path.join(self.config.cached_dir, "topic_words_matrix.p")
        self.topic_top_words_file = os.path.join(self.config.cached_dir, "topic_top_words.p")


    def _create_files_db(self):
        config_file = "/home/rohola/codes/topical_language_generation/configs/wiki_database.json"
        wikidata = WikiDatabase(config_file, self.tokenizer)
        docs = []
        for i, tokens in enumerate(wikidata):
            docs.append(tokens)
            if i > 50000:
                break

        id2word_wiki = Dictionary(docs)
        id2word_wiki.filter_extremes(no_below=20, no_above=0.01)

        id2word_wiki.save(self.wiki_dict_file)

        wiki_corpus = WikiCorpus(docs, id2word_wiki)
        MmCorpus.serialize(self.mm_corpus_file, wiki_corpus)

    def _create_files1(self):
        dir = "/media/rohola/data/dialog_systems/alexa_prize_topical_chat_dataset/reading_sets/"
        wikidata = TopicalDataset(dir, self.tokenizer)
        doc_stream = (tokens for tokens in wikidata)

        id2word_wiki = Dictionary(doc_stream)
        id2word_wiki.filter_extremes(no_below=20, no_above=0.2)

        id2word_wiki.save(self.wiki_dict_file)

        wiki_corpus = WikiCorpus(wikidata, id2word_wiki)
        MmCorpus.serialize(self.mm_corpus_file, wiki_corpus)

    def _create_files(self):
        wikidata = WikiData(self.config.dataset_dir, self.tokenizer)
        doc_stream = (tokens for tokens in wikidata)

        id2word_wiki = Dictionary(doc_stream)
        id2word_wiki.filter_extremes(no_below=20, no_above=0.1)

        id2word_wiki.save(self.wiki_dict_file)

        wiki_corpus = WikiCorpus(wikidata, id2word_wiki)
        MmCorpus.serialize(self.mm_corpus_file, wiki_corpus)

    def _run_model(self):
        id2word_wiki = Dictionary.load(self.wiki_dict_file)
        mm_corpus = MmCorpus(self.mm_corpus_file)

        #to be removed
        #mm_corpus = ClippedCorpus(mm_corpus, 4000)

        tfidf_model = TfidfModel(mm_corpus, id2word=id2word_wiki)

        corpus = tfidf_model[mm_corpus]
        MmCorpus.serialize(self.wiki_tfidf_file, corpus)

        self.model = LsiModel(corpus,
                              num_topics=self.config.num_topics,
                              id2word=id2word_wiki,
                              chunksize=self.config.chunksize)

        MmCorpus.serialize(self.wiki_lsi_file, self.model[corpus])
        self.model.save(self.model_file)

    def get_model(self):
        return LsiModel.load(self.model_file)

    def get_topic_words_matrix(self):
        if not os.path.isfile(self.topic_words_matrix_file):
            lsi_model = self.get_model()
            topic_words = lsi_model.get_topics()  # K X V' (num_topics x selected_vocab_size)
            topic_word_matrix = np.zeros(
                (self.config.num_topics, self.tokenizer.tokenizer.vocab_size))  # K x V (num_topics x vocab_size)

            for i in range(len(lsi_model.id2word)):
                j = self.tokenizer.tokenizer.convert_tokens_to_ids(lsi_model.id2word[i])
                topic_word_matrix[:, j] = topic_words[:, i]

            pickle.dump(topic_word_matrix, open(self.topic_words_matrix_file, 'wb'))
        else:
            topic_word_matrix = pickle.load(open(self.topic_words_matrix_file, 'rb'))
        return topic_word_matrix


    def get_topic_words(self, num_words=None):
        if not os.path.isfile(self.topic_top_words_file):
            if not num_words:
                num_words = len(self.dictionary)

            model = self.get_model()
            topic_words = model.show_topics(self.config.num_topics,
                                                     num_words=num_words,
                                                     formatted=False)
            pickle.dump(topic_words, open(self.topic_top_words_file, 'wb'))
        else:
            topic_words = pickle.load(open(self.topic_top_words_file, 'rb'))
        return topic_words


if __name__ == "__main__":
    config_file = "/home/rohola/codes/topical_language_generation/configs/wiki_lsi_config.json"
    lsi_model_wiki = LSIModelWiki(config_file)
    lsi_model_wiki._create_files_db()
    #lsi_model_wiki._create_files1()
    lsi_model_wiki._run_model()
    #m = lsi_model_wiki.get_model()
    #tw = lsi_model_wiki.get_topic_words_matrix()
    twords = lsi_model_wiki.get_topic_words(10)
    topic_words =[t[1] for t in twords]
    for topic in topic_words:
        print(topic)