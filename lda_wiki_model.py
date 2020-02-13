from gensim.utils import ClippedCorpus
from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus

from datasets.wiki_database import WikiDatabase
from datasets.wikidata import WikiData, WikiCorpus
import numpy as np
import os
import pickle
from configs import LDAConfig, LDAWikiConfig
from topical_tokenizers import SpacyTokenizer, TransformerGPT2Tokenizer, SimpleTokenizer


class LDAModelWiki:

    def __init__(self, config_file):
        self.config = LDAWikiConfig.from_json_file(config_file)
        if self.config.tokenizer == "gpt":
            self.tokenizer = TransformerGPT2Tokenizer(self.config.cached_dir)
        elif self.config.tokenizer == "simple":
            self.tokenizer = SimpleTokenizer(self.config.cached_dir)

        self.wiki_dict_file = os.path.join(self.config.cached_dir, "wiki_dict")
        self.mm_corpus_file = os.path.join(self.config.cached_dir, "wiki_bow.mm")
        self.model_file = os.path.join(self.config.cached_dir, "lda_model.p")
        self.all_topic_tokens_file = os.path.join(self.config.cached_dir, "all_topic_tokens.p")

    def _create_files_db(self):
        config_file = "/home/rohola/codes/topical_language_generation/configs/wiki_database.json"
        wikidata = WikiDatabase(config_file, self.tokenizer)
        docs = []
        for i, tokens in enumerate(wikidata):
            docs.append(tokens)
            if i > 50000:
                break

        dictionary_wiki = Dictionary(docs)
        dictionary_wiki.filter_extremes(no_below=100, no_above=0.01)

        dictionary_wiki.save(self.wiki_dict_file)

        wiki_corpus = WikiCorpus(docs, dictionary_wiki)
        MmCorpus.serialize(self.mm_corpus_file, wiki_corpus)

    def _create_files(self):
        wikidata = WikiData(self.config.dataset_dir, self.tokenizer)
        doc_stream = (tokens for tokens in wikidata)
        dictionary_wiki = Dictionary(doc_stream)
        dictionary_wiki.filter_extremes(no_below=20, no_above=0.1)

        dictionary_wiki.save(self.wiki_dict_file)

        wiki_corpus = WikiCorpus(wikidata, dictionary_wiki)
        MmCorpus.serialize(self.mm_corpus_file, wiki_corpus)

    def _run_model(self):
        id2word_wiki = Dictionary.load(self.wiki_dict_file)
        mm_corpus = MmCorpus(self.mm_corpus_file)

        #mm_corpus = ClippedCorpus(mm_corpus, 4000)

        self.model = LdaModel(mm_corpus,
                             num_topics=self.config.num_topics,
                             id2word=id2word_wiki,
                             alpha=self.config.alpha,
                             chunksize=self.config.chunksize,
                             iterations=self.config.iterations,
                             passes=self.config.passes
                             )

        self.model.save(self.model_file)

    def get_model(self):
        return LdaModel.load(self.model_file)

    def get_all_topic_tokens(self):
        if not os.path.isfile(self.all_topic_tokens_file):
            model = self.get_model()
            dictionary_wiki = Dictionary.load(self.wiki_dict_file)
            temp = dictionary_wiki[0]
            id2token = dictionary_wiki.id2token
            tokenid_probs = [model.get_topic_terms(i, len(id2token)) for i in range(self.config.num_topics)]
            all_topic_tokens = [[(id2token[i], p) for i, p in tokenid_probs_topic] for tokenid_probs_topic in
                                tokenid_probs]
            pickle.dump(all_topic_tokens, open(self.all_topic_tokens_file, 'wb'))
        else:
            all_topic_tokens = pickle.load(open(self.all_topic_tokens_file, 'rb'))

        return all_topic_tokens


    def get_psi_matrix(self):
        psi_matrix_file = os.path.join(self.config.cached_dir, "psi_matrix.p")
        if not os.path.isfile(psi_matrix_file):
            try:
                model = self.model
            except:
                model = self.get_model()

            id2token_wiki = Dictionary.load(self.wiki_dict_file)
            temp = id2token_wiki[0]
            id2token = id2token_wiki.id2token
            psi_matrix = np.zeros((self.config.num_topics, self.tokenizer.tokenizer.vocab_size))
            matrix = model.get_topics()  # a matrix of k x V' (num_topic x selected_vocab_size)
            for i in range(len(id2token)):
                j = self.tokenizer.tokenizer.convert_tokens_to_ids(id2token[i])
                psi_matrix[:, j] = matrix[:, i]

            pickle.dump(psi_matrix, open(psi_matrix_file, 'wb'))
        else:
            psi_matrix = pickle.load(open(psi_matrix_file, 'rb'))

        return psi_matrix


if __name__ == "__main__":
    config_file = "/home/rohola/codes/topical_language_generation/configs/wiki_lda_config.json"
    lda_model_wiki = LDAModelWiki(config_file)
    #lda_model_wiki._create_files_db()
    #lda_model_wiki._create_files()
    #lda_model_wiki._run_model()
    #m = lda_model_wiki.get_model()
    #lda_model_wiki.get_psi_matrix()
    topic_tokens = lda_model_wiki.get_all_topic_tokens()
    for tt in topic_tokens:
        print(tt)
