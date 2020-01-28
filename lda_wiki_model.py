from gensim.utils import ClippedCorpus
from gensim.models import LdaModel
from gensim.corpora import Dictionary, MmCorpus

from datasets.wikidata import WikiData, WikiCorpus
import numpy as np
import os
import pickle
from configs import LDAConfig, LDAWikiConfig
from topical_tokenizers import SpacyTokenizer, TransformerGPT2Tokenizer

class LDAModelWiki:

    def __init__(self, config_file):
        self.config = LDAWikiConfig.from_json_file(config_file)
        if self.config.tokenizer == "gpt":
            self.tokenizer = TransformerGPT2Tokenizer(self.config.cached_dir)

        self.wiki_dict_file = os.path.join(self.config.cached_dir, "wiki_dict")
        self.mm_corpus_file = os.path.join(self.config.cached_dir, "wiki_bow.mm")
        self.model_file = os.path.join(self.config.cached_dir, "lda_model.p")

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

        clipped_corpus = ClippedCorpus(mm_corpus, 4000)

        self.model = LdaModel(clipped_corpus,
                             num_topics=self.config.num_topics,
                             id2word=id2word_wiki,
                             alpha=self.config.alpha,
                             chunksize=self.config.chunksize,
                             iterations=self.config.iterations,
                             passes=self.config.passes
                             )

        self.model.save(self.model_file)

    def get_model(self):
        model_file = os.path.join(self.config.cached_dir, "model.p")
        return LdaModel.load(model_file)


    def get_psi_matrix(self):
        psi_matrix_file = os.path.join(self.config.cached_dir, "psi_matrix.p")
        if not os.path.isfile(psi_matrix_file):
            model = self.get_model()
            id2word_wiki = Dictionary.load(self.wiki_dict_file)
            temp = id2word_wiki[0]
            id2token = id2word_wiki.id2token
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
    #lda_model_wiki._run_model()
    m = lda_model_wiki.get_model()
    lda_model_wiki.get_psi_matrix()
