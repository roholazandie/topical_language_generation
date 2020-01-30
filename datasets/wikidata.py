from gensim import utils
from gensim.corpora.wikicorpus import filter_wiki, init_to_ignore_interrupt
import multiprocessing
#from topical_tokenizers import TransformerGPT2Tokenizer
import os
import json
from collections import Counter
from gensim.utils import ClippedCorpus
from gensim.models import LdaModel

from transformers import GPT2TokenizerFast
from transformers import GPT2Tokenizer


IGNORED_NAMESPACES = [
    'Wikipedia', 'Category', 'File', 'Portal', 'Template',
    'MediaWiki', 'User', 'Help', 'Book', 'Draft', 'WikiProject',
    'Special', 'Talk'
]

ARTICLE_MIN_WORDS = 50


class WikiData:

    def __init__(self, dirname, tokenizer):
        self.tokenizer = tokenizer #tokenizer_class.from_pretrained(model_name_or_path)
        self.dirname = dirname

    def extract_wiki_pages(self, json_path):
        for root, dirs, files in os.walk(json_path):
            for file in files:
                with open(root + "/" + file) as fr:
                    text = fr.read()
                    json_text = text.replace("}", "},")
                    json_text = json_text[::-1].replace(",", "", 1)[::-1]
                    json_text = "[" + json_text + "]"
                    docs = json.loads(json_text)
                    for doc in docs:
                        yield doc

    def process_article(self, args):
        text, title, page_id = args[0], args[1], args[2]
        text = filter_wiki(text)
        #token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = self.tokenizer.tokenize(text)
        yield tokens, title, page_id


    def wiki_token_ids(self, dirname):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0

        texts = ((doc["text"], doc["title"], doc["id"]) for doc in self.extract_wiki_pages(dirname))


        try:
            # process the corpus in smaller chunks of docs, because multiprocessing.Pool
            # is dumb and would load the entire input into RAM at once...
            for group in utils.chunkize(texts, chunksize=1000, maxsize=1):
                for g in group:
                    for tokens_ids, title, pageid in self.process_article(g):
                        articles_all += 1
                        positions_all += len(tokens_ids)
                        # article redirects and short stubs are pruned here
                        if len(tokens_ids) < ARTICLE_MIN_WORDS or any(
                                title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                            continue
                        articles += 1
                        positions += len(tokens_ids)
                        yield tokens_ids

        except KeyboardInterrupt:
            print(
                "user terminated iteration over Wikipedia corpus after %i documents with %i positions "
                "(total %i articles, %i positions before pruning articles shorter than %i words)",
                articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
            )
        else:
            print(
                "finished iterating over Wikipedia corpus of %i documents with %i positions "
                "(total %i articles, %i positions before pruning articles shorter than %i words)",
                articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS
            )
            length = articles  # cache corpus length



    def __iter__(self):
        # for ids in self.wiki_token_ids(self.dirname):
        #     yield list(dict(Counter(ids)).items())
        for ids in self.wiki_token_ids(self.dirname):
            yield ids


class WikiCorpus:

    def __init__(self, wikidata, dictionary):
        self.wikidata = wikidata
        self.dictionary = dictionary

    def __iter__(self):
        for tokens in self.wikidata:
            yield self.dictionary.doc2bow(tokens)


if __name__ == "__main__":
    from gensim.corpora import Dictionary, MmCorpus
    import time

    # t1 = time.time()
    # dirname = "/media/rohola/data/clean_wiki_text_json"
    # wikidata = WikiData(dirname)
    #
    # doc_stream = (tokens for tokens in wikidata)
    # id2word_wiki = Dictionary(doc_stream)
    # id2word_wiki.filter_extremes(no_below=20, no_above=0.1)
    # id2word_wiki.save('/home/rohola/codes/topical_language_generation/caches/wiki_dict')
    #
    # #id2word_wiki = id2word_wiki.load('/home/rohola/codes/topical_language_generation/caches/wiki_dict')
    #
    #
    # wiki_corpus = WikiCorpus(wikidata, id2word_wiki)
    # MmCorpus.serialize('/home/rohola/codes/topical_language_generation/caches/wiki_bow.mm', wiki_corpus)

    # t2 = time.time()
    # print(t2-t1)

    id2word_wiki = Dictionary.load('/home/rohola/codes/topical_language_generation/caches/wiki_cache/wiki_dict')
    temp = id2word_wiki[0]

    mm_corpus = MmCorpus('/home/rohola/codes/topical_language_generation/caches/wiki_cache/wiki_bow.mm')
    #print(mm_corpus)
    #print(next(iter(mm_corpus)))

    clipped_corpus = ClippedCorpus(mm_corpus, 4000)
    lda_model = LdaModel(clipped_corpus,
                         num_topics=10,
                         id2word=id2word_wiki,
                         alpha=0.001,
                         chunksize=200,
                         iterations=400,
                         passes=10
                         )
    s = lda_model.print_topics(10)
    #lda_model.top_topics()
    print(s)