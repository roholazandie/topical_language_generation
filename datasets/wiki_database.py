from gensim import utils
from gensim.corpora.wikicorpus import filter_wiki, init_to_ignore_interrupt
import multiprocessing
#from topical_tokenizers import TransformerGPT2Tokenizer
import os
import json
from configs import DatabseConfig
from collections import Counter
from gensim.utils import ClippedCorpus
from gensim.models import LdaModel

from topical_tokenizers import TransformerGPT2Tokenizer
from transformers import GPT2TokenizerFast
from pymongo import MongoClient

IGNORED_NAMESPACES = [
    'Wikipedia', 'Category', 'File', 'Portal', 'Template',
    'MediaWiki', 'User', 'Help', 'Book', 'Draft', 'WikiProject',
    'Special', 'Talk'
]

ARTICLE_MIN_WORDS = 50


class WikiDatabase:

    def __init__(self, config_file, tokenizer):
        self.dbconfig = DatabseConfig.from_json_file(config_file)
        self.tokenizer = tokenizer #tokenizer_class.from_pretrained(model_name_or_path)
        client = MongoClient()
        self.db = client[self.dbconfig.database_name]


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
        yield text, tokens, title, page_id


    def populate_database(self,):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0

        texts = ((doc["text"], doc["title"], doc["id"]) for doc in self.extract_wiki_pages(self.dbconfig.dataset_dir))

        try:
            # process the corpus in smaller chunks of docs, because multiprocessing.Pool
            # is dumb and would load the entire input into RAM at once...
            for group in utils.chunkize(texts, chunksize=1000, maxsize=1):
                for g in group:
                    for text, tokens_ids, title, pageid in self.process_article(g):
                        articles_all += 1
                        positions_all += len(tokens_ids)
                        # article redirects and short stubs are pruned here
                        if len(tokens_ids) < ARTICLE_MIN_WORDS or any(
                                title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                            continue
                        articles += 1
                        positions += len(tokens_ids)
                        document = {"article": articles,
                                    "title": title,
                                    "text": text,
                                    "token_ids": tokens_ids,
                                    "pageid": pageid,
                                    }
                        self.db[self.dbconfig.collection_name].insert_one(document)

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
        for document in self.db[self.dbconfig.collection_name].find(): #todo: not sure this is working
            yield document

    def __getitem__(self, item):
        return self.db[self.dbconfig.collection_name].find_one({"article": item})



if __name__ == "__main__":
    config_file = "/home/rohola/codes/topical_language_generation/configs/wiki_database.json"
    cached_dir = "/home/rohola/codes/topical_language_generation/caches"
    tokenizer = TransformerGPT2Tokenizer(cached_dir)
    wiki_database = WikiDatabase(config_file, tokenizer)
    wiki_database.populate_database()

    #print(wiki_database[4])