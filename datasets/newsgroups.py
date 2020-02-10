from sklearn.datasets import fetch_20newsgroups
import re
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.utils import simple_preprocess

from topical_tokenizers import TransformerGPT2Tokenizer, SpacyTokenizer


class NewsgroupDataset:

    def __init__(self, dataset_dir, tokenizer, do_tokenize=True):
        self.tokenizer = tokenizer
        self.do_tokenize = do_tokenize
        newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
        newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True)
        self.newsgroups = newsgroups_train.data + newsgroups_test.data
        self.newsgroups = [re.sub('\S*@\S*\s?', '', sent) for sent in self.newsgroups]
        # Remove new line characters
        self.newsgroups = [re.sub('\s+', ' ', sent) for sent in self.newsgroups]
        # Remove distracting single quotes
        self.newsgroups = [re.sub("\'", "", sent) for sent in self.newsgroups]
        self.stop_words = list(STOP_WORDS)


    def _process_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    # def _process_text(self, text):
    #     return text.split()


    def __iter__(self):
        for content in self.newsgroups:
            content = " ".join([token for token in simple_preprocess(str(content), deacc=True) if token not in self.stop_words])
            if self.do_tokenize:
                yield self._process_text(content)
            else:
                yield content



if __name__ == "__main__":
    cached_dir = "/home/rohola/cached_models"
    #tokenizer = TransformerGPT2Tokenizer(cached_dir)
    tokenizer = SpacyTokenizer(dict_dir="")
    news_dataset = NewsgroupDataset("", tokenizer)
    for article in news_dataset:
        print(article)
